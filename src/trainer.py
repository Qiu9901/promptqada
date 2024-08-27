import os, shutil
import sys
import json
import torch
from torch import nn
from torch.nn.parallel.data_parallel import DataParallel
import datasets
from tqdm import tqdm
import dill
import warnings

from typing import *

try:
    from typing import OrderedDict
except ImportError:
    from collections import OrderedDict
from .utilis import normalize_answer,compute_f1
from .modeling import PromptForClassification,PromptForGeneration
from .utilis import PromptDataLoader,postprocess_preds,postprocess_actuals,get_metrics,postprocess_preds_t5
from .config_utilis import AllConfig
from openprompt.utils.logging import logger
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim import Adam

import nltk


class BaseRunner(object):
    def __init__(self,
                 model: [PromptForClassification,PromptForGeneration, nn.Module],
                 config: AllConfig = None,
                 train_dataloader: Optional[PromptDataLoader] = None,
                 valid_dataloader: Optional[PromptDataLoader] = None,
                 test_dataloader: Optional[PromptDataLoader] = None):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

        self.model.to(config.TrainingConfig.device)

        self.cur_epoch = 0
        self.best_score = None
        self.global_step = 0
        self.best_dev_score = None
        self.best_test_score = None

    def set_stop_criterion(self):
        """Total training steps, either controlled by num_training_steps or num_epochs"""
        if self.config.TrainingConfig.max_train_steps is not None:
            if self.config.TrainingConfig.num_train_epochs is not None:
                logger.warning("num_training_steps set explicitly, num_epochs is not in use.")
            self.num_training_steps = self.config.TrainingConfig.max_train_steps
            self.num_epochs = 10000
        else:
            if self.config.TrainingConfig.num_train_epochs is None:
                raise RuntimeError("At least num_training_steps & num_epochs should be specified.")
            self.num_training_steps = self.steps_per_epoch * self.config.TrainingConfig.num_train_epochs
            self.num_epochs = self.config.TrainingConfig.num_train_epochs

    @property
    def steps_per_epoch(self):
        batches = len(self.train_dataloader)
        return batches // self.config.TrainingConfig.gradient_accumulation_steps

    @property
    def inner_model(self):
        return self.model.module if isinstance(self.model, DataParallel) else self.model

    def configure_optimizer(self):
        """config the optimizer and scheduler for

        1. model

        2. template

        3. verbalizer(optional)
        """
        optimizers, schedulers = [], []

        if not self.config.TrainingConfig.freeze_plm:
            no_decay = ["bias", "LayerNorm.weight"]
            weight_decay = self.config.TrainingConfig.weight_decay
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.inner_model.plm.named_parameters() if
                            not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in self.inner_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
            plm_optimizer = Adam(
                optimizer_grouped_parameters, lr=self.config.TrainingConfig.plm_learning_rate
            )
            optimizers.append(plm_optimizer)
            schedulers.append(get_linear_schedule_with_warmup(plm_optimizer,
                                                              num_warmup_steps=self.config.TrainingConfig.num_warmup_steps,
                                                              num_training_steps=self.num_training_steps))

        if self.config.TrainingConfig.template_learning_rate is not None:
            optimizer_grouped_parameters = [
                {'params': [p for name, p in self.inner_model.template.named_parameters() if
                            'raw_embedding' not in name],'weight_decay': 0.001}
            ]
            # adam
            template_optimizer=Adam(optimizer_grouped_parameters,lr=self.config.TrainingConfig.template_learning_rate)
            optimizers.append(template_optimizer)
            template_scheduler = get_linear_schedule_with_warmup(
                template_optimizer,
                num_warmup_steps=0,
                num_training_steps=self.num_training_steps
            )
            schedulers.append(template_scheduler)

        # TODO: Add template and verbalizer optimizer/scheduler
        self.optimizers = optimizers
        self.schedulers = schedulers

    def save_metric(self, validation_score: float, test_score: float, mode="steps", final=False):
        metric_path = os.path.join(self.config.TrainingConfig.output_dir, "metric.csv")
        if not os.path.exists(metric_path):
            with open(metric_path, "w") as fout:
                fout.write(f"mode,{mode},acc\n")

        if final:
            with open(metric_path, "a") as fout:
                fout.write('{},{},{}\n'.format("best_dev", self.global_step, validation_score))
                fout.write('{},{},{}\n'.format("best_test", self.global_step, test_score))
        else:
            with open(metric_path, "a") as fout:
                fout.write('{},{},{}\n'.format("dev", self.global_step, validation_score))
                fout.write('{},{},{}\n'.format("test", self.global_step, test_score))

    def inference_epoch(self, split: str):
        outputs = []
        self.model.eval()
        with torch.no_grad():
            data_loader = self.valid_dataloader if split == "validation" else self.test_dataloader
            for batch_idx, batch in enumerate(tqdm(data_loader, desc=split)):
                batch = batch.to(self.config.TrainingConfig.device).to_dict()
                outputs.append(self.inference_step(batch, batch_idx))

        metrics = self.inference_epoch_end(split, outputs)

        return metrics

    def training_epoch(self, epoch):
        self.model.train()
        self.model.zero_grad()

        total_loss = 0.0
        sum_loss = 0.0


        with tqdm(total=self.steps_per_epoch, desc=f"train epoch: {epoch}") as pbar:

            for batch_idx, batch in enumerate(self.train_dataloader):
                batch = batch.to(self.config.TrainingConfig.device).to_dict()
                # step need special trainer to rewrite
                loss = self.training_step(batch, batch_idx)

                if self.config.TrainingConfig.gradient_accumulation_steps > 1:
                    loss = loss / self.config.TrainingConfig.gradient_accumulation_steps

                sum_loss += loss.item()
                loss.backward()
                if (batch_idx + 1) % self.config.TrainingConfig.gradient_accumulation_steps == 0:
                    pbar.set_postfix({'loss': sum_loss})

                    if self.config.TrainingConfig.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.TrainingConfig.max_grad_norm)

                    for optimizer in self.optimizers:
                        optimizer.step()
                    for scheduler in self.schedulers:
                        scheduler.step()
                    for optimizer in self.optimizers:
                        optimizer.zero_grad()

                    total_loss += sum_loss
                    sum_loss = 0
                    self.global_step += 1
                    pbar.update(1)

                if self.config.TrainingConfig.num_dev_steps:
                    if self.global_step % self.config.TrainingConfig.num_dev_steps == 0:
                        val_score = self.inference_epoch("validation")
                        logger.info(
                            f"training epoch: {self.cur_epoch}, validation score: F1: {val_score['F1']} EM: {val_score['EM']}")

                        self.save_metric(val_score, mode=f"steps_{self.global_step}", split='val')

                        if self.best_dev_score == None:
                            self.best_dev_score = val_score

                        if val_score['F1'] >= self.best_dev_score['F1']:
                            self.best_dev_score = val_score
                            self.save_checkpoint(ckpt='PromptGeneration')

                        logger.info(
                            f"training steps: {self.global_step}, validation score: {val_score['F1']}")

            if self.global_step >= self.num_training_steps:
                logger.info(
                    f"Training epoch {epoch}, num_steps {self.global_step}, avg_loss: {total_loss / self.steps_per_epoch:.4f}, total_loss: {total_loss:.4f}")
                return -1  # an indicator of stopping the training
        if self.config.TrainingConfig.max_train_steps is None:
            logger.info(
                f"Training epoch {epoch}, num_steps {self.global_step},  avg_loss: {total_loss / self.steps_per_epoch:.4f}, total_loss: {total_loss:.4f}")
        return 1

    def on_fit_start(self):
        """Some initialization works"""
        pass

    def checkpoint_path(self, ckpt: str) -> str:
        # TODO: fix path
        return os.path.join(os.path.join(self.config.TrainingConfig.output_dir), f'{ckpt}.ckpt')

    def save_checkpoint(self, ckpt: str, save_state=False, extra: dict = {}, copy: str = None):
        # if self.clean: return
        logger.info(f"Saving checkpoint {self.checkpoint_path(ckpt)}...")
        state_dict = {
            "state_dict": self.inner_model.state_dict(),
        }
        state_dict.update(extra)

        if save_state:
            state_dict["optimizer"] = [opt.state_dict() if isinstance(opt, torch.optim.Optimizer) else None for opt in
                                       self.optimizers]
            with warnings.catch_warnings(record=True):
                state_dict["scheduler"] = [
                    sch.state_dict() if isinstance(sch, torch.optim.lr_scheduler._LRScheduler) else None for sch in
                    self.schedulers]

            state_dict.update({
                "cur_epoch": self.cur_epoch,
                "best_score": self.best_score,
                "global_step": self.global_step,
            })
        torch.save(state_dict, self.checkpoint_path(ckpt), pickle_module=dill)
        if copy:
            logger.info(f"Copying checkpoint {self.checkpoint_path(ckpt)} to {self.checkpoint_path(copy)}...")
            shutil.copyfile(self.checkpoint_path(ckpt), self.checkpoint_path(copy))
        logger.info(f"Save Checkpoint finished")

    def load_checkpoint(self, ckpt: str, load_state=False) -> bool:
        logger.info(f"Loading Checkpoint {self.checkpoint_path(ckpt)}...")
        try:
            state_dict = torch.load(self.checkpoint_path(ckpt), pickle_module=dill, map_location="cpu")
        except FileNotFoundError:
            logger.warning(f"Checkpoint {self.checkpoint_path(ckpt)} not found")
            return False

        # load state to model
        self.model = self.inner_model
        self.model.load_state_dict(state_dict['state_dict'])

        if load_state:
            # load state to optimizers
            for optimizer, op_state in zip(self.optimizers, state_dict['optimizer']):
                if isinstance(optimizer, torch.optim.Optimizer):
                    optimizer.load_state_dict(op_state)
            for scheduler, sc_state in zip(self.schedulers, state_dict['scheduler']):
                if isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):
                    with warnings.catch_warnings(record=True):
                        scheduler.load_state_dict(sc_state)

            # load training state
            self.cur_epoch = state_dict['cur_epoch'] + 1
            self.best_score = state_dict['best_score']
            self.global_step = state_dict['global_step']
        logger.info(f"Load Checkpoint finished")
        return True

    def fit(self, ckpt: Optional[str] = None,augment_answer: Optional[bool] = False,augment_question:Optional[bool]=False):
        self.set_stop_criterion()
        self.configure_optimizers()

        if ckpt:
            if not self.load_checkpoint(ckpt):
                logger.warning("Train from scratch instead ...")
        if self.cur_epoch == 0:
            self.on_fit_start()

        for self.cur_epoch in range(self.cur_epoch, self.num_epochs):
            continue_training = self.training_epoch(self.cur_epoch)

            score = self.inference_epoch("validation")
            copy = None
            if self.best_score is None or ((score - self.best_score) >= 0) == self.config.checkpoint.higher_better:
                copy = 'best'
                self.best_score = score
            self.save_checkpoint('last', extra={"validation_metric": score}, copy=copy)
            if continue_training == -1:
                logger.info("Stop training by reaching maximum num_training_steps")
                break

        return self.best_score

    def test(self, ckpt: Optional[str] = None) -> dict:
        if ckpt:
            if not self.load_checkpoint(ckpt, load_state=False):
                logger.error("Test cannot be performed")
                exit()
        return self.inference_epoch("test")

    def run(self, ckpt: Optional[str] = None,augment_answer: Optional[bool] = False,augment_question:Optional[bool]=False) -> dict:
        res = self.fit(ckpt=ckpt,augment_answer=augment_answer,augment_question=augment_question)
        return res

    def save_results(self, split, results: dict):
        for name, values in results.items():
            file_name = os.path.join(self.config.TrainingConfig.output_dir, f"{split}_{name}.txt")
            with open(file_name, 'w') as fout:
                for value in values:
                    print(value, file=fout)


class QuestionAnswerRunner(BaseRunner):
    def __init__(self,
                 model: [PromptForGeneration,nn.Module],
                 config= AllConfig,
                 train_dataloader: Optional[PromptDataLoader]=None,
                 valid_dataloader: Optional[PromptDataLoader]=None,
                 test_dataloader: Optional[PromptDataLoader]=None,
                 generate_dataloader:Optional[PromptDataLoader]=None,
                 pretrain_dataloader: Optional[PromptDataLoader] = None,
                 save_ckp:Optional[bool]=True,
                 whether_save_metric:Optional[bool]=True,
                 whether_test:Optional[bool]=True,
                 whether_validation: Optional[bool] = True,
                 whether_filter_data: Optional[bool] = False,
                 question_augment: Optional[bool]=False,
                 answer_augment: Optional[bool]=False,
                 del_ckp: Optional[bool]=True,
                 loss_function: Optional[Callable] = None,
                 after: Optional[bool]=False):
        super().__init__(model=model,
                         config=config,
                         train_dataloader=train_dataloader,
                         valid_dataloader=valid_dataloader,
                         test_dataloader=test_dataloader,
                         )
        self.del_ckp=del_ckp
        self.question_augment=question_augment
        self.answer_augment=answer_augment
        self.whether_filter_data=whether_filter_data
        self.whether_test=whether_test
        self.whether_save_metric=whether_save_metric
        self.whether_validation=whether_validation
        self.save_ckp=save_ckp
        self.generate_dataloader=generate_dataloader
        self.pretrain_dataloader=pretrain_dataloader
        self.after=after
        # TODO: other loss function
        self.loss_function = nn.CrossEntropyLoss()


    def save_metric(self, score: dict,mode="steps",split="val"):
        metric_path = os.path.join(self.config.TrainingConfig.output_dir, "metric.csv")
        if not os.path.exists(metric_path):
            with open(metric_path, "w") as fout:
                fout.write(f"{mode},F1,EM\n")

        with open(metric_path, "a") as fout:
            fout.write('{},{},{}\n'.format(mode,score['F1'],score['EM']))



    def inference_step(self,batch,batch_idx,num_beam=1,num_return_sequences=1):
        act_anwers=batch['answer']
        if self.question_augment:
            act_anwers=batch['question']
        if "bart" in self.config.ModelConfig.model_name_or_path:
            max_length=50
        elif "t5" in self.config.ModelConfig.model_name_or_path:
            max_length=25
        else:
            raise "This type mode don't implementation"
        generated_ids = self.model.generate(batch,
                                            max_length=max_length,
                                            num_beams=num_beam,
                                            early_stopping=True,
                                            num_return_sequences=num_return_sequences,)
        preds = [self.train_dataloader.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                 for g in
                 generated_ids]
        return preds,act_anwers


    def filter_question(self,question_dict:dict):
        """
        filter don't full question format example
        :param question_dict: {"train_0_0":[{"question":...,"answer":...,"context":....},]
        :return:
        """
        question_words = ["what ", "why ", "when ", "where ",
                          "name ", "is ", "how ", "do ", "does ",
                          "which ", "are ", "could ", "would ",
                          "should ", "has ", "have ", "whom ", "whose ", "don't "]
        result={}
        for key in question_dict.keys():
            questions=question_dict[key]
            # filter the same question
            temp=[x["question"] for x in questions]
            temp_answer=[x["answer"] for x in questions]
            flags=[False]*len(temp)
            # filter same question
            for i in range(len(temp)):
                for j in range(i+1,len(temp)):
                    if temp[i]==temp[j]:
                        flags[j]=True

            for index,question_context in enumerate(questions):
                if flags[index]:
                    continue
                q=question_context["question"]
                temp = q.lower()
                # temp = word_tokenize(temp)
                # filter answer in question
                if temp_answer[index].lower() in temp:
                    continue

                if len(temp)>0:
                    if any(x in temp for x in question_words) and '?' in temp:
                        if key in result.keys():
                            result[key].append({"question":q,"answer":temp_answer[index],"context":question_context["context"]})
                        else:
                            result[key]=[{"question":q,"answer":temp_answer[index],"context":question_context["context"]}]

        return result

    def filter_answer(self,answer_dict:dict):
        """
        filter don't full question format example
        :param answer_dict: {"train_0_0":[{"question":...,"answer":...,"context":....},]
        :return:
        """
        result = {}
        for key in answer_dict.keys():
            answer=answer_dict[key]
            for question_context in answer:
                a=question_context["answer"]
                if a in question_context["context"]:
                    if key in result.keys():
                        result[key].append({"question":question_context["question"],"answer":a,"context":question_context["context"]})
                    else:
                        result[key]=[{"question":question_context["question"],"answer":a,"context":question_context["context"]}]
        return result


    def save_generate_question_answer(self,question_answer_dic:dict,only_question=True,filter=False):
        """
        save QA to file
        :param question_answer_dic:{"train_0_0",[{},{}]
        :return:
        """
        if only_question:
            output_dir=os.path.join("mrqa-few-shot-augment-question",self.config.DataConfig.task_name)
        else:
            output_dir = os.path.join("mrqa-few-shot-augment-answer", self.config.DataConfig.task_name)

        if filter:
            output_dir = os.path.join("mrqa-few-shot-augment-data", self.config.DataConfig.task_name)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        output_file = os.path.join(output_dir, f'generate-{self.config.DataConfig.task_name}-train-seed-{self.config.DataConfig.seed}-num-examples-{self.config.DataConfig.k_shot}.jsonl')
        if self.after==False:
            if os.path.exists(output_file):
                os.remove(output_file)
                logger.info(f"Successful move question/answer in {output_file}")


            head = [{"header": {"dataset": self.config.DataConfig.task_name, "split": "augment train data"}}]
            with open(output_file, 'a') as fout:
                fout.write(str(head) + '\r\n')

        count=0
        for key in question_answer_dic.keys():
            count+=len(question_answer_dic[key])
            for question_answer_context in question_answer_dic[key]:
                paragraph = {}
                paragraph["guid"]=key
                paragraph["context"] = question_answer_context["context"]
                paragraph["qas"] = []
                paragraph["qas"].append({ "question": question_answer_context["question"],"answers": [question_answer_context["answer"]],
                                          })

                with open(output_file, 'a') as fout:
                    fout.write(json.dumps(paragraph))
                    fout.write('\r\n')
        if self.after or filter:
            logger.info(f"Generate totally question number {count}.")
        return output_file


    def generate_question(self,num_beam=1,num_return_sequences=1):
        # todo: one question can generate a lot of answers
        if "bart" in self.config.ModelConfig.model_name_or_path:
            max_length=50
        elif "t5" in self.config.ModelConfig.model_name_or_path:
            max_length=25
        else:
            raise "This type mode don't implementation"

        outputs={}
        self.model.eval()
        with torch.no_grad():

            for batch_idx, batch in enumerate(tqdm(self.generate_dataloader, desc="generate question...")):
                batch = batch.to(self.config.TrainingConfig.device).to_dict()
                generated_ids = self.model.generate(batch,
                                                max_length=max_length,
                                                num_beams=num_beam,
                                                early_stopping=True,
                                                num_return_sequences=num_return_sequences,
                                                generate_question=True
                                                )

                preds = [self.train_dataloader.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                            for g in
                            generated_ids]
                train_ids_list=batch["guid"]
                for start in range(0,len(preds),num_return_sequences):
                    pred=preds[start:start+num_return_sequences]
                    answer=batch["answer"][start // num_return_sequences]
                    for i,question_with_answer in enumerate(pred):
                        if "Question:" in question_with_answer:
                            pred[i]={"question":question_with_answer.split('Question:')[1].split('Answer')[0].strip(),
                                     "answer":answer,"context":batch["context"][start//num_return_sequences]}

                        else:
                            pred[i] = {"question": "",
                                        "answer": answer,"context":batch["context"][start//num_return_sequences]}
                    outputs[train_ids_list[start//num_return_sequences]]=pred

        # {"train_0_0":[{question:,answer:}],}
        # outputs=self.filter_question(outputs)
        output_file=self.save_generate_question_answer(outputs,only_question=True)
        return output_file

    def generate_answer(self,  num_beam=1, num_return_sequences=1):
        if "bart" in self.config.ModelConfig.model_name_or_path:
            max_length = 50
        elif "t5" in self.config.ModelConfig.model_name_or_path:
            max_length = 25
        else:
            raise "This type mode don't implementation"

        outputs = {}
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.generate_dataloader, desc="generate answer...")):
                batch = batch.to(self.config.TrainingConfig.device).to_dict()
                generated_ids = self.model.generate(batch,
                                                    max_length=max_length,
                                                    num_beams=num_beam,
                                                    early_stopping=True,
                                                    num_return_sequences=num_return_sequences, )

                preds = [self.train_dataloader.tokenizer.decode(g, skip_special_tokens=True,
                                                                clean_up_tokenization_spaces=True)
                         for g in
                         generated_ids]
                train_ids_list = batch["guid"]
                for start in range(0, len(preds), num_return_sequences):
                    question=batch["question"][start // num_return_sequences]
                    context=batch["context"][start // num_return_sequences]
                    pred = preds[start:start + num_return_sequences]
                    for i, question_with_answer in enumerate(pred):
                        if "Answer:" in question_with_answer:
                            answer=question_with_answer.split("Answer:")[1].split("Context")[0].strip()
                            if '.' in answer:
                                answer=answer[:-1]
                            pred[i] = {
                                "question": question,
                                "answer": answer, "context": context}
                        else:
                            pred[i] = {"question": question,
                                       "answer": "", "context": context}
                    outputs[train_ids_list[start // num_return_sequences]]=pred

        outputs = self.filter_answer(outputs)
        output_file_path=self.save_generate_question_answer(outputs, only_question=False)
        return output_file_path

    def postprocess_preds(self,predictions):
        processed_preds = []
        for p in predictions:
            if self.question_augment:
                if 'Question:' in p:
                    processed_preds.append(p.split('Question:')[1].split('Answer')[0].strip())
                else:
                    processed_preds.append("")
            else:
                if 'Answer:' in p:
                    processed_preds.append(p.split('Answer:')[1].split('Context')[0].strip())
                else:
                    processed_preds.append("")
        return processed_preds



    def inference_epoch_end(self, split,outputs):
        pred_answers,act_answers=[],[]
        for pred,act in outputs:
            pred_answers.extend(pred)
            act_answers.extend(act)


        if "bart" in self.config.ModelConfig.model_name_or_path:
            processed_preds=self.postprocess_preds(pred_answers)
            processed_actuals=postprocess_actuals(act_answers)
        elif "t5" in self.config.ModelConfig.model_name_or_path:
            processed_preds = postprocess_preds_t5(pred_answers)
            processed_actuals = postprocess_actuals(act_answers)
        else:
            raise "This type mode don't implementation"


        if self.question_augment:
            rouge = datasets.load_metric('src/rouge.py')
            cur_metrics = rouge.compute(predictions=pred_answers, references=act_answers, rouge_types=['rougeL'])
            return {'rougeL':cur_metrics['rougeL'].mid.fmeasure}
        else:
            cur_metrics = get_metrics(processed_preds, processed_actuals)

        return {'F1':cur_metrics[0],'EM':cur_metrics[1]}

    def training_step(self, batch, batch_idx):
        outputs=self.model(batch)
        loss=outputs[0]

        return loss

    def filter_QA(self):
        # In here, test data_loader is augment data, we filter can't entiredly match QA pair.
        # todo: one question can generate a lot of answers
        if "bart" in self.config.ModelConfig.model_name_or_path:
            max_length = 50
        elif "t5" in self.config.ModelConfig.model_name_or_path:
            max_length = 25
        else:
            raise "This type mode don't implementation"

        outputs = {}
        self.model.eval()
        with torch.no_grad():
            logger.info(f"Original augmented data num: {len(self.test_dataloader.raw_dataset)}")
            for batch_idx, batch in enumerate(tqdm(self.test_dataloader, desc="filter question...")):
                batch = batch.to(self.config.TrainingConfig.device).to_dict()
                generated_ids = self.model.generate(batch,
                                                    max_length=max_length,
                                                    num_beams=1,
                                                    early_stopping=True,
                                                    num_return_sequences=1, )

                preds = [self.train_dataloader.tokenizer.decode(g, skip_special_tokens=True,
                                                                clean_up_tokenization_spaces=True)
                         for g in
                         generated_ids]
                train_ids_list = batch["guid"]
                for i in range(0, len(preds)):
                    pred = preds[i]
                    if 'Answer:' in pred:
                        pred=pred.split('Answer:')[1].split('Context')[0].strip()
                    else:
                        pred=""
                    answer = batch["answer"][i]
                    if normalize_answer(pred)==normalize_answer(answer):
                        outputs[train_ids_list[i]] = [{"question":batch["question"][i],
                                     "answer":answer,"context":batch["context"][i]}]

        output_file = self.save_generate_question_answer(outputs, filter=True)
        return output_file


    def pretraining(self,ckpt: Optional[str] = None):
        self.set_stop_criterion()
        self.configure_optimizer()

        if ckpt:
            if not self.load_checkpoint(ckpt):
                logger.warning("Train from scratch instead ...")

        if self.cur_epoch == 0:
            self.on_fit_start()
        mask_pro = self.pretrain_dataloader.mask_prob
        for cur_epoch in range(0,self.num_epochs):
            self.pretrain_dataloader.mask_prob = ((cur_epoch + 1) / self.num_epochs) * mask_pro
            continue_training = self.training_epoch(cur_epoch)
            if continue_training == -1:
                logger.info("Pretraining Done!")
                break


    def fit(self,ckpt: Optional[str] = None,augment_answer: Optional[bool] = False,augment_question:Optional[bool]=False):

        self.set_stop_criterion()
        self.configure_optimizer()

        if ckpt:
            if not self.load_checkpoint(ckpt):
                logger.warning("Train from scratch instead ...")

        if self.cur_epoch == 0:
            self.on_fit_start()
        mask_pro=self.train_dataloader.mask_prob
        for self.cur_epoch in range(self.cur_epoch,self.num_epochs):
            self.train_dataloader.mask_prob = ((self.cur_epoch+1) / self.num_epochs) * mask_pro
            continue_training = self.training_epoch(self.cur_epoch)

            if self.config.TrainingConfig.max_train_steps==None:

                if self.whether_validation:
                    val_score=self.inference_epoch("validation")
                    if self.question_augment:
                        logger.info(
                        f"training epoch: {self.cur_epoch}, Rouge-L: {val_score['rougeL']}")
                    else:
                        logger.info(
                    f"training epoch: {self.cur_epoch}, validation score: F1: {val_score['F1']} EM: {val_score['EM']}")

                    if self.whether_save_metric:
                        self.save_metric(val_score, mode=f"epochs_{self.cur_epoch}",split='val')

                    if self.best_dev_score == None:
                        self.best_dev_score = val_score
                        self.save_checkpoint(ckpt='PromptGeneration')

                    metric='F1'
                    if self.question_augment:
                        metric='rougeL'

                    if val_score[metric] > self.best_dev_score[metric]:
                        self.best_dev_score = val_score
                        if self.save_ckp:
                            self.save_checkpoint(ckpt='PromptGeneration')

            if continue_training==-1:
                logger.info("Stop training by reaching maximum num_training_steps")
                break
        metric = 'F1'
        if self.question_augment:
            metric = 'rougeL'

        if self.whether_save_metric:
            self.save_metric(self.best_dev_score,mode='best_val',split='val')
        if self.whether_validation:
            logger.info(f"Best validation score {metric}: {self.best_dev_score[metric]}")

        # when last=True, we have augmented quesiton, then we will agument answer.
        if augment_answer:
            if self.whether_validation:
                logger.info("loading best validation model weight...")
                self.load_checkpoint("PromptGeneration")
            answer_path = self.generate_answer(num_beam=1, num_return_sequences=1)
            return answer_path

        if augment_question:
            if self.save_ckp and self.whether_validation:
                logger.info("loading best validation model weight...")
                self.load_checkpoint("PromptGeneration")
            question_augment_path=self.generate_question(num_beam=1,num_return_sequences=1)
            if self.del_ckp:
                if os.path.isfile(self.checkpoint_path("PromptGeneration")):
                    path = self.checkpoint_path("PromptGeneration")
                    os.remove(self.checkpoint_path("PromptGeneration"))
                    logger.info(f"Successful move filter ckp from {path}")
            return question_augment_path

        if self.whether_filter_data:
            logger.info("loading best validation model weight...")
            self.load_checkpoint("PromptGeneration")
            augment_data_path=self.filter_QA()
            if self.del_ckp:
                if os.path.isfile(self.checkpoint_path("PromptGeneration")):
                    path=self.checkpoint_path("PromptGeneration")
                    os.remove(self.checkpoint_path("PromptGeneration"))
                    logger.info(f"Successful move filter ckp from {path}")
            return augment_data_path

        if self.whether_test:
            if self.save_ckp:
                logger.info("loading best validation model weight...")
                self.load_checkpoint("PromptGeneration")
            test_score = self.inference_epoch("test")
            if self.whether_save_metric:
                self.save_metric(test_score, mode='best_test', split='test')
            return self.best_dev_score, test_score



        return self.best_dev_score




