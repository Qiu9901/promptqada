"""
this py is used for fine-tuning bart in few-shot setting, which we don't use any prompt.
"""
import torch
import torch.nn as nn
from transformers import (
    AdamW,
    set_seed,
    get_scheduler
)
from transformers.models.bart import (
    BartTokenizerFast,
    BartTokenizer,

)



import yaml
import argparse
import json
import logging
import math
import os
import random
import json
from pathlib import Path

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


import transformers
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
# from transformers import BartForConditionalGeneration,BartConfig
from src.plm_modeling import BartForConditionalGeneration,BartConfig
from transformers import T5ForConditionalGeneration,T5Config,T5Tokenizer
# from src.plm_modeling import BartForSequenceClassification,BartForConditionalGeneration,BartConfig
from src.prompt_prototype import BARTTokenizerWrapper,T5TokenizerWrapper
from src.utilis import check_path,PromptDataLoader
from src.config_utilis import DataConfig,ModelConfig,TrainingConfig,AllConfig
from src.prompt_mix_template import MixedTemplate
from src.modeling import PromptForGeneration

from src.trainer import ProtoVerbClassificationRunner,QuestionAnswerRunner
from src.processors import QuestionAnswerProcessor,DataProcessor
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument("--config_path",
                        type=str,
                        required=True,
                        help="All parameters file path, this is needed."
                        )
    parser.add_argument("--run_time",
                        type=str,
                        default=None,
                        help="Bash output file name prefix."
                        )
    parser.add_argument("--plm_learning_rate",
                        type=float,
                        default=None,
                        help="This hyper-parameters need be fine.")
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="This hyper-parameters need be fine.")
    parser.add_argument("--k_shot",
                        type=int,
                        default=None,
                        help="Training example number")
    parser.add_argument("--per_device_train_batch_size",
                        type=int,
                        default=None,
                        help="This hyper-parameters need be fine.")
    parser.add_argument("--model_name_or_path",
                        type=str,
                        default=None,
                        help="The using pretrained model.")
    parser.add_argument("--task_name",
                        type=str,
                        default=None,
                        help="The using dataset.")
    parser.add_argument("--bash",
                        type=bool,
                        default=None,
                        help="Bash will change output dir.")
    parser.add_argument("--example_idx",
                        type=int,
                        default=None,
                        help="Bash will change output dir.")
    args = parser.parse_args()

    return args


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    args=parse_args()

    config={}
    with open(args.config_path,'r',encoding='utf-8') as f:
        result = yaml.load_all(f.read(), Loader=yaml.FullLoader)
        for key,value in zip(['data','model','training'],result):
            config[key]=value



    if args.plm_learning_rate is not None:
        config['training']['plm_learning_rate']=args.plm_learning_rate

    if args.run_time is not None:
        config['training']['bash_run_time']=args.run_time

    if args.task_name is not None:
        config['data']['task_name']=args.task_name

    if args.seed is not None:
        config['data']['seed'] = args.seed

    if args.k_shot is not None:
        config['data']['k_shot']=args.k_shot

    if args.per_device_train_batch_size is not None:
        config['model']['per_device_train_batch_size']=args.per_device_train_batch_size

    if args.model_name_or_path is not None:
        config['model']['model_name_or_path']=args.model_name_or_path

    if args.bash:
        config['data']['bash']=args.bash


    config['training']['output_dir']='prompt_temp'
    config=AllConfig(dataConfig=DataConfig(**config['data']),
                     modelConfig=ModelConfig(**config['model']),
                     trainingConfig=TrainingConfig(**config['training']))
    set_seed(config.DataConfig.seed)
    # Set random seeds and deterministic pytorch for reproducibility
    torch.backends.cudnn.deterministic = True
    check_path(config.TrainingConfig.output_dir)
    # config.save()

    processor = QuestionAnswerProcessor(data_dir=config.DataConfig.data_file,
                                        dataset_name=config.DataConfig.task_name,
                                        data_seed=config.DataConfig.seed,
                                        train_size=config.DataConfig.k_shot)


    train_examples = processor.get_example(split='train', multi_answer=False)

    dev_examples = processor.get_example(split='dev', multi_answer=False)

    augment_examples=processor.get_example(file_path=f'mrqa-few-shot-entity/{config.DataConfig.task_name}/{config.DataConfig.task_name}-train-seed-{config.DataConfig.seed}-num-examples-{config.DataConfig.k_shot}.jsonl',multi_answer=False)
    print(len(augment_examples))
    fore_train_list=[x for x in range(config.DataConfig.k_shot//2)]
    after_train_list=[x for x in range(config.DataConfig.k_shot//2,config.DataConfig.k_shot)]
    fore_train_example=train_examples[:len(train_examples)//2]
    after_train_example=train_examples[len(train_examples)//2:]

    fore_augment_examples=[]
    after_augment_examples = []
    for idx,example in enumerate(augment_examples):
        guid = f"augment_{idx}"
        if int(example.guid.split('_')[1].split('_')[0]) in fore_train_list:
            example.guid=guid
            fore_augment_examples.append(example)
        elif int(example.guid.split('_')[1].split('_')[0]) in after_train_list:
            example.guid = guid
            after_augment_examples.append(example)
        else:
            raise "Error idx in augment example"


    if "bart" in config.ModelConfig.model_name_or_path:
        plm_tokenizer = BartTokenizer.from_pretrained(config.ModelConfig.model_name_or_path)
        plm_config=BartConfig.from_pretrained(config.ModelConfig.model_name_or_path)
        # In generation mode, we only need original model
        plm_model = BartForConditionalGeneration.from_pretrained(
        config.ModelConfig.model_name_or_path,
        )

        tokenizer_wrapper = BARTTokenizerWrapper(max_src_length=config.ModelConfig.src_max_length,
                                             max_trg_length=config.ModelConfig.target_max_length,
                                             tokenizer=plm_tokenizer,
                                             config=plm_config)
    elif "t5" in config.ModelConfig.model_name_or_path:
        plm_tokenizer = T5Tokenizer.from_pretrained(config.ModelConfig.model_name_or_path)
        plm_config = T5Config.from_pretrained(config.ModelConfig.model_name_or_path)
        # In generation mode, we only need original model
        plm_model = T5ForConditionalGeneration.from_pretrained(
            config.ModelConfig.model_name_or_path,
        )

        tokenizer_wrapper = T5TokenizerWrapper(max_src_length=config.ModelConfig.src_max_length,
                                                 max_trg_length=config.ModelConfig.target_max_length,
                                                 tokenizer=plm_tokenizer,
                                                 config=plm_config)
    else:
        raise "No implementation this type model"

    train_question_src_template=MixedTemplate(model=plm_model,tokenizer=plm_tokenizer,
                                 text=config.ModelConfig.generate_train_template_question)
    train_question_trg_template=MixedTemplate(model=plm_model,tokenizer=plm_tokenizer,
                                 text=config.ModelConfig.generate_train_template_question_target)


    question_generate_tempate=MixedTemplate(model=plm_model,tokenizer=plm_tokenizer,
                                 text=config.ModelConfig.generate_template_question)
    src_template=MixedTemplate(model=plm_model,tokenizer=plm_tokenizer,text=config.ModelConfig.template)





    train_dataloader = PromptDataLoader(dataset=fore_train_example,
                                        template=train_question_src_template,
                                        target_template=train_question_trg_template,
                                        tokenizer_wrapper=tokenizer_wrapper,
                                        tokenizer=plm_tokenizer,
                                        batch_size=config.ModelConfig.per_device_train_batch_size,
                                        padding=config.ModelConfig.pad_to_max_length,
                                        config=plm_config,
                                        shuffle=True,
                                        train=True,
                                        mask_prob=0)


    generate_dataloader=PromptDataLoader(dataset=after_augment_examples,
                                        template=question_generate_tempate,
                                        tokenizer_wrapper=tokenizer_wrapper,
                                        tokenizer=plm_tokenizer,
                                        batch_size=config.ModelConfig.per_device_eval_batch_size,
                                        padding=config.ModelConfig.pad_to_max_length,
                                        config=plm_config,
                                        shuffle=False,
                                        train=False)


    dev_dataloader = PromptDataLoader(dataset=dev_examples,
                                      template=train_question_src_template,
                                      tokenizer_wrapper=tokenizer_wrapper,
                                      tokenizer=plm_tokenizer,
                                      batch_size=config.ModelConfig.per_device_eval_batch_size,
                                      padding=config.ModelConfig.pad_to_max_length,
                                      config=plm_config,
                                      shuffle=False,
                                      train=False)



    prompt_model=PromptForGeneration(plm=plm_model,
                                    template=src_template,
                                    )


    question_file=generate_augment_data(prompt_model, train_dataloader, dev_dataloader,generate_dataloader,config,processor,plm_config,
                          plm_model,plm_tokenizer,tokenizer_wrapper,augment_question=True)

    logger.info(f"Fore question file is saved at {question_file}")


    if "bart" in config.ModelConfig.model_name_or_path:
        plm_tokenizer = BartTokenizer.from_pretrained(config.ModelConfig.model_name_or_path)
        plm_config=BartConfig.from_pretrained(config.ModelConfig.model_name_or_path)
        # In generation mode, we only need original model
        plm_model = BartForConditionalGeneration.from_pretrained(
        config.ModelConfig.model_name_or_path,
        )

        tokenizer_wrapper = BARTTokenizerWrapper(max_src_length=config.ModelConfig.src_max_length,
                                             max_trg_length=config.ModelConfig.target_max_length,
                                             tokenizer=plm_tokenizer,
                                             config=plm_config)
    elif "t5" in config.ModelConfig.model_name_or_path:
        plm_tokenizer = T5Tokenizer.from_pretrained(config.ModelConfig.model_name_or_path)
        plm_config = T5Config.from_pretrained(config.ModelConfig.model_name_or_path)
        # In generation mode, we only need original model
        plm_model = T5ForConditionalGeneration.from_pretrained(
            config.ModelConfig.model_name_or_path,
        )

        tokenizer_wrapper = T5TokenizerWrapper(max_src_length=config.ModelConfig.src_max_length,
                                                 max_trg_length=config.ModelConfig.target_max_length,
                                                 tokenizer=plm_tokenizer,
                                                 config=plm_config)
    else:
        raise "No implementation this type model"

    train_question_src_template=MixedTemplate(model=plm_model,tokenizer=plm_tokenizer,
                                 text=config.ModelConfig.generate_train_template_question)
    train_question_trg_template=MixedTemplate(model=plm_model,tokenizer=plm_tokenizer,
                                 text=config.ModelConfig.generate_train_template_question_target)


    question_generate_tempate=MixedTemplate(model=plm_model,tokenizer=plm_tokenizer,
                                 text=config.ModelConfig.generate_template_question)
    src_template=MixedTemplate(model=plm_model,tokenizer=plm_tokenizer,text=config.ModelConfig.template)





    train_dataloader = PromptDataLoader(dataset=after_train_example,
                                        template=train_question_src_template,
                                        target_template=train_question_trg_template,
                                        tokenizer_wrapper=tokenizer_wrapper,
                                        tokenizer=plm_tokenizer,
                                        batch_size=config.ModelConfig.per_device_train_batch_size,
                                        padding=config.ModelConfig.pad_to_max_length,
                                        config=plm_config,
                                        shuffle=True,
                                        train=True,
                                        mask_prob=0)


    generate_dataloader=PromptDataLoader(dataset=fore_augment_examples,
                                        template=question_generate_tempate,
                                        tokenizer_wrapper=tokenizer_wrapper,
                                        tokenizer=plm_tokenizer,
                                        batch_size=config.ModelConfig.per_device_eval_batch_size,
                                        padding=config.ModelConfig.pad_to_max_length,
                                        config=plm_config,
                                        shuffle=False,
                                        train=False)


    dev_dataloader = PromptDataLoader(dataset=dev_examples,
                                      template=train_question_src_template,
                                      tokenizer_wrapper=tokenizer_wrapper,
                                      tokenizer=plm_tokenizer,
                                      batch_size=config.ModelConfig.per_device_eval_batch_size,
                                      padding=config.ModelConfig.pad_to_max_length,
                                      config=plm_config,
                                      shuffle=False,
                                      train=False)



    prompt_model=PromptForGeneration(plm=plm_model,
                                    template=src_template,
                                    )


    question_file=generate_augment_data(prompt_model, train_dataloader, dev_dataloader,generate_dataloader,config,processor,plm_config,
                          plm_model,plm_tokenizer,tokenizer_wrapper,augment_question=True,after=True)

    logger.info(f"After question file is saved at {question_file}")



def generate_augment_data(prompt_model:[nn.Module],
            train_dataloader:PromptDataLoader,
            dev_dataloader:PromptDataLoader,
            generate_dataloader:PromptDataLoader,
            config:AllConfig,
            augment_answer=False,
            augment_question=False,
            after=False
                          ):

    logger.info(f"train example number: {len(train_dataloader.tensor_dataset)}")
    logger.info(f"generate example number: {len(generate_dataloader.tensor_dataset)}")
    logger.info(f"validation example number: {len(dev_dataloader.tensor_dataset)}")
    logger.info(f"gradient accumulation steps: {config.TrainingConfig.gradient_accumulation_steps}")
    logger.info(
        f"train batch size: {config.TrainingConfig.gradient_accumulation_steps * config.ModelConfig.per_device_train_batch_size}")


    config.TrainingConfig.num_train_epochs=20
    if augment_question:
        config.TrainingConfig.num_train_epochs = 35


    if augment_question:
        runner = QuestionAnswerRunner(model=prompt_model,
                                      config=config,
                                      train_dataloader=train_dataloader,
                                      valid_dataloader=dev_dataloader,
                                      generate_dataloader=generate_dataloader,
                                      save_ckp=True,
                                      whether_save_metric=False,
                                      whether_validation=True,
                                      question_augment=True,
                                      del_ckp=True,
                                      after=after
                                      )
    file_path=runner.run(ckpt=None,augment_answer=augment_answer,augment_question=augment_question)


    return file_path




if __name__ == "__main__":
    main()