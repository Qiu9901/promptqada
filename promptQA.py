"""
this py is used for fine-tuning bart in few-shot setting, which we don't use any prompt.
"""
import torch.nn as nn
from transformers import set_seed
from transformers.models.bart import BartTokenizer
import yaml
import argparse
import logging
import torch

from src.plm_modeling import BartForConditionalGeneration,BartConfig
from transformers import T5ForConditionalGeneration,T5Config,T5Tokenizer
from src.prompt_prototype import BARTTokenizerWrapper,T5TokenizerWrapper
from src.utilis import check_path,PromptDataLoader
from src.config_utilis import DataConfig,ModelConfig,TrainingConfig,AllConfig
from src.prompt_mix_template import MixedTemplate
from src.modeling import PromptForGeneration

from src.trainer import ,QuestionAnswerRunner
from src.processors import QuestionAnswerProcessor
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
    parser.add_argument("--mask_prob",
                        type=float,
                        default=0.3,
                        help="mask probability")
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
    parser.add_argument("--augment",
                        type=str,
                        default=None,
                        help="Bash will change output dir.")
    parser.add_argument("--augment_val",
                        type=str,
                        default=None,
                        help="Bash will change output dir.")
    parser.add_argument("--device",
                        type=str,
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

    if args.device is not None:
        config['training']['device']=args.device


    config=AllConfig(dataConfig=DataConfig(**config['data']),
                     modelConfig=ModelConfig(**config['model']),
                     trainingConfig=TrainingConfig(**config['training']))
    set_seed(config.DataConfig.seed)
    # Set random seeds and deterministic pytorch for reproducibility
    torch.backends.cudnn.deterministic = True
    check_path(config.TrainingConfig.output_dir)
    config.save()
    alpha=0
    augment=False
    if args.augment==None:
        processor = QuestionAnswerProcessor(data_dir=config.DataConfig.data_file,
                                        dataset_name=config.DataConfig.task_name,
                                        data_seed=config.DataConfig.seed,
                                        train_size=config.DataConfig.k_shot,)
    else:
        processor = QuestionAnswerProcessor(data_dir=config.DataConfig.data_file,
                                            dataset_name=config.DataConfig.task_name,
                                            data_seed=config.DataConfig.seed,
                                            train_size=config.DataConfig.k_shot,
                                            augment=True)
        if config.DataConfig.k_shot==16:
            alpha=0.1
        elif config.DataConfig.k_shot==32:
            alpha=0.2
        elif config.DataConfig.k_shot==64:
            alpha=0.3
        elif config.DataConfig.k_shot==128:
            alpha=0.4
        elif config.DataConfig.k_shot==256:
            alpha=0.5
        else:
            print("error k shot")
        augment=True

    train_examples = processor.get_example(split='train', multi_answer=False)

    dev_multianswer_examples = processor.get_example(split='dev', multi_answer=True)


    if args.augment_val!=None:
        augment_examples=processor.get_example(file_path=processor.dev_augment_file,multi_answer=True)
        dev_multianswer_examples.extend(augment_examples)

    test_multianswer_examples = processor.get_example(split='test', multi_answer=True)


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

    src_template=MixedTemplate(model=plm_model,tokenizer=plm_tokenizer,text=config.ModelConfig.template)
    target_template=MixedTemplate(model=plm_model,tokenizer=plm_tokenizer,text=config.ModelConfig.target_template)

    train_dataloader = PromptDataLoader(dataset=train_examples,
                                        template=src_template,
                                        target_template=target_template,
                                        tokenizer_wrapper=tokenizer_wrapper,
                                        tokenizer=plm_tokenizer,
                                        batch_size=config.ModelConfig.per_device_train_batch_size,
                                        padding=config.ModelConfig.pad_to_max_length,
                                        config=plm_config,
                                        shuffle=True,
                                        train=True,
                                        augment=augment,alpha=alpha)


    dev_dataloader = PromptDataLoader(dataset=dev_multianswer_examples,
                                      template=src_template,
                                      tokenizer_wrapper=tokenizer_wrapper,
                                      tokenizer=plm_tokenizer,
                                      batch_size=config.ModelConfig.per_device_eval_batch_size,
                                      padding=config.ModelConfig.pad_to_max_length,
                                      config=plm_config,
                                      shuffle=False,
                                      train=False)

    test_dataloader = PromptDataLoader(dataset=test_multianswer_examples,
                                      template=src_template,
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

    training(prompt_model, train_dataloader, dev_dataloader,test_dataloader,config)


def training(prompt_model:[nn.Module],
            train_dataloader:PromptDataLoader,
            dev_dataloader:PromptDataLoader,
            test_dataloader:PromptDataLoader,
            config:AllConfig,
            pretraining_dataloader=None,
            ):
    logger.info(f"train example number: {len(train_dataloader.tensor_dataset)}")
    logger.info(f"validation example number: {len(dev_dataloader.tensor_dataset)}")
    logger.info(f"test example number: {len(test_dataloader.tensor_dataset)}")
    logger.info(f"gradient accumulation steps: {config.TrainingConfig.gradient_accumulation_steps}")
    logger.info(
        f"train batch size: {config.TrainingConfig.gradient_accumulation_steps * config.ModelConfig.per_device_train_batch_size}")
    if pretraining_dataloader!=None:
        logger.info(f"pretrain example number: {len(pretraining_dataloader.tensor_dataset)}")


    runner=QuestionAnswerRunner(model=prompt_model,
                                config=config,
                                train_dataloader=train_dataloader,
                                valid_dataloader=dev_dataloader,
                                test_dataloader=test_dataloader,
                                )

    dev,test=runner.run()
    logger.info(f"Test F1:{test['F1']},EM:{test['EM']}")
    path=config.TrainingConfig.output_dir
    path=path.split('seed')[0]
    with open(path+'result.txt','a') as fout:
        fout.write(str(config.DataConfig.seed)+" "+str(dev['F1'])+" "+str(test['F1'])+'\r\n')



if __name__ == "__main__":
    main()