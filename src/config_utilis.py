import json

import yaml
import os
from typing import *
from transformers import SchedulerType
from .utilis import check_path
import time
class DataConfig(object):
    """
    This data config class, this parameters is used for getting special task data.
    """
    def __init__(self,
                 task_name: str,
                 seed: int,
                 k_shot: int,
                 data_file: str,
                 bash: bool=False,
                 ):
        """
        :param task_name: special task name, e.g. SST-2
        :param seed: data and train seed
        :param k_shot: per-k instances for one class
        :param data_file: store dataset path
        """
        self.task_name=task_name
        self.seed=seed
        self.k_shot=k_shot
        self.data_file=data_file
        self.bash=bash


class ModelConfig(object):
    """
    This model config class, this parameters is used for getting model to be trained.
    """
    def __init__(self,
                 model_name_or_path: str,
                 prompt: Optional[bool]=False,
                 template: Optional[str] = None,
                 target_template: Optional[str]=None,
                 generate_train_template_answer: Optional[str]=None,
                 generate_train_template_answer_target: Optional[str]=None,
                 generate_train_template_question: Optional[str]=None,
                 generate_train_template_question_target: Optional[str]=None,
                 generate_template_question: Optional[str]=None,
                 verbalizer: Optional[str]=None,
                 src_max_length: Optional[int]=-1,
                 target_max_length: Optional[int]=-1,
                 pad_to_max_length: Optional[int]=True,
                 per_device_train_batch_size: Optional[int]=8,
                 per_device_eval_batch_size: Optional[int]=8,
                 use_slow_tokenizer: Optional[bool]=True,
                 lr_scheduler_type: Optional[SchedulerType]="linear",
                 ):
        """
        :param model_name_or_path: pretrained model path.
        :param prompt: whether or not use prompt.
        :param template: e.g. {"placeholder":"text_a"}*A*{"mask"}*movie.
        :param target_template: it's for target text parse, e.g. {"placeholder":"text_a"}*A*{"mask"}*movie.
        :param verbalizer: e.g. {"0":"irresistible","1":"pathetic"}
        :param max_length: max encoder/decoder input length.
        :param pad_to_max_length: whether or not static pad all input into max_length.
        :param per_device_train_batch_size: train batch size.
        :param per_device_eval_batch_size: dev batch size.
        :param use_slow_tokenizer: whether or not use slow tokenizer.
        :param lr_scheduler_type: optimizing schedule.
        """
        self.model_name_or_path=model_name_or_path
        self.prompt=prompt

        self.template=template
        self.target_template=target_template

        self.generate_train_template_answer=generate_train_template_answer
        self.generate_train_template_answer_target=generate_train_template_answer_target

        self.generate_train_template_question=generate_train_template_question
        self.generate_train_template_question_target=generate_train_template_question_target
        self.generate_template_question=generate_template_question

        self.verbalizer=verbalizer
        self.src_max_length=src_max_length
        self.target_max_length=target_max_length
        self.pad_to_max_length=pad_to_max_length
        self.per_device_train_batch_size=per_device_train_batch_size
        self.per_device_eval_batch_size=per_device_eval_batch_size
        self.use_slow_tokenizer=use_slow_tokenizer
        self.lr_scheduler_type=lr_scheduler_type


class TrainingConfig(object):
    """
    This training config class, this parameters is used for training and inference phrase.
    """
    def __init__(self,
                 freeze_plm: Optional[bool]=False,
                 plm_learning_rate: Optional[float]=5e-6,
                 template_learning_rate: Optional[float] = None,
                 weight_decay: Optional[float]=0.0,
                 num_train_epochs: Optional[int]=None,
                 max_train_steps: Optional[int]=None,
                 num_dev_steps: Optional[int]=None,
                 gradient_accumulation_steps: Optional[int]=1,
                 max_grad_norm: Optional[float]=1.0,
                 num_warmup_steps: Optional[int]=0,
                 output_dir:Optional[str]="prompt_result",
                 device: Optional[str]="cuda:0",
                 save_checkpoint: Optional[bool]=False,
                 bash_run_time: Optional[str]=None):
        """
        :param freeze_plm: whether or not free plm.
        :param plm_learning_rate: plm learning rate.
        :param template_learning_rate: template learning rate.
        :param weight_decay: plm weight decay.
        :param num_train_epochs: train epochs.
        :param max_train_steps: max train steps.
        :param num_dev_steps: num dev steps.
        :param gradient_accumulation_steps: add batch size.
        :param max_grad_norm: clip max grad.
        :param num_warmup_steps: plm num warwup steps.
        :param output_dir: checkpoint and result output dir.
        :param bash_run_time: bash output dir.
        :param device: which GPU will be used.
        """
        self.freeze_plm=freeze_plm
        self.plm_learning_rate=plm_learning_rate
        self.template_learning_rate=template_learning_rate
        self.weight_decay=weight_decay
        self.num_train_epochs=num_train_epochs
        self.max_train_steps=max_train_steps
        self.num_dev_steps=num_dev_steps
        self.gradient_accumulation_steps=gradient_accumulation_steps
        self.max_grad_norm=max_grad_norm
        self.num_warmup_steps=num_warmup_steps
        self.output_dir=output_dir
        self.bash_run_time=bash_run_time
        self.device=device
        self.save_checkpoint=save_checkpoint

class AllConfig(object):
    def __init__(self,
                 dataConfig: DataConfig,
                 modelConfig: ModelConfig,
                 trainingConfig: TrainingConfig):
        self.DataConfig=dataConfig
        self.ModelConfig=modelConfig
        if not dataConfig.bash:
            output_dir = os.path.join(trainingConfig.output_dir, time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())),"k_shot_" + str(dataConfig.k_shot)+'_'+str(dataConfig.task_name),
                                  f'seed_{dataConfig.seed}_LR_{trainingConfig.plm_learning_rate}_BS_{modelConfig.per_device_train_batch_size}')
        else:
            output_dir = os.path.join(trainingConfig.output_dir,'bash_'+str(dataConfig.task_name),
                                      "k_shot_" + str(dataConfig.k_shot) + '_' + str(dataConfig.task_name),
                                      trainingConfig.bash_run_time,
                                      f'seed_{dataConfig.seed}_LR_{trainingConfig.plm_learning_rate}_BS_{modelConfig.per_device_train_batch_size}')
        self.TrainingConfig=trainingConfig
        self.TrainingConfig.output_dir=output_dir

    def save(self):
        """
        save all parameters to output dir
        """
        output_dir=os.path.join(self.TrainingConfig.output_dir,'log.txt')
        result={}
        for key,value in zip(['DataConfig','ModelConfig','TrainingConfig'],[self.DataConfig,self.ModelConfig,self.TrainingConfig]):
            result[key]=vars(value)
        result=json.dumps(result)
        with open(output_dir,'w') as f:
            f.write(result)


