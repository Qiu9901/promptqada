import json
import logging
from typing import *
from abc import abstractmethod
# from openprompt.data_utils.utils import InputExample
from .data_utilis import InputFeatures,InputExample
from transformers.data.metrics import glue_compute_metrics
import os
import pandas as pd
logger = logging.getLogger()


class QuestionAnswerProcessor():
    def __init__(self,
                 data_dir:str,
                 dataset_name:str,
                 data_seed:int,
                 train_size:int,
                 augment=False,
                 filter_augment_data=False):

        self.data_dir=data_dir
        self.dataset_name=dataset_name
        self.data_seed=data_seed
        self.train_size=train_size
        self.augment=augment
        self.train_data_file=f'{data_dir}/{dataset_name}/{dataset_name}-train-seed-{data_seed}-num-examples-{train_size}.jsonl'
        if filter_augment_data:
            self.train_augment_file=f'mrqa-few-shot-augment-question/{dataset_name}/generate-{dataset_name}-train-seed-{data_seed}-num-examples-{train_size}.jsonl'

        if augment:
            self.train_augment_file = f'mrqa-few-shot-augment-filter/{dataset_name}/generate-{dataset_name}-train-seed-{data_seed}-num-examples-{train_size}.jsonl'

        self.dev_data_file=f'{data_dir}-val/{dataset_name}/{dataset_name}-val-seed-{data_seed}-num-examples-{train_size}.jsonl'
        self.test_data_file=f'{data_dir}/{dataset_name}/dev.jsonl'

    def whitespace_tokenize(self,text):
        text = text.strip()
        if not text:
            return []
        tokens = text.split()
        return tokens

    def get_example(self,split='train',multi_answer=False,file_path=None):
        examples=[]
        if file_path==None:
            if split=="train":
                lines=open(self.train_data_file,"r",encoding='utf-8').read().splitlines()

            elif split=="dev":
                lines = open(self.dev_data_file, "r", encoding='utf-8').read().splitlines()
            elif split=="test":
                lines = open(self.test_data_file, "r", encoding='utf-8').read().splitlines()
            else:
                raise "Error file split->[train,dev,test]"
        else:
            lines=open(file_path,"r",encoding="utf-8").read().splitlines()

        for context_idx,line in enumerate(lines[1:]):
            paragraph=json.loads(line)
            context=" ".join(self.whitespace_tokenize(paragraph["context"]))
            qas=paragraph["qas"]

            for question_idx,qa in enumerate(qas):
                question_text=qa["question"]
                question_text = " ".join(self.whitespace_tokenize(question_text))
                multi_answers=[]

                if multi_answer:
                    # for eval
                    for answer in qa["answers"]:
                        cleaned_answer_text=" ".join(self.whitespace_tokenize(answer))
                        multi_answers.append(cleaned_answer_text)
                else:
                    # train only use first answer
                    answer=qa["answers"][0]
                    cleaned_answer_text=" ".join(self.whitespace_tokenize(answer))
                question_type=""
                if 'question_type' in qa.keys():
                    question_type=qa['question_type']

                guid=f"{split}_{context_idx}_{question_idx}"

                if "train_id" in paragraph.keys():
                    guid=paragraph["train_id"]

                if multi_answers:
                    example=InputExample(
                    guid=guid,
                    context=context,
                    question=question_text,
                    answer=multi_answers,
                    question_type=question_type
                    )
                else:
                    example = InputExample(
                        guid=guid,
                        context=context,
                        question=question_text,
                        answer=cleaned_answer_text,
                        question_type=question_type
                    )
                examples.append(example)

        if self.augment:
            if split=="train":
                if os.path.exists(self.train_augment_file)==False:
                    logger.info(f"File {self.train_augment_file} is not exist!\nMay be augmented data is 0.")
                lines = open(self.train_augment_file, "r", encoding="utf-8").read().splitlines()
                for context_idx, line in enumerate(lines[1:]):
                    paragraph = json.loads(line)
                    context = " ".join(self.whitespace_tokenize(paragraph["context"]))
                    qas = paragraph["qas"]

                    for question_idx, qa in enumerate(qas):
                        question_text = qa["question"]
                        question_text = " ".join(self.whitespace_tokenize(question_text))
                        multi_answers = []

                        if multi_answer:
                            # for eval
                            for answer in qa["answers"]:
                                cleaned_answer_text = " ".join(self.whitespace_tokenize(answer))
                                multi_answers.append(cleaned_answer_text)
                        else:
                            # train only use first answer
                            answer = qa["answers"][0]
                            cleaned_answer_text = " ".join(self.whitespace_tokenize(answer))

                        guid = f"{split}_{context_idx}_{question_idx}"
                        if multi_answers:
                            example = InputExample(
                                guid=guid,
                                context=context,
                                question=question_text,
                                answer=multi_answers,
                                augment=True
                                    )
                        else:
                            example = InputExample(
                                guid=guid,
                                context=context,
                                question=question_text,
                                answer=cleaned_answer_text,
                                augment=True
                            )
                        examples.append(example)

        return examples


    def get_augment_example(self,split='train',multi_answer=False,file_path=None):
        examples=[]

        lines=open(self.train_augment_file,"r",encoding="utf-8").read().splitlines()

        for context_idx,line in enumerate(lines[1:]):
            paragraph=json.loads(line)
            context=" ".join(self.whitespace_tokenize(paragraph["context"]))
            qas=paragraph["qas"]

            for question_idx,qa in enumerate(qas):
                question_text=qa["question"]
                question_text = " ".join(self.whitespace_tokenize(question_text))
                multi_answers=[]

                if multi_answer:
                    # for eval
                    for answer in qa["answers"]:
                        cleaned_answer_text=" ".join(self.whitespace_tokenize(answer))
                        multi_answers.append(cleaned_answer_text)
                else:
                    # train only use first answer
                    answer=qa["answers"][0]
                    cleaned_answer_text=" ".join(self.whitespace_tokenize(answer))

                guid=f"{split}_{context_idx}_{question_idx}"
                if multi_answers:
                    example=InputExample(
                    guid=guid,
                    context=context,
                    question=question_text,
                    answer=multi_answers
                    )
                else:
                    example = InputExample(
                        guid=guid,
                        context=context,
                        question=question_text,
                        answer=cleaned_answer_text
                    )
                examples.append(example)

        return examples

