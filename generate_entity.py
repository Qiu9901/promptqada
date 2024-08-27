import spacy
import argparse
import logging
import os
import random
import numpy as np
import json
from typing import *
from tqdm.auto import tqdm
import copy
import pickle

logger = logging.getLogger(__name__)

class InputExample(object):
    """A raw input example consisting of segments of text,
    a label for classification task or a target sequence of generation task.
    Other desired information can be passed via meta.

    Args:
        guid (:obj:`str`, optional): A unique identifier of the example.
        question (:obj:`str`, optional): The placeholder for sequence of text.
        context (:obj:`str`, optional): A secend sequence of text, which is not always necessary.
        answer (:obj:`str`, optional): A secend sequence of text, which is not always necessary.
        label (:obj:`int`, optional): The label id of the example in classification task.
        tgt_text (:obj:`Union[str,List[str]]`, optional):  The target sequence of the example in a generation task..
        meta (:obj:`Dict`, optional): An optional dictionary to store arbitrary extra information for the example.
    """

    def __init__(self,
                 guid = None,
                 question = "",
                 context = "",
                 answer: Union[list,str]="",
                 label = None,
                 meta: Optional[Dict] = None,
                 tgt_text: Optional[Union[str,List[str]]] = None
                ):

        self.guid = guid
        self.question = question
        self.context = context
        self.answer=answer

        self.label = label
        self.meta = meta if meta else {}
        self.tgt_text = tgt_text

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        r"""Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        r"""Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def keys(self, keep_none=False):
        return [key for key in self.__dict__.keys() if getattr(self, key) is not None]

    @staticmethod
    def load_examples(path: str) -> List['InputExample']:
        """Load a set of input examples from a file"""
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    @staticmethod
    def save_examples(examples: List['InputExample'], path: str) -> None:
        """Save a set of input examples to a file"""
        with open(path, 'wb') as fh:
            pickle.dump(examples, fh)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")

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
    args = parser.parse_args()

    return args

class QuestionAnswerProcessor():
    def __init__(self,
                 data_dir:str,
                 dataset_name:str,
                 data_seed:int,
                 train_size:int):

        self.data_dir=data_dir
        self.dataset_name=dataset_name
        self.data_seed=data_seed
        self.train_size=train_size

        self.train_data_file=f'{data_dir}/{dataset_name}/{dataset_name}-train-seed-{data_seed}-num-examples-{train_size}.jsonl'
        self.dev_data_file=f'{data_dir}/{dataset_name}/{dataset_name}-train-seed-{data_seed}-num-examples-1024.jsonl'
        self.test_data_file=f'{data_dir}/{dataset_name}/dev.jsonl'

    def whitespace_tokenize(self,text):
        text = text.strip()
        if not text:
            return []
        tokens = text.split()
        return tokens

    def get_example(self,split='train',multi_answer=False):
        examples=[]
        if split=="train":
            lines=open(self.train_data_file,"r",encoding='utf-8').read().splitlines()
        elif split=="dev":
            lines = open(self.dev_data_file, "r", encoding='utf-8').read().splitlines()
        elif split=="test":
            lines = open(self.test_data_file, "r", encoding='utf-8').read().splitlines()
        else:
            raise "Error file split->[train,dev,test]"

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


def get_context_entity(context:str,answer:str):
    """
    Here we don't filter any candidate answer.
    :param context:
    :return:
    """
    doc=nlp(context)

    result_count={}

    for ent in doc.ents:
        entity_text=ent.text
        entity_type=ent.label_
        if entity_text==answer:
            continue
        if entity_type not in keeping_types:
            continue
        if len(entity_text)>20:
            continue
        # print(ent.text, ent.start_char, ent.end_char, ent.label_)
        if entity_text not in result_count.keys():
            result_count[entity_text]=[1,entity_type,rule_dict[entity_type]]
        else:
            result_count[entity_text][0]+=1

    return result_count


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    data_dir="../mrqa-few-shot"
    tasks=["bioasq"]
    # tasks=['squad']
    seeds=[44]
    shots=[256]
    output_path="mrqa-few-shot-entity"

    for task_name in tasks:
        output_dir=os.path.join(output_path,task_name)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        for shot in shots:
            for seed in seeds:
                random.seed(seed)
                np.random.seed(seed)
                processor = QuestionAnswerProcessor(data_dir=data_dir,
                                        dataset_name=task_name,
                                        data_seed=seed,
                                        train_size=shot)
                output_file=os.path.join(output_dir,f'{task_name}-train-seed-{seed}-num-examples-{shot}.jsonl')
                if os.path.exists(output_file):
                    os.remove(output_file)
                    logging.info(f"removing {output_file}")

                head = [{"header": {"dataset": task_name, "split": "entity_answer"}}]
                with open(output_file, 'a') as fout:
                    fout.write(str(head) + '\r\n')

                # train data preprocessing
                train_examples = processor.get_example(split='train', multi_answer=False)
                for example in tqdm(train_examples,total=len(train_examples),desc=f'processing {task_name}_{shot}_{seed}'):
                    train_id=example.guid
                    context=example.context
                    question=""
                    answer=example.answer
                    # [(text,type),()]
                    entity_dict=get_context_entity(context,answer)

                    for entity in entity_dict.keys():
                        for question_type in entity_dict[entity][2]:
                            paragraph={}
                            paragraph["context"] = context
                            paragraph["qas"] = []
                            paragraph["train_id"]=train_id

                            paragraph["qas"].append(
                            {"answers": [entity], "question": question,"question_type":question_type}
                            )

                            with open(output_file,'a') as fout:
                                fout.write(json.dumps(paragraph))
                                fout.write('\r\n')











if __name__=="__main__":
    nlp=spacy.load("en_core_web_trf")
    keeping_types=('CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART')
    rule_dict={
        "PERSON":["Who"],
        'NORP': ["Who"],
        'ORG': ["Who"],
        'GPE':["Where"],
        'LOC': ["Where"],
        'FAC': ["Where"],
        'PRODUCT':["What"],
        'EVENT': ["What"],
        'LANGUAGE': ["What"],
        'LAW': ["What"],
        'WORK_OF_ART': ["What"],
        'DATE':["When"],
        'TIME': ["When"],
        'PERCENT':["How many","How much"],
        'CARDINAL': ["How many","How much"],
        'MONEY': ["How many","How much"],
        'ORDINAL': ["How many","How much"],
        'QUANTITY': ["How many","How much"],
    }
    main()