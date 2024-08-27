"""
this is for original fine-tuning data utils
"""
import torch
from torch.utils.data import DataLoader,Dataset
from torch.utils.data.sampler import RandomSampler
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.dummy_pt_objects import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from numpy.random import permutation, poisson
from .data_utilis import InputFeatures,InputExample
from .prompt_base import Template,TokenizerWrapper
from .prompt_prototype import BARTTokenizerWrapper
from tqdm.std import tqdm
from copy import deepcopy
import os
from typing import *
import logging
import collections
import re
import string
import numpy as np
import math
logger = logging.getLogger()

def check_path(path):
    # d = os.path.dirname(path)
    if not os.path.exists(path):
        os.makedirs(path)


def postprocess_preds(predictions):
    processed_preds = []
    for p in predictions:
        if 'Answer:' in p:
            processed_preds.append(p.split('Answer:')[1].split('Context')[0].strip())
        else:
            processed_preds.append("")
    return processed_preds

def postprocess_preds_t5(predictions):
    processed_preds = []
    for p in predictions:
        processed_preds.append(p)
        # if 'Answer:' in p:
        #     processed_preds.append(p.split('Answer:')[1].split('Context')[0].strip())
        # else:
        #     processed_preds.append("")
    return processed_preds

def postprocess_actuals(actuals):

    return actuals




def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
  return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
  gold_toks = get_tokens(a_gold)
  pred_toks = get_tokens(a_pred)
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1

def get_metrics(preds, actual_labels):
    f1_sum = 0.0
    exact_sum = 0.0
    for p, actuals in zip(preds, actual_labels):
        f1_sum += max([compute_f1(a, p) for a in actuals])
        exact_sum += max([compute_exact(a, p) for a in actuals])
    return f1_sum / len(preds), exact_sum / len(preds)






class PromptDataLoader(object):
    """
    1.tokenize text data, padding to max_length
    2.convert idx to tensor
    3.build dataloader
    """
    def __init__(self,
                 dataset: Union[Dataset,List],
                 template: Template,
                 tokenizer_wrapper: BARTTokenizerWrapper,
                 tokenizer: PreTrainedTokenizer,
                 target_template: Optional[Template]=None,
                 batch_size: Optional[int]=8,
                 padding: Optional[bool]=True,
                 config: Optional[PretrainedConfig]=None,
                 shuffle: Optional[bool]=True,
                 drop_last: Optional[bool]=False,
                 teacher_forcing: Optional[bool] = False,
                 train: Optional[bool]=False,
                 mask_prob: Optional[float]=0.3,
                 poisson_lambda: Optional[float] = 3,
                 alpha: Optional[float]=0.01,
                 augment: Optional[bool]=False,
                 decoder_max_length: Optional[int] = -1,
                 predict_eos_token: Optional[bool] = True,
                 truncate_method: Optional[str] = "tail",
                 ):
        self.train=train
        self.config=config
        self.augment=augment
        self.raw_dataset=dataset
        self.tokenizer=tokenizer
        self.wrapped_dataset=[]
        self.tensor_dataset=[]
        self.alpha=alpha

        self.template=template
        self.target_template=target_template

        self.tokenizer_wrapper=tokenizer_wrapper

        self.padding=padding
        self.teacher_forcing=teacher_forcing
        self.mask_prob=mask_prob
        self.poisson_lambda=poisson_lambda
        self.wrap()
        # for i in range(1):
        #     logger.info(f"wrapped instances: {self.wrapped_dataset[i]}")


        if train:
            source_length,target_length=[],[]
            for i,wrapped_example in enumerate(self.wrapped_dataset):
                sl,tl=self.tokenizer_wrapper.compute_max_seq_length(wrapped_example,self.train)
                source_length.append(sl)
                target_length.append(tl)

            max_src_length=min(800,max(source_length))
            max_trg_length=min(800,max(target_length))
            self.tokenizer_wrapper.max_src_length=800
            self.tokenizer_wrapper.max_trg_length=max_trg_length

        else:
            max_src_length=800
            self.tokenizer_wrapper.max_src_length = max_src_length


        logger.info(f"source max length: {self.tokenizer_wrapper.max_src_length}, target max length: {self.tokenizer_wrapper.max_trg_length}")

        self.tokenize()

        self.shuffle=shuffle
        self.batch_size=batch_size
        if shuffle:
            sampler=RandomSampler(self.tensor_dataset)
        else:
            sampler=None
        if self.train:
            self.dataloader = DataLoader(
                self.tensor_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                collate_fn=self.collate_fun_mask_full_word,
                drop_last=drop_last
            )
        else:
            self.dataloader = DataLoader(
                self.tensor_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                collate_fn=self.collate_fun,
                drop_last=drop_last
            )


    def collate_fun(self,batch):
        """
        pack batch data
        :param batch: [InputFeatures,]
        :return: InputFeatures
        """
        keys = list(set(batch[0].keys()).intersection(set(batch[0].tensorable_keys)))
        if "answer" in batch[0].keys():
            keys.append("answer")
        keys.append("guid")
        keys.append("context")
        keys.append("question")
        keys.append("question_type")
        result = {}
        for key in keys:
            if not(key == "answer" or key =="guid" or key=="context" or key=="question" or key=="question_type"):
                result[key] = torch.stack([elem[key] for elem in batch], dim=0)
            else:
                result[key] = [elem[key] for elem in batch]

        return InputFeatures(**result)

    def collate_fun_mask(self,batch):
        """
        random mask question tokens
        :param batch:
        :return: InputFeatures
        """
        keys = list(set(batch[0].keys()).intersection(set(batch[0].tensorable_keys)))
        if "answer" in batch[0].keys():
            keys.append("answer")
        result = {}
        for key in keys:
            if key != "answer":
                result[key] = torch.stack([elem[key] for elem in batch], dim=0)
            else:
                result[key] = [elem[key] for elem in batch]

        # we will change input_ids for mask language modeling
        inputs=result['input_ids']
        probability_matrix = torch.full(inputs.shape, self.mask_prob)
        sepcial_token_mask=result['special_token_ids'].bool()
        probability_matrix.masked_fill_(sepcial_token_mask,value=0.0)
        masked_indices=torch.bernoulli(probability_matrix).bool()

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token
        indeces_replaced=torch.bernoulli(torch.full(inputs.shape,0.8)).bool() & masked_indices
        inputs[indeces_replaced]=self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random=torch.bernoulli(torch.full(inputs.shape,0.5)).bool() & masked_indices & ~indeces_replaced
        random_words=torch.randint(len(self.tokenizer),inputs.shape,dtype=torch.long)
        inputs[indices_random]=random_words[indices_random]
        result['input_ids']=inputs
        return InputFeatures(**result)

    def collate_fun_mask_full_word(self,batch):
        keys = list(set(batch[0].keys()).intersection(set(batch[0].tensorable_keys)))
        if "answer" in batch[0].keys():
            keys.append("answer")
        result = {}
        for key in keys:
            if key != "answer":
                result[key] = torch.stack([elem[key] for elem in batch], dim=0)
            else:
                result[key] = [elem[key] for elem in batch]

        weight=[]
        alpha=self.alpha
        count_augment=0
        for feature in batch:
            if feature.augment:
                weight.append(alpha)
            else:
                weight.append(1-alpha)
                count_augment+=1

        result['weight']=torch.tensor(weight,dtype=torch.float)
        if self.augment==False or count_augment==len(batch):
            result['weight']=None

        if self.mask_prob==0:
            return InputFeatures(**result)

        inputs=result["input_ids"]
        inputs=np.array(inputs)
        labels = result['input_ids']
        labels=np.array(labels)
        special_tokens_mask = result['special_token_ids'].bool()
        special_tokens_mask = np.array(special_tokens_mask, dtype=bool)
        is_token = ~(labels == self.tokenizer.pad_token_id) & ~special_tokens_mask
        num_to_mask = int(math.ceil(is_token.astype(float).sum() * self.mask_prob))

        # if num_to_mask==0:
        #     return InputFeatures(**result)


        lengths = poisson(lam=self.poisson_lambda, size=(num_to_mask,))
        while np.cumsum(lengths, 0)[-1] < num_to_mask:
            lengths = np.concatenate([lengths, poisson(lam=self.poisson_lambda, size=(num_to_mask,))])
        lengths = lengths[lengths > 0]

        # trim to about num_to_mask tokens
        idx = np.argmin(np.abs(np.cumsum(lengths, 0) - num_to_mask)) + 1
        lengths = lengths[: idx + 1]


        token_indices = np.argwhere(is_token == 1)
        span_starts = permutation(token_indices.shape[0])[: lengths.shape[0]]

        # prepare mask
        masked_indices = np.array(token_indices[span_starts])
        mask = np.full_like(labels, fill_value=False)

        # mask span start indices
        for mi in masked_indices:
            mask[tuple(mi)] = True
        lengths -= 1

        # fill up spans
        max_index = labels.shape[1] - 1
        remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)
        while np.any(remaining):
            masked_indices[remaining, 1] += 1
            for mi in masked_indices:
                mask[tuple(mi)] = True
            lengths -= 1
            remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)

        # place the mask tokens
        mask[np.where(special_tokens_mask)] = False
        inputs[np.where(mask)] = self.tokenizer.mask_token_id


        # labels[np.where(special_tokens_mask)] = -100

        # remove mask tokens that are not starts of spans
        to_remove = (mask == 1) & np.roll((mask == 1), 1, 1)
        new_inputs = np.full_like(labels, fill_value=self.tokenizer.pad_token_id)

        # splits = list(map(lambda x: x.reshape(-1),  np.split(inputs_copy, indices_or_sections=2, axis=0))
        for i, example in enumerate(np.split(inputs, indices_or_sections=new_inputs.shape[0], axis=0)):
            new_example = example[0][~to_remove[i]]

            if self.tokenizer.pad_token_id in new_example.tolist():
                pad_position = new_example.tolist().index(self.tokenizer.pad_token_id)
            else:
                pad_position=len(new_example)
            new_inputs[i, 0: new_example.shape[0]] = new_example
            result['attention_mask'][i, :]=1
            result['attention_mask'][i, pad_position:] = 0


        result['input_ids']=torch.tensor(new_inputs,dtype=torch.long)

        # batching now fixed
        return InputFeatures(**result)


    def wrap(self):
        """
        TODO: wrap example
        """
        if isinstance(self.raw_dataset,Dataset) or isinstance(self.raw_dataset,List):
            assert len(self.raw_dataset)>0, 'The dataset to be wrapped is empty.'
            for idx,example in enumerate(self.raw_dataset):
                wrapped_example=self.template.wrap_one_example(example)
                if self.train:
                    wrapped_target_example=self.target_template.wrap_one_example(example)
                    self.wrapped_dataset.append([wrapped_example,wrapped_target_example])
                else:
                    self.wrapped_dataset.append(wrapped_example)
        else:
            raise ValueError("dataset can't be wrapped!")


    def tokenize(self):
        """
        tokenize original text in to tensor data
        TODO: use wrapped_tokenizer tokenize the wrapped text in to tensor data
        """
        for idex,wrapped_example in tqdm(enumerate(self.wrapped_dataset),desc="tokenizing"):
            if self.train:
                self.tensor_dataset.append(InputFeatures(**self.tokenizer_wrapper.tokenize_one_example(wrapped_example,self.train)
                                        ,**wrapped_example[0][1]).to_tensor())
            else:
                self.tensor_dataset.append(InputFeatures(
                    **self.tokenizer_wrapper.tokenize_one_example(wrapped_example, self.train)
                    , **wrapped_example[1]).to_tensor())

    def __len__(self):
        return  len(self.dataloader)


    def __iter__(self,):
        return self.dataloader.__iter__()
