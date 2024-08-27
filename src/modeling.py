import torch
import torch.nn as nn
from typing import *
from .data_utilis import InputExample, InputFeatures
from transformers.utils.dummy_pt_objects import PreTrainedModel
from .prompt_base import TokenizerWrapper,Template, Verbalizer


class PromptModel(nn.Module):
    def __init__(self,
                 plm: PreTrainedModel,
                 template: Optional[Template]=None,
                 verbalizer: Optional[Verbalizer]=None,
                 freeze_plm: bool =False,
                 plm_eval_mode: bool=False,
                 ):
        super().__init__()
        self.plm=plm
        self.template=template
        self.verbalizer=verbalizer
        self.freeze_plm=freeze_plm
        self.plm_eval_mode=plm_eval_mode
        # free pretrained model
        if freeze_plm:
            for param in self.plm.parameters():
                param.requires_grad=False
        # free pretrained model and close dropout/bach_normal and so on.
        if plm_eval_mode:
            self.plm.eval()
            for param in self.plm.parameters():
                param.requires_grad=False


    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            if not (self.plm_eval_mode and 'plm' in name and mode):
                module.train(mode)
        return self

    def forward(self,batch: Union[Dict,InputFeatures]):
        # we don't compute loss in pretrained model inside.
        batch =self.template.process_batch(batch)
        forward_keys=["input_ids","inputs_embeds","attention_mask","decoder_input_ids","decoder_label_ids","weight"]
        input_batch={key:batch[key] for key in batch if key in forward_keys}
        if "decoder_label_ids" in input_batch.keys():
            input_batch['labels']=input_batch['decoder_label_ids']
            input_batch.pop('decoder_label_ids')
        outputs = self.plm(**input_batch, output_hidden_states=True)
        outputs = self.template.post_processing_outputs(outputs)
        return outputs



class PromptForGeneration(nn.Module):

    def __init__(self,
                 plm: PreTrainedModel,
                 template: Template,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False):
        super().__init__()
        self.prompt_model = PromptModel(plm, template,freeze_plm, plm_eval_mode)

    @property
    def plm(self):
        return self.prompt_model.plm

    @property
    def template(self):
        return self.prompt_model.template

    @property
    def device(self):
        # TODO: multi-devices
        return self.plm.device


    @property
    def tokenizer(self):
        r'''Utility property, to get the tokenizer more easily.
        '''
        return self.template.tokenizer


    def forward(self,batch:Union[Dict,InputFeatures]):
        """
        Get the logits of label words
        :return: The logits of the label words (obtained by the current verbalizer).
        """
        outputs=self.prompt_model(batch)

        return outputs

    def generate(self,batch:Union[Dict,InputFeatures],max_length=30,num_beams=1,early_stopping=True,num_return_sequences=1,generate_question=False):
        batch = self.template.process_batch(batch)
        forward_keys = ["input_ids", "attention_mask"]
        input_batch = {key: batch[key] for key in batch if key in forward_keys}
        input_batch["num_return_sequences"]=num_return_sequences
        input_batch["max_length"]=max_length
        input_batch["num_beams"]=num_beams
        input_batch["early_stopping"]=early_stopping
        if generate_question:
            input_batch["do_sample"] = True
            input_batch["top_p"] = 0.9
        generated_ids=self.plm.generate(**input_batch)
        return generated_ids



