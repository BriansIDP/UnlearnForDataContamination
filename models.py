import argparse
import logging
import math
import os
from time import time
from copy import deepcopy
import random
import json
from copy import deepcopy
from tqdm import tqdm

import numpy as np
import six
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft import PeftConfig, PeftModel
from torch.nn.utils.rnn import pad_sequence


letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
class Model(torch.nn.Module):
    def __init__(
        self,
        model_path,
        tokenizer,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_module=["q_proj", "v_proj"],
        pool="average",
        losstype="kl",
        middle_dim=128,
        uselora=False,
    ):
        super(Model, self).__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            cache_dir="/data/milsrg1/huggingface/cache/gs534/cache",
            trust_remote_code=True,
        )
        self.uselora = uselora
        if self.uselora:
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_module,
            )
            self.llm = get_peft_model(self.llm, self.peft_config)
        self.middle_dim = middle_dim
        self.tokenizer = tokenizer
        self.losstype = losstype

    def merge_and_reload(self, outpath, new_lora_kwargs, save=True, load_new=True):
        self.llm = self.llm.merge_and_unload()
        if save:
            self.llm.save_pretrained(outpath)
        if load_new:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=new_lora_kwargs["lora_rank"],
                lora_alpha=new_lora_kwargs["lora_alpha"],
                lora_dropout=new_lora_kwargs["lora_dropout"],
                target_modules=new_lora_kwargs["lora_module"],
            )
            self.llm = get_peft_model(self.llm, peft_config)

    def add_adapter(self, new_lora_kwargs):
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=new_lora_kwargs["lora_rank"],
            lora_alpha=new_lora_kwargs["lora_alpha"],
            lora_dropout=new_lora_kwargs["lora_dropout"],
            target_modules=new_lora_kwargs["lora_module"],
        )
        self.llm.add_adapter(peft_config, adapter_name="lora_1")

    def delete_adapter(self):
        self.llm.delete_adapter(adapter_names=["lora_1"])

    def unfreeze_model(self):
        for name, param in self.llm.named_parameters():
            # if "self_attn" in name:
            param.requires_grad = True

    def forward(
        self,
        inputs,
        labels,
        input_masks=None,
        unlearn_target=None,
        alpha=1.0,
        return_hidden=False,
    ):
        attention_mask = torch.ones_like(inputs)
        outputs = self.llm(
            input_ids=inputs,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = outputs.logits[:, :-1]
        if "fixed" in self.losstype:
            with torch.no_grad():
                with self.llm.disable_adapter():
                    orig_outputs = self.llm(
                        input_ids=inputs,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                orig_logits = orig_outputs.logits[:, :-1]
                # self.llm.enable_adapters()

        if unlearn_target is not None:
            logps = torch.log_softmax(logits[torch.arange(input_masks.size(0)), input_masks.sum(dim=-1)-1], dim=-1)
            choicelist = torch.tensor([self.tokenizer.encode(c)[-1] for c in letters])
            if "fixed" in self.losstype:
                orig_logps = torch.log_softmax(orig_logits[torch.arange(input_masks.size(0)), input_masks.sum(dim=-1)-1], dim=-1)
                probs = torch.exp(orig_logps).detach().data
            else:
                probs = torch.exp(logps).detach().data
            if "norm" in self.losstype:
                logps = torch.log_softmax(logps[:, choicelist], dim=-1)
                probs = probs[:, choicelist] / probs[:, choicelist].sum(dim=-1, keepdim=True)
                unlearn_target = torch.where(choicelist.to(unlearn_target.device)[:, None] == unlearn_target)[0]
            y_bar_cs = probs[torch.arange(unlearn_target.size(0)), unlearn_target]
            y_bar_cs = torch.clip(y_bar_cs, min=0.0001, max=0.99)
            if "bary" in self.losstype:
                labels = y_bar_cs * alpha # factor to be changed
            else:
                labels = torch.clip(labels, min=0.0001, max=0.99)
            target_dist = ((1 - labels) / (1 - y_bar_cs)).unsqueeze(1) * probs
            target_dist[torch.arange(unlearn_target.size(0)), unlearn_target] = labels
            if "kl" in self.losstype:
                loss = - (target_dist * logps).sum(dim=-1).mean()
            elif "mse" in self.losstype:
                loss = ((target_dist - torch.exp(logps)) ** 2).sum(dim=-1).mean()
        else:
            labels = labels[:, 1:]
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            if return_hidden:
                loss = (loss, logits)
        return loss

    def generate(self, inputs, temperature=1.0, do_sample=True, max_new_tokens=512, return_dict=False):
        attention_mask = torch.ones_like(inputs)
        generate_ids = self.llm.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            attention_mask=attention_mask,
            temperature=temperature,
            top_p=0.9,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            output_logits=return_dict,
            return_dict_in_generate=return_dict,
        )
        return generate_ids