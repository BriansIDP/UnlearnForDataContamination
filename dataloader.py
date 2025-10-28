import os
import re
import math
import ast
import pathlib
import random
from typing import Optional, Dict
from tqdm import tqdm
import json
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import transformers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(
        self,
        data_path,
        tokenizer,
        unlearnmode=False,
        validation=False,
    ):
        super(SupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        with open(data_path) as fin:
            self.data = json.load(fin)
        if validation and isinstance(self.data, dict):
            newdata = []
            for qid, datapiece in self.data.items():
                datapiece["bar_y"] = datapiece["distribution"]
                newdata.append(datapiece)
            self.data = newdata
        self.unlearnmode = unlearnmode
        self.letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        self.validation = validation

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        label_mask = 1
        datapiece = self.data[idx]
        if "options" in datapiece:
            if isinstance(datapiece["options"], list):
                options = {self.letters[idx]: opt for idx, opt in enumerate(datapiece["options"])}
            else:
                options = datapiece["options"]
            choices = "\n".join(["{}. {}".format(key, value) for key, value in options.items()])
            question = "{}\nChoose from:\n{}\nRespond only with the letter of the correct option.".format(datapiece["question"], choices)
        else:
            question = datapiece["question"]
        messages = [
            # {"role": "system", "content": "You are helpful AI assistant"},
            {"role": "user", "content": question},
        ]
        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        messages.append({"role": "assistant", "content": datapiece["answer"]})
        complete_inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt"
        )[0]
        if self.unlearnmode:
            if "bar_y" in datapiece and "tildeyc" not in datapiece:
                if self.validation:
                    complete_labels = datapiece["bar_y"]
                else:
                    complete_labels = datapiece["bar_y"][self.letters.index(datapiece["answer"])]
            else:
                complete_labels = datapiece["tildeyc"]
        else:
            complete_labels = torch.cat([complete_inputs[:model_inputs.size(1)]*0-100, complete_inputs[model_inputs.size(1):]], dim=-1)
        return model_inputs, complete_inputs, complete_labels, datapiece["answer"]

def collate_fn(batch):
    input_ids = pad_sequence([sample[0][0] for sample in batch], batch_first=True)
    input_masks = input_ids != 0
    complete_inputs = pad_sequence([sample[1] for sample in batch], batch_first=True)
    if isinstance(batch[0][2], float) or isinstance(batch[0][2], list):
        complete_labels = torch.tensor([sample[2] for sample in batch])
    else:
        complete_labels = pad_sequence([sample[2] for sample in batch], batch_first=True, padding_value=-100)
    answer = [sample[3] for sample in batch]
    # bar_y_orig = [sample[4] for sample in batch] if batch[0][4] is not None else None
    return input_ids, input_masks, complete_inputs, complete_labels, answer