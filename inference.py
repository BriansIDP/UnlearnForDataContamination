import os
import re
import random
import argparse
import math
import pickle
import time
import json
from collections import OrderedDict

from tqdm import tqdm
import torch
from torch.optim import SGD, AdamW
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from models import Model


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

def logging(s, logfile, logging_=True, log_=True):
    if logging_:
        print(s)
    if log_:
        with open(logfile, 'a+') as f_log:
            f_log.write(s + '\n')


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is"
        "The correct option is",
        "Best answer:"
        "Best option:",
        "Answer:",
        "Option:",
        "The correct answer",
        "The correct option",
        "The final answer is:\n",
        "<answer>",
    ]
    for answer_prefix in answer_prefixes:
        s = s.split(answer_prefix)[-1]
        # s = s.replace(answer_prefix, "")
    if s == "":
        return s
    if s[0].lower() == s[0]:
        s = s[0].upper() + s[1:]
    if len(s.split()) > 10 and not re.search("[ABCDEFGHIJ]", s):
        return ""
    matches = re.search(r'[ABCDEFGHIJ]', s)
    if matches is None:
        return ""
    return matches[0]


def main(args):
    # Load model
    model_path = args.model_path
    if os.path.exists(os.path.join(args.model_path, "model_config.json")):
        with open(os.path.join(args.model_path, "model_config.json")) as fin:
            train_args = json.load(fin)
            model_path = train_args["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir="/mnt/bn/tiktok-mm-5/aiic/users/guangzhisun/UnlearnForDataContamination/cache", trust_remote_code=True)
    lora_kwargs = {}
    lora_path = os.path.join(args.model_path, "lora_config.json")
    if os.path.exists(lora_path) and not args.origmodel:
        with open(lora_path) as fin:
            lora_kwargs = json.load(fin)
    if os.path.exists(os.path.join(args.model_path, "base_model")):
        model_path = os.path.join(args.model_path, "base_model")
    model = Model(
        model_path,
        tokenizer,
        **lora_kwargs
    )

    if not args.origmodel:
        modelpath = os.path.join(args.model_path, args.model_ckpt, "pytorch_model.pt")
        trained_params = torch.load(modelpath)
        msg = model.load_state_dict(trained_params, strict=False)
        model.merge_and_reload("basemodel", None, save=False, load_new=False)
        # print(msg)
    model = model.to(device)
    if not args.get_movements:
        model.eval()
    else:
        model.unfreeze_model()

    with open(args.testfile) as fin:
        testdata = json.load(fin)

    results = []
    total = 0
    hits = 0
    for datapiece in tqdm(testdata):
        if args.outputlogp:
            origpiece = datapiece
            datapiece = datapiece["alt_question"]
            origpiece.pop("alt_question")
        choices = "\n".join(["{}. {}".format(key, value) for key, value in datapiece["options"].items()])
        question = "{}\nChoose from the following options only:\n{}\nOnly output the letter of the correct option. Do not respond with anything else.".format(datapiece["question"], choices)
        messages = [
            # {"role": "system", "content": "You are helpful AI assistant"},
            {"role": "user", "content": question},
        ]
        if args.get_movements:
            # Add adapter
            model.add_adapter(lora_kwargs)
            # initialize optimizer stuff
            # optimizer = SGD(model.parameters(), lr=1e-8)
            optimizer = AdamW(model.parameters(), lr=1e-4)
            optimizer.zero_grad()
            model_inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            messages.append({"role": "assistant", "content": datapiece["answer"]})
            complete_inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt"
            ).to(device)
            input_masks = complete_inputs != 0
            complete_labels = torch.cat([complete_inputs[:, :model_inputs.size(1)]*0-100, complete_inputs[:, model_inputs.size(1):]], dim=-1).to(device)
            loss_curve = []
            distribution_curve = []
            for iteration in range(2):
                loss, logits = model(complete_inputs, complete_labels, input_masks, return_hidden=True)
                choicelist = torch.tensor([tokenizer.encode(c)[-1] for c in letters])
                last_distribution = torch.softmax(logits[0, -2], dim=-1)[choicelist]
                print("Loss: {:.2f}".format(loss.item()))
                loss_curve.append(loss.item())
                distribution_curve.append(last_distribution.tolist())
                loss.backward()
                optimizer.step()
            datapiece["loss_curve"] = loss_curve
            datapiece["distribution_curve"] = distribution_curve
            results.append(datapiece)
            model.delete_adapter()
            total += 1
        else:
            model_inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)
            # model_inputs = tokenizer([text], return_tensors="pt").to(device)
            with torch.no_grad():
                generated_ids = model.generate(
                    model_inputs,
                    max_new_tokens=32,
                    do_sample=args.do_sample,
                    return_dict=args.outputlogp or args.allchoices,
                )
            if args.outputlogp:
                correct_choice = tokenizer.encode(datapiece["answer"])[-1]
                prob_tildeyc = torch.softmax(generated_ids["logits"][0], dim=-1)[0, correct_choice].item()
                generated_ids = generated_ids["sequences"]
                datapiece = origpiece
                datapiece["tildeyc"] = prob_tildeyc
            elif args.allchoices:
                choicelist = torch.tensor([tokenizer.encode(c)[-1] for c in letters])
                allprobs = torch.softmax(generated_ids["logits"][0], dim=-1)[0, choicelist]
                # allprobs = allprobs / allprobs.sum()
                generated_ids = generated_ids["sequences"]
                datapiece["bar_y"] = allprobs.tolist()
            generated_ids = generated_ids[:, model_inputs.size(1):]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # print(response)
            datapiece["pred_str"] = response
            datapiece["pred"] = extract_characters_regex(response)
            if datapiece["pred"] == datapiece["answer"]:
                hits += 1
            total += 1
            results.append(datapiece)
            if total % 1000 == 0 and total != 0:
                print("Accuracy: {}/{}={:.3f}".format(hits, total, hits/total*100))
    print("Accuracy: {}/{}={:.3f}".format(hits, total, hits/total*100))
    assert args.outfile.endswith("json")
    with open(args.outfile, "w") as fout:
        json.dump(results, fout, indent=4)

if __name__ == "__main__":
    ## Parameter groups
    parser = argparse.ArgumentParser(description="LLM finetuning")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./hf_models",
        help="Path to the model file",
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="",
        help="Checkpoint of the model file",
    )
    parser.add_argument(
        "--testfile",
        type=str,
        default="dataset/gt_nbest_sel.json",
        help="Path to the model file",
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default='./log.txt',
        help="Path to the log file",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default='./output.json',
        help="output file",
    )
    parser.add_argument(
        "--origmodel",
        action='store_true',
        help="Use original LLM",
    )
    parser.add_argument(
        "--do_generation",
        action='store_true',
        help="Run generation",
    )
    parser.add_argument(
        "--do_sample",
        action='store_true',
        help="Run generation with sampling",
    )
    parser.add_argument(
        "--outputlogp",
        action='store_true',
        help="Output log probability",
    )
    parser.add_argument(
        "--allchoices",
        action='store_true',
        help="Output log probability",
    )
    parser.add_argument(
        "--get_movements",
        action='store_true',
        help="Find the change when trained further",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=1,
        help="Number of samples to draw",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=4,
        help="LoRA alpha",
    )
    args = parser.parse_args()
    main(args)