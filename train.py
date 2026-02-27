import os
import random
import argparse
import math
import pickle
import time
import json
from collections import OrderedDict
from sklearn.metrics import precision_recall_curve, auc
from scipy import stats

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import SchedulerType, get_scheduler
from torch.optim import AdamW, SGD
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft import PeftConfig, PeftModel
from torch.utils.data import DataLoader
import accelerate
from accelerate import Accelerator
from torch.nn.utils.rnn import pad_sequence

from models import Model
from dataloader import SupervisedDataset, collate_fn


accelerator = Accelerator()
device = accelerator.device
random.seed(1)
torch.manual_seed(1)
letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def logging(s, logfile, logging_=True, log_=True):
    if logging_:
        print(s)
    if log_:
        with open(logfile, 'a+') as f_log:
            f_log.write(s + '\n')


def save_checkpoint(model, tokenizer, outputdir, epoch, step):
    fulloutput = os.path.join(outputdir, "checkpoint.{}.{}".format(epoch, step))
    os.system(f"mkdir -p {fulloutput}")
    checkpoint = OrderedDict()
    for k, v in model.named_parameters():
        if v.requires_grad:
            checkpoint[k] = v
    torch.save(checkpoint, f'{fulloutput}/pytorch_model.pt')
    # save tokenizer
    tokenizer.save_pretrained(fulloutput)
    # save configuration
    model.llm.config.save_pretrained(fulloutput)
    return checkpoint

def main(rank, args, world_size):
    # Save model configuration
    with open(os.path.join(args.outputdir, 'model_config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    with open(args.lora_config) as fin:
        lora_kwargs = json.load(fin)
    if args.load_from != "":
        old_lora_path = os.path.join(os.path.dirname(args.load_from), "lora_config.json")
        new_lora_kwargs = lora_kwargs
        with open(old_lora_path) as fin:
            lora_kwargs = json.load(fin)
    os.system("cp {} {}".format(args.lora_config, os.path.join(args.outputdir, 'lora_config.json')))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        cache_dir="/mnt/bn/tiktok-mm-5/aiic/users/guangzhisun/UnlearnForDataContamination/cache",
        trust_remote_code=True,
    )
    model = Model(
        args.model_path,
        tokenizer,
        losstype=args.losstype,
        **lora_kwargs,
    )
    if args.probetype != "":
        model.initialize_probe(args.probetype)
        model.probelayer = args.probelayer

    if args.load_from != "":
        modelpath = os.path.join(args.load_from, "pytorch_model.pt")
        trained_params = torch.load(modelpath)
        msg = model.load_state_dict(trained_params, strict=False)
        model.merge_and_reload(os.path.join(args.outputdir, 'base_model/'), new_lora_kwargs)

    traindata = SupervisedDataset(
        args.train_data_path,
        tokenizer,
        unlearnmode=args.unlearnmode,
        probe=args.probetype,
    )
    valdata = SupervisedDataset(
        args.val_data_path,
        tokenizer,
        unlearnmode=args.unlearnmode,
        validation=True,
        probe=args.probetype,
    )
    train_dataloader = DataLoader(
        traindata,
        batch_size=args.batch_size,
        shuffle=True if "order" not in args.train_data_path else False,
        # sampler=DistributedSampler(traindata),
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        valdata,
        batch_size=1,
        shuffle=False,
        # sampler=DistributedSampler(traindata),
        collate_fn=collate_fn,
    )

    ## Optimiser
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if accelerator.state.deepspeed_plugin is None or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config:
        if args.probetype != "":
            optimizer = SGD(optimizer_grouped_parameters, lr=args.learning_rate)
        else:
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    else:
        optimizer = accelerate.utils.DummyOptim(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    num_warmup_steps = args.num_warmup_steps * max_train_steps

    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )
    else:
        lr_scheduler = accelerate.utils.DummyScheduler(
            optimizer, total_num_steps=max_train_steps, warmup_num_steps=num_warmup_steps
        )
    model, optimizer, train_dataloader, lr_scheduler, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler, val_dataloader)

    for v, parameter in model.llm.named_parameters():
        if "lora" not in v:
            parameter.requires_grad = False

    print("Start training")
    best_val_loss = 10000
    for epoch in range(args.num_train_epochs):
        model.train()
        model = train_one_epoch(
            args,
            epoch,
            model,
            train_dataloader,
            traindata,
            optimizer,
            lr_scheduler,
            tokenizer,
            rank,
            world_size,
        )
        accuracy = eval_one_epoch(
            args,
            epoch,
            model,
            val_dataloader,
            tokenizer,
            rank,
            world_size,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        # Save models
        if accelerator.is_main_process:
            logging(f"Saving at Epoch {epoch} | Learning rate: {current_lr}", args.logfile)
            # save_checkpoint(model, tokenizer, args.outputdir, epoch)
        save_checkpoint(model, tokenizer, args.outputdir, epoch, "final")


def train_one_epoch(
    args,
    epoch,
    model,
    train_dataloader,
    traindata,
    optimizer,
    lr_scheduler,
    tokenizer,
    rank,
    world_size,
):
    optimizer.zero_grad()
    trainsize = len(train_dataloader)
    start = time.time()
    for i, batch in enumerate(train_dataloader):
        input_ids, input_masks, complete_inputs, complete_labels, answer = batch
        if args.unlearnmode:
            answer_ids = torch.tensor([tokenizer.encode(ans)[1] for ans in answer]).to(input_ids.device)
            loss = model(complete_inputs, complete_labels, input_masks, unlearn_target=answer_ids, alpha=args.alpha)
        elif args.probetype != "":
            loss, classpred = model(input_ids, complete_labels, input_masks)
        else:
            loss = model(complete_inputs, complete_labels, input_masks)
        loss = loss / args.gradient_accumulation_steps
        # loss.backward()
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if (i + 1) % args.log_interval == 0 and accelerator.is_main_process:
            elasped_time = time.time() - start
            logging(f"Epoch {epoch} | Batch {i}/{trainsize} | Loss: {loss:.3f} | time {elasped_time}", args.logfile)

        if (i + 1) % args.save_interval == 0 and accelerator.is_main_process:
            logging(f"Saving at Step {i+1}", args.logfile)
            save_checkpoint(model, tokenizer, args.outputdir, epoch, i+1)

    return model


def eval_one_epoch(
    args,
    epoch,
    model,
    val_dataloader,
    tokenizer,
    rank,
    world_size,
):
    total_loss = 0
    total_sample = 0
    if args.probetype != "":
        total_loss, total_sample = [], []
    start = time.time()
    with torch.no_grad():
        logging("="*89, args.logfile)
        for i, batch in tqdm(enumerate(val_dataloader)):
            input_ids, input_masks, complete_inputs, complete_labels, answer = batch
            if args.unlearnmode:
                letter_ids = torch.tensor([tokenizer.encode(ans)[1] for ans in letters])
                pred = model.generate(input_ids, do_sample=False, return_dict=True, max_new_tokens=2)
                bary = torch.softmax(pred["logits"][0], dim=-1)
                bary = bary[torch.arange(bary.size(0)), letter_ids]
                pred = torch.sqrt(((complete_labels - bary) ** 2).mean(dim=-1))
                total_loss += pred.mean()
            elif args.probetype != "":
                loss, classoutput = model(input_ids, complete_labels)
                if "alpha" not in args.probetype:
                    classoutput = torch.softmax(classoutput, dim=-1)[:, -1]
                else:
                    classoutput = classoutput[:, -1]
                total_sample.extend(classoutput.tolist())
                total_loss.extend(complete_labels.tolist())
                # total_loss += ((classoutput > 0.5) == complete_labels).sum()
            else:
                pred = model.generate(input_ids, do_sample=False, max_new_tokens=32)
                pred = tokenizer.decode(pred[0, input_ids.size(1):], skip_special_tokens=True)
                if pred == answer[0]:
                    total_loss += 1
                elif len(pred.split()) > 2 and i % 10 == 0:
                    print("Answer:", answer)
                    print("Prediction:", pred)
                    if i > 100:
                        break
            if args.probetype == "":
                total_sample += 1
        logging("="*89, args.logfile)

    elasped_time = time.time() - start
    if args.probetype == "":
        accuracy = total_loss / max(total_sample, 1.0)
    else:
        if "alpha" in args.probetype:
            accuracy = stats.pearsonr(total_loss, total_sample)[0]
        else:
            precision, recall, thresholds = precision_recall_curve(total_loss, total_sample)
            accuracy = auc(recall, precision)
        total_sample = len(total_sample)
    logging("="*89, args.logfile)
    logging(f"Epoch {epoch} | Validation Samples {total_sample} | Validation Acc: {accuracy} | time {elasped_time}", args.logfile)
    logging("="*89, args.logfile)
    return accuracy


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
        "--train_data_path",
        type=str,
        default="./hf_models",
        help="Path to the train data file",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default="./hf_models",
        help="Path to the validation data file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0, help="Weight decay."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=float, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default='./log.txt',
        help="Path to the log file",
    )
    parser.add_argument(
        "--outputdir",
        type=str,
        default='./exp/clip_vlm',
        help="Path to the output dir",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="log interval",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=0,
        help="Saving interval",
    )
    parser.add_argument(
        "--master_port",
        type=str,
        default='12355',
        help="Master port number",
    )
    parser.add_argument(
        "--lora_config",
        type=str,
        default="data/lora_config.json",
        help="LoRA configuration",
    )
    parser.add_argument(
        "--losstype",
        type=str,
        default="mse",
        help="type of loss to train forget model",
    )
    parser.add_argument(
        "--load_from",
        type=str,
        default="",
        help="path to load checkpoint from",
    )
    parser.add_argument(
        "--unlearnmode",
        action='store_true',
        help="Train with unlearning mode",
    )
    parser.add_argument(
        "--probetype",
        type=str,
        default="",
        help="Train with linear probes",
    )
    parser.add_argument(
        "--probelayer",
        type=int,
        default=-1,
        help="Which layer to probe",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="alpha",
    )
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    print(world_size)
    main(0, args, world_size)