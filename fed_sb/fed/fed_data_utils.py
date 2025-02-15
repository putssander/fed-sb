import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import numpy as np
import pandas as pd
from peft import get_peft_model, LoraConfig, TaskType
from fed_sb.utils.data_utils import *


def create_client_dataloaders(dataset, args):
    client_data = [[] for _ in range(args.num_clients)]
    for data in dataset:
        client_idx = np.random.randint(args.num_clients)
        client_data[client_idx].append(data)
    return [
        DataLoader(cd, batch_size=args.batch_size, shuffle=True) for cd in client_data
    ]


def create_client_datamodules_it(dataset,tokenizer, args):
    client_data = [[] for _ in range(args.num_clients)]
    for data in dataset:
        client_idx = np.random.randint(args.num_clients)
        client_data[client_idx].append(data)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return [ dict(train_dataset=cd, data_collator=data_collator)  for cd in client_data ]


def create_client_datamodules_cr(dataset,tokenizer, args):
    client_data = [[] for _ in range(args.num_clients)]
    for data in dataset:
        client_idx = np.random.randint(args.num_clients)
        client_data[client_idx].append(data)

        data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            )
       
    return [
        dict(train_dataset=cd, data_collator=data_collator) 
    for cd in client_data ]