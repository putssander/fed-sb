import torch
    # Import modules for initialization if not already imported
import math
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)
from datasets import load_dataset
import numpy as np
from peft import (
    get_peft_model,
    AdaLoraModel,
    AdaLoraConfig,
    TaskType,
    LoraConfig,
    prepare_model_for_kbit_training,
)
from fed_sb.utils.data_utils import *
import argparse
from copy import deepcopy
from tqdm import tqdm

from peft.utils import _get_submodules

def create_model_tokenizer_it(args):

    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        device_map="auto",
        torch_dtype = torch.bfloat16
    ) 
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        model_max_length=args.max_seq_length,
        padding="max_length",
    )

    tokenizer.pad_token_id = tokenizer.eos_token_id

    #model.to(args.device)

    return model, tokenizer

def create_model_tokenizer_cr(args):

    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        device_map="auto",
        torch_dtype = torch.bfloat16) 
    
    if "llama" in args.model:

        if "Llama-3" in args.model:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model,
                use_fast=True,
                model_max_length=args.max_seq_length,
                padding="max_length",
            )
        else:
            tokenizer = LlamaTokenizer.from_pretrained(
                args.model,
                use_fast=True,
                model_max_length=args.max_seq_length,
                padding="max_length",
            )

    else:

        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            use_fast=True,
            model_max_length=args.max_seq_length,
            padding="max_length",
        )

    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"


    return model, tokenizer

def create_peft_model_it(model, args):

    peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            task_type="CAUSAL_LM",
        )

    model = get_peft_model(model, peft_config)

    return model, peft_config

def create_peft_FFA_model_it(model, args):

    peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            task_type="CAUSAL_LM",
        )

    model = get_peft_model(model, peft_config)

    # This loop ensures that every lora_A parameter is (re)initialized
    # in a deterministic manner and then frozen.
    base_seed = 42
    for name, param in model.named_parameters():
        i = 0
        if "lora_A" in name:
            # Create a unique seed for each parameter (using hash of the parameter name)
            unique_seed = base_seed + i
            i += 1
            with torch.random.fork_rng(devices=[param.device]):
                torch.random.manual_seed(unique_seed)
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            param.requires_grad = False

    return model, peft_config

def create_peft_model_cr(model, args):

    peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

    model = get_peft_model(model, peft_config)

    return model, peft_config

def create_peft_FFA_model_cr(model, args):

    peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

    model = get_peft_model(model, peft_config)

    base_seed = 42
    for name, param in model.named_parameters():
        i = 0
        if "lora_A" in name:
            # Create a unique seed for each parameter (using hash of the parameter name)
            unique_seed = base_seed + i
            i += 1
            with torch.random.fork_rng(devices=[param.device]):
                torch.random.manual_seed(unique_seed)
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            param.requires_grad = False

    return model, peft_config