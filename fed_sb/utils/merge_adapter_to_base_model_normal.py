from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
import argparse
import torch
import gc
import os
import json
from pathlib import Path
from safetensors import safe_open
from .initialization_utils import find_and_initialize


def merge_normal(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda"},
    )

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

    if "llama" in args.base_model:
        if "Llama-3" in args.base_model:
            tokenizer = AutoTokenizer.from_pretrained(
                args.base_model,
                use_fast=True,
                device_map={"": "cuda"},
            )
        else:
            tokenizer = LlamaTokenizer.from_pretrained(
                args.base_model,
                use_fast=True,
                device_map={"": "cuda"},
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            use_fast=True,
            device_map={"": "cuda"},
        )

    # Instead of manually loading config and using get_peft_model,
    # directly load the PEFT model using from_pretrained
    model = PeftModel.from_pretrained(
        model,
        args.adapter,
    )

    print("merging the LoRA into the base model.")
    model = model.merge_and_unload()
    print("Saving the merged model to disk.")
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge Adapter to Base Model')
    parser.add_argument('--base_model', type=str)
    parser.add_argument('--adapter', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    merge_normal(args)