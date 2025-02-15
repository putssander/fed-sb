import torch
from transformers import AutoModel
from transformers import AutoTokenizer
from peft import PeftModel
from transformers import AutoModelForCausalLM
from safetensors import safe_open
from safetensors.torch import load_file
import argparse
import os
import json
from fed_sb.fed.fed_agg import *
import yaml
from fed_sb.utils import merge_adapter_to_base_model, merge_adapter_to_base_model_normal
from fed_sb.utils.initialization_utils import *
import shutil

def load_model_with_lora(base_model_name, lora_weights_path, args):
    """
    Load a base model with LoRA weights
    
    Args:
        base_model_name (str): Hugging Face model name or path to base model
        lora_weights_path (str): Path to saved LoRA weights directory
    
    Returns:
        model: Combined model with LoRA weights
        tokenizer: Associated tokenizer
    """
    # 1. Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map={"": "cuda"},
        torch_dtype=torch.bfloat16
    )
    
    if "llama" in base_model_name:
        if "Llama-3" in base_model_name:
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                use_fast=True,
                device_map={"": "cuda"},
            )
        else:
            tokenizer = LlamaTokenizer.from_pretrained(
                base_model_name,
                use_fast=True,
                device_map={"": "cuda"},
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            use_fast=True,
            device_map={"": "cuda"},
        )
    
    tokenizer.pad_token = tokenizer.eos_token

    
    # 3. Load and apply LoRA weights
    model = PeftModel.from_pretrained(
        base_model,
        lora_weights_path,
        device_map={"": "cuda"},
    )
    
    return model, tokenizer

def load_model_with_lora_sb(base_model_name, lora_weights, lora_weights_path, args):
    """
    Load a base model with LoRA weights
    
    Args:
        base_model_name (str): Hugging Face model name or path to base model
        lora_weights_path (str): Path to saved LoRA weights directory
    
    Returns:
        model: Combined model with LoRA weights
        tokenizer: Associated tokenizer
    """
    # 1. Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map={"": "cuda"},
        torch_dtype=torch.bfloat16
    )
    
    # 2. Load the tokenizer
    if "llama" in base_model_name:
        if "Llama-3" in base_model_name:
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                use_fast=True,
                device_map={"": "cuda"},
            )
        else:
            tokenizer = LlamaTokenizer.from_pretrained(
                base_model_name,
                use_fast=True,
                device_map={"": "cuda"},
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            use_fast=True,
            device_map={"": "cuda"},
        )
    
    tokenizer.pad_token = tokenizer.eos_token

    
    # 3. Load and apply LoRA weights
    model = PeftModel.from_pretrained(
        base_model,
        lora_weights_path,
        device_map={"": "cuda"},
    )
    
    with open("config/reconstruct_config.yaml", 'r') as stream:
        reconstr_config = yaml.load(stream, Loader=yaml.FullLoader)

    with open(os.path.join(lora_weights_path, "adapter_config.json")) as f:
        lora_config_dict = json.load(f)
        lora_config = LoraConfig(**lora_config_dict)
        
    adapter_name = "default"
    peft_config_dict = {adapter_name: lora_config}
    
    print(lora_config)
    
    reconstr_config['svd']['rank'] = args.lora_r

    find_and_initialize(model, peft_config_dict, adapter_name=adapter_name, 
                        reconstr_type='svd', 
                        writer=None, reconstruct_config=reconstr_config)

    model_state_dict = model.state_dict()
    for key in model_state_dict.keys():
        if ('lora_A' in key) or ('lora_B' in key):
            model_state_dict[key] = lora_weights[key]

    model.load_state_dict(model_state_dict)
    
    return model, tokenizer


def load_aggregated_merge_adapters(args):
    """
    Load n adapters from the specified directory and attach them to the model.
    
    Args:
        directory (str): The directory containing the adapter files.
        n (int): The number of adapters to load.
        
    Returns:
        model (torch.nn.Module): The model with the loaded adapters.
    """
    # Load the base model
    dir_path = args.dir_path
    model_name = args.model_name
    agg_type = args.agg_type

    # Iterate through the folders in the directory
    adapter_state_dicts = []
    for folder_name in os.listdir(dir_path):
        if "final_model" in folder_name:
            adapter_path = os.path.join(dir_path, folder_name)
            adapter_path = os.path.join(adapter_path, "adapter_model.safetensors")
            # Load the adapter
            adapter = load_file(adapter_path, "cuda")
            # Merge the adapter into the model
            adapter_state_dicts.append(adapter)
            
    client_models = []
    for adapter_state_dict in adapter_state_dicts:
        new_weights = {}
        for key, value in adapter_state_dict.items():

            if args.agg_type == "fed-sb":
                if 'lora_A' in key:                # Insert 'default' after 'lora_A'
                    new_key = key.replace('lora_A', 'lora_A.default')
                elif 'lora_B' in key:
                    # Insert 'default' after 'lora_B'
                    new_key = key.replace('lora_B', 'lora_B.default')
                elif 'lora_latent' in key:
                    new_key = key.replace('_lora_latent', '.default_lora_latent')
                else:
                    new_key = key

            else:
                if 'lora_A' in key:                # Insert 'default' after 'lora_A'
                    new_key = key.replace('lora_A', 'lora_A.default')
                elif 'lora_B' in key:
                    # Insert 'default' after 'lora_B'
                    new_key = key.replace('lora_B', 'lora_B.default')
                else:
                    new_key = key                

            new_weights[new_key] = value

        client_models.append(new_weights)
    
    
    if args.agg_type == "fed-sb":
        global_model, tokenizer = load_model_with_lora_sb(model_name, new_weights, dir_path+"/final_model_0", args)
    else:
        global_model, tokenizer = load_model_with_lora(model_name, dir_path+"/final_model_0", args)


        
    if agg_type == "fedex":
        global_model = aggregate_models_fedex(global_model, client_models, args)
    elif agg_type == "fed-it":
        global_model = aggregate_models_fed_it(global_model, client_models)
    elif agg_type == "ffa":
        global_model = aggregate_models_ffa_2(global_model, client_models)
    elif agg_type == "fed-sb":
        global_model = aggregate_models_fed_sb(global_model, client_models)

    if args.agg_type == "fed-sb":
        for param in global_model.parameters():
            param.data = param.data.contiguous()

    save_directory_final_model = os.path.join(dir_path, "final_model")
    global_model.save_pretrained(save_directory_final_model)

    # Clean up final model directories used for aggregation
    for i in range(len(client_models)):
        model_dir = os.path.join(dir_path, f"final_model_{i}")
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
            print(f"Deleted {model_dir}")

    global_model = global_model.merge_and_unload()
    
    
    save_directory_merged_model = os.path.join(dir_path, "merged_model")
    global_model.save_pretrained(save_directory_merged_model)
    
    save_directory_tokenizer = os.path.join(dir_path, "merged_model")
    tokenizer.save_pretrained(save_directory_tokenizer)
    
    return save_directory_merged_model


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dir_path", type=str, default="model_dir")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--agg_type", type=str, default="fedex")
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=512)

    args = parser.parse_args()

    args.lora_alpha = args.lora_r

    save_directory = load_aggregated_merge_adapters(args)