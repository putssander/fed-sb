import torch
import os
from datetime import datetime
import json


def tensor_to_list(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_list(v) for v in obj]
    else:
        return obj


def save_dict_to_json(data_dict, args, base_path):
    # Create a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"compare_dict_rounds_{timestamp}.json"
    file_path = os.path.join(base_path, filename)

    # Ensure the directory exists
    os.makedirs(base_path, exist_ok=True)

    # Combine data_dict and args
    combined_dict = {"args": vars(args), "data": data_dict}

    # Convert tensors to lists
    json_serializable_dict = tensor_to_list(combined_dict)

    # Write JSON data to the file
    with open(file_path, "w") as json_file:
        json.dump(json_serializable_dict, json_file, indent=2)

    print(f"Data and args saved to {file_path}")

def extract_lora_state_dicts(model):
    # Initialize dictionary to store LoRA state dicts
    lora_state_dicts = {}
    
    # Get the full state dict
    state_dict = model.state_dict()
    
    # Iterate through all keys and filter for LoRA
    for key in state_dict.keys():
        if 'lora' in key.lower():  # Case-insensitive check for 'lora'
            # Detach tensor from computational graph and create a copy
            lora_state_dicts[key] = state_dict[key].detach().clone()
            
    return lora_state_dicts

def load_lora_state_dict(model, state_dict):
    """
    Load only the LoRA weights into the model.
    Args:
        model: The PEFT model
        state_dict: Dictionary containing the LoRA state
    """
    for name, param in model.named_parameters():
        if 'lora_' in name:
            if name in state_dict:
                param.data = state_dict[name].clone()