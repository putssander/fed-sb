import torch
from torch.utils.data import DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
# from data_utils import *
from fed_sb.models import *
from sklearn.metrics import matthews_corrcoef
import numpy as np
import torch.nn as nn

def aggregate_models_fed_it(global_model, client_dicts):

    global_dict = global_model.state_dict()

    for k in global_dict.keys():
        if "lora" in k:  # Only aggregate LoRA parameters
            global_dict[k] = torch.stack(
                [client_dicts[i][k].float() for i in range(len(client_dicts))], 0
            ).mean(0)

        if "classifier" in k:
            global_dict[k] = torch.stack(
                [client_dicts[i][k].float() for i in range(len(client_dicts))], 0
            ).mean(0)

    global_model.load_state_dict(global_dict)

    return global_model

def aggregate_models_fed_sb(global_model, client_dicts):

    global_dict = global_model.state_dict()
    # client_dicts = [client_model.state_dict() for client_model in client_models]
    for k in global_dict.keys():
        if "default_lora_latent_mapping" in k:  # Only aggregate LoRA parameters
            global_dict[k] = torch.stack(
                [client_dicts[i][k].float() for i in range(len(client_dicts))], 0
            ).mean(0)

        if "classifier" in k:
            global_dict[k] = torch.stack(
                [client_dicts[i][k].float() for i in range(len(client_dicts))], 0
            ).mean(0)

    global_model.load_state_dict(global_dict)

    return global_model

def aggregate_models_ffa(global_model, client_dicts):

    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        if "lora_B" in k:  # Only aggregate LoRA B parameters
            global_dict[k] = torch.stack(
                [client_dicts[i][k].float() for i in range(len(client_dicts))], 0
            ).mean(0)

        if "classifier" in k:
            global_dict[k] = torch.stack(
                [client_dicts[i][k].float() for i in range(len(client_dicts))], 0
            ).mean(0)

    global_model.load_state_dict(global_dict)

    return global_model

def aggregate_models_fedex(global_model, client_dicts, args):
    printer = 0 
    global_dict = global_model.state_dict()

    for k in global_dict.keys():
        if "classifier" in k:
            global_dict[k] = torch.stack(
                [client_dicts[i][k].float() for i in range(len(client_dicts))], 0
            ).mean(0)

    for client_dict in client_dicts:
        for k in global_dict.keys():
            if "classifier" in k:
                client_dict[k] = global_dict[k]

    for name, module in global_model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            lora_A_keys = name + ".lora_A.default.weight"
            lora_B_keys = name + ".lora_B.default.weight"
            base_layer_keys = name + ".base_layer.weight"

            lora_A_weights = torch.stack(
                [client_dict[lora_A_keys].detach() for client_dict in client_dicts]
            )
            lora_B_weights = torch.stack(
                [client_dict[lora_B_keys].detach() for client_dict in client_dicts]
            )

            # M shape: (d, k)
            M = sum(
                lora_B_weights[i] @ lora_A_weights[i] for i in range(len(client_dicts))
            ) / len(client_dicts)
            
            lora_A_avg = lora_A_weights.mean(0)
            lora_B_avg = lora_B_weights.mean(0)

            scaling_factor = args.lora_alpha / args.lora_r

            residue = M - lora_B_avg @ lora_A_avg
            
            global_dict[name + ".lora_A.default.weight"] = lora_A_avg
            global_dict[name + ".lora_B.default.weight"] = lora_B_avg
            global_dict[name + ".base_layer.weight"] += residue* scaling_factor
             

    global_model.load_state_dict(global_dict)
    return global_model
