import zipfile
import urllib.request
import os

import argparse
import pandas as pd
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore")
import torch
import torch.nn as nn
import transformers
from torch.utils.data import TensorDataset
from transformers.data.processors.utils import InputExample
from transformers.data.processors.glue import glue_convert_examples_to_features
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from peft import get_peft_model, LoraConfig, TaskType
from opacus import PrivacyEngine
import numpy as np
import wandb
import yaml
import math
from fed_sb.utils.data_utils import *
from fed_sb.models import *
from fed_sb.utils.initialization_utils import *
from fed_sb.utils.gradient_utils import *
from fed_sb.utils.misc import *
from fed_sb.utils.merge_adapter_to_base_model import *

from fed_sb.fed.fed_data_utils import *
from fed_sb.fed.fed_agg import *
from fed_sb.fed.fed_utils import *
from fed_sb.utils.data_utils import *
from fed_sb.models import *
from fed_sb.utils.initialization_utils import *
from fed_sb.utils.gradient_utils import *
from fed_sb.utils.misc import *

LABEL_LIST = ['contradiction', 'entailment', 'neutral']
MAX_SEQ_LENGHT = 128

def accuracy(preds, labels):
    return (preds == labels).mean()

# define evaluation cycle
def evaluate(model, test_dataloader, device):
    model.eval()

    loss_arr = []
    accuracy_arr = []

    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         batch[3]}

            outputs = model(**inputs)
            loss, logits = outputs[:2]

            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
            labels = inputs['labels'].detach().cpu().numpy()

            loss_arr.append(loss.item())
            accuracy_arr.append(accuracy(preds, labels))

    model.train()
    return np.mean(loss_arr), np.mean(accuracy_arr)

def _create_examples(df, set_type):
    """ Convert raw dataframe to a list of InputExample. Filter malformed examples
    """
    examples = []
    for index, row in df.iterrows():
        if row['gold_label'] not in LABEL_LIST:
            continue
        if not isinstance(row['sentence1'], str) or not isinstance(row['sentence2'], str):
            continue

        guid = f"{index}-{set_type}"
        examples.append(
            InputExample(guid=guid, text_a=row['sentence1'], text_b=row['sentence2'], label=row['gold_label']))
    return examples

def _df_to_features(df, set_type, tokenizer):
    """ Pre-process text. This method will:
    1) tokenize inputs
    2) cut or pad each sequence to MAX_SEQ_LENGHT
    3) convert tokens into ids

    The output will contain:
    `input_ids` - padded token ids sequence
    `attention mask` - mask indicating padded tokens
    `token_type_ids` - mask indicating the split between premise and hypothesis
    `label` - label
    """
    examples = _create_examples(df, set_type)

    #backward compatibility with older transformers versions
    legacy_kwards = {}
    from packaging import version
    if version.parse(transformers.__version__) < version.parse("2.9.0"):
        legacy_kwards = {
            "pad_on_left": False,
            "pad_token": tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            "pad_token_segment_id": 0,
        }

    return glue_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        label_list=LABEL_LIST,
        max_length=MAX_SEQ_LENGHT,
        output_mode="classification",
        **legacy_kwards,
    )

def _features_to_dataset(features):
    """ Convert features from `_df_to_features` into a single dataset
    """
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long
    )
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long
    )
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(
        all_input_ids, all_attention_mask, all_token_type_ids, all_labels
    )

    return dataset

def train_snli(args):

    wandb.init(project="project-name")
    wandb.config.update(args)

    if args.dataset_not_processed:
        snli_folder = os.path.join(args.data_dir, "snli_1.0")
        os.listdir(snli_folder)
        
        train_path =  os.path.join(snli_folder, "snli_1.0_train.txt")
        dev_path = os.path.join(snli_folder, "snli_1.0_dev.txt")

        df_train = pd.read_csv(train_path, sep='\t')
        df_test = pd.read_csv(dev_path, sep='\t')

        tokenizer = BertTokenizer.from_pretrained(args.model_name)

        train_features = _df_to_features(df_train, "train", tokenizer)
        test_features = _df_to_features(df_test, "test", tokenizer)

        train_dataset = _features_to_dataset(train_features)
        test_dataset = _features_to_dataset(test_features)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
        test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.batch_size)

        # Save dataloaders to disk
        save_dir = os.path.join(args.data_dir, "processed_dataloaders")
        os.makedirs(save_dir, exist_ok=True)

        train_save_path = os.path.join(save_dir, "train_dataloader.pt")
        test_save_path = os.path.join(save_dir, "test_dataloader.pt")

        # Save the datasets and their properties
        torch.save({
            'dataset': train_dataset,
            'batch_size': args.batch_size,
            'sampler': None  # Default sampler
        }, train_save_path)

        torch.save({
            'dataset': test_dataset,
            'batch_size': args.batch_size,
            'sampler': SequentialSampler(test_dataset)
        }, test_save_path)

        print(f"Saved train dataloader to {train_save_path}")
        print(f"Saved test dataloader to {test_save_path}")

    else:
        # Modified else condition to load DataLoader objects directly
        train_loaded = torch.load(os.path.join(args.data_dir, "processed_dataloaders", "train_dataloader.pt"))
        test_loaded = torch.load(os.path.join(args.data_dir, "processed_dataloaders", "test_dataloader.pt"))

        train_dataset = train_loaded['dataset']
        test_dataset = test_loaded['dataset']
        train_batch_size = train_loaded['batch_size']
        test_batch_size = test_loaded['batch_size']
        test_sampler = test_loaded['sampler']

        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=test_batch_size)

    if args.agg_type == 'fed-sb':

        model_name = args.model_name
        config = BertConfig.from_pretrained(
            model_name,
            num_labels=3,
        )
        tokenizer = BertTokenizer.from_pretrained(
            model_name,
            do_lower_case=False,
        )
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            config=config,
        )
        model = model.to(args.device)

        named_grads = None

        total_training_steps = len(train_dataloader) * args.epochs
        eff_lr = args.lr/total_training_steps

        named_grads = estimate_and_process_grads_torch_snli(
            model=model,
            dataloader=train_dataloader,
            lr=eff_lr,
            num_samples=550,
        )

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # our particular task is sequence classification
            inference_mode=False,  # Enable training mode
            r=args.lora_r,  # Low-rank dimension
            lora_alpha=args.lora_alpha,  # Alpha scaling factor
            lora_dropout=args.lora_dropout,  # Dropout for LoRA layers
        )

        model_with_lora = get_peft_model(model, lora_config)

        with open("config/reconstruct_config.yaml", 'r') as stream:
            reconstr_config = yaml.load(stream, Loader=yaml.FullLoader)
        
        adapter_name = "default"
        peft_config_dict = {adapter_name: lora_config}

        reconstr_config['svd']['rank'] = args.lora_r


        named_grads_new = {}
        for keys in named_grads.keys():
            keys_new = 'base_model.model.' + keys
            named_grads_new[keys_new] = named_grads[keys]

        find_and_initialize_grad(
            model=model_with_lora,
            peft_config=peft_config_dict,
            adapter_name=adapter_name,
            reconstr_type='svd',
            reconstruct_config=reconstr_config,
            writer=None,
            named_grads=named_grads_new,
        )


        trainable_params = sum(p.numel() for p in model_with_lora.parameters() if p.requires_grad)
        print(f"Total trainable parameters with LoRA: {trainable_params:,}") # ~1M

        trainable_params = sum(p.numel() for p in model_with_lora.parameters() if p.requires_grad)
        print(f"Total trainable parameters with LoRA SB after freezing: {trainable_params:,}") 

        trainable_params_lora = sum(p.numel() for (name, p) in model_with_lora.named_parameters() if p.requires_grad 
        and 'lora' in name)
        print(f"Total trainable parameters with LoRA SB after freezing and type LoRA: {trainable_params_lora:,}")

        wandb.log({
            "trainable_params_lora": trainable_params,
            "trainable_params_only_lora": trainable_params_lora,
        })
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8)
        model = model.train()

        privacy_engine = PrivacyEngine()
        criterion = nn.CrossEntropyLoss(reduction="mean")

        model_lora, optimizer_lora, criterion_lora, train_dataloader = (
            privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_dataloader,
                criterion=criterion,
                target_delta=1 / len(train_dataloader),
                target_epsilon=args.epsilon,
                epochs=args.epochs,
                max_grad_norm=args.max_grad_norm,
                grad_sample_mode="ghost",
            )
        )

        model_lora = model_lora.to(args.device)
        model_lora = model_lora.train()

        for epoch in range(1, args.epochs + 1):
            losses = []
            for step, batch in enumerate(tqdm(train_dataloader)):
                optimizer_lora.zero_grad()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3],
                }
                outputs = model_lora(**inputs)  # output = loss, logits, hidden_states, attentions
                loss = criterion_lora(outputs[1], batch[3])
                loss.backward()
                optimizer_lora.step()
                losses.append(loss.item())
                wandb.log({
                    "train_loss": loss.item()
                })
                if (step > 0 and step % args.log_interval == 0):
                    train_loss = np.mean(losses)
                    eval_loss, eval_accuracy = evaluate(model_lora, test_dataloader, args.device)
                    eps = privacy_engine.get_epsilon(1 / len(train_dataloader))
                    wandb.log({
                        "eval_loss": eval_loss,
                        "eval_accuracy": eval_accuracy,
                        "epsilon": eps
                    })
                    print(
                        f"Epoch: {epoch} | "
                        f"Step: {step} | "
                        f"Train loss: {train_loss:.3f} | "
                        f"Eval loss: {eval_loss:.3f} | "
                        f"Eval accuracy: {eval_accuracy:.3f} | "
                        f"ɛ: {eps:.2f}"
                    )
        # Final evaluation after training completes
        eval_loss, eval_accuracy = evaluate(model_lora, test_dataloader, args.device)
        eps = privacy_engine.get_epsilon(1 / len(train_dataloader))
        wandb.log({
            "eval_loss": eval_loss,
            "eval_accuracy": eval_accuracy,
            "epsilon": eps
        })
        print(
            f"=== Final Evaluation ===\n"
            f"Eval loss: {eval_loss:.3f} | "
            f"Eval accuracy: {eval_accuracy:.3f} | "
            f"Final ɛ: {eps:.2f}"
        )

    else:
        model_name = args.model_name
        config = BertConfig.from_pretrained(
            model_name,
            num_labels=3,
        )
        tokenizer = BertTokenizer.from_pretrained(
            model_name,
            do_lower_case=False,
        )
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            config=config,
        )
        

        model = model.to(args.device)



        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # our particular task is sequence classification
            inference_mode=False,  # Enable training mode
            r=args.lora_r,  # Low-rank dimension
            lora_alpha=args.lora_alpha,  # Alpha scaling factor
            lora_dropout=args.lora_dropout,  # Dropout for LoRA layers
        )

        model_with_lora = get_peft_model(model, lora_config)
        
        if args.agg_type == 'ffa':
            for name, param in model_with_lora.named_parameters():
                i = 0
                if "lora_A" in name:
                    unique_seed = 42 + i
                    i += 1
                    with torch.random.fork_rng(devices=[param.device]):
                        torch.random.manual_seed(unique_seed)
                        nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                    param.requires_grad = False
                    
        trainable_params = sum(p.numel() for p in model_with_lora.parameters() if p.requires_grad)
        print(f"Total trainable parameters with LoRA: {trainable_params:,}") # ~1M

        trainable_params = sum(p.numel() for p in model_with_lora.parameters() if p.requires_grad)
        print(f"Total trainable parameters with LoRA after freezing: {trainable_params:,}") 

        trainable_params_lora = sum(p.numel() for (name, p) in model_with_lora.named_parameters() if p.requires_grad 
        and 'lora' in name)
        print(f"Total trainable parameters with LoRA after freezing and type LoRA: {trainable_params_lora:,}")

        wandb.log({
            "trainable_params_lora": trainable_params,
            "trainable_params_only_lora": trainable_params_lora,
        })
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8)
        model = model.train()

        privacy_engine = PrivacyEngine()
        criterion = nn.CrossEntropyLoss(reduction="mean")

        model_lora, optimizer_lora, criterion_lora, train_dataloader = (
            privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_dataloader,
                criterion=criterion,
                target_delta=1 / len(train_dataloader),
                target_epsilon=args.epsilon,
                epochs=args.epochs,
                max_grad_norm=args.max_grad_norm,
                grad_sample_mode="ghost",
            )
        )

        model_lora = model_lora.to(args.device)
        model_lora = model_lora.train()

        for epoch in range(1, args.epochs + 1):
            losses = []
            for step, batch in enumerate(tqdm(train_dataloader)):
                optimizer_lora.zero_grad()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3],
                }
                outputs = model_lora(**inputs)  # output = loss, logits, hidden_states, attentions
                loss = criterion_lora(outputs[1], batch[3])
                loss.backward()
                optimizer_lora.step()
                losses.append(loss.item())
                wandb.log({
                    "train_loss": loss.item()
                })
                if (step > 0 and step % args.log_interval == 0):
                    train_loss = np.mean(losses)
                    eval_loss, eval_accuracy = evaluate(model_lora, test_dataloader, args.device)
                    eps = privacy_engine.get_epsilon(1 / len(train_dataloader))
                    wandb.log({
                        "eval_loss": eval_loss,
                        "eval_accuracy": eval_accuracy,
                        "epsilon": eps
                    })
                    print(
                        f"Epoch: {epoch} | "
                        f"Step: {step} | "
                        f"Train loss: {train_loss:.3f} | "
                        f"Eval loss: {eval_loss:.3f} | "
                        f"Eval accuracy: {eval_accuracy:.3f} | "
                        f"ɛ: {eps:.2f}"
                    )

        # Final evaluation after training completes
        eval_loss, eval_accuracy = evaluate(model_lora, test_dataloader, args.device)
        eps = privacy_engine.get_epsilon(1 / len(train_dataloader))
        wandb.log({
            "eval_loss": eval_loss,
            "eval_accuracy": eval_accuracy,
            "epsilon": eps
        })
        print(
            f"=== Final Evaluation ===\n"
            f"Eval loss: {eval_loss:.3f} | "
            f"Eval accuracy: {eval_accuracy:.3f} | "
            f"Final ɛ: {eps:.2f}"
        )

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="bert-base-cased")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_physical_batch_size", type=int, default=8)
    parser.add_argument("--data_dir", type=str, default="path to data")
    parser.add_argument("--dataset_not_processed", action='store_true')
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    parser.add_argument("--log_interval", type=int, default=2500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=3)
    parser.add_argument("--agg_type", type=str, default="fed-sb")

    args = parser.parse_args()

    if args.agg_type == 'fed-sb':
        args.lora_alpha = args.lora_r

    args.dataset_not_processed = True

    train_snli(args)

if __name__ == "__main__":
    main()