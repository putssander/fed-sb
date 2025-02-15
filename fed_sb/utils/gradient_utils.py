import torch
from tqdm.auto import tqdm
from copy import deepcopy
from typing import Dict, List
from accelerate import Accelerator
import math
import gc
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from .offload_utils_for_quant import show_gpu_and_cpu_memory, OffloadContext

def get_record_gradient_hook(model, record_dict):
    """
    Creates a hook to record the gradients of a model's parameters into a dictionary.

    Args:
        model (torch.nn.Module): The model whose gradients will be recorded.
        record_dict (dict): A dictionary to store the recorded gradients.
    """

    def record_gradient_hook(grad):
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                if n not in record_dict:
                    record_dict[n] = p.grad.detach().cpu()
                else:
                    record_dict[n] += p.grad.detach().cpu()
                p.grad = None
        return grad

    return record_gradient_hook


def estimate_and_process_grads_torch(
    model,
    dataloader,
    lr,
    num_samples=170,
    quant_flag=False,
    origin_type="bf16",
    quant_type="nf4",
    no_split_module_classes=None,
) -> Dict[str, torch.Tensor]:
    """
    Estimates and processes gradients using batch-wise computation.
    Returns a dictionary of processed gradients.
    
    Args:
        model: The PyTorch model
        dataloader: DataLoader instance
        lr: Learning rate
        num_samples: Total number of samples to process
        quant_flag: Whether to use quantization
        origin_type: Original data type
        quant_type: Quantization type
        no_split_module_classes: Module classes to not split
    
    Returns:
        Dict[str, torch.Tensor]: Processed gradients
    """
    #batch_size = dataloader.batch_size
    accelerator = Accelerator()
    
    if accelerator and model.device.type != "cuda":
        if not quant_flag:
            model.to(accelerator.device)
        else:
            model.to("cpu")
    
    model.train()
    dataloader = accelerator.prepare(dataloader)
    
    running_grads_sum = {}
    named_grads = {}
    total_samples = 0
    
    with OffloadContext(
        model=model,
        named_grads=named_grads,
        quant_flag=quant_flag,
        origin_type=origin_type,
        quant_type=quant_type,
        no_split_module_classes=no_split_module_classes,
    ):
        for batch in tqdm(dataloader, desc="Computing gradients"):
            current_batch_size = len(batch['input_ids'])
            samples_to_process = min(current_batch_size, num_samples - total_samples)
            
            if samples_to_process <= 0:
                break
                
            # Process only the needed portion of the batch
            batch = {k: v[:samples_to_process].to(accelerator.device) for k, v in batch.items()}
            
            if accelerator.is_main_process:
                print(f"Processing batch with {samples_to_process} samples")
            
            # Forward pass
            outputs = model(**batch)
            
            # Normalize loss by batch size to maintain scale
            (outputs.loss / samples_to_process).backward()
            
            # Record gradients
            get_record_gradient_hook(model, named_grads)(None)
            
            # Accumulate gradients
            for name, grad in named_grads.items():
                if name not in running_grads_sum:
                    running_grads_sum[name] = grad.detach().cpu()
                else:
                    running_grads_sum[name] += grad.detach().cpu()
            
            # Clear gradients
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = None
            
            total_samples += samples_to_process
            named_grads.clear()
            del outputs
            torch.cuda.empty_cache()

    # Process final gradients
    processed_grads = {}
    
    # Synchronize for distributed training
    if accelerator and accelerator.num_processes > 1:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print("Processing final gradients")
        for name in running_grads_sum:
            grad = running_grads_sum[name].to(accelerator.device)
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            running_grads_sum[name] = grad.cpu()
    
    # Process gradients
    for name, grad in running_grads_sum.items():
        processed_grads[name] = (-1 * lr * torch.sign(grad))
    
    if accelerator.is_main_process:
        print("Finished processing gradients")

    return processed_grads

def estimate_and_process_grads_torch_snli(
    model,
    dataloader,
    lr,
    num_samples=170,
    quant_flag=False,
    origin_type="bf16",
    quant_type="nf4",
    no_split_module_classes=None,
) -> Dict[str, torch.Tensor]:
    """
    Estimates and processes gradients using batch-wise computation.
    Returns a dictionary of processed gradients.
    
    Args:
        model: The PyTorch model
        dataloader: DataLoader instance
        lr: Learning rate
        num_samples: Total number of samples to process
        quant_flag: Whether to use quantization
        origin_type: Original data type
        quant_type: Quantization type
        no_split_module_classes: Module classes to not split
    
    Returns:
        Dict[str, torch.Tensor]: Processed gradients
    """
    accelerator = Accelerator()
    
    if accelerator and model.device.type != "cuda":
        if not quant_flag:
            model.to(accelerator.device)
        else:
            model.to("cpu")
    
    model.train()
    dataloader = accelerator.prepare(dataloader)

    running_grads_sum = {}
    named_grads = {}
    total_samples = 0
    
    with OffloadContext(
        model=model,
        named_grads=named_grads,
        quant_flag=quant_flag,
        origin_type=origin_type,
        quant_type=quant_type,
        no_split_module_classes=no_split_module_classes,
    ):
        for batch in tqdm(dataloader, desc="Computing gradients"):
            # Move each tensor in the batch to the accelerator device
            batch = tuple(t.to(accelerator.device) for t in batch)
            current_batch_size = batch[0].size(0)
            samples_to_process = min(current_batch_size, num_samples - total_samples)
            
            if samples_to_process <= 0:
                break
                
            # Process only the needed portion of the batch
            input_ids = batch[0][:samples_to_process]
            attention_mask = batch[1][:samples_to_process]
            token_type_ids = batch[2][:samples_to_process]
            labels = batch[3][:samples_to_process]
            
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "labels": labels,
            }
            
            if accelerator.is_main_process:
                print(f"Processing batch with {samples_to_process} samples")
            
            # Forward pass
            outputs = model(**inputs)
            
            # Normalize loss by batch size to maintain scale
            (outputs.loss / samples_to_process).backward()
            
            # Record gradients
            get_record_gradient_hook(model, named_grads)(None)
            
            # Accumulate gradients
            for name, grad in named_grads.items():
                if name not in running_grads_sum:
                    running_grads_sum[name] = grad.detach().cpu()
                else:
                    running_grads_sum[name] += grad.detach().cpu()
            
            # Clear gradients
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = None
            
            total_samples += samples_to_process
            named_grads.clear()
            del outputs
            torch.cuda.empty_cache()

    # Process final gradients
    processed_grads = {}
    
    # Synchronize for distributed training
    if accelerator and accelerator.num_processes > 1:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print("Processing final gradients")
        for name in running_grads_sum:
            grad = running_grads_sum[name].to(accelerator.device)
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            running_grads_sum[name] = grad.cpu()
    
    # Process gradients
    for name, grad in running_grads_sum.items():
        processed_grads[name] = (-1 * lr * torch.sign(grad))
    
    if accelerator.is_main_process:
        print("Finished processing gradients")

    return processed_grads

# def dp_estimate_and_process_grads_torch(
#     model,
#     dataloader,
#     lr,
#     num_samples=170,
#     quant_flag=False,
#     origin_type="bf16",
#     quant_type="nf4",
#     no_split_module_classes=None,
# ) -> Dict[str, torch.Tensor]:
#     """
#     Estimates and processes gradients using batch-wise computation.
#     Returns a dictionary of processed gradients.
    
#     Args:
#         model: The PyTorch model
#         dataloader: DataLoader instance
#         lr: Learning rate
#         num_samples: Total number of samples to process
#         quant_flag: Whether to use quantization
#         origin_type: Original data type
#         quant_type: Quantization type
#         no_split_module_classes: Module classes to not split
    
#     Returns:
#         Dict[str, torch.Tensor]: Processed gradients
#     """
#     batch_size = dataloader.batch_size
#     accelerator = Accelerator()
#     rank = accelerator.local_process_index  # Get rank
#     world_size = accelerator.num_processes  # Get total processes

#     print(f"[DEBUG] Rank {rank}: Accelerator initialized with {world_size} processes.")
#     print(f"[DEBUG] Rank {rank}: Model initial device - {next(model.parameters()).device}")
#     print(f"[DEBUG] Accelerator device - {accelerator.device}")
    
#     # if accelerator and model.device.type != "cuda":
#     #     if not quant_flag:
#     #         model.to(accelerator.device)
#     #     else:
#     #         model.to("cpu")
#     # if accelerator and (hasattr(model, "device") and model.device.type != "cuda" or not hasattr(model, "device") and next(model.parameters()).device.type != "cuda"):
#     #     if not quant_flag:
#     #         model.to(accelerator.device)
#     #     else:
#     #         model.to("cpu")
#     if quant_flag:
#         model.to("cpu")
        
#     model.train()
#     dataloader = accelerator.prepare(dataloader)
    
#     # checking devices of all objects
#     print(f"======[DEBUG] Rank {rank}: Model device - {next(model.parameters()).device}======")
#     for batch in dataloader:
#         if isinstance(batch, dict):
#             for key, value in batch.items():
#                 print(f"======[DEBUG] Rank {rank}: Dataloader batch key '{key}' device - {value.device}======")
#         elif isinstance(batch, (list, tuple)):
#             for i, value in enumerate(batch):
#                 print(f"======[DEBUG] Rank {rank}: Dataloader batch index {i} device - {value.device}======")
#         else:
#             print(f"======[DEBUG] Rank {rank}: Dataloader batch device - {batch.device}======")
#         break  # Only check the first batch

    
    
#     running_grads_sum = {}
#     named_grads = {}
#     total_samples = 0
    
#     with OffloadContext(
#         model=model,
#         named_grads=named_grads,
#         quant_flag=quant_flag,
#         origin_type=origin_type,
#         quant_type=quant_type,
#         no_split_module_classes=no_split_module_classes,
#     ):
#         for batch in tqdm(dataloader, desc="Computing gradients"):
#             current_batch_size = len(batch['input_ids'])
#             samples_to_process = min(current_batch_size, num_samples - total_samples)
            
#             if samples_to_process <= 0:
#                 break
                
#             # Process only the needed portion of the batch
#             batch = {k: v[:samples_to_process].to(accelerator.device) for k, v in batch.items()}
            
#             if accelerator.is_main_process:
#                 print(f"Processing batch with {samples_to_process} samples")
            
#             # Forward pass
#             outputs = model(**batch)
            
#             # Normalize loss by batch size to maintain scale
#             (outputs.loss / samples_to_process).backward()
            
#             # Record gradients
#             get_record_gradient_hook(model, named_grads)(None)
            
#             # Accumulate gradients
#             for name, grad in named_grads.items():
#                 if name not in running_grads_sum:
#                     running_grads_sum[name] = grad.detach().cpu()
#                 else:
#                     running_grads_sum[name] += grad.detach().cpu()
            
#             # Clear gradients
#             for param in model.parameters():
#                 if param.grad is not None:
#                     param.grad = None
            
#             total_samples += samples_to_process
#             named_grads.clear()
#             del outputs
#             torch.cuda.empty_cache()

#     # Process final gradients
#     processed_grads = {}
    
#     # Synchronize for distributed training
#     if accelerator and accelerator.num_processes > 1:
#         accelerator.wait_for_everyone()
#         if accelerator.is_main_process:
#             print("Processing final gradients")
#         for name in running_grads_sum:
#             grad = running_grads_sum[name].to(accelerator.device)
#             dist.all_reduce(grad, op=dist.ReduceOp.SUM)
#             running_grads_sum[name] = grad.cpu()
    
#     # Process gradients
#     for name, grad in running_grads_sum.items():
#         processed_grads[name] = (-1 * lr * torch.sign(grad))
    
#     if accelerator.is_main_process:
#         print("Finished processing gradients")

#     return processed_grads

import torch
import torch.distributed as dist
from tqdm import tqdm
from typing import Dict

def dp_estimate_and_process_grads_torch(
    model,
    dataloader,
    lr,
    num_samples=170,
    quant_flag=False,
    origin_type="bf16",
    quant_type="nf4",
    no_split_module_classes=None,
) -> Dict[str, torch.Tensor]:
    """
    Enhanced version with explicit device management for rotary embeddings
    """
    device_of_model = next(model.parameters()).device
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"Rank {rank}: Model device - {device_of_model}")
    
    # Force model to be fully on one device
    model = model.to(device_of_model)
    
    # Ensure rotary embeddings are on correct device
    for module in model.modules():
        if hasattr(module, 'rotary_emb'):
            module.rotary_emb = module.rotary_emb.to(device_of_model)
            # If rotary_emb has inv_freq, ensure it's on the correct device
            if hasattr(module.rotary_emb, 'inv_freq'):
                module.rotary_emb.inv_freq = module.rotary_emb.inv_freq.to(device_of_model)
    
    if quant_flag:
        model.to("cpu")
    
    model.train()
    running_grads_sum = {}
    named_grads = {}
    total_samples = 0
    
    with OffloadContext(
        model=model,
        named_grads=named_grads,
        quant_flag=quant_flag,
        origin_type=origin_type,
        quant_type=quant_type,
        no_split_module_classes=no_split_module_classes,
    ):
        for batch in tqdm(dataloader, desc="Computing gradients"):
            current_batch_size = len(batch['input_ids'])
            samples_to_process = min(current_batch_size, num_samples - total_samples)
            
            if samples_to_process <= 0:
                break
            
            # Move batch to device and ensure all tensors are on same device
            processed_batch = {}
            for k, v in batch.items():
                if v.device != device_of_model:
                    processed_batch[k] = v[:samples_to_process].to(device_of_model, non_blocking=True)
                else:
                    processed_batch[k] = v[:samples_to_process]
                
                # Double-check device placement
                if processed_batch[k].device != device_of_model:
                    raise RuntimeError(f"Tensor {k} on wrong device: {processed_batch[k].device} vs {device_of_model}")
            
            # Verify model device placement
            for name, param in model.named_parameters():
                if param.device != device_of_model:
                    print(f"WARNING: Parameter {name} on wrong device: {param.device}")
                    param.data = param.data.to(device_of_model)
            
            if rank == 0:
                print(f"Processing batch with {samples_to_process} samples")
            
            # Forward pass
            torch.cuda.synchronize()  # Ensure all tensors are ready
            outputs = model(**processed_batch)
            
            # Rest of the processing remains the same
            (outputs.loss / samples_to_process).backward()
            get_record_gradient_hook(model, named_grads)(None)
            
            for name, grad in named_grads.items():
                if name not in running_grads_sum:
                    running_grads_sum[name] = grad.detach().cpu()
                else:
                    running_grads_sum[name] += grad.detach().cpu()
            
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = None
            
            total_samples += samples_to_process
            named_grads.clear()
            del outputs
            torch.cuda.empty_cache()

    # Process final gradients
    processed_grads = {}
    
    if world_size > 1:
        dist.barrier()
        if rank == 0:
            print("Processing final gradients")
        for name in running_grads_sum:
            grad = running_grads_sum[name].to(device_of_model)
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            running_grads_sum[name] = grad.cpu()
    
    for name, grad in running_grads_sum.items():
        processed_grads[name] = (-1 * lr * torch.sign(grad))
    
    if rank == 0:
        print("Finished processing gradients")

    return processed_grads

import torch
from tqdm import tqdm
from typing import Dict

def dp_estimate_and_process_grads_torch_noparallel(
    model,
    dataloader,
    lr,
    num_samples=170,
    quant_flag=False,
    origin_type="bf16",
    quant_type="nf4",
    no_split_module_classes=None,
) -> Dict[str, torch.Tensor]:
    """
    Enhanced version with explicit device management for rotary embeddings
    """
    device_of_model = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device_of_model)
    
    print(f"Model device - {device_of_model}")
    
    # Ensure rotary embeddings are on correct device
    for module in model.modules():
        if hasattr(module, 'rotary_emb'):
            module.rotary_emb = module.rotary_emb.to(device_of_model)
            # If rotary_emb has inv_freq, ensure it's on the correct device
            if hasattr(module.rotary_emb, 'inv_freq'):
                module.rotary_emb.inv_freq = module.rotary_emb.inv_freq.to(device_of_model)
    
    if quant_flag:
        model.to("cpu")
    
    model.train()
    running_grads_sum = {}
    named_grads = {}
    total_samples = 0
    
    with OffloadContext(
        model=model,
        named_grads=named_grads,
        quant_flag=quant_flag,
        origin_type=origin_type,
        quant_type=quant_type,
        no_split_module_classes=no_split_module_classes,
    ):
        for batch in tqdm(dataloader, desc="Computing gradients"):
            current_batch_size = len(batch['input_ids'])
            samples_to_process = min(current_batch_size, num_samples - total_samples)
            
            if samples_to_process <= 0:
                break
            
            # Move batch to device and ensure all tensors are on same device
            processed_batch = {}
            for k, v in batch.items():
                if v.device != device_of_model:
                    processed_batch[k] = v[:samples_to_process].to(device_of_model, non_blocking=True)
                else:
                    processed_batch[k] = v[:samples_to_process]
                
                # Double-check device placement
                if processed_batch[k].device != device_of_model:
                    raise RuntimeError(f"Tensor {k} on wrong device: {processed_batch[k].device} vs {device_of_model}")
            
            # Verify model device placement
            for name, param in model.named_parameters():
                if param.device != device_of_model:
                    print(f"WARNING: Parameter {name} on wrong device: {param.device}")
                    param.data = param.data.to(device_of_model)
            
            print(f"Processing batch with {samples_to_process} samples")
            
            # Forward pass
            torch.cuda.synchronize()  # Ensure all tensors are ready
            outputs = model(**processed_batch)
            
            # Rest of the processing remains the same
            (outputs.loss / samples_to_process).backward()
            get_record_gradient_hook(model, named_grads)(None)
            
            for name, grad in named_grads.items():
                if name not in running_grads_sum:
                    running_grads_sum[name] = grad.detach().cpu()
                else:
                    running_grads_sum[name] += grad.detach().cpu()
            
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = None
            
            total_samples += samples_to_process
            named_grads.clear()
            del outputs
            torch.cuda.empty_cache()

    # Process final gradients
    processed_grads = {}
    
    for name, grad in running_grads_sum.items():
        processed_grads[name] = (-1 * lr * torch.sign(grad))
    
    print("Finished processing gradients")

    return processed_grads