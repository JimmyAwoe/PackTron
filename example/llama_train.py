"""
Simple example demonstrating PackTron data loading with LLaMA model training.

This example shows how to use PackTron for efficient sentence packing
with transformers library to train a LLaMA model in a multi-GPU setup.

Prerequisites:
1. Preprocess your data using PackTron's preprocess_data.py to generate .bin and .idx files
2. Install required packages: transformers, torch, etc.

Usage (2 GPUs):
    torchrun --nproc-per-node=2 llama_pretrain.py

Usage (single GPU for testing):
    python llama_pretrain.py
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import LlamaForCausalLM, LlamaConfig
import sys
import os
import argparse
import logging

# Add parent directory to path to import PackTron modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataloader import create_dataloader
from utils.dataset_config import PackTronConfig
from utils.tokenizer import build_tokenizer
from utils.log import log_single_rank

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PackTron LLaMA Training Example')
    
    # Model configuration
    parser.add_argument('--model-config', type=str, required=True,
                        help='Path to LLaMA model config JSON file (e.g., test/llama_60m.json)')
    parser.add_argument('--tokenizer-model', type=str, default=None,
                        help='Tokenizer model name or path (default: use model-config directory)')
    
    # Data configuration
    parser.add_argument('--data-path', nargs='*', type=str, required=True,
                        help='Data path prefix')
    parser.add_argument('--cache-dir', type=str, default='./cache',
                        help='Cache directory for PackTron indices (default: ./cache)')
    parser.add_argument('--split-config', type=str, default='98,2',
                        help='Data split ratios: train,valid')
    
    # Training configuration
    parser.add_argument('--sequence-length', type=int, default=4096,
                        help='Sequence length for training (default: 4096)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size per GPU (default: 4)')
    parser.add_argument('--train-iters', type=int, default=1000,
                        help='Number of training iterations')
    parser.add_argument('--eval-iters', type=int, default=10,
                        help='Number of evaluation iterations per evaluation')
    parser.add_argument('--eval-interval', type=int, default=100,
                        help='Evaluation interval in training steps')
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                        help='Learning rate (default: 1e-5)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    # Optional: checkpoint and logging
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save checkpoints (default: None, no saving)')
    parser.add_argument('--log-interval', type=int, default=1,
                        help='Logging interval in steps (default: 10)')
    
    return parser.parse_args()


def main():
    """Main training function"""
    
    # Parse command line arguments
    args = parse_args()

    # Initialize the process group
    dist.init_process_group(backend='nccl')
    
    # Configure logging to output to console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
        
    # Set the device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    log_single_rank(logger, logging.INFO, "Loading LLaMA model and tokenizer...")
    
    tokenizer = build_tokenizer(args)
    
    # Load model 
    llama_config = LlamaConfig.from_json_file(args.model_config)
    model = LlamaForCausalLM(llama_config)
    model = model.to(device)
    
    # Wrap model with DDP for multi-GPU training
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # ==================== Create PackTron DataLoader ====================
    log_single_rank(logger, logging.INFO, "Creating PackTron DataLoader...")

    # Calculate number of evaluations and total eval_iters for PackTronConfig
    num_evals = args.train_iters // args.eval_interval
    total_eval_iters = num_evals * args.eval_iters
    
    log_single_rank(logger, logging.INFO, 
                    f"Training configuration: train_iters={args.train_iters}, "
                    f"eval_interval={args.eval_interval}, eval_iters_per_eval={args.eval_iters}, "
                    f"total_evaluations={num_evals}, total_eval_iters={total_eval_iters}")

    # PackTron data configuration
    config = PackTronConfig(
        path_to_cache=args.cache_dir,
        random_seed=args.random_seed,
        split_config=args.split_config,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        data_path=args.data_path,
        train_iters=args.train_iters,
        eval_iters=total_eval_iters,  # Total eval iterations across all evaluations
    )

    train_dataloader, eval_dataloader = create_dataloader(
        tokenizer=tokenizer,
        config=config,
        rank=rank,
        world_size=world_size,
        consumed_samples=0
    )
    
    # ==================== Training Loop ====================
    log_single_rank(logger, logging.INFO, "Starting training...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    total_loss = 0
    num_batches = 0
    
    # Simple training loop
    for step, batch in enumerate(train_dataloader):
        if step >= config.train_iters:
            break
            
        tokens = batch['tokens'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device).float()
        
        # Forward pass
        outputs = model(
            input_ids=tokens,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (step + 1) % args.log_interval == 0:
            avg_loss = total_loss / num_batches
            log_single_rank(logger, logging.INFO, f"Step {step + 1}/{config.train_iters}, Loss: {avg_loss:.4f}, "
                        f"Batches processed: {num_batches}")
        
        # Evaluation
        if (step + 1) % args.eval_interval == 0:
            log_single_rank(logger, logging.INFO, f"Starting evaluation at step {step + 1}...")
            model.eval()
            eval_loss = 0.0
            eval_batches = 0
            
            with torch.no_grad():
                for eval_step, eval_batch in enumerate(eval_dataloader):
                    if eval_step >= args.eval_iters:
                        break
                    
                    eval_tokens = eval_batch['tokens'].to(device)
                    eval_labels = eval_batch['labels'].to(device)
                    eval_attention_mask = eval_batch['attention_mask'].to(device).float()
                    
                    eval_outputs = model(
                        input_ids=eval_tokens,
                        attention_mask=eval_attention_mask,
                        labels=eval_labels
                    )
                    
                    eval_loss += eval_outputs.loss.item()
                    eval_batches += 1
            
            avg_eval_loss = eval_loss / eval_batches if eval_batches > 0 else 0.0
            log_single_rank(logger, logging.INFO, 
                          f"Evaluation at step {step + 1}: Average loss over {eval_batches} batches = {avg_eval_loss:.4f}")
            
            model.train()  # Switch back to training mode
        
    log_single_rank(logger, logging.INFO, f"Training completed!")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

