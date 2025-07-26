#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) Training Script

This script provides a complete SFT training pipeline with:
- HuggingFace model loading with memory optimizations
- Gradient accumulation for large effective batch sizes
- Comprehensive logging and evaluation
- Model saving and checkpointing

Usage:
    python train_sft.py --model-path Qwen/Qwen2.5-Math-1.5B-Instruct --data-path data/sft_train.jsonl
"""

import argparse
import json
import logging
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from tqdm import tqdm
import wandb
from typing import List, Dict, Optional

from helper import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
    log_generations
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        self.examples = []
        with open(data_path, 'r') as f:
            for line in f:
                example = json.loads(line)
                self.examples.append(example)
        
        logger.info(f"Loaded {len(self.examples)} training examples from {data_path}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Extract prompt and response
        prompt = example.get("prompt", example.get("instruction", ""))
        response = example.get("response", example.get("output", ""))
        
        return {
            "prompt": prompt,
            "response": response,
            "original_example": example
        }


def collate_fn(batch, tokenizer):
    """Collate function for DataLoader that tokenizes prompts and responses."""
    prompts = [item["prompt"] for item in batch]
    responses = [item["response"] for item in batch]
    
    # Tokenize using our SFT function
    tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
    
    return {
        "input_ids": tokenized["input_ids"],
        "labels": tokenized["labels"],
        "response_mask": tokenized["response_mask"],
        "prompts": prompts,
        "responses": responses,
        "original_examples": [item["original_example"] for item in batch]
    }


def load_model_and_tokenizer(model_path: str, device: str):
    """Load HuggingFace model and tokenizer with optimizations."""
    logger.info(f"Loading model and tokenizer from {model_path}")
    
    # Load model with memory optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto" if device == "auto" else None,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move model to device if not using device_map="auto"
    if device != "auto":
        model = model.to(device)
    
    logger.info(f"Model loaded with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    logger.info(f"Model device: {next(model.parameters()).device}")
    
    return model, tokenizer


def evaluate_model(model, tokenizer, eval_dataset, device, max_eval_examples=50):
    """Evaluate model on a small subset of data."""
    model.eval()
    
    # Create eval dataloader
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=4, 
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    
    all_prompts = []
    all_generations = []
    all_ground_truths = []
    all_metrics = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dataloader):
            if len(all_prompts) >= max_eval_examples:
                break
                
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)
            
            # Get log probabilities
            log_probs_result = get_response_log_probs(
                model, input_ids, labels, return_token_entropy=True
            )
            
            # Compute metrics
            batch_loss = F.cross_entropy(
                model(input_ids).logits.view(-1, model.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
                reduction='none'
            ).view(labels.shape)
            
            # Masked average loss per example
            masked_loss = (batch_loss * response_mask).sum(dim=1) / response_mask.sum(dim=1).clamp(min=1)
            
            # Collect data for logging
            for i in range(len(batch["prompts"])):
                all_prompts.append(batch["prompts"][i])
                all_generations.append(batch["responses"][i])  # Ground truth as "generation" for eval
                all_ground_truths.append(batch["responses"][i])
                
                metrics = {
                    "loss": masked_loss[i].item(),
                    "avg_log_prob": log_probs_result["log_probs"][i][response_mask[i].bool()].mean().item() if response_mask[i].sum() > 0 else 0.0,
                    "entropy": log_probs_result["token_entropy"][i][response_mask[i].bool()].mean().item() if response_mask[i].sum() > 0 else 0.0,
                    "response_length": response_mask[i].sum().item()
                }
                all_metrics.append(metrics)
    
    # Compute aggregate metrics
    avg_loss = sum(m["loss"] for m in all_metrics) / len(all_metrics)
    avg_log_prob = sum(m["avg_log_prob"] for m in all_metrics) / len(all_metrics)
    avg_entropy = sum(m["entropy"] for m in all_metrics) / len(all_metrics)
    
    logger.info(f"Evaluation - Loss: {avg_loss:.4f}, Avg Log Prob: {avg_log_prob:.4f}, Entropy: {avg_entropy:.4f}")
    
    return {
        "eval_loss": avg_loss,
        "eval_avg_log_prob": avg_log_prob,
        "eval_entropy": avg_entropy,
        "prompts": all_prompts,
        "generations": all_generations,
        "ground_truths": all_ground_truths,
        "metrics": all_metrics
    }


def train_sft(
    model_path: str,
    train_data_path: str,
    eval_data_path: Optional[str] = None,
    output_dir: str = "sft_output",
    learning_rate: float = 1e-5,
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_length: int = 512,
    save_steps: int = 500,
    eval_steps: int = 200,
    logging_steps: int = 50,
    device: str = "auto",
    use_wandb: bool = False,
    wandb_project: str = "sft-training"
):
    """Main SFT training function."""
    
    # Initialize wandb if requested
    if use_wandb:
        wandb.init(project=wandb_project, config={
            "model_path": model_path,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "max_length": max_length
        })
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path, device)
    model.train()
    
    # Create datasets
    train_dataset = SFTDataset(train_data_path, tokenizer, max_length)
    eval_dataset = SFTDataset(eval_data_path, tokenizer, max_length) if eval_data_path else None
    
    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps
    )
    
    logger.info(f"Training for {num_epochs} epochs, {num_training_steps} total steps")
    logger.info(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    
    # Training loop
    global_step = 0
    running_loss = 0.0
    
    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        
        # Initialize progress bar
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)
            
            # Get log probabilities from model
            log_probs_result = get_response_log_probs(model, input_ids, labels)
            
            # Perform SFT microbatch training step
            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=log_probs_result["log_probs"],
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps
            )
            
            running_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Gradient accumulation: update weights every N steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging
                if global_step % logging_steps == 0:
                    avg_loss = running_loss / logging_steps
                    logger.info(f"Step {global_step}: Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.2e}")
                    
                    if use_wandb:
                        wandb.log({
                            "train_loss": avg_loss,
                            "learning_rate": scheduler.get_last_lr()[0],
                            "epoch": epoch,
                            "step": global_step
                        })
                    
                    running_loss = 0.0
                
                # Evaluation
                if eval_dataset and global_step % eval_steps == 0:
                    logger.info(f"Running evaluation at step {global_step}")
                    eval_results = evaluate_model(model, tokenizer, eval_dataset, device)
                    
                    if use_wandb:
                        wandb.log({
                            "eval_loss": eval_results["eval_loss"],
                            "eval_avg_log_prob": eval_results["eval_avg_log_prob"],
                            "eval_entropy": eval_results["eval_entropy"],
                            "step": global_step
                        })
                    
                    # Log generations
                    log_generations(
                        prompts=eval_results["prompts"][:3],
                        generations=eval_results["generations"][:3],
                        ground_truths=eval_results["ground_truths"][:3],
                        metrics=eval_results["metrics"][:3],
                        step=global_step,
                        epoch=epoch,
                        output_file=f"{output_dir}/eval_generations_step_{global_step}",
                        save_json=True,
                        max_examples=3
                    )
                    
                    model.train()  # Return to training mode
                
                # Save checkpoint
                if global_step % save_steps == 0:
                    checkpoint_dir = f"{output_dir}/checkpoint-{global_step}"
                    logger.info(f"Saving checkpoint to {checkpoint_dir}")
                    
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
    
    # Final save
    logger.info(f"Saving final model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Final evaluation
    if eval_dataset:
        logger.info("Running final evaluation")
        eval_results = evaluate_model(model, tokenizer, eval_dataset, device)
        
        # Log final generations
        log_generations(
            prompts=eval_results["prompts"][:5],
            generations=eval_results["generations"][:5],
            ground_truths=eval_results["ground_truths"][:5],
            metrics=eval_results["metrics"][:5],
            step=global_step,
            epoch=num_epochs,
            output_file=f"{output_dir}/final_eval_generations",
            save_json=True,
            max_examples=5
        )
    
    if use_wandb:
        wandb.finish()
    
    logger.info("Training completed successfully!")


def main():
    # Run training
    train_sft(
        model_path="Qwen/Qwen2.5-Math-1.5B-Instruct",
        train_data_path="data/sft_train.jsonl",
        eval_data_path="data/sft_eval.jsonl",
        output_dir="sft_output",
        learning_rate=1e-5,
        num_epochs=3,
        batch_size=4,
        gradient_accumulation_steps=4,
        max_length=512,
        save_steps=500,
        eval_steps=200,
        logging_steps=50,
        device="auto",
        use_wandb=False,
        wandb_project="sft-training"
    )


if __name__ == "__main__":
    main() 