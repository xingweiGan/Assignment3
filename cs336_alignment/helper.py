"""
Supervised Fine-Tuning utilities for CS336 Assignment 5.

This module provides functions for tokenizing prompts and outputs for SFT training.
"""

import torch
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Union
from transformers import PreTrainedTokenizer, PreTrainedModel


def tokenize_prompt_and_output(
    prompt_strs: List[str], 
    output_strs: List[str], 
    tokenizer: PreTrainedTokenizer
) -> dict[str, torch.Tensor]:
    """
    Tokenize the prompt and output strings, and construct a mask that is 1 for the response 
    tokens and 0 for other tokens (prompt or padding).
    
    Args:
        prompt_strs: List of prompt strings.
        output_strs: List of output strings.
        tokenizer: Tokenizer to use for tokenization.
        
    Returns:
        dict[str, torch.Tensor]. Let prompt_and_output_lens be a list containing the lengths of
        the tokenized prompt and output strings. Then the returned dictionary should have the
        following keys:
        - input_ids: torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
          the tokenized prompt and output strings, with the final token sliced off.
        - labels: torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
          shifted input ids, i.e., the input ids without the first token.
        - response_mask: torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
          a mask on the response tokens in the labels.
    """
    if len(prompt_strs) != len(output_strs):
        raise ValueError(f"Number of prompts ({len(prompt_strs)}) must match number of outputs ({len(output_strs)})")
    
    batch_size = len(prompt_strs)
    if batch_size == 0:
        raise ValueError("Empty batch provided")
    
    # Tokenize prompts and outputs separately
    tokenized_prompts = []
    tokenized_outputs = []
    prompt_and_output_lens = []
    
    for prompt_str, output_str in zip(prompt_strs, output_strs):
        # Tokenize prompt and output separately
        # add_special_tokens=False to avoid adding extra BOS/EOS tokens in the middle
        prompt_tokens = tokenizer.encode(prompt_str, add_special_tokens=False)
        output_tokens = tokenizer.encode(output_str, add_special_tokens=False)
        
        # Handle special tokens properly - add BOS at the beginning if tokenizer expects it
        if tokenizer.bos_token_id is not None and len(prompt_tokens) > 0:
            # Only add BOS if it's not already there
            if prompt_tokens[0] != tokenizer.bos_token_id:
                prompt_tokens = [tokenizer.bos_token_id] + prompt_tokens
        
        tokenized_prompts.append(prompt_tokens)
        tokenized_outputs.append(output_tokens)
        prompt_and_output_lens.append(len(prompt_tokens) + len(output_tokens))
    
    # Find maximum length and prepare for padding
    max_length = max(prompt_and_output_lens)
    
    # Initialize tensors
    input_ids = torch.full((batch_size, max_length - 1), tokenizer.pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_length - 1), tokenizer.pad_token_id, dtype=torch.long)
    response_mask = torch.zeros((batch_size, max_length - 1), dtype=torch.long)
    
    # Fill tensors for each example
    for i, (prompt_tokens, output_tokens) in enumerate(zip(tokenized_prompts, tokenized_outputs)):
        # Concatenate prompt and output tokens
        full_tokens = prompt_tokens + output_tokens
        prompt_len = len(prompt_tokens)
        total_len = len(full_tokens)
        
        # Slice off the final token for input_ids (standard for next-token prediction)
        sequence_len = min(total_len - 1, max_length - 1)
        
        if sequence_len > 0:
            # Fill input_ids (without the last token)
            input_ids[i, :sequence_len] = torch.tensor(full_tokens[:sequence_len], dtype=torch.long)
            
            # Fill labels (shifted by 1, i.e., without the first token)
            labels[i, :sequence_len] = torch.tensor(full_tokens[1:sequence_len + 1], dtype=torch.long)
            
            # Create response mask: 1 for response tokens, 0 for prompt tokens
            # Response tokens start after the prompt
            response_start = max(0, prompt_len - 1)  # -1 because we removed first token in labels
            response_end = sequence_len
            
            if response_start < response_end:
                response_mask[i, response_start:response_end] = 1
    
    # Handle padding token labels - typically set to -100 to ignore in loss computation
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100
    
    return {
        "input_ids": input_ids,
        "labels": labels, 
        "response_mask": response_mask
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).
    
    Args:
        logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size)
                containing unnormalized logits.
                
    Returns:
        torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token
        prediction.
    """
    # Compute log probabilities using log_softmax for numerical stability
    # log_softmax(x) = x - logsumexp(x) which is numerically stable
    log_probs = torch.log_softmax(logits, dim=-1)  # shape: (batch_size, seq_len, vocab_size)
    
    # Compute probabilities from log probabilities
    probs = torch.exp(log_probs)  # shape: (batch_size, seq_len, vocab_size)
    
    # Compute entropy: H = -∑ p(x) * log(p(x))
    # This is equivalent to: H = -∑ probs * log_probs
    entropy = -torch.sum(probs * log_probs, dim=-1)  # shape: (batch_size, seq_len)
    
    return entropy


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Get per-token conditional log-probabilities (given the previous tokens) from a causal language model,
    and optionally the entropy of the model's next-token distribution.
    
    Args:
        model: PreTrainedModel HuggingFace model used for scoring (placed on the correct device
               and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor shape (batch_size, sequence_length), concatenated prompt +
                   response tokens as produced by your tokenization method.
        labels: torch.Tensor shape (batch_size, sequence_length), labels as produced by your
                tokenization method.
        return_token_entropy: bool If True, also return per-token entropy by calling compute_entropy.
        
    Returns:
        dict[str, torch.Tensor].
        "log_probs" shape (batch_size, sequence_length), conditional log-probabilities log pθ(xt|x<t).
        "token_entropy" optional, shape (batch_size, sequence_length), per-token entropy
                        for each position (present only if return_token_entropy=True).
    """
    # Get logits from the model
    with torch.no_grad():  # Ensure no gradients unless explicitly needed
        outputs = model(input_ids)
        logits = outputs.logits  # shape: (batch_size, sequence_length, vocab_size)
    
    # Compute log probabilities using log_softmax for numerical stability
    log_probs_dist = torch.log_softmax(logits, dim=-1)  # shape: (batch_size, seq_len, vocab_size)
    
    # Get log probabilities for the actual tokens in labels
    # We need to gather the log probs for the specific tokens
    batch_size, seq_len = labels.shape
    
    # Create a mask for valid (non-ignored) labels
    # Labels with value -100 are typically ignored in loss computation
    valid_mask = (labels != -100)
    
    # Initialize log_probs tensor
    log_probs = torch.full_like(labels, 0.0, dtype=torch.float32)
    
    # Only compute log probs for valid positions
    if valid_mask.any():
        # Get the indices where labels are valid
        valid_labels = labels.clone()
        valid_labels[~valid_mask] = 0  # Set invalid labels to 0 temporarily for gathering
        
        # Gather log probabilities for the actual tokens
        # torch.gather gathers values along the vocab dimension using labels as indices
        gathered_log_probs = torch.gather(
            log_probs_dist, 
            dim=-1, 
            index=valid_labels.unsqueeze(-1)
        ).squeeze(-1)  # shape: (batch_size, sequence_length)
        
        # Only keep log probs for valid positions, set others to 0
        log_probs = gathered_log_probs * valid_mask.float()
    
    # Prepare return dictionary
    result = {"log_probs": log_probs}
    
    # Optionally compute and return token entropy
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        result["token_entropy"] = token_entropy
    
    return result


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Sum over a dimension and normalize by a constant, considering only those elements where mask == 1.
    
    Args:
        tensor: torch.Tensor The tensor to sum and normalize.
        mask: torch.Tensor Same shape as tensor; positions with 1 are included in the sum.
        normalize_constant: float the constant to divide by for normalization.
        dim: int | None the dimension to sum along before normalization. If None, sum over all
             dimensions.
             
    Returns:
        torch.Tensor the normalized sum, where masked elements (mask == 0) don't contribute to
        the sum.
    """
    # Verify tensor and mask have the same shape
    if tensor.shape != mask.shape:
        raise ValueError(f"Tensor and mask must have the same shape. "
                        f"Got tensor: {tensor.shape}, mask: {mask.shape}")
    
    # Apply mask to tensor - only elements where mask == 1 contribute
    # Convert mask to same dtype as tensor for proper multiplication
    mask_float = mask.float()
    masked_tensor = tensor * mask_float
    
    # Sum over the specified dimension(s)
    if dim is None:
        # Sum over all dimensions
        masked_sum = torch.sum(masked_tensor)
    else:
        # Sum over the specified dimension
        masked_sum = torch.sum(masked_tensor, dim=dim)
    
    # Normalize by the constant
    if normalize_constant == 0:
        raise ValueError("normalize_constant cannot be zero")
    
    normalized_result = masked_sum / normalize_constant
    
    return normalized_result


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.
    
    Args:
        policy_log_probs: torch.Tensor (batch_size, sequence_length), per-token log-probabilities 
                         from the SFT policy being trained.
        response_mask: torch.Tensor (batch_size, sequence_length), 1 for response tokens, 
                      0 for prompt/padding.
        gradient_accumulation_steps: int Number of microbatches per optimizer step.
        normalize_constant: float The constant by which to divide the sum. It is fine to leave this as 1.0.
        
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss: scalar tensor. The microbatch loss, adjusted for gradient accumulation. 
              We return this so we can log it.
        metadata: Dict with metadata from the underlying loss call, and any other statistics 
                 you might want to log.
    """
    # Verify input shapes match
    if policy_log_probs.shape != response_mask.shape:
        raise ValueError(f"policy_log_probs and response_mask must have the same shape. "
                        f"Got {policy_log_probs.shape} and {response_mask.shape}")
    
    # Compute cross-entropy loss: we want to maximize log probabilities, so minimize negative log probs
    # For SFT, the loss is the negative log likelihood of the target tokens
    negative_log_probs = -policy_log_probs
    
    # Use masked_normalize to compute the loss only over response tokens
    # Sum over all dimensions and normalize by the provided constant
    masked_loss = masked_normalize(
        tensor=negative_log_probs,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=None  # Sum over all dimensions to get scalar loss
    )
    
    # Scale loss for gradient accumulation
    # When accumulating gradients over multiple microbatches, we divide the loss
    # by the number of accumulation steps so that the effective gradient magnitude
    # remains the same as if we processed the full batch at once
    scaled_loss = masked_loss / gradient_accumulation_steps
    
    # Compute backward pass
    scaled_loss.backward()
    
    # Compute metadata for logging
    with torch.no_grad():
        # Count number of response tokens for normalization info
        num_response_tokens = response_mask.sum()
        
        # Compute average loss per response token
        if num_response_tokens > 0:
            avg_loss_per_token = masked_loss / num_response_tokens
        else:
            avg_loss_per_token = torch.tensor(0.0, device=masked_loss.device)
        
        # Compute some useful statistics
        metadata = {
            "num_response_tokens": num_response_tokens,
            "avg_loss_per_token": avg_loss_per_token,
            "unscaled_loss": masked_loss,  # Loss before gradient accumulation scaling
            "policy_log_probs_mean": masked_normalize(
                policy_log_probs, response_mask, 
                normalize_constant=max(num_response_tokens.item(), 1),
                dim=None
            ),
            "policy_log_probs_std": torch.std(policy_log_probs[response_mask.bool()]) if num_response_tokens > 0 else torch.tensor(0.0)
        }
    
    # Return the scaled loss (for logging purposes) and metadata
    return scaled_loss, metadata


def log_generations(
    prompts: List[str],
    generations: List[str],
    ground_truths: Optional[List[str]] = None,
    metrics: Optional[List[dict]] = None,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    output_file: Optional[Union[str, Path]] = None,
    max_examples: int = 5,
    truncate_length: int = 200,
    console_output: bool = True,
    save_json: bool = False,
    logger: Optional[logging.Logger] = None,
    prefix: str = "GENERATION"
) -> None:
    """
    Log generations from a model for monitoring and debugging purposes.
    Enhanced for SFT/RL training with detailed reward and entropy analysis.
    
    Args:
        prompts: List of input prompts.
        generations: List of model generations corresponding to prompts.
        ground_truths: Optional list of ground truth responses.
        metrics: Optional list of metric dictionaries for each example.
                Expected keys in metrics dict:
                - 'format_reward': reward for correct format
                - 'answer_reward': reward for correct answer  
                - 'total_reward': total reward score
                - 'token_entropy': per-token entropy values or average entropy
                - 'is_correct': boolean indicating if response is correct
        step: Optional training step number.
        epoch: Optional training epoch number.
        output_file: Optional path to save generations to file.
        max_examples: Maximum number of examples to log (to avoid spam).
        truncate_length: Maximum length to display for each text (characters).
        console_output: Whether to log to console.
        save_json: Whether to save detailed JSON output to file.
        logger: Optional logger instance. If None, uses default logger.
        prefix: Prefix for log messages.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Validate inputs
    if len(prompts) != len(generations):
        raise ValueError(f"Number of prompts ({len(prompts)}) must match number of generations ({len(generations)})")
    
    if ground_truths is not None and len(ground_truths) != len(prompts):
        raise ValueError(f"Number of ground truths ({len(ground_truths)}) must match number of prompts ({len(prompts)})")
    
    if metrics is not None and len(metrics) != len(prompts):
        raise ValueError(f"Number of metrics ({len(metrics)}) must match number of prompts ({len(prompts)})")
    
    # Prepare header info
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header_parts = [f"[{timestamp}]", prefix]
    
    if step is not None:
        header_parts.append(f"Step {step}")
    if epoch is not None:
        header_parts.append(f"Epoch {epoch}")
    
    header = " ".join(header_parts)
    
    # Limit number of examples to avoid overwhelming output
    num_examples = min(len(prompts), max_examples)
    
    if console_output:
        logger.info(f"{header} - Showing {num_examples}/{len(prompts)} examples")
        logger.info("=" * 80)
    
    # Prepare data for JSON output if requested
    json_data = {
        "timestamp": timestamp,
        "step": step,
        "epoch": epoch,
        "total_examples": len(prompts),
        "examples": []
    }
    
    # Compute response lengths for each generation
    response_lengths = [len(gen.split()) for gen in generations]
    
    # Log each example
    for i in range(num_examples):
        example_data = {
            "index": i,
            "prompt": prompts[i],
            "generation": generations[i],
            "response_length": response_lengths[i]
        }
        
        if ground_truths is not None:
            example_data["ground_truth"] = ground_truths[i]
        
        if metrics is not None:
            example_data["metrics"] = metrics[i]
        
        # Add to JSON data
        json_data["examples"].append(example_data)
        
        # Console output
        if console_output:
            logger.info(f"\n--- Example {i+1} ---")
            
            # Log prompt (truncated if needed)
            prompt_display = prompts[i]
            if len(prompt_display) > truncate_length:
                prompt_display = prompt_display[:truncate_length] + "..."
            logger.info(f"PROMPT: {prompt_display}")
            
            # Log generation (truncated if needed)
            generation_display = generations[i]
            if len(generation_display) > truncate_length:
                generation_display = generation_display[:truncate_length] + "..."
            logger.info(f"GENERATION: {generation_display}")
            logger.info(f"RESPONSE_LENGTH: {response_lengths[i]} tokens")
            
            # Log ground truth if available
            if ground_truths is not None:
                gt_display = ground_truths[i]
                if len(gt_display) > truncate_length:
                    gt_display = gt_display[:truncate_length] + "..."
                logger.info(f"GROUND_TRUTH: {gt_display}")
            
            # Enhanced metrics logging with focus on rewards and entropy
            if metrics is not None:
                metric_dict = metrics[i]
                
                # Log reward information prominently
                reward_parts = []
                if 'format_reward' in metric_dict:
                    reward_parts.append(f"format: {metric_dict['format_reward']:.3f}")
                if 'answer_reward' in metric_dict:
                    reward_parts.append(f"answer: {metric_dict['answer_reward']:.3f}")
                if 'total_reward' in metric_dict:
                    reward_parts.append(f"total: {metric_dict['total_reward']:.3f}")
                
                if reward_parts:
                    logger.info(f"REWARDS: {', '.join(reward_parts)}")
                
                # Log token entropy if available
                if 'token_entropy' in metric_dict:
                    entropy_val = metric_dict['token_entropy']
                    if isinstance(entropy_val, torch.Tensor):
                        if entropy_val.numel() > 1:
                            # If it's a tensor with multiple values, compute mean
                            avg_entropy = entropy_val.mean().item()
                        else:
                            avg_entropy = entropy_val.item()
                    else:
                        avg_entropy = entropy_val
                    logger.info(f"AVG_TOKEN_ENTROPY: {avg_entropy:.4f}")
                
                # Log correctness if available
                if 'is_correct' in metric_dict:
                    logger.info(f"CORRECT: {metric_dict['is_correct']}")
                
                # Log any other metrics
                other_metrics = []
                for k, v in metric_dict.items():
                    if k not in ['format_reward', 'answer_reward', 'total_reward', 'token_entropy', 'is_correct']:
                        if isinstance(v, (float, torch.Tensor)):
                            if isinstance(v, torch.Tensor):
                                v = v.item() if v.numel() == 1 else v.mean().item()
                            other_metrics.append(f"{k}: {v:.4f}")
                        else:
                            other_metrics.append(f"{k}: {v}")
                
                if other_metrics:
                    logger.info(f"OTHER_METRICS: {', '.join(other_metrics)}")
    
    if console_output:
        logger.info("=" * 80)
        
        # Enhanced aggregate statistics
        if metrics is not None:
            _log_enhanced_aggregate_metrics(metrics, response_lengths, logger, prefix)
    
    # Save to file if requested
    if output_file is not None:
        _save_generations_to_file(
            json_data, output_file, save_json, 
            prompts, generations, ground_truths, metrics,
            step, epoch, timestamp
        )


def _log_enhanced_aggregate_metrics(
    metrics: List[dict], 
    response_lengths: List[int],
    logger: logging.Logger, 
    prefix: str
) -> None:
    """Log enhanced aggregate statistics with focus on rewards, entropy, and length analysis."""
    if not metrics:
        return
    
    logger.info(f"{prefix} - Enhanced Aggregate Metrics:")
    
    # Response length analysis
    avg_length = sum(response_lengths) / len(response_lengths)
    logger.info(f"  avg_response_length: {avg_length:.2f} tokens")
    
    # Length analysis by correctness
    correct_lengths = []
    incorrect_lengths = []
    
    for i, metric_dict in enumerate(metrics):
        if 'is_correct' in metric_dict:
            if metric_dict['is_correct']:
                correct_lengths.append(response_lengths[i])
            else:
                incorrect_lengths.append(response_lengths[i])
    
    if correct_lengths:
        avg_correct_length = sum(correct_lengths) / len(correct_lengths)
        logger.info(f"  avg_response_length_correct: {avg_correct_length:.2f} tokens ({len(correct_lengths)} examples)")
    
    if incorrect_lengths:
        avg_incorrect_length = sum(incorrect_lengths) / len(incorrect_lengths)
        logger.info(f"  avg_response_length_incorrect: {avg_incorrect_length:.2f} tokens ({len(incorrect_lengths)} examples)")
    
    # Reward statistics
    reward_keys = ['format_reward', 'answer_reward', 'total_reward']
    for reward_key in reward_keys:
        values = []
        for metric_dict in metrics:
            if reward_key in metric_dict:
                val = metric_dict[reward_key]
                if isinstance(val, torch.Tensor):
                    val = val.item()
                if isinstance(val, (int, float)):
                    values.append(val)
        
        if values:
            mean_val = sum(values) / len(values)
            logger.info(f"  avg_{reward_key}: {mean_val:.4f} (over {len(values)} examples)")
    
    # Token entropy statistics
    entropy_values = []
    for metric_dict in metrics:
        if 'token_entropy' in metric_dict:
            entropy_val = metric_dict['token_entropy']
            if isinstance(entropy_val, torch.Tensor):
                if entropy_val.numel() > 1:
                    entropy_values.append(entropy_val.mean().item())
                else:
                    entropy_values.append(entropy_val.item())
            else:
                entropy_values.append(entropy_val)
    
    if entropy_values:
        avg_entropy = sum(entropy_values) / len(entropy_values)
        logger.info(f"  avg_token_entropy: {avg_entropy:.4f} (over {len(entropy_values)} examples)")
    
    # Correctness statistics
    correct_count = 0
    total_with_correctness = 0
    for metric_dict in metrics:
        if 'is_correct' in metric_dict:
            total_with_correctness += 1
            if metric_dict['is_correct']:
                correct_count += 1
    
    if total_with_correctness > 0:
        accuracy = correct_count / total_with_correctness
        logger.info(f"  accuracy: {accuracy:.4f} ({correct_count}/{total_with_correctness})")
    
    # Other standard metrics
    all_keys = set()
    for metric_dict in metrics:
        all_keys.update(metric_dict.keys())
    
    # Skip keys we've already logged
    skip_keys = {'format_reward', 'answer_reward', 'total_reward', 'token_entropy', 'is_correct'}
    
    for key in sorted(all_keys - skip_keys):
        values = []
        for metric_dict in metrics:
            if key in metric_dict:
                val = metric_dict[key]
                if isinstance(val, torch.Tensor):
                    val = val.item() if val.numel() == 1 else val.mean().item()
                if isinstance(val, (int, float)):
                    values.append(val)
        
        if values:
            mean_val = sum(values) / len(values)
            logger.info(f"  avg_{key}: {mean_val:.4f} (over {len(values)} examples)")


def _save_generations_to_file(
    json_data: dict,
    output_file: Union[str, Path],
    save_json: bool,
    prompts: List[str],
    generations: List[str],
    ground_truths: Optional[List[str]],
    metrics: Optional[List[dict]],
    step: Optional[int],
    epoch: Optional[int],
    timestamp: str
) -> None:
    """Save generations to file in requested format with enhanced metrics."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if save_json:
        # Save as JSON
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    # Save as human-readable text
    txt_path = output_path.with_suffix('.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Model Generations Log\n")
        f.write(f"Timestamp: {timestamp}\n")
        if step is not None:
            f.write(f"Step: {step}\n")
        if epoch is not None:
            f.write(f"Epoch: {epoch}\n")
        f.write(f"Total Examples: {len(prompts)}\n")
        f.write("=" * 80 + "\n\n")
        
        response_lengths = [len(gen.split()) for gen in generations]
        
        for i, (prompt, generation) in enumerate(zip(prompts, generations)):
            f.write(f"--- Example {i+1} ---\n")
            f.write(f"PROMPT:\n{prompt}\n\n")
            f.write(f"GENERATION:\n{generation}\n\n")
            f.write(f"RESPONSE_LENGTH: {response_lengths[i]} tokens\n\n")
            
            if ground_truths is not None:
                f.write(f"GROUND_TRUTH:\n{ground_truths[i]}\n\n")
            
            if metrics is not None:
                metric_dict = metrics[i]
                
                # Write reward information prominently
                f.write("REWARDS:\n")
                for reward_key in ['format_reward', 'answer_reward', 'total_reward']:
                    if reward_key in metric_dict:
                        val = metric_dict[reward_key]
                        if isinstance(val, torch.Tensor):
                            val = val.item()
                        f.write(f"  {reward_key}: {val:.4f}\n")
                f.write("\n")
                
                # Write entropy information
                if 'token_entropy' in metric_dict:
                    entropy_val = metric_dict['token_entropy']
                    if isinstance(entropy_val, torch.Tensor):
                        if entropy_val.numel() > 1:
                            avg_entropy = entropy_val.mean().item()
                        else:
                            avg_entropy = entropy_val.item()
                    else:
                        avg_entropy = entropy_val
                    f.write(f"AVG_TOKEN_ENTROPY: {avg_entropy:.4f}\n\n")
                
                # Write correctness
                if 'is_correct' in metric_dict:
                    f.write(f"CORRECT: {metric_dict['is_correct']}\n\n")
                
                # Write other metrics
                f.write("OTHER_METRICS:\n")
                skip_keys = {'format_reward', 'answer_reward', 'total_reward', 'token_entropy', 'is_correct'}
                for k, v in metric_dict.items():
                    if k not in skip_keys:
                        if isinstance(v, torch.Tensor):
                            v = v.item() if v.numel() == 1 else v.mean().item()
                        f.write(f"  {k}: {v}\n")
                f.write("\n")
            
            f.write("-" * 40 + "\n\n")
        
        # Add enhanced aggregate metrics
        if metrics is not None:
            f.write("ENHANCED AGGREGATE METRICS:\n")
            
            # Response length analysis
            avg_length = sum(response_lengths) / len(response_lengths)
            f.write(f"  avg_response_length: {avg_length:.2f} tokens\n")
            
            # Length by correctness
            correct_lengths = []
            incorrect_lengths = []
            
            for i, metric_dict in enumerate(metrics):
                if 'is_correct' in metric_dict:
                    if metric_dict['is_correct']:
                        correct_lengths.append(response_lengths[i])
                    else:
                        incorrect_lengths.append(response_lengths[i])
            
            if correct_lengths:
                avg_correct_length = sum(correct_lengths) / len(correct_lengths)
                f.write(f"  avg_response_length_correct: {avg_correct_length:.2f} tokens ({len(correct_lengths)} examples)\n")
            
            if incorrect_lengths:
                avg_incorrect_length = sum(incorrect_lengths) / len(incorrect_lengths)
                f.write(f"  avg_response_length_incorrect: {avg_incorrect_length:.2f} tokens ({len(incorrect_lengths)} examples)\n")
            
            # Standard aggregate metrics computation
            all_keys = set()
            for metric_dict in metrics:
                all_keys.update(metric_dict.keys())
            
            for key in sorted(all_keys):
                values = []
                for metric_dict in metrics:
                    if key in metric_dict:
                        val = metric_dict[key]
                        if isinstance(val, torch.Tensor):
                            val = val.item() if val.numel() == 1 else val.mean().item()
                        if isinstance(val, (int, float)):
                            values.append(val)
                
                if values:
                    mean_val = sum(values) / len(values)
                    if key == 'is_correct':
                        # For correctness, show as accuracy
                        f.write(f"  accuracy: {mean_val:.4f} ({sum(values)}/{len(values)} correct)\n")
                    else:
                        f.write(f"  avg_{key}: {mean_val:.4f} (over {len(values)} examples)\n")
