"""
Baseline evaluation script for GSM8K test set using r1_zero prompting.

This script:
1. Loads testing examples from /data/gsm8k/test.jsonl
2. Formats them as string prompts using the r1_zero prompt template
3. Generates outputs for each example using vLLM
4. Calculates evaluation metrics using the question_only_reward_fn
5. Serializes examples, model generations, and evaluation scores to disk
"""

import json
import logging
import os
from pathlib import Path
from statistics import mean
from typing import Callable, List

from tqdm import tqdm
from vllm import LLM, SamplingParams
from xopen import xopen

from .drgrpo_grader import question_only_reward_fn

logger = logging.getLogger(__name__)


def load_r1_zero_prompt() -> str:
    """Load the r1_zero prompt template."""
    prompt_path = Path(__file__).parent / "prompts" / "r1_zero.prompt"
    with open(prompt_path, 'r') as f:
        return f.read().strip()


def load_gsm8k_test_data(data_path: str = "/data/gsm8k/test.jsonl") -> List[dict]:
    """Load GSM8K test examples from jsonl file."""
    examples = []
    
    # Try relative path first, then absolute
    if not os.path.exists(data_path):
        # Try relative to project root
        project_root = Path(__file__).parent.parent
        relative_path = project_root / data_path.lstrip("/")
        if relative_path.exists():
            data_path = str(relative_path)
    
    with xopen(data_path) as f:
        for line in f:
            examples.append(json.loads(line))
    
    logger.info(f"Loaded {len(examples)} GSM8K test examples from {data_path}")
    return examples


def format_prompts(examples: List[dict], prompt_template: str) -> List[str]:
    """Format GSM8K examples using the r1_zero prompt template."""
    prompts = []
    for example in examples:
        # Format the prompt by substituting the question
        formatted_prompt = prompt_template.format(question=example["question"])
        prompts.append(formatted_prompt)
    
    logger.info(f"Formatted {len(prompts)} prompts using r1_zero template")
    return prompts


def extract_ground_truth_answer(answer_text: str) -> str:
    """Extract the numerical answer from GSM8K ground truth text."""
    # GSM8K answers end with #### followed by the numerical answer
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    return answer_text.strip()


def analyze_result_categories(all_metrics: List[dict[str, float]]) -> dict[str, int]:
    """
    Analyze evaluation results and categorize them based on reward components.
    
    Categories:
    1. Correct (format=1, answer=1): Both format and answer are correct
    2. Format correct, answer wrong (format=1, answer=0): Properly formatted but incorrect answer
    3. Format incorrect (format=0, answer=0): Cannot parse answer or completely wrong format
    
    Args:
        all_metrics: List of metric dictionaries from reward function
        
    Returns:
        Dictionary with category names and their counts
    """
    categories = {
        "Correct (format=1, answer=1)": 0,
        "Format correct, answer wrong (format=1, answer=0)": 0, 
        "Format incorrect (format=0, answer=0)": 0
    }
    
    for metrics in all_metrics:
        format_reward = metrics.get("format_reward", 0)
        answer_reward = metrics.get("answer_reward", 0)
        
        if format_reward == 1.0 and answer_reward == 1.0:
            categories["Correct (format=1, answer=1)"] += 1
        elif format_reward == 1.0 and answer_reward == 0.0:
            categories["Format correct, answer wrong (format=1, answer=0)"] += 1
        elif format_reward == 0.0 and answer_reward == 0.0:
            categories["Format incorrect (format=0, answer=0)"] += 1
        else:
            # This should not happen with the current reward function, but let's log it
            logger.warning(f"Unexpected reward combination: format={format_reward}, answer={answer_reward}")
    
    return categories


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
    ground_truths: List[str],
    output_path: str = "baseline_results.jsonl",
    model_name: str = "unknown"
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    
    Args:
        vllm_model: The vLLM model instance
        reward_fn: Function that takes (response, ground_truth) and returns metrics dict
        prompts: List of formatted prompts
        eval_sampling_params: Sampling parameters for generation
        ground_truths: List of ground truth answers
        output_path: Path to save results
        model_name: Name of the model being evaluated
    """
    logger.info(f"Generating responses for {len(prompts)} prompts...")
    
    # Generate model responses
    raw_responses = vllm_model.generate(prompts, eval_sampling_params)
    responses = []
    for output in raw_responses:
        response = output.outputs[0].text.strip()
        responses.append(response)
    
    assert len(responses) == len(prompts) == len(ground_truths)
    logger.info(f"Generated {len(responses)} responses")
    
    # Calculate metrics for each example
    all_metrics = []
    results = []
    
    for i, (prompt, response, ground_truth) in enumerate(tqdm(
        zip(prompts, responses, ground_truths), 
        desc="Computing evaluation metrics",
        total=len(prompts)
    )):
        # Calculate metrics using the reward function
        metrics = reward_fn(response, ground_truth)
        all_metrics.append(metrics)
        
        # Prepare result record
        result = {
            "example_id": i,
            "prompt": prompt,
            "model_response": response,
            "ground_truth": ground_truth,
            "metrics": metrics,
            "model_name": model_name,
            "sampling_params": {
                "temperature": eval_sampling_params.temperature,
                "top_p": eval_sampling_params.top_p,
                "max_tokens": eval_sampling_params.max_tokens,
            }
        }
        results.append(result)
    
    # Calculate aggregate metrics
    aggregate_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            metric_values = [metrics[key] for metrics in all_metrics]
            aggregate_metrics[key] = mean(metric_values)
    
    # Analyze result categories
    category_counts = analyze_result_categories(all_metrics)
    
    # Log aggregate metrics
    logger.info("=== Evaluation Results ===")
    for key, value in sorted(aggregate_metrics.items()):
        logger.info(f"{key}: {value:.4f}")
    
    # Log category analysis
    logger.info("=== Result Categories ===")
    total_examples = len(all_metrics)
    for category, count in category_counts.items():
        percentage = (count / total_examples) * 100
        logger.info(f"{category}: {count}/{total_examples} ({percentage:.1f}%)")
    
    # Save results to disk
    logger.info(f"Saving results to {output_path}")
    with xopen(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    # Save aggregate metrics separately
    summary_path = output_path.replace(".jsonl", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "aggregate_metrics": aggregate_metrics,
            "category_analysis": category_counts,
            "total_examples": len(prompts),
            "model_name": model_name,
        }, f, indent=2)
    
    logger.info(f"Saved aggregate metrics to {summary_path}")


def analyze_existing_results(results_path: str) -> None:
    """
    Analyze an existing results file and print category breakdown.
    
    Args:
        results_path: Path to the JSONL results file
    """
    logger.info(f"Loading existing results from {results_path}")
    
    all_metrics = []
    with xopen(results_path) as f:
        for line in f:
            result = json.loads(line)
            all_metrics.append(result["metrics"])
    
    # Analyze categories
    category_counts = analyze_result_categories(all_metrics)
    
    # Print analysis
    print("=== Result Categories Analysis ===")
    total_examples = len(all_metrics)
    for category, count in category_counts.items():
        percentage = (count / total_examples) * 100
        print(f"{category}: {count}/{total_examples} ({percentage:.1f}%)")


def main():
    """Main evaluation function for GSM8K baseline."""
    import sys

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check if user wants to analyze existing results
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        if len(sys.argv) < 3:
            print("Usage: python baseline.py --analyze <results_file.jsonl>")
            return
        results_path = sys.argv[2]
        analyze_existing_results(results_path)
        return
    
    # Load components
    logger.info("Loading r1_zero prompt template...")
    prompt_template = load_r1_zero_prompt()
    
    logger.info("Loading GSM8K test data...")
    test_examples = load_gsm8k_test_data()
    
    # Format prompts
    prompts = format_prompts(test_examples, prompt_template)
    
    # Extract ground truth answers
    ground_truths = [
        extract_ground_truth_answer(example["answer"]) 
        for example in test_examples
    ]
    
    # Initialize vLLM model with Qwen 2.5 Math 1.5B
    model_name_or_path = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-Math-1.5B-Instruct")
    logger.info(f"Initializing vLLM model: {model_name_or_path}")
    
    vllm_model = LLM(
        model=model_name_or_path,
        tensor_parallel_size=1,  # Adjust based on available GPUs
        trust_remote_code=True,
        max_model_len=4096,  # Qwen 2.5 supports up to 32k context length
        download_dir=None,  # Use default HF cache directory
    )
    
    # Set up sampling parameters for evaluation
    eval_sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic for evaluation
        top_p=1.0,
        max_tokens=512,  # Sufficient for mathematical reasoning
        stop=["</answer>"]  # Stop at end of answer tag
    )
    
    # Run evaluation
    evaluate_vllm(
        vllm_model=vllm_model,
        reward_fn=question_only_reward_fn,
        prompts=prompts,
        eval_sampling_params=eval_sampling_params,
        ground_truths=ground_truths,
        output_path="gsm8k_baseline_results.jsonl",
        model_name=model_name_or_path
    )
    
    logger.info("Baseline evaluation completed successfully!")


if __name__ == "__main__":
    main()
