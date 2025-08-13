
from unittest.mock import patch
from transformers import PreTrainedModel
from vllm import LLM
import torch
from vllm.model_executor import set_random_seed as vllm_set_random_seed
import pandas as pd
from tests.adapters import run_tokenize_prompt_and_output, run_get_response_log_probs, run_sft_microbatch_train_step
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from datasets import load_dataset
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.math_baseline import evaluate_vllm

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)

    # Monkeypatch from TRL:
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model 
    llm_model.load_weights(state_dict.items())


#--------Usage---------------

#------Part A:---------------------------------
#Load the pretrained model-Qwen
# --- setup (same process) ---
MODEL_ID = "Qwen/Qwen2.5-Math-1.5B"
device_train = "cuda:0"   # GPU A
device_eval  = "cuda:1"   # GPU B

# Policy model on GPU A
#1. Tokenize SFT dataset- Here we use the original form 
#!!!!!! # of unique egs
num_unique=128
df = next(pd.read_json("data/gsm8k/train.jsonl", lines=True, chunksize=num_unique))
sft_q = df["question"].tolist()
sft_a = df["answer"].tolist()
#print(sft_q[127])

model_id="Qwen/Qwen2.5-Math-1.5B"
tokenizer=AutoTokenizer.from_pretrained(model_id)
data_tokenized = run_tokenize_prompt_and_output(sft_q, sft_a, tokenizer)


#2. Compute the loss and gradient of a batch sampled from SFT dataset AND on test.jsonl to get validation accuracy (Use R1-format)

#!!!!!! gradient_accumulation_steps and microbatch_size
gradient_accumulation_steps=4
microbatch_size=4
num_microbatch=num_unique/microbatch_size
num_egs_per_microbatch=microbatch_size*gradient_accumulation_steps


#For first MICROBATCH!!!!!!

model= AutoModelForCausalLM.from_pretrained(model_id).to(device_train).train()
# NEW: optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)
optimizer.zero_grad()

#Start to update
input_ids= data_tokenized["input_ids"][:num_egs_per_microbatch].to(device_train)
labels= data_tokenized["labels"][:num_egs_per_microbatch].to(device_train)
response_mask= data_tokenized["response_mask"][:num_egs_per_microbatch].to(device_train)

log_probs=run_get_response_log_probs(model,input_ids,labels)["log_probs"]
loss,metadata=run_sft_microbatch_train_step(log_probs,response_mask,gradient_accumulation_steps)

loss.backward()
optimizer.step()

print(loss)
print(next(model.parameters()).device)


#------Part B:---------------------------------
# vLLM on GPU B (inference-only)
llm = init_vllm(model_id=MODEL_ID, device=device_eval, seed=42) 
load_policy_into_vllm_instance(model, llm)



"""
#Start to evaluate- EDIT HERE!!!
ds = load_dataset("openai/gsm8k", "main")

# Lists of gold answers aligned with each split
train_gold = [extract_gsm8k_gold(a) for a in ds["train"]["answer"]]
prompts = r1_prompts_from_train(ds)
sampling = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=192,
    stop=["</answer>"],
    include_stop_str_in_output=True,
)
evaluate_vllm(llm,r1_zero_reward_fn,prompts,train_gold,sampling)
"""

