
from unittest.mock import patch
from transformers import PreTrainedModel
from vllm import LLM, SamplingParams
import torch
from vllm.model_executor import set_random_seed as vllm_set_random_seed
import pandas as pd
from tests.adapters import run_tokenize_prompt_and_output, run_get_response_log_probs, run_sft_microbatch_train_step
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.math_baseline import r1_prompts_from_train, extract_gsm8k_gold, evaluate_vllm


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

def r1_answer_from_train(ds):
    """
    Take a list of GSM8K answer strings and:
      - replace the final line prefix "#### " with "</think><answer>"
      - append " </answer>" at the end
    """
    return [a.replace("\n#### ", "\n</think> <answer>") + "</answer>" for a in ds]


#--------Usage---------------

#------Part A & B Set-up:---------------------------------
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
#Adjusting to r1-style prompts and answers
prompt_train=r1_prompts_from_train(sft_q)
answer_train = r1_answer_from_train(sft_a)


model_id="Qwen/Qwen2.5-Math-1.5B"
tokenizer=AutoTokenizer.from_pretrained(model_id)
data_tokenized = run_tokenize_prompt_and_output(prompt_train, answer_train, tokenizer)



# Set-up on GPU B (inference-only)
llm = init_vllm(model_id=MODEL_ID, device=device_eval, seed=42) 
#now let's evaluate its performance
df_eval = pd.read_json("data/gsm8k/test.jsonl", lines=True)
sft_q_eval = df_eval["question"].tolist()
sft_a_eval = df_eval["answer"].tolist()
prompts_eval = r1_prompts_from_train(sft_q_eval)
ground_truth_eval = [extract_gsm8k_gold(a) for a in sft_a_eval]
sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=192,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

#---------------------------------------------------------------
#TRAINING
# Compute the loss and gradient of a batch sampled from SFT dataset AND on test.jsonl to get validation accuracy (Use R1-format)

#!!!!!! gradient_accumulation_steps and microbatch_size
gradient_accumulation_steps=4
microbatch_size=4
num_microbatch=num_unique/microbatch_size
num_egs_per_microbatch=microbatch_size*gradient_accumulation_steps
num_mbs = num_unique // num_egs_per_microbatch  # 128 // 16 = 8

model= AutoModelForCausalLM.from_pretrained(model_id).to(device_train).train()
# NEW: optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)
optimizer.zero_grad()



#Start to update-for whole training loop:
#make data and model in same GPU.
for mb in range(num_mbs):
    base = mb * num_egs_per_microbatch
    end  = base + num_egs_per_microbatch

    # slice one macro-batch (16 examples) on CPU
    ids  = data_tokenized["input_ids"][base:end]
    lbls = data_tokenized["labels"][base:end]
    msk  = data_tokenized["response_mask"][base:end]

    # split into 4 microbatches of 4 each, accumulate grads
    for step in range(gradient_accumulation_steps):  # 0..3
        s = step * microbatch_size
        e = s + microbatch_size

        mb_ids = ids[s:e].to(device_train, non_blocking=True)
        mb_lbl = lbls[s:e].to(device_train, non_blocking=True)
        mb_msk = msk[s:e].to(device_train, non_blocking=True)

        mb_logp = run_get_response_log_probs(model, mb_ids, mb_lbl)["log_probs"]

        # scale manually for accumulation; pass grad_accum=1 to avoid double-scaling
        mb_loss, _ = run_sft_microbatch_train_step(mb_logp, mb_msk, gradient_accumulation_steps=1)
        print(f"loss: { (mb_loss):.4f}")
        (mb_loss / gradient_accumulation_steps).backward()

    optimizer.step()
    optimizer.zero_grad()
    load_policy_into_vllm_instance(model, llm)
    evaluate_vllm(llm, r1_zero_reward_fn, prompts_eval, ground_truth_eval, sampling, mb)





