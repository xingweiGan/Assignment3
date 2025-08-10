from datasets import load_dataset

ds = load_dataset("openai/gsm8k", "main")

""" See an eg of loaded dataset
# Look at a sample from train split
print("Sample from train split:")
sample = ds['train'][3]
print(f"Question: {sample['question'][:200]}...")
print(f"Answer: {sample['answer'][:200]}...")
print()
"""

#Transform the ds to r1-format prompts (list)
def r1_prompts_from_train(ds):
    header = (
        "A conversation between User and Assistant. "
        "The Assistant first thinks step-by-step inside <think>...</think>, "
        "then gives ONLY the final result inside <answer>...</answer>.\n"
    )
    return [f"{header}User: {q}\nAssistant: <think>" for q in ds["train"]["question"]]

prompts = r1_prompts_from_train(ds)
print(len(prompts))  # print one element
