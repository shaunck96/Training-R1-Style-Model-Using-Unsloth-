# Training-R1-Style-Model-Using-Unsloth-

# Llama 3.1 (8B) - GRPO Fine-tuning with Unsloth

This repository demonstrates how to fine-tune the **Meta Llama 3.1 (8B) Instruct** model using Unsloth's **GRPO** reinforcement learning method. You can run it on a **free** Tesla T4 Google Colab instance. Below is a high-level walkthrough of the code and usage instructions.

---

## Overview

- **Unsloth** is a library for training, finetuning, and deploying LLMs in various ways, including reinforcement learning (RL).  
- **GRPO** (Guided Reinforcement Policy Optimization) is an RL approach integrated within Unsloth, using custom reward functions to guide model updates.
- In this notebook, you will:
  1. Install Unsloth & dependencies.
  2. Load the Llama 3.1 (8B) Instruct model (4-bit or 16-bit precision).
  3. Prepare data for Chain-of-Thought style training.
  4. Train the model using **GRPO** with multiple **reward functions**.
  5. Evaluate and generate model outputs (inference).
  6. Save or push the fine-tuned model in different formats (float16, 4-bit, LoRA adapters, GGUF, etc.).

---
## 1. Requirements

- Python 3.8+
- A GPU with enough VRAM (e.g., a Tesla T4 in Google Colab).
- [Unsloth](https://github.com/unslothai/unsloth) library.
- [vLLM](https://github.com/vllm-project/vllm) for fast inference.
- [Hugging Face Transformers](https://github.com/huggingface/transformers), [Accelerate](https://github.com/huggingface/accelerate), plus standard dependencies (`datasets`, `trl`, `torch`).

---
## 2. Installation

```
pip install unsloth vllm
pip install --upgrade pillow
```

If you’re in Google Colab, simply run these commands in a cell.

---
## 3. Model & Environment Setup

```python
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

from unsloth import is_bfloat16_supported
import torch

# Model parameters
max_seq_length = 512
lora_rank = 32

# Load base Llama 3.1 8B Instruct
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/meta-Llama-3.1-8B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,      # Set False for 16-bit
    fast_inference=True,    # vLLM for fast inference
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.6,
)

# Prepare LoRA with PEFT
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
```

---
## 4. Data Preparation

Below, we load and preprocess **GSM8k** as an example. You can adapt these steps for your own dataset. We structure each sample into a conversation-like format and extract the correct answer.

```python
import re
from datasets import load_dataset, Dataset

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def extract_hash_answer(text: str) -> str:
    if "####" not in text:
        return ""
    return text.split("####")[1].strip()

def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })
    return data

dataset = get_gsm8k_questions()
```

---
## 5. Reward Functions

Multiple reward functions are combined to score the generated answers:

- **Correctness**: Compares the extracted answer with ground truth.
- **Integer Check**: Checks if the answer is a digit.
- **Strict Format**: Ensures valid `<reasoning>...</reasoning><answer>...</answer>` format.
- **Soft Format**: A more lenient check for the correct tags.
- **XML Count**: Rewards partial correctness for each XML tag.

```python
def correctness_reward_func(prompts, completions, answer, **kwargs):
    ...

def int_reward_func(completions, **kwargs):
    ...

def strict_format_reward_func(completions, **kwargs):
    ...

def soft_format_reward_func(completions, **kwargs):
    ...

def xmlcount_reward_func(completions, **kwargs):
    ...
```

---
## 6. Training with GRPO

Initialize **GRPO** configuration and start the training:

```python
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    use_vllm=True,
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=6,
    max_prompt_length=256,
    max_completion_length=200,
    max_steps=250,   # Modify as needed
    save_steps=250,
    max_grad_norm=0.1,
    report_to="none",
    output_dir="outputs",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
```

You should see logs with metrics like `reward`, `kl`, etc. as training progresses.

---
## 7. Inference

After training, test the model’s performance via `fast_generate`:

```python
from vllm import SamplingParams

text = tokenizer.apply_chat_template(
    [
      {"role": "system", "content": "Respond as a helpful assistant."},
      {"role": "user", "content": "Calculate pi."}
    ],
    tokenize=False,
    add_generation_prompt=True
)

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=1024,
)

# Generate without LoRA
output = model.fast_generate(
    [text],
    sampling_params=sampling_params,
    lora_request=None,
)[0].outputs[0].text
print(output)

# Generate with your trained LoRA
model.save_lora("grpo_saved_lora")  # Save LoRA if not yet saved

output_with_lora = model.fast_generate(
    [text],
    sampling_params=sampling_params,
    lora_request=model.load_lora("grpo_saved_lora"),
)[0].outputs[0].text
print(output_with_lora)
```

---
## 8. Saving & Exporting Models

Unsloth supports multiple methods to **save or merge** model weights.

### 8.1 Saving to Float16 or 4-bit

```python
# Merge to 16-bit
model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")

# Merge to 4-bit
model.save_pretrained_merged("model", tokenizer, save_method="merged_4bit")
```

To push to Hugging Face:
```python
model.push_to_hub_merged(
    repo_id="username/your_model_4bit",
    tokenizer=tokenizer,
    save_method="merged_4bit",
    token="YOUR_HF_ACCESS_TOKEN",
)
```

### 8.2 Saving LoRA Adapters

```python
# Save LoRA adapters only
model.save_lora("grpo_saved_lora")

# Or merge LoRA and save
model.save_pretrained_merged("model_lora", tokenizer, save_method="lora")
```
Or push to Hugging Face:
```python
model.push_to_hub_merged(
    repo_id="username/lora_adapters",
    tokenizer=tokenizer,
    save_method="lora",
    token="YOUR_HF_ACCESS_TOKEN",
)
```

### 8.3 GGUF / llama.cpp Conversion

Convert the model to GGUF format for llama.cpp, Jan, Open WebUI, etc.:

```python
# Basic 8-bit quantization
model.save_pretrained_gguf("model_gguf", tokenizer)  # defaults to q8_0

# Or choose f16, q4_k_m, q5_k_m, etc.
model.save_pretrained_gguf("model_fp16", tokenizer, quantization_method="f16")
model.save_pretrained_gguf("model_q4", tokenizer, quantization_method="q4_k_m")

# Push multiple quantizations to HF
model.push_to_hub_gguf(
    repo_id="username/your_llama_gguf",
    tokenizer=tokenizer,
    quantization_method=["q4_k_m", "q8_0"],
    token="YOUR_HF_ACCESS_TOKEN",
)
```
