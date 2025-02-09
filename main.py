# import sys; modules = list(sys.modules.keys())
# for x in modules: sys.modules.pop(x) if "PIL" in x or "google" in x else None
# 
# !pip install unsloth vllm
# !pip install --upgrade pillow
# # If you are running this notebook on local, you need to install `diffusers` too
# # !pip install diffusers
# # Temporarily install a specific TRL nightly version
# !pip install git+https://github.com/huggingface/trl.git@e95f9fb74a3c3647b86f251b7e230ec51c64b72b

from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)



from unsloth import is_bfloat16_supported
import torch
max_seq_length = 1024 
lora_rank = 8 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-7B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
)

import re
from datasets import load_dataset, Dataset
import pandas as pd
import json

# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
     <reasoning>
     ...your detailed internal monologue...
     </reasoning>
     <answer>
     ```cpp
     ...your final C++ code...
     ```
     </answer>
There should be no extra text or tags before, between, or after these sections.

"""



def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def format_dataset(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    def format_prompt(row):
        return {
            'prompt': [  # Changed from 'messages' to 'prompt'
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': row['description']}
            ],
            'answer': row['solution']  # Added explicit answer field
        }

    dataset = Dataset.from_pandas(df)
    formatted_dataset = dataset.map(format_prompt)
    return formatted_dataset

dataset = format_dataset('/content/atcoder_training_data.json')

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"\nResponse:\n{responses[0]}")
    return [10.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def format_adherence_reward(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        content = completion[0]["content"]
        xml_count = count_xml(content)
        valid_structure = True

        # Basic tag presence check
        if "<reasoning>" not in content: valid_structure = False
        if "</reasoning>" not in content: valid_structure = False
        if "<answer>" not in content: valid_structure = False
        if "</answer>" not in content: valid_structure = False

        # Order validation
        reasoning_pos = content.find("<reasoning>")
        answer_pos = content.find("<answer>")
        if answer_pos != -1 and reasoning_pos != -1 and answer_pos < reasoning_pos:
            valid_structure = False

        # Content length sanity checks
        reasoning_content = content.split("<reasoning>")[-1].split("</reasoning>")[0].strip()
        answer_content = content.split("<answer>")[-1].split("</answer>")[0].strip()
        if len(reasoning_content) < 30 or len(answer_content) < 5:
            valid_structure = False

        rewards.append((0.5*len(reasoning_content)+0.5*len(answer_content))/100+5*xml_count if valid_structure else 0.0)
    return rewards


def cpp_syntax_reward(completions, **kwargs):
    rewards = []
    cpp_patterns = [
        r"#include\s*<",
        r"using\s+namespace\s+std",
        r"std::",
        r"int\s+main\s*\("
    ]
    for completion in completions:
        content = completion[0]["content"]
        score = sum(1 for pattern in cpp_patterns if re.search(pattern, content))
        rewards.append(score * 3)  # 0.5 per C++ pattern found
    return rewards


def reasoning_coherence_reward(prompts, completions, **kwargs):
    """Rewards logical step-by-step progression"""
    rewards = []
    transition_phrases = [
        "problem analysis", "key observations",
        "algorithm chosen", "edge cases considered",
        "complexity analysis", "alternative approaches"
    ]

    for completion in completions:
        content = completion[0]["content"].lower()
        reasoning_block = content.split("<reasoning>")[-1].split("</reasoning>")[0]

        score = 0.0
        lines = [line.strip() for line in reasoning_block.split("\n") if line.strip()]

        # Step progression scoring
        step_markers = sum(1 for line in lines if line.startswith(("step", "1.", "2.", "3.", "finally")))
        score += (step_markers)

        # Conceptual markers
        concept_hits = sum(1 for phrase in transition_phrases if phrase in reasoning_block)
        score += (concept_hits)

        # Continuity check
        if len(lines) >= 3 and any(">" in line or "-" in line for line in lines):
            score += 0.5

        rewards.append(max(0.0, score))
    return rewards

def logical_consistency_reward(prompts, completions, **kwargs):
    """Ensures reasoning connects to final answer"""
    rewards = []
    for completion in completions:
        content = completion[0]["content"]
        reasoning = content.split("<reasoning>")[-1].split("</reasoning>")[0].lower()
        answer = content.split("<answer>")[-1].split("</answer>")[0].lower()

        # Answer justification check
        justification_score = 1.0 if any(
            term in reasoning for term in ["therefore", "thus", "hence", "conclusion"]
        ) else 0.3

        # Numerical consistency
        answer_numbers = set(re.findall(r'\b\d+\b', answer))
        reasoning_numbers = set(re.findall(r'\b\d+\b', reasoning))
        number_score = min(1.0, len(answer_numbers & reasoning_numbers) * 0.5)

        rewards.append(justification_score + number_score)
    return rewards

def cp_thinking_reward(prompts, completions, **kwargs):
    """Competitive programming specific patterns"""
    rewards = []
    thinking_markers = {
        "time complexity": 0.5,
        "space optimization": 0.4,
        "edge case": 0.6,
        "test input": 0.3,
        "constraints analysis": 0.5,
        "algorithm selection": 0.6,
        "algorithm": 0.5,
        "data structure": 0.5,
    }

    for completion in completions:
        reasoning = completion[0]["content"].split("<reasoning>")[-1].split("</reasoning>")[0].lower()
        score = 0.0

        # Density-based scoring
        total_hits = sum(reasoning.count(phrase) for phrase in thinking_markers)
        unique_hits = sum(1 for phrase in thinking_markers if phrase in reasoning)
        score = total_hits * 0.2 + unique_hits * 0.4

        rewards.append(score)
    return rewards

def answer_confidence_reward(prompts, completions, **kwargs):
    """Rewards definitive solutions"""
    rewards = []
    confidence_indicators = {
        "clearly": 0.2,
        "obviously": 0.2,
        "definitely": 0.2,
        "conclusively": 0.2,
        "maybe": 0.2,
        "perhaps": 0.2,
        "possibly": 0.2,
        "I think": 0.2
    }

    for completion in completions:
        content = completion[0]["content"].lower()
        reasoning = content.split("<reasoning>")[-1].split("</reasoning>")[0]


        score = 0.0
        # Positive markers
        for phrase, value in confidence_indicators.items():
            score += content.count(phrase) * value

        rewards.append(score)
    return rewards


def strict_start_reward(prompts, completions, **kwargs):
    """Enforces the response starts with <reasoning> with no preceding text"""
    rewards = []
    for completion in completions:
        content = completion[0]["content"]

        # Find first non-whitespace character position
        first_non_ws = next((i for i, c in enumerate(content) if not c.isspace()), None)

        # Case 1: No <reasoning> tag at all
        if "<reasoning>" not in content:
            rewards.append(-5.0)
            continue

        reasoning_pos = content.find("<reasoning>")

        # Case 2: Perfect start (first non-whitespace is <reasoning>)
        if first_non_ws == reasoning_pos:
            rewards.append(5.0)

        # Case 3: Text before reasoning (even whitespace)
        elif reasoning_pos > 0:
            # Calculate penalty severity
            text_before = content[:reasoning_pos]
            penalty = len(text_before) / 50.0  # Max 1.0 penalty for 50+ chars
            rewards.append(penalty)

        # Case 4: <reasoning> exists but malformed position
        else:
            rewards.append(0.0)

    return rewards


def introspection_reward_func(prompts, completions, **kwargs) -> list[float]:
    rewards = []

    introspection_phrases = [
        "think",
        "thinking",
        "wait",
        "we can do it",
        "this can be correct",
        "this can be done",
        "perhaps",
        "maybe",
        "let's try",
        "alternative",
        "consider",
        "upon reflection",
        "wonder",
        "let's examine",
        "thinking about",
        "after some thought",
        "suspect",
        "believe",
        "reflection",
        "it seems that",
        "conclude",
        "my",
        "hypothesize",
        "analyze",
        "reason",
        "considering",
        "possibility",
        "might be",
        "recall",
        "estimate",
        "deduce",
        "oh"
    ]

    for completion in completions:
        content = completion[0]["content"]
        score = 0.0

        # Check if the response has a <reasoning> section
        if "<reasoning>" in content and "</reasoning>" in content:
            # Extract the reasoning section and convert to lower-case for matching
            reasoning = content.split("<reasoning>")[-1].split("</reasoning>")[0].lower()

            # Count occurrences of each introspective phrase and add bonus for each occurrence
            for phrase in introspection_phrases:
                count = reasoning.count(phrase)
                if count > 0:
                    score += 0.5 * count  # Increase reward by 0.5 per occurrence

        rewards.append(min(score, 10.0))

    return rewards

def strict_output_format_reward(prompts, completions, **kwargs) -> list[float]:
    rewards = []
    for completion in completions:
        content = completion[0]["content"].strip()
        valid = True

        # Check that each required tag appears exactly once.
        if (content.count("<reasoning>") != 1 or
            content.count("</reasoning>") != 1 or
            content.count("<answer>") != 1 or
            content.count("</answer>") != 1):
            valid = False

        if valid:
            # Get positions of the tags.
            r_open  = content.find("<reasoning>")
            r_close = content.find("</reasoning>")
            a_open  = content.find("<answer>")
            a_close = content.find("</answer>")

            # Enforce the correct order:
            #   <reasoning> ... </reasoning> must come before <answer> ... </answer>
            if not (r_open < r_close < a_open < a_close):
                valid = False

            # Ensure there is no extra text before the <reasoning> tag or after the </answer> tag.
            prefix = content[:r_open].strip()
            suffix = content[a_close + len("</answer>"):].strip()
            if prefix or suffix:
                valid = False

        # Assign reward: full reward if valid, strict penalty otherwise.
        rewards.append(10.0 if valid else 0.0)

    return rewards

def cpp_markdown_reward(prompts, completions, **kwargs) -> list[float]:
    rewards = []
    for completion in completions:
        content = completion[0]["content"]
        answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", content, re.DOTALL)
        if not answer_match:
            rewards.append(0.0)
            continue

        answer_block = answer_match.group(1).strip()
        lines = answer_block.splitlines()
        # Need at least three lines: a starting fence, at least one line of code, and an ending fence.
        if len(lines) < 3:
            rewards.append(0.0)
            continue

        start_line = lines[0].strip()
        end_line = lines[-1].strip()

        # Count occurrences on the starting and ending lines.
        start_marker_count = start_line.count("```cpp")
        end_marker_count = end_line.count("```")

        # If there's no proper starting or ending marker, reward is 0.
        if start_marker_count < 1 or end_marker_count < 1:
            rewards.append(0.0)
            continue

        # Start with a full reward.
        reward = 5.0

        # Penalize extra markers on the first line (beyond the first occurrence).
        if start_marker_count > 1:
            reward -= (start_marker_count - 1)
        # Penalize extra markers on the last line.
        if end_marker_count > 1:
            reward -= (end_marker_count - 1)

        # Additionally, check the inner lines for any extraneous markdown fences.
        for line in lines[1:-1]:
            line_strip = line.strip()
            if line_strip in ["```", "```cpp"]:
                reward -= 1.0

        # Ensure the reward is not negative.
        rewards.append(max(0.0, reward))
    return rewards

import wandb
wandb.init(project="atcoder-v7", entity="anshul-6273-freelance")


from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 3e-4,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_steps = 5,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    #per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 2, # Decrease if out of memory
    max_prompt_length = 256,
    max_completion_length = 512,
    num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 100,
    save_steps = 50,
    max_grad_norm = 0.1,
    report_to = "wandb", # Can use Weights & Biases
    output_dir = "outputs",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        #xmlcount_reward_func,
        #soft_format_reward_func,
        #strict_format_reward_func,
        #int_reward_func,
        format_adherence_reward,
        correctness_reward_func,
        #answer_confidence_reward,
        cp_thinking_reward,
        reasoning_coherence_reward,
        strict_start_reward,
        cpp_syntax_reward,
        introspection_reward_func,
        strict_output_format_reward,
        cpp_markdown_reward
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()

question = """
implement dijsktra in C++
"""

text = tokenizer.apply_chat_template([
    {"role" : "user", "content" : question},
], tokenize = False, add_generation_prompt = True)

from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 0.8,
    top_p = 0.95,
    max_tokens = 2096,
)
output = model.fast_generate(
    [text],
    sampling_params = sampling_params,
    lora_request = None,
)[0].outputs[0].text

print(output)


model.save_lora("grpo_saved_lora_final")


text = tokenizer.apply_chat_template([
    {"role" : "system", "content" : SYSTEM_PROMPT},
    {"role" : "user", "content" : question},
], tokenize = False, add_generation_prompt = True)

from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 1,
    top_p = 0.95,
    max_tokens = 8192*2,
)
output = model.fast_generate(
    text,
    sampling_params = sampling_params,
    lora_request = model.load_lora("grpo_saved_lora_final"),
)[0].outputs[0].text

print(output)

# Merge to 16bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# Merge to 4bit
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

# Just LoRA adapters
if False: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = "")


# Save to 8bit Q8_0
if False: model.save_pretrained_gguf("model", tokenizer,)
# Remember to go to https://huggingface.co/settings/tokens for a token!
# And change hf to your username!
if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# Save to 16bit GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("anshul6273/Qwen2.5-7B-Atcoder-Reasoning-v1-GGUF", tokenizer, quantization_method = "f16", token = "")

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("Newmodel", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("anshul6273/newExp", tokenizer, quantization_method = "q4_k_m", token = "")

# Save to multiple GGUF options - much faster if you want multiple!
if True:
    model.push_to_hub_gguf(
        "anshul6273/Qwen2.5-7B-Atcoder-Reasoning-v1-GGUF", # Change hf to your username!
        tokenizer,
        quantization_method = ["f16", "q8_0", "q5_k_m",],
        token = "",
    )

from huggingface_hub import HfApi

# Initialize client
api = HfApi()

# Upload large folder (resumable, multi-threaded)
api.upload_large_folder(
    folder_path="/content/anshul6273/Qwen2.5-7B-Atcoder-Reasoning-v1-GGUF",
    repo_id="anshul6273/Qwen2.5-7B-Atcoder-Reasoning-v1-GGUF",
    repo_type="model"
)

