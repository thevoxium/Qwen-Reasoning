# Qwen-Reasoning
## Overview
This project trains the Qwen2.5-7B-Instruct model to REASON. The model is trained using GRPO with multiple custom reward functions to ensure:

- Structured output with clear reasoning and code sections
- High-quality C++ implementations
- Proper competitive programming thinking patterns
- Coherent step-by-step explanations

## Sample Output

**Prompt**: How many r's in the word strawberry?
**Response**:
<reasoning>
I am thinking about how to solve this problem. The question is asking for the number of 'r' characters in the word "strawberry". We need to iterate through the string and count the 'r's. Let's consider the string and count the characters.
</reasoning>
<answer>
```cpp
#include <iostream>
#include <string>

int main() {
    std::string word = "strawberry";
    int count = 0;
    for (char c : word) {
        if (c == 'r') count++;
    }
    std::cout << "Number of 'r's in 'strawberry': " << count << std::endl;
    return 0;
}
```
</answer>


## Key Points
- **Format-Driven Responses**: Generates responses in a strict XML format:
  ```
  <reasoning>
  Step-by-step problem analysis and solution approach
  </reasoning>
  <answer>
  </answer>
  ```

- **Multiple Reward Functions**:
  - Format adherence validation
  - C++ syntax checking
  - Competitive programming thought process evaluation
  - Reasoning coherence assessment
  - Solution correctness verification
  - Strict output formatting
  - Code markdown validation


## Installation

```bash
pip install unsloth vllm
pip install --upgrade pillow
pip install git+https://github.com/huggingface/trl.git@e95f9fb74a3c3647b86f251b7e230ec51c64b72b
```

## Model Configuration
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-7B-Instruct",
    max_seq_length = 1024,
    load_in_4bit = True,
    fast_inference = True,
    max_lora_rank = 8,
    gpu_memory_utilization = 0.5,
)
```

## Training
The model is trained using GRPO with the following key parameters:
- Learning rate: 3e-4
- Adam betas: (0.9, 0.99)
- Weight decay: 0.1
- Cosine learning rate schedule
- Mixed precision training (bfloat16 where supported)
- Wandb integration for training monitoring

## Using the Model
### Via Hugging Face
GGUF models are available at: [Qwen2.5-7B-Atcoder-Reasoning-v1-GGUF](https://huggingface.co/anshul6273/Qwen2.5-7B-Atcoder-Reasoning-v1-GGUF/)

### Using llama.cpp
```bash
~/llama.cpp/build/bin/llama-server -m ~/Downloads/Qwen2.5-7B-Atcoder-Reasoning-v1-Q5_K_M.gguf \
    --port 42069 \
    --temp 0.8 \
    --top_p 0.95 \
    --alias Qwen2.5-7B-Instruct-Atcoder
```

## Memory Optimization
- Adjust `gpu_memory_utilization` (default: 0.5) if experiencing OOM errors
- Use `gradient_accumulation_steps` to control batch size
- Configure `num_generations` based on available memory
- Choose appropriate quantization method for inference

## Available GGUF Variants
- F16 (Full precision)
- Q8_0 (8-bit quantization)
- Q5_K_M

## Acknowledgments
- [DeepSeek R1 Paper](https://arxiv.org/abs/2501.12948)
- [Unsloth's R1 Reasoning Blog](https://unsloth.ai/blog/r1-reasoning)
- [Original Implementation Reference](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb)

## License
This project follows the licensing terms of Qwen2.5-7B-Instruct and its dependencies.
