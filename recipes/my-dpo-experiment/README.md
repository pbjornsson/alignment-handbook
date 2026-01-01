# Getting Started with DPO Fine-tuning

This guide will help you fine-tune a model using Direct Preference Optimization (DPO).

## Prerequisites

**Important:** DPO requires a **supervised fine-tuned (SFT) model**, NOT a base model!

- ❌ Don't use: `meta-llama/Llama-2-7b` (base model)
- ✅ Do use: `meta-llama/Llama-2-7b-chat-hf` (SFT model)

Why? DPO aligns an already instruction-tuned model with preferences. Base models don't follow instructions well enough for DPO to be effective.

## Step 1: Choose Your Starting Model

Popular SFT models to start with:
- `HuggingFaceTB/SmolLM2-1.7B-Instruct` (small, fast)
- `microsoft/Phi-3-mini-4k-instruct` (3.8B, high quality)
- `meta-llama/Llama-3.2-3B-Instruct` (requires approval)
- Or use your own SFT model from previous training!

## Step 2: Choose Your Dataset

### Recommended for Beginners:
**`HuggingFaceH4/ultrafeedback_binarized`**
- General-purpose helpfulness
- 64k high-quality preference pairs
- Used in Zephyr-7B (SOTA at release)

### Other Options:

**For Safety:**
- `HuggingFaceH4/cai-conversation-harmless` - Constitutional AI
- `Anthropic/hh-rlhf` - Helpfulness & Harmlessness

**For Code:**
- `HuggingFaceH4/CodeFeedback-Filtered-Instruction`

**For Conversations:**
- `OpenAssistant/oasst1`
- `HuggingFaceH4/chatbot_arena_conversations`

### Mix Multiple Datasets:
```yaml
dataset_mixture:
  datasets:
    - id: HuggingFaceH4/ultrafeedback_binarized
      split: train_prefs
      weight: 0.7  # 70% from ultrafeedback
    - id: HuggingFaceH4/cai-conversation-harmless
      split: train_prefs
      weight: 0.3  # 30% from safety data
```

## Step 3: Understand the Data Format

DPO datasets need two columns:

**`chosen`**: The preferred response
```json
[
  {"role": "user", "content": "What is the capital of France?"},
  {"role": "assistant", "content": "The capital of France is Paris."}
]
```

**`rejected`**: The dispreferred response
```json
[
  {"role": "user", "content": "What is the capital of France?"},
  {"role": "assistant", "content": "I'm not sure, maybe Lyon?"}
]
```

## Step 4: Configure Your Training

Edit `config_starter.yaml`:

### Key Parameters to Adjust:

**`beta`** (DPO temperature)
- Higher (0.5): Stay closer to reference model (more conservative)
- Lower (0.01): Deviate more from reference (more aggressive)
- Default: 0.1 is a good starting point

**`learning_rate`**
- Typical range: `1e-7` to `5e-6`
- Start low: `5e-7`

**`per_device_train_batch_size`**
- T4 GPU (16GB): 1-2
- A100 GPU (40GB): 4-8
- Adjust based on memory

**`gradient_accumulation_steps`**
- Effective batch size = `batch_size * gradient_accumulation_steps * num_gpus`
- Target effective batch size: 16-64

**`num_train_epochs`**
- Start with 1 epoch
- Monitor eval loss to avoid overfitting

## Step 5: Run Training

### On Colab:
```python
# In your Colab notebook
%cd /content/alignment-handbook
!accelerate launch --config_file recipes/accelerate_configs/ddp.yaml \
    --num_processes=1 \
    scripts/run_dpo.py \
    recipes/my-dpo-experiment/config_starter.yaml
```

### Locally (if you have GPU):
```bash
cd alignment-handbook
accelerate launch --config_file recipes/accelerate_configs/ddp.yaml \
    --num_processes=1 \
    scripts/run_dpo.py \
    recipes/my-dpo-experiment/config_starter.yaml
```

### Override Config from Command Line:
```bash
accelerate launch --config_file recipes/accelerate_configs/ddp.yaml \
    scripts/run_dpo.py \
    recipes/my-dpo-experiment/config_starter.yaml \
    --per_device_train_batch_size=2 \
    --num_train_epochs=2 \
    --learning_rate=1e-6
```

## Step 6: Monitor Training

### Metrics to Watch:

**`rewards/chosen`**: Should be positive and increase
**`rewards/rejected`**: Should be negative and decrease
**`rewards/margins`**: Difference between chosen and rejected (should increase)
**`eval/loss`**: Should decrease (but watch for overfitting)

### Using TensorBoard:
```bash
tensorboard --logdir data/my-first-dpo
```

### Using Weights & Biases:
Add to config:
```yaml
report_to:
  - wandb
```

## Step 7: Evaluate Your Model

Compare your DPO model against the base SFT model:

```python
from transformers import pipeline

# Load your DPO model
generator = pipeline("text-generation", model="data/my-first-dpo")

# Test it
prompt = "What are the benefits of exercise?"
response = generator(prompt, max_length=200)
print(response)
```

## Tips for Success

### Memory Issues?
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps` to 8
- Enable `gradient_checkpointing: true`
- Reduce `max_length` to 512

### Model Not Improving?
- Check your base model is SFT, not base model
- Try increasing `beta` (e.g., 0.5)
- Reduce learning rate
- Check data quality (view some examples)

### Training Too Slow?
- Use smaller model (SmolLM2-1.7B)
- Reduce dataset size (use `weight: 0.1` for 10% of data)
- Use QLoRA instead of full fine-tuning

### Want Better Results?
- Train for more epochs (2-3)
- Mix multiple preference datasets
- Use larger, higher-quality SFT base model
- Tune `beta` hyperparameter

## Next Steps

1. **Create custom preference data** for your specific use case
2. **Experiment with different base models** and datasets
3. **Try ORPO** (combines SFT + DPO in one step)
4. **Benchmark** your model on MT-Bench or AlpacaEval

## Common Datasets Reference

| Dataset | Size | Domain | Best For |
|---------|------|--------|----------|
| `ultrafeedback_binarized` | 64k | General | Overall helpfulness |
| `cai-conversation-harmless` | 42k | Safety | Harmlessness |
| `hh-rlhf` | 169k | General | Helpfulness + Safety |
| `orca_dpo_pairs` | 13k | Reasoning | Complex reasoning |
| `stack-exchange-paired` | 10M | Technical | Technical Q&A |

## Troubleshooting

**"CUDA out of memory"**
- Set `per_device_train_batch_size: 1`
- Set `gradient_accumulation_steps: 8`
- Enable gradient checkpointing
- Reduce sequence lengths

**"Dataset has no 'chosen' column"**
- Wrong dataset split (use `train_prefs` not `train_sft`)
- Dataset not formatted for DPO
- Check dataset card on Hugging Face

**"Loss is NaN"**
- Lower learning rate (try 1e-7)
- Check data quality
- Ensure using BF16/FP16

**"Model outputs gibberish"**
- Start from SFT model, not base model
- Reduce learning rate
- Check if overfitting (eval loss increasing)
