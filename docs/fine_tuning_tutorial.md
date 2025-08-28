# Fine-Tuning Tutorial

This comprehensive tutorial will guide you through the process of fine-tuning language models on your domain-specific data using the LLM Optimization Platform.

## Table of Contents

1. [Overview](#overview)
2. [Dataset Preparation](#dataset-preparation)
3. [Training Configuration](#training-configuration)
4. [Fine-Tuning Process](#fine-tuning-process)
5. [Model Evaluation](#model-evaluation)
6. [Deployment and Usage](#deployment-and-usage)
7. [Advanced Techniques](#advanced-techniques)
8. [Troubleshooting](#troubleshooting)

## Overview

### What is Fine-Tuning?

Fine-tuning is the process of taking a pre-trained language model and training it further on your specific dataset. This allows the model to:

- **Adapt to your domain:** Learn domain-specific terminology and patterns
- **Improve performance:** Better understand your specific use cases
- **Maintain general knowledge:** Retain the broad knowledge from pre-training
- **Reduce costs:** Often smaller fine-tuned models can outperform larger general models

### When to Fine-Tune

Consider fine-tuning when:
- ✅ You have domain-specific data (legal, medical, technical)
- ✅ You need consistent formatting or style
- ✅ General models don't perform well on your tasks
- ✅ You want to reduce inference costs
- ✅ You have at least 100-1000 high-quality examples

### Supported Models

The platform supports fine-tuning these base models:
- **GPT-2** (small, medium, large)
- **DistilGPT-2** (faster, smaller)
- **Custom models** (bring your own base model)

## Dataset Preparation

### Dataset Format

Your dataset should be in **JSONL** (JSON Lines) format, with each line containing a prompt-response pair:

```json
{"prompt": "What is machine learning?", "response": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."}
{"prompt": "Explain neural networks", "response": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information."}
{"prompt": "Define deep learning", "response": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data."}
```

### Dataset Quality Guidelines

#### 1. **High-Quality Examples**
```json
// ✅ Good example - Clear, specific, well-formatted
{"prompt": "How do I implement a binary search algorithm in Python?", "response": "Here's a Python implementation of binary search:\n\n```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    \n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    \n    return -1\n```\n\nThis algorithm has O(log n) time complexity and works on sorted arrays."}

// ❌ Poor example - Vague, incomplete
{"prompt": "coding", "response": "use python"}
```

#### 2. **Consistent Format**
Maintain consistent formatting across all examples:
- Use the same prompt style
- Keep response format uniform
- Include similar levels of detail

#### 3. **Diverse Examples**
Include variety in your dataset:
- Different question types
- Various difficulty levels
- Multiple aspects of your domain

### Dataset Creation Example

Let's create a dataset for a customer support chatbot:

```python
import json

# Sample customer support data
support_data = [
    {
        "prompt": "How do I reset my password?",
        "response": "To reset your password:\n1. Go to the login page\n2. Click 'Forgot Password'\n3. Enter your email address\n4. Check your email for reset instructions\n5. Follow the link to create a new password"
    },
    {
        "prompt": "What are your business hours?",
        "response": "Our customer support is available:\n- Monday-Friday: 9 AM - 6 PM EST\n- Saturday: 10 AM - 4 PM EST\n- Sunday: Closed\n\nFor urgent issues, please use our emergency contact form."
    },
    {
        "prompt": "How do I cancel my subscription?",
        "response": "To cancel your subscription:\n1. Log into your account\n2. Go to 'Account Settings'\n3. Click 'Subscription'\n4. Select 'Cancel Subscription'\n5. Confirm cancellation\n\nNote: You'll retain access until your current billing period ends."
    }
]

# Save as JSONL
with open('customer_support_dataset.jsonl', 'w') as f:
    for item in support_data:
        f.write(json.dumps(item) + '\n')
```

### Dataset Validation

Always validate your dataset before training:

```python
from fine_tuning.dataset_validator import DatasetValidator

validator = DatasetValidator()
result = validator.validate_file('customer_support_dataset.jsonl')

print(f"Dataset is valid: {result.is_valid}")
print(f"Total examples: {result.total_examples}")
print(f"Average prompt length: {result.avg_prompt_length}")
print(f"Average response length: {result.avg_response_length}")

if result.issues:
    print("Issues found:")
    for issue in result.issues:
        print(f"- {issue}")
```

### Dataset Splitting

The platform automatically splits your dataset:
- **Training set:** 80% (used for training)
- **Validation set:** 20% (used for evaluation)

You can customize this split:

```python
from fine_tuning.dataset_tokenizer import DatasetTokenizer

tokenizer = DatasetTokenizer()
train_data, val_data = tokenizer.split_dataset(
    'customer_support_dataset.jsonl',
    validation_split=0.2  # 20% for validation
)
```

## Training Configuration

### Basic Configuration

Create a training configuration:

```python
from fine_tuning.training_config import TrainingConfig

config = TrainingConfig(
    # Model settings
    base_model="gpt2",  # or "distilgpt2", "gpt2-medium"
    
    # Training parameters
    epochs=3,
    batch_size=4,
    learning_rate=5e-5,
    warmup_steps=100,
    
    # Optimization
    use_lora=True,  # Enables LoRA for efficient training
    lora_rank=16,
    
    # Monitoring
    eval_steps=100,
    save_steps=500,
    logging_steps=50
)
```

### Configuration Parameters Explained

#### **Model Parameters**
- `base_model`: The pre-trained model to fine-tune
- `max_length`: Maximum sequence length (default: 512)

#### **Training Parameters**
- `epochs`: Number of training epochs (3-5 recommended)
- `batch_size`: Training batch size (start with 4, adjust based on memory)
- `learning_rate`: Learning rate (5e-5 is a good starting point)
- `warmup_steps`: Gradual learning rate increase (10% of total steps)

#### **Memory Optimization**
- `use_lora`: Enable LoRA for memory-efficient training
- `lora_rank`: LoRA rank (8-64, higher = more parameters)
- `gradient_accumulation_steps`: Accumulate gradients (if memory limited)

#### **Monitoring**
- `eval_steps`: How often to evaluate on validation set
- `save_steps`: How often to save checkpoints
- `logging_steps`: How often to log training metrics

### Advanced Configuration

For production use cases:

```python
config = TrainingConfig(
    base_model="gpt2-medium",
    epochs=5,
    batch_size=8,
    learning_rate=3e-5,
    warmup_steps=200,
    
    # Advanced optimization
    use_lora=True,
    lora_rank=32,
    lora_alpha=64,
    lora_dropout=0.1,
    
    # Regularization
    weight_decay=0.01,
    max_grad_norm=1.0,
    
    # Scheduling
    lr_scheduler_type="cosine",
    
    # Early stopping
    early_stopping_patience=3,
    early_stopping_threshold=0.01,
    
    # Monitoring
    eval_steps=50,
    save_steps=200,
    logging_steps=25,
    
    # Output
    output_dir="./models/customer-support-v2",
    save_total_limit=3  # Keep only 3 best checkpoints
)
```

## Fine-Tuning Process

### Step 1: Initialize the Service

```python
from fine_tuning.fine_tuning_service import FineTuningService

service = FineTuningService()
```

### Step 2: Start Training

```python
# Start the training job
job = service.start_training(
    dataset_path="customer_support_dataset.jsonl",
    config=config,
    experiment_name="customer-support-v1"
)

print(f"Training job started!")
print(f"Job ID: {job.job_id}")
print(f"Output directory: {job.output_dir}")
```

### Step 3: Monitor Progress

```python
import time

while True:
    status = service.get_training_status(job.job_id)
    
    print(f"Status: {status.status}")
    print(f"Epoch: {status.current_epoch}/{config.epochs}")
    print(f"Step: {status.current_step}")
    print(f"Training Loss: {status.current_loss:.4f}")
    
    if status.validation_loss:
        print(f"Validation Loss: {status.validation_loss:.4f}")
    
    if status.status in ["completed", "failed"]:
        break
    
    time.sleep(30)  # Check every 30 seconds
```

### Step 4: Training Metrics

Monitor key metrics during training:

```python
# Get detailed training metrics
metrics = service.get_training_metrics(job.job_id)

print("Training Progress:")
for epoch_metrics in metrics.epochs:
    print(f"Epoch {epoch_metrics.epoch}:")
    print(f"  Training Loss: {epoch_metrics.train_loss:.4f}")
    print(f"  Validation Loss: {epoch_metrics.val_loss:.4f}")
    print(f"  Learning Rate: {epoch_metrics.learning_rate:.2e}")
    print(f"  Duration: {epoch_metrics.duration:.2f}s")
```

### Understanding Training Metrics

#### **Loss Curves**
- **Training Loss:** Should decrease steadily
- **Validation Loss:** Should decrease but may plateau
- **Gap between losses:** Large gap indicates overfitting

#### **Good Training Signs**
- ✅ Steady decrease in training loss
- ✅ Validation loss follows training loss
- ✅ No sudden spikes or instability
- ✅ Model converges before max epochs

#### **Warning Signs**
- ⚠️ Training loss not decreasing
- ⚠️ Validation loss increasing (overfitting)
- ⚠️ Loss oscillating wildly
- ⚠️ Very slow convergence

## Model Evaluation

### Automatic Evaluation

The platform provides automatic evaluation metrics:

```python
from evaluator.metrics_calculator import MetricsCalculator

calculator = MetricsCalculator()
results = calculator.evaluate_model(
    model_path="./models/customer-support-v1",
    test_dataset="test_dataset.jsonl"
)

print("Evaluation Results:")
print(f"BLEU Score: {results.bleu:.3f}")
print(f"ROUGE-L: {results.rouge_l:.3f}")
print(f"Perplexity: {results.perplexity:.2f}")
```

### Manual Testing

Test your model with sample prompts:

```python
from api.services.text_generator import TextGenerator

generator = TextGenerator()

# Test prompts
test_prompts = [
    "How do I reset my password?",
    "What are your business hours?",
    "I need help with billing"
]

for prompt in test_prompts:
    response = generator.generate(
        prompt=prompt,
        model_id="customer-support-v1",
        max_tokens=100,
        temperature=0.7
    )
    
    print(f"Prompt: {prompt}")
    print(f"Response: {response['text']}")
    print("-" * 50)
```

### Comparison with Base Model

Compare your fine-tuned model with the base model:

```python
from evaluator.prompt_evaluator import PromptEvaluator

evaluator = PromptEvaluator()

comparison = evaluator.compare_models(
    prompts=test_prompts,
    model_a="gpt2",  # Base model
    model_b="customer-support-v1",  # Fine-tuned model
    criteria=["relevance", "helpfulness", "accuracy"]
)

print("Model Comparison:")
for criterion, scores in comparison.items():
    print(f"{criterion}:")
    print(f"  Base Model: {scores['model_a']:.3f}")
    print(f"  Fine-tuned: {scores['model_b']:.3f}")
    print(f"  Improvement: {scores['improvement']:.3f}")
```

## Deployment and Usage

### Save and Register Model

```python
# Save the trained model
model_info = service.save_model(
    job_id=job.job_id,
    model_name="customer-support-v1",
    description="Customer support chatbot trained on FAQ data",
    tags=["customer-support", "faq", "chatbot"]
)

print(f"Model saved: {model_info.model_id}")
```

### Use via API

Once saved, use your model through the API:

```bash
curl -X POST http://localhost:5000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "How do I reset my password?",
    "model_id": "customer-support-v1",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Use in Python

```python
import requests

def ask_support_bot(question):
    response = requests.post(
        "http://localhost:5000/api/v1/generate",
        json={
            "prompt": question,
            "model_id": "customer-support-v1",
            "max_tokens": 150,
            "temperature": 0.7
        }
    )
    
    if response.status_code == 200:
        return response.json()["text"]
    else:
        return "Sorry, I couldn't process your request."

# Test the bot
questions = [
    "How do I reset my password?",
    "What are your business hours?",
    "How do I cancel my subscription?"
]

for question in questions:
    answer = ask_support_bot(question)
    print(f"Q: {question}")
    print(f"A: {answer}")
    print()
```

## Advanced Techniques

### 1. Multi-Task Fine-Tuning

Train on multiple related tasks:

```json
{"prompt": "FAQ: How do I reset my password?", "response": "To reset your password: 1. Go to login page..."}
{"prompt": "SUPPORT: User can't access account", "response": "I'll help you regain access. First, let's try..."}
{"prompt": "BILLING: Question about charges", "response": "I can help explain your billing. Let me review..."}
```

### 2. Few-Shot Learning Enhancement

Include examples in your prompts:

```json
{"prompt": "Examples:\nQ: How do I login?\nA: Go to the login page and enter credentials.\n\nQ: How do I reset my password?", "response": "To reset your password:\n1. Go to the login page\n2. Click 'Forgot Password'..."}
```

### 3. Instruction Following

Train the model to follow specific instructions:

```json
{"prompt": "Instruction: Provide a helpful, concise answer to customer questions.\n\nCustomer: How do I cancel my subscription?", "response": "To cancel your subscription:\n1. Log into your account\n2. Go to Account Settings..."}
```

### 4. Domain Adaptation

Gradually adapt to your domain:

1. **Stage 1:** General domain data (broad coverage)
2. **Stage 2:** Domain-specific data (focused training)
3. **Stage 3:** Task-specific data (fine-tuning)

### 5. Hyperparameter Optimization

Use automated hyperparameter search:

```python
from fine_tuning.hyperparameter_optimizer import HyperparameterOptimizer

optimizer = HyperparameterOptimizer()

best_config = optimizer.optimize(
    dataset_path="customer_support_dataset.jsonl",
    search_space={
        "learning_rate": [1e-5, 3e-5, 5e-5, 1e-4],
        "batch_size": [2, 4, 8],
        "lora_rank": [8, 16, 32],
        "epochs": [3, 5, 7]
    },
    metric="validation_loss",
    trials=20
)

print(f"Best configuration: {best_config}")
```

## Troubleshooting

### Common Issues and Solutions

#### 1. **Out of Memory Errors**

**Symptoms:**
- CUDA out of memory
- Process killed during training

**Solutions:**
```python
# Reduce batch size
config.batch_size = 2

# Enable gradient accumulation
config.gradient_accumulation_steps = 4

# Use LoRA
config.use_lora = True
config.lora_rank = 8

# Use smaller model
config.base_model = "distilgpt2"
```

#### 2. **Poor Model Performance**

**Symptoms:**
- Model generates irrelevant responses
- High validation loss
- No improvement over base model

**Solutions:**
```python
# Check dataset quality
validator = DatasetValidator()
result = validator.validate_file("dataset.jsonl")
print(result.quality_report)

# Increase training data
# Add more diverse examples
# Improve prompt formatting

# Adjust hyperparameters
config.learning_rate = 3e-5  # Lower learning rate
config.epochs = 5  # More epochs
config.warmup_steps = 200  # More warmup
```

#### 3. **Overfitting**

**Symptoms:**
- Training loss decreases but validation loss increases
- Large gap between training and validation metrics

**Solutions:**
```python
# Add regularization
config.weight_decay = 0.01
config.dropout = 0.1

# Early stopping
config.early_stopping_patience = 3

# More validation data
# Reduce model complexity
config.lora_rank = 8  # Smaller LoRA rank
```

#### 4. **Slow Training**

**Symptoms:**
- Training takes too long
- Low GPU utilization

**Solutions:**
```python
# Increase batch size (if memory allows)
config.batch_size = 8

# Use gradient accumulation
config.gradient_accumulation_steps = 2

# Optimize data loading
config.dataloader_num_workers = 4

# Use mixed precision
config.fp16 = True
```

#### 5. **Model Not Loading**

**Symptoms:**
- Errors when loading saved model
- Missing model files

**Solutions:**
```bash
# Check model directory
ls -la ./models/customer-support-v1/

# Verify required files
# - config.json
# - pytorch_model.bin (or model.safetensors)
# - tokenizer files

# Re-save model if needed
service.save_model(job_id, force_overwrite=True)
```

### Debugging Training

#### Enable Detailed Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# Monitor training progress
service.start_training(
    dataset_path="dataset.jsonl",
    config=config,
    debug=True,
    verbose=True
)
```

#### Check Training Data

```python
# Inspect tokenized data
from fine_tuning.dataset_tokenizer import DatasetTokenizer

tokenizer = DatasetTokenizer()
tokenized = tokenizer.tokenize_dataset("dataset.jsonl", "gpt2")

print(f"Max sequence length: {max(len(seq) for seq in tokenized)}")
print(f"Average sequence length: {sum(len(seq) for seq in tokenized) / len(tokenized)}")

# Check for truncation
truncated = sum(1 for seq in tokenized if len(seq) >= 512)
print(f"Truncated sequences: {truncated}/{len(tokenized)}")
```

#### Monitor GPU Usage

```bash
# Monitor GPU memory and utilization
nvidia-smi -l 1

# Check GPU processes
nvidia-smi pmon
```

### Performance Optimization

#### Dataset Optimization

```python
# Optimize dataset size
def optimize_dataset(input_file, output_file, target_size=1000):
    """Keep most diverse examples up to target size."""
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # Load and vectorize prompts
    prompts = []
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data['prompt'])
    
    # Calculate diversity scores
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(prompts)
    
    # Select diverse examples
    selected_indices = []
    remaining_indices = list(range(len(prompts)))
    
    # Start with random example
    selected_indices.append(remaining_indices.pop(0))
    
    while len(selected_indices) < target_size and remaining_indices:
        # Find most diverse remaining example
        selected_vectors = vectors[selected_indices]
        max_min_distance = -1
        best_idx = None
        
        for idx in remaining_indices:
            distances = cosine_similarity(vectors[idx:idx+1], selected_vectors)
            min_distance = distances.min()
            
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                best_idx = idx
        
        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
    
    # Save optimized dataset
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for i, line in enumerate(f_in):
            if i in selected_indices:
                f_out.write(line)
    
    print(f"Optimized dataset: {len(selected_indices)} examples")

# Use the optimizer
optimize_dataset("large_dataset.jsonl", "optimized_dataset.jsonl", 1000)
```

## Best Practices Summary

### 📊 **Data Quality**
- Use high-quality, diverse examples
- Maintain consistent formatting
- Validate datasets before training
- Include 100-1000+ examples minimum

### ⚙️ **Configuration**
- Start with recommended hyperparameters
- Use LoRA for memory efficiency
- Enable early stopping
- Monitor validation metrics

### 🔍 **Monitoring**
- Track training and validation loss
- Watch for overfitting signs
- Monitor resource usage
- Save regular checkpoints

### 🚀 **Optimization**
- Use appropriate model sizes
- Optimize batch size for your hardware
- Enable mixed precision training
- Consider gradient accumulation

### 🧪 **Evaluation**
- Test on held-out data
- Compare with base models
- Use multiple evaluation metrics
- Collect human feedback

### 📈 **Iteration**
- Start simple, then optimize
- Experiment with different approaches
- Keep detailed experiment logs
- Version your datasets and models

---

**Ready to fine-tune your first model?** 🎯

Start with a small dataset and basic configuration, then iterate and improve based on your results. Remember that fine-tuning is an iterative process - don't expect perfect results on the first try!

**Next:** Check out the [Prompt Optimization Guide](prompt_optimization_guide.md) to learn how to get the best performance from your fine-tuned models.