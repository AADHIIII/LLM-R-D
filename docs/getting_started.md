# Getting Started Guide

Welcome to the LLM Optimization Platform! This guide will help you get up and running quickly with fine-tuning models, generating text, and optimizing prompts.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [First Steps](#first-steps)
5. [Basic Usage](#basic-usage)
6. [Next Steps](#next-steps)

## Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed
- **Docker** (optional, for containerized deployment)
- **API Keys** for commercial models (OpenAI, Anthropic)
- **Basic knowledge** of REST APIs and JSON

### System Requirements

- **Memory:** 8GB RAM minimum (16GB recommended for fine-tuning)
- **Storage:** 10GB free space minimum
- **GPU:** Optional but recommended for fine-tuning (CUDA-compatible)

## Installation

### Option 1: Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/llm-optimization-platform.git
   cd llm-optimization-platform
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize the database:**
   ```bash
   python -c "from database.connection import init_database; init_database()"
   ```

### Option 2: Docker Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/llm-optimization-platform.git
   cd llm-optimization-platform
   ```

2. **Build and run with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# API Keys (required for commercial models)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Database Configuration
DATABASE_URL=sqlite:///./llm_optimization.db

# API Configuration
API_HOST=0.0.0.0
API_PORT=5000
DEBUG=false

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Security
SECRET_KEY=your_secret_key_here
API_KEY_REQUIRED=true
```

### API Keys Setup

#### OpenAI API Key
1. Visit [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Create a new API key
3. Add it to your `.env` file

#### Anthropic API Key
1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Generate an API key
3. Add it to your `.env` file

### Configuration Validation

Test your configuration:

```bash
python validate_setup.py
```

This will check:
- ✅ Environment variables
- ✅ Database connectivity
- ✅ API key validity
- ✅ Required directories

## First Steps

### 1. Start the API Server

**Local installation:**
```bash
python run_api.py
```

**Docker installation:**
```bash
docker-compose up api
```

The API will be available at `http://localhost:5000`

### 2. Verify Installation

Check the health endpoint:

```bash
curl http://localhost:5000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "1.0.0"
}
```

### 3. List Available Models

```bash
curl http://localhost:5000/api/v1/models
```

You should see commercial models (if API keys are configured) and any fine-tuned models.

## Basic Usage

### Generate Text with a Commercial Model

```bash
curl -X POST http://localhost:5000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain artificial intelligence in simple terms",
    "model_id": "gpt-3.5-turbo",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Upload and Validate a Dataset

1. **Prepare your dataset** in JSONL format:
   ```json
   {"prompt": "What is machine learning?", "response": "Machine learning is a subset of AI..."}
   {"prompt": "Explain neural networks", "response": "Neural networks are computing systems..."}
   ```

2. **Validate the dataset:**
   ```bash
   python -c "
   from fine_tuning.dataset_validator import DatasetValidator
   validator = DatasetValidator()
   result = validator.validate_file('path/to/your/dataset.jsonl')
   print('Valid:', result.is_valid)
   print('Issues:', result.issues)
   "
   ```

### Start a Fine-Tuning Job

```python
from fine_tuning.fine_tuning_service import FineTuningService
from fine_tuning.training_config import TrainingConfig

# Create training configuration
config = TrainingConfig(
    base_model="gpt2",
    epochs=3,
    batch_size=4,
    learning_rate=5e-5
)

# Initialize service and start training
service = FineTuningService()
job = service.start_training(
    dataset_path="path/to/your/dataset.jsonl",
    config=config,
    output_dir="./models/my-fine-tuned-model"
)

print(f"Training job started: {job.job_id}")
```

### Monitor Training Progress

```python
# Check training status
status = service.get_training_status(job.job_id)
print(f"Status: {status.status}")
print(f"Progress: {status.progress}%")
print(f"Current loss: {status.current_loss}")
```

### Use Your Fine-Tuned Model

Once training is complete:

```bash
curl -X POST http://localhost:5000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Your custom prompt here",
    "model_id": "my-fine-tuned-model",
    "max_tokens": 100
  }'
```

## Web Interface

### Access the Dashboard

1. **Start the web interface:**
   ```bash
   # If using Docker
   docker-compose up frontend
   
   # If running locally (in a separate terminal)
   cd web_interface/frontend
   npm install
   npm start
   ```

2. **Open your browser:**
   Navigate to `http://localhost:3000`

### Key Features

- **📊 Dashboard:** Overview of experiments and models
- **🤖 Model Management:** Upload datasets and start training
- **✨ Prompt Testing:** Compare outputs across models
- **📈 Analytics:** Cost tracking and performance metrics
- **👥 Feedback:** Rate and improve model outputs

## Common Tasks

### Compare Multiple Models

```bash
curl -X POST http://localhost:5000/api/v1/generate/batch \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": [
      "Explain quantum computing",
      "What is blockchain technology?"
    ],
    "model_id": "gpt-4",
    "max_tokens": 50
  }'
```

### Track API Costs

```bash
curl -X POST http://localhost:5000/api/v1/cost/track \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "gpt-4",
    "input_tokens": 100,
    "output_tokens": 50,
    "latency_ms": 1500
  }'
```

### Get Usage Analytics

```bash
curl http://localhost:5000/api/v1/cost/analytics?days=7
```

## Troubleshooting

### Common Issues

#### 1. "Model not found" Error
**Problem:** Trying to use a model that isn't available.

**Solution:**
```bash
# List available models
curl http://localhost:5000/api/v1/models

# Check model status
curl http://localhost:5000/api/v1/models/gpt-4
```

#### 2. API Key Authentication Errors
**Problem:** Invalid or missing API keys.

**Solution:**
1. Verify your `.env` file has the correct API keys
2. Test API key validity:
   ```bash
   curl http://localhost:5000/api/v1/commercial/test
   ```

#### 3. Out of Memory During Fine-Tuning
**Problem:** Insufficient memory for training.

**Solutions:**
- Reduce batch size in training config
- Use gradient accumulation
- Enable LoRA (Low-Rank Adaptation):
  ```python
  config = TrainingConfig(
      base_model="gpt2",
      batch_size=2,  # Reduced batch size
      use_lora=True,  # Enable LoRA
      lora_rank=16
  )
  ```

#### 4. Slow API Responses
**Problem:** High latency in text generation.

**Solutions:**
- Use smaller models for development
- Implement request caching
- Check system resources:
  ```bash
  curl http://localhost:5000/api/v1/status
  ```

### Getting Help

1. **Check the logs:**
   ```bash
   tail -f logs/app.log
   ```

2. **Run diagnostics:**
   ```bash
   python validate_setup.py --verbose
   ```

3. **Monitor system health:**
   ```bash
   curl http://localhost:5000/api/v1/monitoring/health
   ```

## Next Steps

Now that you have the platform running, explore these advanced features:

### 🎯 [Fine-Tuning Tutorial](fine_tuning_tutorial.md)
Learn how to fine-tune models on your specific domain data.

### 🔍 [Prompt Optimization Guide](prompt_optimization_guide.md)
Discover techniques for optimizing prompts and comparing model performance.

### 💰 [Cost Management](cost_management.md)
Set up budgets, track usage, and optimize costs across different models.

### 🔧 [Advanced Configuration](advanced_configuration.md)
Configure monitoring, logging, and production deployment.

### 📚 [API Reference](api_documentation.html)
Complete API documentation with interactive examples.

## Best Practices

### Security
- 🔐 Never commit API keys to version control
- 🛡️ Use environment variables for sensitive configuration
- 🔒 Enable API key authentication in production
- 📝 Regularly rotate API keys

### Performance
- ⚡ Use appropriate model sizes for your use case
- 📊 Monitor resource usage and costs
- 🎯 Implement caching for frequently used prompts
- 🔄 Use batch processing for multiple requests

### Data Management
- 📁 Organize datasets with clear naming conventions
- ✅ Always validate datasets before training
- 💾 Backup trained models and configurations
- 📈 Track experiment metadata and results

### Development Workflow
- 🧪 Start with small datasets for testing
- 📋 Use version control for configurations
- 🔍 Monitor training progress and metrics
- 🎯 Set up automated testing for critical paths

## Support

- 📖 **Documentation:** [Full documentation](README.md)
- 🐛 **Issues:** [GitHub Issues](https://github.com/your-org/llm-optimization-platform/issues)
- 💬 **Discussions:** [GitHub Discussions](https://github.com/your-org/llm-optimization-platform/discussions)
- 📧 **Email:** support@llm-optimization.com

---

**Ready to start optimizing?** 🚀

Continue with the [Fine-Tuning Tutorial](fine_tuning_tutorial.md) to create your first custom model!