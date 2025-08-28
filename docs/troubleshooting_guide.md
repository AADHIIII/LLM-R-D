# Troubleshooting Guide

This comprehensive guide helps you diagnose and resolve common issues with the LLM Optimization Platform.

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Installation Issues](#installation-issues)
3. [Configuration Problems](#configuration-problems)
4. [API and Connection Issues](#api-and-connection-issues)
5. [Fine-Tuning Problems](#fine-tuning-problems)
6. [Performance Issues](#performance-issues)
7. [Model Loading and Usage](#model-loading-and-usage)
8. [Database and Storage Issues](#database-and-storage-issues)
9. [Monitoring and Logging](#monitoring-and-logging)
10. [Common Error Messages](#common-error-messages)

## Quick Diagnostics

### System Health Check

Run the built-in diagnostic tool:

```bash
python validate_setup.py --verbose
```

This checks:
- ✅ Python version and dependencies
- ✅ Environment variables
- ✅ Database connectivity
- ✅ API key validity
- ✅ File permissions
- ✅ GPU availability (if applicable)

### API Health Check

```bash
# Basic health check
curl http://localhost:5000/api/v1/health

# Detailed system status
curl http://localhost:5000/api/v1/status

# Commercial API connectivity
curl http://localhost:5000/api/v1/commercial/test
```

### Log Analysis

Check recent logs for errors:

```bash
# View recent API logs
tail -f logs/api.log

# View application logs
tail -f logs/app.log

# Search for errors
grep -i error logs/*.log | tail -20
```

## Installation Issues

### Problem: Python Version Compatibility

**Symptoms:**
- Import errors during installation
- Syntax errors when running scripts
- Package compatibility warnings

**Solution:**
```bash
# Check Python version
python --version

# Should be 3.8 or higher
# If not, install correct version:
# macOS with Homebrew:
brew install python@3.9

# Ubuntu/Debian:
sudo apt update
sudo apt install python3.9 python3.9-pip

# Create virtual environment with correct Python
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Problem: Dependency Installation Failures

**Symptoms:**
- `pip install` fails with compilation errors
- Missing system dependencies
- Version conflicts

**Solution:**
```bash
# Update pip and setuptools
pip install --upgrade pip setuptools wheel

# Install system dependencies (Ubuntu/Debian)
sudo apt install build-essential python3-dev

# Install system dependencies (macOS)
xcode-select --install

# For CUDA-related issues
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Clean install if needed
pip uninstall -r requirements.txt -y
pip install -r requirements.txt --no-cache-dir
```

### Problem: Docker Installation Issues

**Symptoms:**
- Docker containers fail to start
- Port conflicts
- Volume mounting issues

**Solution:**
```bash
# Check Docker status
docker --version
docker-compose --version

# Fix port conflicts
docker-compose down
# Edit docker-compose.yml to change ports
# Then restart:
docker-compose up -d

# Fix volume permissions
sudo chown -R $USER:$USER ./data ./logs ./models

# Clean rebuild
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

## Configuration Problems

### Problem: Environment Variables Not Loading

**Symptoms:**
- "API key not found" errors
- Configuration defaults being used
- Database connection failures

**Solution:**
```bash
# Check if .env file exists
ls -la .env

# Verify environment variables are loaded
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('OPENAI_API_KEY:', 'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET')
print('DATABASE_URL:', os.getenv('DATABASE_URL', 'NOT SET'))
"

# Fix .env file format (no spaces around =)
# ❌ OPENAI_API_KEY = your_key_here
# ✅ OPENAI_API_KEY=your_key_here

# Restart application after fixing .env
```

### Problem: Database Configuration Issues

**Symptoms:**
- "Database connection failed" errors
- Tables not found
- Permission denied errors

**Solution:**
```bash
# Check database file permissions
ls -la *.db

# Initialize database if needed
python -c "
from database.connection import init_database
init_database()
print('Database initialized successfully')
"

# For PostgreSQL/MySQL issues:
# Check connection string format
# postgresql://user:password@localhost:5432/dbname
# mysql://user:password@localhost:3306/dbname

# Test database connection
python -c "
from database.connection import db_manager
with db_manager.get_session() as session:
    print('Database connection successful')
"
```

### Problem: API Key Validation Failures

**Symptoms:**
- "Invalid API key" errors
- Commercial models unavailable
- Authentication failures

**Solution:**
```bash
# Test OpenAI API key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# Test Anthropic API key
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3-sonnet-20240229", "max_tokens": 10, "messages": [{"role": "user", "content": "Hi"}]}'

# Verify API keys in platform
python -c "
from api.services.commercial_api_service import CommercialAPIService
service = CommercialAPIService()
results = service.test_connections()
print(results)
"
```

## API and Connection Issues

### Problem: API Server Won't Start

**Symptoms:**
- "Port already in use" errors
- Import errors on startup
- Configuration validation failures

**Solution:**
```bash
# Check if port is in use
lsof -i :5000
# Kill process if needed:
kill -9 <PID>

# Check for import errors
python -c "
try:
    from api.app import create_app
    app = create_app()
    print('App creation successful')
except Exception as e:
    print(f'Import error: {e}')
"

# Start with debug mode
python run_api.py --debug

# Check configuration
python -c "
from api.config import DevelopmentConfig
config = DevelopmentConfig()
print('Config loaded successfully')
"
```

### Problem: API Requests Timing Out

**Symptoms:**
- Requests hang indefinitely
- Timeout errors
- Slow response times

**Solution:**
```bash
# Check system resources
curl http://localhost:5000/api/v1/status

# Monitor API performance
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:5000/api/v1/health

# Create curl-format.txt:
echo "
     time_namelookup:  %{time_namelookup}s
        time_connect:  %{time_connect}s
     time_appconnect:  %{time_appconnect}s
    time_pretransfer:  %{time_pretransfer}s
       time_redirect:  %{time_redirect}s
  time_starttransfer:  %{time_starttransfer}s
                     ----------
          time_total:  %{time_total}s
" > curl-format.txt

# Increase timeout in requests
python -c "
import requests
response = requests.get(
    'http://localhost:5000/api/v1/health',
    timeout=30  # Increase timeout
)
print(response.json())
"
```

### Problem: CORS Issues

**Symptoms:**
- Browser console shows CORS errors
- Frontend can't connect to API
- "Access-Control-Allow-Origin" errors

**Solution:**
```python
# Check CORS configuration in api/app.py
from flask_cors import CORS

app = Flask(__name__)
CORS(app, 
     origins=['http://localhost:3000', 'http://localhost:8080'],
     methods=['GET', 'POST', 'PUT', 'DELETE'],
     allow_headers=['Content-Type', 'Authorization'])

# For development, allow all origins (not for production):
CORS(app, origins="*")
```

## Fine-Tuning Problems

### Problem: Out of Memory During Training

**Symptoms:**
- "CUDA out of memory" errors
- Process killed during training
- System becomes unresponsive

**Solution:**
```python
# Reduce batch size
config = TrainingConfig(
    batch_size=2,  # Reduce from 4 or 8
    gradient_accumulation_steps=4,  # Maintain effective batch size
    use_lora=True,  # Enable LoRA
    lora_rank=8,   # Reduce LoRA rank
)

# Use smaller model
config.base_model = "distilgpt2"  # Instead of "gpt2-medium"

# Enable gradient checkpointing
config.gradient_checkpointing = True

# Use mixed precision
config.fp16 = True

# Monitor GPU memory
import torch
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

### Problem: Training Not Converging

**Symptoms:**
- Loss not decreasing
- Validation metrics not improving
- Training appears stuck

**Solution:**
```python
# Check learning rate
config = TrainingConfig(
    learning_rate=5e-5,  # Try different values: 1e-5, 3e-5, 1e-4
    warmup_steps=100,    # Increase warmup
    lr_scheduler_type="cosine",  # Try different schedulers
)

# Verify dataset quality
from fine_tuning.dataset_validator import DatasetValidator
validator = DatasetValidator()
result = validator.validate_file('dataset.jsonl')
print(f"Dataset quality score: {result.quality_score}")
print(f"Issues: {result.issues}")

# Check for data leakage
# Ensure training and validation sets don't overlap

# Monitor gradients
config.max_grad_norm = 1.0  # Gradient clipping
config.logging_steps = 10   # More frequent logging
```

### Problem: Model Overfitting

**Symptoms:**
- Training loss decreases but validation loss increases
- Large gap between training and validation metrics
- Poor performance on new data

**Solution:**
```python
# Add regularization
config = TrainingConfig(
    weight_decay=0.01,     # L2 regularization
    dropout=0.1,           # Dropout rate
    early_stopping_patience=3,  # Stop if no improvement
    early_stopping_threshold=0.01,
)

# Reduce model complexity
config.lora_rank = 8  # Smaller LoRA rank

# Increase validation split
from fine_tuning.dataset_tokenizer import DatasetTokenizer
tokenizer = DatasetTokenizer()
train_data, val_data = tokenizer.split_dataset(
    'dataset.jsonl',
    validation_split=0.3  # Increase from 0.2
)

# Add more training data if possible
# Implement data augmentation
```

### Problem: Training Crashes or Fails

**Symptoms:**
- Training stops unexpectedly
- Error messages during training
- Corrupted model checkpoints

**Solution:**
```python
# Enable checkpointing
config = TrainingConfig(
    save_steps=100,        # Save more frequently
    save_total_limit=5,    # Keep more checkpoints
    resume_from_checkpoint=True,
)

# Add error handling
try:
    job = service.start_training(
        dataset_path="dataset.jsonl",
        config=config
    )
except Exception as e:
    print(f"Training failed: {e}")
    # Check logs for detailed error
    
# Verify dataset before training
validator = DatasetValidator()
if not validator.validate_file('dataset.jsonl').is_valid:
    print("Dataset validation failed")
    exit(1)

# Check disk space
import shutil
free_space = shutil.disk_usage('.').free / (1024**3)
print(f"Free disk space: {free_space:.2f} GB")
if free_space < 5:  # Less than 5GB
    print("Warning: Low disk space")
```

## Performance Issues

### Problem: Slow Text Generation

**Symptoms:**
- High latency for API requests
- Timeouts during generation
- Poor user experience

**Solution:**
```python
# Optimize generation parameters
response = generator.generate(
    prompt=prompt,
    model_id=model_id,
    max_tokens=100,      # Reduce if possible
    temperature=0.7,
    do_sample=True,      # Faster than beam search
    pad_token_id=tokenizer.eos_token_id,
)

# Use model caching
from functools import lru_cache

@lru_cache(maxsize=5)
def load_model(model_id):
    return AutoModelForCausalLM.from_pretrained(model_id)

# Implement request batching
def batch_generate(prompts, model_id, batch_size=4):
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        batch_results = model.generate(batch)
        results.extend(batch_results)
    return results

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

### Problem: High Memory Usage

**Symptoms:**
- System running out of RAM
- Swap usage increasing
- Application becomes slow

**Solution:**
```python
# Implement model unloading
import gc
import torch

def unload_model():
    global model
    if 'model' in globals():
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Use model rotation for multiple models
class ModelManager:
    def __init__(self, max_models=2):
        self.models = {}
        self.max_models = max_models
        self.usage_order = []
    
    def get_model(self, model_id):
        if model_id not in self.models:
            if len(self.models) >= self.max_models:
                # Unload least recently used model
                lru_model = self.usage_order.pop(0)
                del self.models[lru_model]
                gc.collect()
            
            self.models[model_id] = load_model(model_id)
        
        # Update usage order
        if model_id in self.usage_order:
            self.usage_order.remove(model_id)
        self.usage_order.append(model_id)
        
        return self.models[model_id]

# Monitor memory usage
import psutil
process = psutil.Process()
memory_info = process.memory_info()
print(f"Memory usage: {memory_info.rss / 1024**2:.2f} MB")
```

### Problem: Database Performance Issues

**Symptoms:**
- Slow query responses
- Database locks
- High CPU usage from database

**Solution:**
```python
# Add database indexes
from database.connection import db_manager

with db_manager.get_session() as session:
    # Add indexes for frequently queried columns
    session.execute("""
        CREATE INDEX IF NOT EXISTS idx_experiments_created_at 
        ON experiments(created_at);
        
        CREATE INDEX IF NOT EXISTS idx_evaluations_model_id 
        ON evaluations(model_id);
        
        CREATE INDEX IF NOT EXISTS idx_evaluations_experiment_id 
        ON evaluations(experiment_id);
    """)
    session.commit()

# Implement connection pooling
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600
)

# Use pagination for large queries
def get_experiments_paginated(page=1, per_page=50):
    offset = (page - 1) * per_page
    return session.query(Experiment)\
                 .offset(offset)\
                 .limit(per_page)\
                 .all()

# Optimize queries
# Use select_related/joinedload for related objects
experiments = session.query(Experiment)\
                    .options(joinedload(Experiment.evaluations))\
                    .all()
```

## Model Loading and Usage

### Problem: Model Not Found

**Symptoms:**
- "Model not found" errors
- Empty model list
- 404 errors when accessing models

**Solution:**
```bash
# Check model directory
ls -la models/

# Verify model files
ls -la models/your-model-name/
# Should contain: config.json, pytorch_model.bin, tokenizer files

# Re-register model if needed
python -c "
from fine_tuning.model_manager import ModelManager
manager = ModelManager()
manager.register_model(
    model_path='./models/your-model-name',
    model_name='your-model-name',
    description='Your model description'
)
"

# Check model registry
curl http://localhost:5000/api/v1/models
```

### Problem: Model Loading Errors

**Symptoms:**
- Import errors when loading models
- Incompatible model formats
- Tokenizer mismatches

**Solution:**
```python
# Verify model compatibility
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    model = AutoModelForCausalLM.from_pretrained('./models/your-model')
    tokenizer = AutoTokenizer.from_pretrained('./models/your-model')
    print("Model loaded successfully")
except Exception as e:
    print(f"Model loading failed: {e}")

# Fix tokenizer issues
tokenizer = AutoTokenizer.from_pretrained('gpt2')  # Use base tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Handle model format conversion
# Convert from older format if needed
model.save_pretrained('./models/your-model', safe_serialization=True)

# Check model configuration
import json
with open('./models/your-model/config.json', 'r') as f:
    config = json.load(f)
    print(f"Model type: {config.get('model_type')}")
    print(f"Architecture: {config.get('architectures')}")
```

### Problem: Inconsistent Model Outputs

**Symptoms:**
- Different outputs for same input
- Unexpected response quality
- Model behavior changes

**Solution:**
```python
# Set random seeds for reproducibility
import torch
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Use consistent generation parameters
generation_config = {
    'max_tokens': 100,
    'temperature': 0.7,
    'top_p': 0.9,
    'do_sample': True,
    'pad_token_id': tokenizer.eos_token_id,
    'eos_token_id': tokenizer.eos_token_id,
}

# Verify model version/checkpoint
print(f"Model config: {model.config}")
print(f"Model name: {model.config.name_or_path}")

# Check for model drift in production
# Implement model versioning and monitoring
```

## Database and Storage Issues

### Problem: Database Connection Failures

**Symptoms:**
- "Connection refused" errors
- Database timeout errors
- Unable to create tables

**Solution:**
```python
# Test database connection
from database.connection import db_manager

try:
    with db_manager.get_session() as session:
        result = session.execute("SELECT 1").fetchone()
        print("Database connection successful")
except Exception as e:
    print(f"Database connection failed: {e}")

# Check database URL format
import os
db_url = os.getenv('DATABASE_URL')
print(f"Database URL: {db_url}")

# For SQLite issues:
# Check file permissions
import stat
db_file = 'llm_optimization.db'
if os.path.exists(db_file):
    file_stat = os.stat(db_file)
    print(f"Database file permissions: {stat.filemode(file_stat.st_mode)}")

# Initialize database if needed
from database.connection import init_database
init_database()
```

### Problem: Disk Space Issues

**Symptoms:**
- "No space left on device" errors
- Failed model saves
- Database write errors

**Solution:**
```bash
# Check disk usage
df -h

# Find large files
du -h --max-depth=1 | sort -hr

# Clean up old models
find ./models -name "*.bin" -mtime +30 -exec ls -lh {} \;
# Remove if safe:
# find ./models -name "*.bin" -mtime +30 -delete

# Clean up logs
find ./logs -name "*.log" -mtime +7 -exec gzip {} \;

# Clean up temporary files
rm -rf /tmp/transformers_cache/*
rm -rf ~/.cache/huggingface/transformers/*

# Set up log rotation
# Add to /etc/logrotate.d/llm-platform:
# /path/to/logs/*.log {
#     daily
#     rotate 7
#     compress
#     delaycompress
#     missingok
#     notifempty
# }
```

### Problem: File Permission Issues

**Symptoms:**
- "Permission denied" errors
- Cannot write to directories
- Model save failures

**Solution:**
```bash
# Fix directory permissions
chmod -R 755 ./models ./logs ./data

# Fix ownership
sudo chown -R $USER:$USER ./models ./logs ./data

# Check current permissions
ls -la models/ logs/ data/

# For Docker issues:
# Add user to docker group
sudo usermod -aG docker $USER
# Logout and login again

# Fix Docker volume permissions
docker-compose exec api chown -R app:app /app/models /app/logs
```

## Monitoring and Logging

### Problem: Missing or Incomplete Logs

**Symptoms:**
- No log files generated
- Empty log files
- Missing error information

**Solution:**
```python
# Configure logging properly
import logging
import os

# Ensure log directory exists
os.makedirs('logs', exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()  # Also log to console
    ]
)

# Test logging
logger = logging.getLogger(__name__)
logger.info("Logging test message")

# Check log file
tail -f logs/app.log
```

### Problem: Monitoring System Not Working

**Symptoms:**
- No metrics being collected
- Monitoring dashboard empty
- Alerts not firing

**Solution:**
```python
# Check monitoring service status
from monitoring.metrics_collector import get_metrics_collector
from monitoring.alerting import get_alert_manager

collector = get_metrics_collector()
alert_manager = get_alert_manager()

print(f"Metrics collector running: {collector.running}")
print(f"Alert manager running: {alert_manager.running}")

# Start monitoring if not running
if not collector.running:
    collector.start()

if not alert_manager.running:
    alert_manager.start()

# Check monitoring database
import sqlite3
conn = sqlite3.connect('monitoring/metrics.db')
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM system_metrics")
count = cursor.fetchone()[0]
print(f"System metrics records: {count}")
conn.close()

# Test metrics collection
curl http://localhost:5000/api/v1/monitoring/health
```

## Common Error Messages

### "CUDA out of memory"

**Cause:** GPU memory exhausted during training or inference.

**Solutions:**
1. Reduce batch size: `config.batch_size = 2`
2. Enable gradient accumulation: `config.gradient_accumulation_steps = 4`
3. Use LoRA: `config.use_lora = True`
4. Use smaller model: `config.base_model = "distilgpt2"`
5. Clear GPU cache: `torch.cuda.empty_cache()`

### "Model not found"

**Cause:** Model path incorrect or model not registered.

**Solutions:**
1. Check model directory: `ls -la models/`
2. Verify model files exist
3. Re-register model in system
4. Check model ID in API call

### "API key not found"

**Cause:** Missing or incorrect API key configuration.

**Solutions:**
1. Check `.env` file exists and has correct format
2. Verify API key is valid
3. Restart application after updating `.env`
4. Test API key with direct API call

### "Connection refused"

**Cause:** Service not running or wrong port/host.

**Solutions:**
1. Check if API server is running: `ps aux | grep python`
2. Verify port is correct: `netstat -tlnp | grep 5000`
3. Check firewall settings
4. Verify host configuration

### "Dataset validation failed"

**Cause:** Invalid dataset format or content.

**Solutions:**
1. Check JSONL format is correct
2. Verify required fields exist
3. Check for empty or malformed entries
4. Validate character encoding (UTF-8)

### "Training job failed"

**Cause:** Various training-related issues.

**Solutions:**
1. Check training logs for specific error
2. Verify dataset quality
3. Check system resources (memory, disk space)
4. Validate training configuration
5. Test with smaller dataset first

## Getting Additional Help

### Enable Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger().setLevel(logging.DEBUG)

# Run with debug flag
python run_api.py --debug

# Enable verbose output
python validate_setup.py --verbose
```

### Collect System Information

```bash
# Create diagnostic report
python -c "
import sys
import platform
import torch
import transformers
import flask

print('=== System Information ===')
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'Flask: {flask.__version__}')

if torch.cuda.is_available():
    print(f'CUDA: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('CUDA: Not available')
"
```

### Contact Support

When contacting support, include:

1. **Error message** (full stack trace)
2. **System information** (from diagnostic script above)
3. **Configuration** (sanitized, no API keys)
4. **Steps to reproduce** the issue
5. **Log files** (recent entries)

**Support Channels:**
- 📧 Email: support@llm-optimization.com
- 🐛 GitHub Issues: [Create Issue](https://github.com/your-org/llm-optimization-platform/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-org/llm-optimization-platform/discussions)

---

**Still having issues?** 🤔

Don't hesitate to reach out for help. Include as much detail as possible about your setup and the specific error you're encountering.