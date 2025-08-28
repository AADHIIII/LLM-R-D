# 🧪 Comprehensive Testing Guide

This guide provides detailed instructions for testing all components of the LLM R&D Platform.

## 📋 Table of Contents

1. [Quick Start Testing](#quick-start-testing)
2. [Frontend Testing](#frontend-testing)
3. [Backend API Testing](#backend-api-testing)
4. [Integration Testing](#integration-testing)
5. [Performance Testing](#performance-testing)
6. [Security Testing](#security-testing)
7. [Automated Testing](#automated-testing)
8. [Manual Testing Scenarios](#manual-testing-scenarios)

## 🚀 Quick Start Testing

### Prerequisites
```bash
# Ensure platform is running
docker-compose ps

# Should show all services as "Up"
# - llm-platform-backend
# - llm-platform-db
# - llm-platform-redis
# - llm-platform-frontend (if using Docker for frontend)
```

### Basic Health Check
```bash
# Test API health
curl http://localhost:9000/api/v1/health

# Expected response:
# {
#   "status": "healthy",
#   "timestamp": "2025-08-28T19:30:05.151480",
#   "version": "1.0.0"
# }
```

## 🎨 Frontend Testing

### 1. Web Interface Access
- **URL**: http://localhost:3000
- **Expected**: React application loads with navigation sidebar

### 2. Navigation Testing
Test all main sections:
- ✅ **Dashboard**: Overview and system status
- ✅ **Prompt Testing**: Multi-model prompt comparison
- ✅ **Fine-tuning**: Model training interface
- ✅ **Analytics**: Cost and performance metrics
- ✅ **Results**: Experiment results and history

### 3. Prompt Testing Interface

#### Test Case 1: Single Prompt, Multiple Models
1. Navigate to "Prompt Testing"
2. Add prompt: "Explain quantum computing in simple terms"
3. Select models: GPT-4, GPT-3.5 Turbo, Claude 3 Sonnet
4. Click "Run Evaluation"
5. **Expected**: 
   - Progress indicator shows
   - Results appear incrementally
   - Metrics displayed (BLEU, ROUGE, etc.)
   - Cost and latency information

#### Test Case 2: Multiple Prompts Comparison
1. Add 3 different prompts:
   - "Write a Python function to sort a list"
   - "Explain machine learning to a 5-year-old"
   - "What are the benefits of renewable energy?"
2. Select 2-3 models
3. Run evaluation
4. **Expected**: Side-by-side comparison view

#### Test Case 3: Human Feedback
1. After getting results, rate responses using thumbs up/down
2. Add qualitative feedback
3. **Expected**: Feedback saved and reflected in metrics

### 4. Fine-tuning Interface

#### Test Case 1: Create New Experiment
1. Navigate to "Fine-tuning"
2. Click "Create New Experiment"
3. Fill form:
   - Name: "Test Customer Support Model"
   - Description: "Testing fine-tuning workflow"
   - Dataset: Select from dropdown
   - Configure training parameters
4. Submit
5. **Expected**: Experiment appears in list with "pending" status

#### Test Case 2: Monitor Training Progress
1. Start a training job (mock)
2. **Expected**: Progress bar updates, status changes to "running"
3. View training logs and metrics

### 5. Analytics Dashboard

#### Test Case 1: Cost Tracking
1. Navigate to "Analytics"
2. View cost breakdown by model
3. Check daily/monthly spending charts
4. **Expected**: Interactive charts with cost data

#### Test Case 2: Performance Metrics
1. View response time charts
2. Check model accuracy comparisons
3. Export performance report
4. **Expected**: Downloadable CSV/PDF reports

## 🔧 Backend API Testing

### 1. Authentication Testing

#### Register New User
```bash
curl -X POST http://localhost:9000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "TestPassword123!",
    "role": "developer"
  }'
```

#### Login and Get Token
```bash
curl -X POST http://localhost:9000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "TestPassword123!"
  }'
```

### 2. Model Management

#### List Available Models
```bash
curl http://localhost:9000/api/v1/models
```

#### Get Model Details
```bash
curl http://localhost:9000/api/v1/models/gpt-4
```

### 3. Text Generation

#### Generate Text (requires auth token)
```bash
curl -X POST http://localhost:9000/api/v1/generate \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a haiku about programming",
    "model": "gpt-3.5-turbo",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### 4. Evaluation Endpoints

#### Run Prompt Evaluation
```bash
curl -X POST http://localhost:9000/api/v1/evaluate \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["What is AI?", "Explain machine learning"],
    "models": ["gpt-4", "gpt-3.5-turbo"],
    "evaluation_criteria": ["accuracy", "coherence", "relevance"]
  }'
```

### 5. Cost Tracking

#### Get Cost Summary
```bash
curl -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  http://localhost:9000/api/v1/costs/summary?period=daily
```

#### Get Detailed Usage
```bash
curl -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  http://localhost:9000/api/v1/costs/usage?start_date=2025-08-01&end_date=2025-08-31
```

## 🔗 Integration Testing

### 1. End-to-End Workflow Test

#### Complete Prompt Testing Flow
```bash
# 1. Register user
USER_RESPONSE=$(curl -s -X POST http://localhost:9000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"e2etest","email":"e2e@test.com","password":"TestPass123!"}')

# 2. Login and get token
TOKEN_RESPONSE=$(curl -s -X POST http://localhost:9000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"e2etest","password":"TestPass123!"}')

TOKEN=$(echo $TOKEN_RESPONSE | jq -r '.access_token')

# 3. List models
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:9000/api/v1/models

# 4. Generate text
curl -X POST http://localhost:9000/api/v1/generate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello world","model":"gpt-3.5-turbo"}'

# 5. Run evaluation
curl -X POST http://localhost:9000/api/v1/evaluate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompts":["Test prompt"],"models":["gpt-3.5-turbo"]}'
```

### 2. Database Integration Test

#### Test Data Persistence
```bash
# Create experiment
EXPERIMENT_RESPONSE=$(curl -s -X POST http://localhost:9000/api/v1/experiments \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Integration Test",
    "description": "Testing data persistence",
    "dataset_id": "test-dataset"
  }')

EXPERIMENT_ID=$(echo $EXPERIMENT_RESPONSE | jq -r '.id')

# Retrieve experiment
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:9000/api/v1/experiments/$EXPERIMENT_ID
```

## ⚡ Performance Testing

### 1. Load Testing

#### API Load Test
```bash
# Install Apache Bench (if not installed)
# macOS: brew install httpie
# Ubuntu: sudo apt-get install apache2-utils

# Test concurrent requests
ab -n 100 -c 10 http://localhost:9000/api/v1/health

# Expected: 
# - Requests per second: > 50
# - Time per request: < 200ms
# - Failed requests: 0
```

#### Custom Load Test Script
```bash
python scripts/load_test.py --concurrent-users 20 --duration 60 --endpoint /api/v1/models
```

### 2. Database Performance

#### Connection Pool Test
```bash
python scripts/db_performance_test.py --connections 50 --queries 1000
```

### 3. Memory and CPU Monitoring

#### Monitor Resource Usage
```bash
# Monitor Docker containers
docker stats

# Expected resource usage:
# - Backend: < 512MB RAM, < 50% CPU
# - Database: < 256MB RAM, < 30% CPU
# - Redis: < 128MB RAM, < 10% CPU
```

## 🔒 Security Testing

### 1. Authentication Security

#### Test Invalid Credentials
```bash
# Should return 401 Unauthorized
curl -X POST http://localhost:9000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"invalid","password":"wrong"}'
```

#### Test Unauthorized Access
```bash
# Should return 401 Authentication Required
curl -X POST http://localhost:9000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"test","model":"gpt-4"}'
```

### 2. Input Validation

#### Test SQL Injection Protection
```bash
curl -X POST http://localhost:9000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin'\'' OR 1=1--","password":"test"}'
```

#### Test XSS Protection
```bash
curl -X POST http://localhost:9000/api/v1/generate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"<script>alert(\"xss\")</script>","model":"gpt-4"}'
```

### 3. Rate Limiting

#### Test Rate Limits
```bash
# Send multiple rapid requests
for i in {1..100}; do
  curl -s http://localhost:9000/api/v1/health > /dev/null &
done
wait

# Should eventually return 429 Too Many Requests
```

## 🤖 Automated Testing

### 1. Backend Unit Tests

#### Run All Tests
```bash
cd 03-LLM-Optimization/llm-optimization-platform
python -m pytest tests/ -v
```

#### Run Specific Test Categories
```bash
# API tests
python -m pytest tests/test_api_integration.py -v

# Authentication tests
python -m pytest tests/test_authentication.py -v

# Database tests
python -m pytest tests/test_database_integration.py -v

# Fine-tuning tests
python -m pytest tests/test_fine_tuning_service.py -v

# Evaluation tests
python -m pytest tests/test_evaluation_engine_integration.py -v
```

#### Generate Coverage Report
```bash
python -m pytest tests/ --cov=. --cov-report=html --cov-report=term
open htmlcov/index.html  # View coverage report
```

### 2. Frontend Unit Tests

#### Run React Tests
```bash
cd web_interface/frontend
npm test

# Run with coverage
npm test -- --coverage --watchAll=false
```

#### Run Specific Component Tests
```bash
# Test prompt testing components
npm test -- --testPathPattern=PromptTesting

# Test fine-tuning components
npm test -- --testPathPattern=FineTuning

# Test analytics components
npm test -- --testPathPattern=Analytics
```

### 3. Integration Tests

#### End-to-End Tests
```bash
# Run full integration test suite
python -m pytest tests/test_end_to_end.py -v

# Run with real API calls (requires API keys)
INTEGRATION_TEST=true python -m pytest tests/test_service_integration.py -v
```

## 📝 Manual Testing Scenarios

### Scenario 1: New User Onboarding
1. **Register**: Create account with valid credentials
2. **Login**: Access platform with new credentials
3. **Explore**: Navigate through all main sections
4. **First Test**: Run simple prompt test with default models
5. **View Results**: Check evaluation metrics and costs

### Scenario 2: Prompt Engineering Workflow
1. **Create Prompts**: Design 3 different prompt variations
2. **Select Models**: Choose mix of commercial and fine-tuned models
3. **Run Comparison**: Execute evaluation across all combinations
4. **Analyze Results**: Compare metrics and identify best performers
5. **Iterate**: Refine prompts based on results
6. **Export**: Download results for further analysis

### Scenario 3: Fine-tuning Pipeline
1. **Upload Dataset**: Prepare and upload training data
2. **Configure Training**: Set hyperparameters and options
3. **Start Training**: Launch fine-tuning job
4. **Monitor Progress**: Track training metrics and logs
5. **Evaluate Model**: Test fine-tuned model performance
6. **Deploy**: Make model available for inference

### Scenario 4: Cost Management
1. **Set Budget**: Configure daily/monthly spending limits
2. **Monitor Usage**: Track costs across different models
3. **Analyze Trends**: Review spending patterns over time
4. **Optimize**: Identify cost-saving opportunities
5. **Alert Testing**: Trigger cost threshold alerts

### Scenario 5: Team Collaboration
1. **Multi-user Setup**: Create accounts with different roles
2. **Share Experiments**: Collaborate on prompt testing
3. **Review Results**: Provide feedback on model outputs
4. **Access Control**: Test role-based permissions
5. **Audit Trail**: Review user activity logs

## 🐛 Common Issues & Troubleshooting

### Issue 1: API Authentication Fails
**Symptoms**: 401 errors on API calls
**Solutions**:
- Check JWT token validity
- Verify user credentials
- Ensure proper Authorization header format

### Issue 2: Model API Timeouts
**Symptoms**: 504 Gateway Timeout errors
**Solutions**:
- Check API key validity
- Verify network connectivity
- Increase timeout settings
- Check model availability

### Issue 3: Database Connection Issues
**Symptoms**: 500 errors, connection refused
**Solutions**:
- Verify PostgreSQL is running
- Check database credentials
- Ensure database exists and is accessible
- Review connection pool settings

### Issue 4: Frontend Build Failures
**Symptoms**: Compilation errors, missing dependencies
**Solutions**:
- Run `npm install` to update dependencies
- Clear node_modules and reinstall
- Check Node.js version compatibility
- Review TypeScript configuration

## 📊 Test Results Documentation

### Expected Performance Benchmarks
- **API Response Time**: < 100ms for simple endpoints
- **Text Generation**: 500-2000ms depending on model
- **Database Queries**: < 50ms for most operations
- **Frontend Load Time**: < 3 seconds initial load
- **Memory Usage**: < 1GB total for all services

### Success Criteria
- ✅ All unit tests pass (>95% coverage)
- ✅ Integration tests complete successfully
- ✅ No security vulnerabilities detected
- ✅ Performance benchmarks met
- ✅ All manual test scenarios work as expected

### Test Environment Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB+ available
- **Storage**: 10GB+ free space
- **Network**: Stable internet for API calls
- **Browser**: Chrome/Firefox/Safari latest versions

---

## 📞 Support

If you encounter issues during testing:
1. Check the [Troubleshooting Guide](docs/troubleshooting_guide.md)
2. Review application logs: `docker-compose logs`
3. Open an issue on GitHub with test details
4. Contact support with test environment information

**Happy Testing! 🚀**