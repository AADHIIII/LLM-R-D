# API Documentation

This directory contains comprehensive API documentation for the LLM Optimization Platform.

## Files

### 📋 API Specification
- **`api_specification.yaml`** - Complete OpenAPI 3.0 specification with all endpoints, schemas, and examples
- **`api_documentation.html`** - Interactive Swagger UI documentation with custom styling and features

### 🧪 Documentation Tests
- **`../tests/test_api_documentation.py`** - Comprehensive test suite to validate documentation accuracy

## Quick Start

### 1. View Interactive Documentation

Open the interactive API documentation in your browser:

```bash
# Serve the documentation locally
cd docs
python -m http.server 8080
# Then open http://localhost:8080/api_documentation.html
```

Or if you have the API server running:
```bash
# Access via the API server (if serving static files)
curl http://localhost:5000/docs/api_documentation.html
```

### 2. Use the OpenAPI Specification

The OpenAPI specification can be used with various tools:

```bash
# Generate client SDKs
openapi-generator generate -i api_specification.yaml -g python -o ./python-client

# Validate the specification
swagger-codegen validate -i api_specification.yaml

# Import into Postman
# File > Import > api_specification.yaml
```

### 3. Test Documentation Accuracy

Run the documentation validation tests:

```bash
cd ..
python -m pytest tests/test_api_documentation.py -v
```

## API Overview

### Base URLs
- **Development:** `http://localhost:5000/api/v1`
- **Production:** `https://api.llm-optimization.com/v1`

### Authentication
All endpoints require API key authentication:
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:5000/api/v1/health
```

### Rate Limits
- **Text Generation:** 100 requests/minute
- **Batch Operations:** 10 requests/minute  
- **Other Endpoints:** 1000 requests/minute

## Endpoint Categories

### 🏥 Health & Status
- `GET /health` - Basic health check
- `GET /status` - Detailed system status
- `GET /ready` - Readiness probe

### 🤖 Models
- `GET /models` - List available models
- `GET /models/{id}` - Get model details

### ✨ Text Generation
- `POST /generate` - Generate text
- `POST /generate/batch` - Batch generation

### 🏢 Commercial APIs
- `GET /commercial/test` - Test API connections
- `GET /commercial/models` - List commercial models

### 💰 Cost Tracking
- `POST /cost/track` - Track API usage
- `POST /cost/estimate` - Estimate costs
- `POST /cost/compare` - Compare model costs
- `GET /cost/budget/status` - Budget status
- `GET /cost/analytics` - Usage analytics

### 👥 Human Feedback
- `PUT /feedback/evaluation/{id}` - Update feedback
- `GET /feedback/stats` - Feedback statistics

### 📊 Monitoring
- `GET /monitoring/health` - Monitoring status
- `GET /monitoring/metrics/system` - System metrics
- `GET /monitoring/alerts` - Active alerts

## Example Usage

### Generate Text
```bash
curl -X POST http://localhost:5000/api/v1/generate \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a short story about AI",
    "model_id": "gpt-4",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Track Costs
```bash
curl -X POST http://localhost:5000/api/v1/cost/track \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "gpt-4",
    "input_tokens": 50,
    "output_tokens": 100,
    "latency_ms": 1500
  }'
```

### Get System Status
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:5000/api/v1/status
```

## Error Handling

All endpoints return standardized error responses:

```json
{
  "error": "validation_error",
  "message": "Invalid request parameters",
  "details": "max_tokens must be between 1 and 2048",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Common Error Codes
- `validation_error` - Invalid request parameters
- `authentication_error` - Missing or invalid API key
- `rate_limit_exceeded` - Too many requests
- `model_not_found` - Requested model unavailable
- `generation_error` - Text generation failed
- `internal_server_error` - Unexpected server error

## Response Formats

### Success Response
```json
{
  "success": true,
  "data": { ... },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Error Response
```json
{
  "success": false,
  "error": "error_code",
  "message": "Error description",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Interactive Features

The Swagger UI documentation includes:

### 🔑 API Key Management
- Click "🔑 Set API Key" button to store your API key
- Automatically includes API key in all test requests
- Persistent storage in browser localStorage

### 🧪 Try It Out
- Interactive request forms for all endpoints
- Real-time request/response testing
- Parameter validation and examples

### 📖 Comprehensive Documentation
- Detailed descriptions for all endpoints
- Request/response schemas with examples
- Parameter validation rules
- Error response documentation

## Validation & Testing

The documentation includes comprehensive tests to ensure accuracy:

### Schema Validation
- OpenAPI 3.0 specification compliance
- All required fields present
- Valid YAML syntax
- Reference integrity

### Endpoint Coverage
- All implemented endpoints documented
- Request/response schemas match implementation
- Examples work with actual API
- Error responses documented

### Integration Testing
- Documentation serves correctly
- Interactive features work
- API key authentication functions
- Examples execute successfully

## Contributing

When adding new endpoints or modifying existing ones:

1. **Update OpenAPI Spec** - Add/modify endpoint in `api_specification.yaml`
2. **Add Examples** - Include request/response examples
3. **Update Tests** - Add validation tests in `test_api_documentation.py`
4. **Test Documentation** - Run tests to ensure accuracy
5. **Update README** - Add any new sections or usage examples

### Documentation Standards

- **Descriptions:** Clear, concise endpoint descriptions
- **Examples:** Working examples for all request/response formats
- **Validation:** Proper parameter validation rules
- **Error Handling:** Document all possible error responses
- **Security:** Include authentication requirements
- **Versioning:** Maintain backward compatibility

## Tools & Resources

### OpenAPI Tools
- [Swagger Editor](https://editor.swagger.io/) - Online OpenAPI editor
- [OpenAPI Generator](https://openapi-generator.tech/) - Generate client SDKs
- [Swagger Codegen](https://swagger.io/tools/swagger-codegen/) - Code generation
- [Postman](https://www.postman.com/) - API testing with OpenAPI import

### Validation Tools
- [Swagger Validator](https://validator.swagger.io/) - Online spec validation
- [OpenAPI Spec Validator](https://pypi.org/project/openapi-spec-validator/) - Python validation
- [Spectral](https://stoplight.io/open-source/spectral/) - OpenAPI linting

### Documentation Hosting
- [Swagger UI](https://swagger.io/tools/swagger-ui/) - Interactive documentation
- [ReDoc](https://redocly.github.io/redoc/) - Alternative documentation renderer
- [GitBook](https://www.gitbook.com/) - Documentation platform
- [Stoplight](https://stoplight.io/) - API documentation platform