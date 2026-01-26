"""
Comprehensive tests for API endpoints with proper authentication handling.
"""
import pytest
import json
import io
from datetime import datetime
from unittest.mock import patch, MagicMock

from api.app import create_app


@pytest.fixture
def app():
    """Create test Flask app with auth skipped."""
    app = create_app('testing')
    app.config['SKIP_AUTH'] = True
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def auth_client(app):
    """Create test client with authentication."""
    with app.test_client() as client:
        yield client


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_returns_healthy(self, client):
        """Test health endpoint returns healthy status."""
        response = client.get('/api/v1/health')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert 'version' in data

    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.cpu_percent')
    @patch('psutil.cpu_count')
    def test_status_returns_system_info(self, mock_cpu_count, mock_cpu_percent,
                                        mock_disk_usage, mock_memory, client):
        """Test status endpoint returns system information."""
        mock_memory.return_value = MagicMock(
            total=16000000000, available=8000000000,
            percent=50.0, used=8000000000
        )
        mock_disk_usage.return_value = MagicMock(
            total=500000000000, free=250000000000, used=250000000000
        )
        mock_cpu_percent.return_value = 30.0
        mock_cpu_count.return_value = 8

        response = client.get('/api/v1/status')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'system' in data
        assert data['system']['memory']['percent'] == 50.0
        assert data['system']['cpu']['percent'] == 30.0

    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_ready_returns_ready_status(self, mock_disk_usage, mock_memory, client):
        """Test readiness endpoint returns ready status."""
        mock_memory.return_value = MagicMock(percent=50.0)
        mock_disk_usage.return_value = MagicMock(free=500000000)

        response = client.get('/api/v1/ready')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['ready'] is True


class TestModelsEndpoints:
    """Test model management endpoints."""

    def test_list_models_returns_commercial_models(self, client):
        """Test listing models includes commercial models."""
        with patch('os.path.exists', return_value=False):
            response = client.get('/api/v1/models')
            assert response.status_code == 200

            data = json.loads(response.data)
            assert 'models' in data
            assert 'count' in data

            model_ids = [m['id'] for m in data['models']]
            assert 'gpt-4' in model_ids or 'gpt-3.5-turbo' in model_ids

    def test_get_model_info_returns_404_for_unknown(self, client):
        """Test getting info for unknown model returns 404."""
        response = client.get('/api/v1/models/unknown-model-xyz')
        assert response.status_code == 404

        data = json.loads(response.data)
        assert data['error'] == 'model_not_found'

    @patch('api.blueprints.models.get_available_models')
    def test_get_model_info_returns_model_details(self, mock_get_models, client):
        """Test getting model info returns correct details."""
        mock_get_models.return_value = [{
            'id': 'test-model',
            'name': 'Test Model',
            'type': 'commercial',
            'provider': 'test',
            'status': 'available',
            'capabilities': ['text-generation']
        }]

        response = client.get('/api/v1/models/test-model')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['model']['id'] == 'test-model'
        assert data['model']['type'] == 'commercial'


class TestGenerateEndpoints:
    """Test text generation endpoints."""

    def test_generate_requires_prompt(self, client):
        """Test generation fails without prompt."""
        response = client.post('/api/v1/generate',
                               json={'model_id': 'gpt-4'},
                               content_type='application/json')
        assert response.status_code == 400

    def test_generate_requires_model_id(self, client):
        """Test generation fails without model_id."""
        response = client.post('/api/v1/generate',
                               json={'prompt': 'test'},
                               content_type='application/json')
        assert response.status_code == 400

    def test_generate_rejects_empty_prompt(self, client):
        """Test generation fails with empty prompt."""
        response = client.post('/api/v1/generate',
                               json={'prompt': '   ', 'model_id': 'gpt-4'},
                               content_type='application/json')
        assert response.status_code == 400

        data = json.loads(response.data)
        assert 'empty' in data['message'].lower()

    def test_generate_validates_max_tokens(self, client):
        """Test generation validates max_tokens range."""
        response = client.post('/api/v1/generate',
                               json={
                                   'prompt': 'test',
                                   'model_id': 'gpt-4',
                                   'max_tokens': 10000
                               },
                               content_type='application/json')
        assert response.status_code == 400

    def test_generate_validates_temperature(self, client):
        """Test generation validates temperature range."""
        response = client.post('/api/v1/generate',
                               json={
                                   'prompt': 'test',
                                   'model_id': 'gpt-4',
                                   'temperature': 5.0
                               },
                               content_type='application/json')
        assert response.status_code == 400

    @patch('api.blueprints.generate.text_generator')
    def test_generate_success_returns_text(self, mock_generator, client):
        """Test successful generation returns generated text."""
        mock_generator.generate.return_value = {
            'text': 'Generated response text',
            'metadata': {'provider': 'openai'}
        }

        response = client.post('/api/v1/generate',
                               json={
                                   'prompt': 'Hello',
                                   'model_id': 'gpt-4'
                               },
                               content_type='application/json')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['text'] == 'Generated response text'
        assert 'metrics' in data
        assert 'timestamp' in data

    def test_batch_generate_limits_batch_size(self, client):
        """Test batch generation enforces size limit."""
        prompts = [f'prompt {i}' for i in range(20)]
        response = client.post('/api/v1/generate/batch',
                               json={'prompts': prompts, 'model_id': 'gpt-4'},
                               content_type='application/json')
        assert response.status_code == 400

        data = json.loads(response.data)
        assert 'batch size' in data['message'].lower()

    def test_batch_generate_validates_prompts(self, client):
        """Test batch generation validates all prompts."""
        response = client.post('/api/v1/generate/batch',
                               json={
                                   'prompts': ['valid', '', 'also valid'],
                                   'model_id': 'gpt-4'
                               },
                               content_type='application/json')
        assert response.status_code == 400

    @patch('api.blueprints.generate.text_generator')
    def test_batch_generate_success(self, mock_generator, client):
        """Test successful batch generation."""
        mock_generator.generate.side_effect = [
            {'text': 'Response 1', 'metadata': {}},
            {'text': 'Response 2', 'metadata': {}}
        ]

        response = client.post('/api/v1/generate/batch',
                               json={
                                   'prompts': ['prompt 1', 'prompt 2'],
                                   'model_id': 'gpt-4'
                               },
                               content_type='application/json')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert len(data['results']) == 2
        assert data['metrics']['successful_generations'] == 2


class TestUploadEndpoints:
    """Test file upload endpoints."""

    @patch('api.blueprints.upload.secure_file_handler')
    @patch('api.blueprints.upload.audit_logger')
    @patch('api.blueprints.upload.InputValidator')
    def test_upload_dataset_requires_file(self, mock_validator, mock_logger,
                                          mock_handler, client):
        """Test dataset upload requires a file."""
        response = client.post('/api/v1/upload/dataset',
                               data={},
                               content_type='multipart/form-data')
        assert response.status_code == 400

        data = json.loads(response.data)
        assert 'no file' in data['message'].lower()

    @patch('api.blueprints.upload.secure_file_handler')
    @patch('api.blueprints.upload.audit_logger')
    @patch('api.blueprints.upload.InputValidator')
    def test_upload_dataset_rejects_empty_filename(self, mock_validator,
                                                    mock_logger, mock_handler, client):
        """Test dataset upload rejects empty filename."""
        data = {'file': (io.BytesIO(b'test data'), '')}
        response = client.post('/api/v1/upload/dataset',
                               data=data,
                               content_type='multipart/form-data')
        assert response.status_code == 400

    @patch('api.blueprints.upload.secure_file_handler')
    @patch('api.blueprints.upload.audit_logger')
    @patch('api.blueprints.upload.InputValidator')
    def test_upload_dataset_success(self, mock_validator, mock_logger,
                                    mock_handler, client):
        """Test successful dataset upload."""
        mock_validator.validate_filename.return_value = True
        mock_validator.sanitize_string.return_value = 'Test description'
        mock_handler.save_uploaded_file.return_value = {
            'secure_filename': 'abc123_data.csv',
            'original_filename': 'data.csv',
            'size': 1024,
            'mime_type': 'text/csv',
            'uploaded_at': datetime.utcnow().isoformat()
        }

        data = {
            'file': (io.BytesIO(b'col1,col2\n1,2\n3,4'), 'data.csv'),
            'name': 'Test Dataset',
            'description': 'Test description'
        }
        response = client.post('/api/v1/upload/dataset',
                               data=data,
                               content_type='multipart/form-data')
        assert response.status_code == 201

        result = json.loads(response.data)
        assert result['success'] is True
        assert 'file_info' in result

    @patch('api.blueprints.upload.secure_file_handler')
    @patch('api.blueprints.upload.audit_logger')
    @patch('api.blueprints.upload.InputValidator')
    def test_upload_model_requires_name(self, mock_validator, mock_logger,
                                        mock_handler, client):
        """Test model upload requires a name."""
        data = {'file': (io.BytesIO(b'model data'), 'model.bin')}
        response = client.post('/api/v1/upload/model',
                               data=data,
                               content_type='multipart/form-data')
        assert response.status_code == 400

        result = json.loads(response.data)
        assert 'name' in result['message'].lower()

    @patch('api.blueprints.upload.secure_file_handler')
    @patch('api.blueprints.upload.audit_logger')
    @patch('api.blueprints.upload.InputValidator')
    def test_delete_file_validates_type(self, mock_validator, mock_logger,
                                        mock_handler, client):
        """Test file deletion validates file type."""
        response = client.delete('/api/v1/upload/invalid_type/file123')
        assert response.status_code == 400

        data = json.loads(response.data)
        assert 'invalid file type' in data['message'].lower()


class TestEvaluateEndpoints:
    """Test evaluation endpoints."""

    def test_evaluate_returns_not_implemented(self, client):
        """Test evaluate endpoint returns 501 Not Implemented."""
        response = client.post('/api/v1/evaluate',
                               json={},
                               content_type='application/json')
        assert response.status_code == 501

        data = json.loads(response.data)
        assert data['error'] == 'not_implemented'


class TestErrorHandling:
    """Test error handling."""

    def test_404_returns_json(self, client):
        """Test 404 errors return JSON response."""
        response = client.get('/api/v1/nonexistent-endpoint')
        assert response.status_code == 404

        data = json.loads(response.data)
        assert data['error'] == 'not_found'
        assert 'timestamp' in data

    def test_invalid_json_returns_400(self, client):
        """Test invalid JSON returns 400 error."""
        response = client.post('/api/v1/generate',
                               data='not valid json',
                               content_type='application/json')
        assert response.status_code == 400


class TestRequestValidation:
    """Test request validation middleware."""

    def test_requires_json_content_type(self, client):
        """Test POST endpoints require JSON content type."""
        response = client.post('/api/v1/generate',
                               data='{"prompt": "test"}',
                               content_type='text/plain')
        assert response.status_code == 400


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
