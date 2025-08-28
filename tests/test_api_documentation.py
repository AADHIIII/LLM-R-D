"""
Tests to validate API documentation accuracy against actual implementation.
"""

import pytest
import yaml
import json
import requests
from typing import Dict, Any, List
from pathlib import Path
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.app import create_app


class TestAPIDocumentation:
    """Test suite to validate API documentation accuracy."""
    
    @pytest.fixture(scope="class")
    def app(self):
        """Create test Flask application."""
        app = create_app('testing')
        app.config['TESTING'] = True
        return app
    
    @pytest.fixture(scope="class")
    def client(self, app):
        """Create test client."""
        return app.test_client()
    
    @pytest.fixture(scope="class")
    def openapi_spec(self):
        """Load OpenAPI specification."""
        spec_path = Path(__file__).parent.parent / 'docs' / 'api_specification.yaml'
        with open(spec_path, 'r') as f:
            return yaml.safe_load(f)
    
    def test_openapi_spec_structure(self, openapi_spec):
        """Test that OpenAPI spec has required structure."""
        # Check required top-level fields
        required_fields = ['openapi', 'info', 'paths', 'components']
        for field in required_fields:
            assert field in openapi_spec, f"Missing required field: {field}"
        
        # Check OpenAPI version
        assert openapi_spec['openapi'].startswith('3.0'), "Should use OpenAPI 3.0+"
        
        # Check info section
        info = openapi_spec['info']
        assert 'title' in info
        assert 'version' in info
        assert 'description' in info
        
        # Check components section has schemas
        assert 'schemas' in openapi_spec['components']
        assert 'securitySchemes' in openapi_spec['components']
    
    def test_health_endpoints_documented(self, openapi_spec, client):
        """Test that health endpoints are properly documented."""
        paths = openapi_spec['paths']
        
        # Check health endpoints exist in spec
        health_endpoints = ['/health', '/status', '/ready']
        for endpoint in health_endpoints:
            assert endpoint in paths, f"Health endpoint {endpoint} not documented"
            assert 'get' in paths[endpoint], f"GET method not documented for {endpoint}"
        
        # Test actual endpoints work
        for endpoint in health_endpoints:
            response = client.get(f'/api/v1{endpoint}')
            assert response.status_code in [200, 503], f"Health endpoint {endpoint} failed"
    
    def test_models_endpoints_documented(self, openapi_spec, client):
        """Test that model endpoints are properly documented."""
        paths = openapi_spec['paths']
        
        # Check models endpoints
        assert '/models' in paths
        assert 'get' in paths['/models']
        
        assert '/models/{model_id}' in paths
        assert 'get' in paths['/models/{model_id}']
        
        # Test actual endpoints
        response = client.get('/api/v1/models')
        assert response.status_code == 200
        
        # Test with a known model ID
        response = client.get('/api/v1/models/gpt-4')
        assert response.status_code in [200, 404]  # 404 if model not available
    
    def test_generation_endpoints_documented(self, openapi_spec, client):
        """Test that generation endpoints are properly documented."""
        paths = openapi_spec['paths']
        
        # Check generation endpoints
        assert '/generate' in paths
        assert 'post' in paths['/generate']
        
        assert '/generate/batch' in paths
        assert 'post' in paths['/generate/batch']
        
        # Check request schemas exist
        generate_schema = paths['/generate']['post']['requestBody']['content']['application/json']['schema']
        assert '$ref' in generate_schema
        
        batch_schema = paths['/generate/batch']['post']['requestBody']['content']['application/json']['schema']
        assert '$ref' in batch_schema
    
    def test_cost_tracking_endpoints_documented(self, openapi_spec):
        """Test that cost tracking endpoints are properly documented."""
        paths = openapi_spec['paths']
        
        # Check cost tracking endpoints
        cost_endpoints = [
            '/cost/track',
            '/cost/estimate', 
            '/cost/compare',
            '/cost/budget/status',
            '/cost/analytics'
        ]
        
        for endpoint in cost_endpoints:
            assert endpoint in paths, f"Cost endpoint {endpoint} not documented"
    
    def test_feedback_endpoints_documented(self, openapi_spec):
        """Test that feedback endpoints are properly documented."""
        paths = openapi_spec['paths']
        
        # Check feedback endpoints
        assert '/feedback/evaluation/{evaluation_id}' in paths
        assert 'put' in paths['/feedback/evaluation/{evaluation_id}']
        
        assert '/feedback/stats' in paths
        assert 'get' in paths['/feedback/stats']
    
    def test_monitoring_endpoints_documented(self, openapi_spec):
        """Test that monitoring endpoints are properly documented."""
        paths = openapi_spec['paths']
        
        # Check monitoring endpoints
        monitoring_endpoints = [
            '/monitoring/health',
            '/monitoring/metrics/system',
            '/monitoring/alerts'
        ]
        
        for endpoint in monitoring_endpoints:
            assert endpoint in paths, f"Monitoring endpoint {endpoint} not documented"
    
    def test_error_response_schema(self, openapi_spec):
        """Test that error response schema is properly defined."""
        schemas = openapi_spec['components']['schemas']
        
        assert 'ErrorResponse' in schemas
        error_schema = schemas['ErrorResponse']
        
        # Check required fields
        required_fields = ['error', 'message', 'timestamp']
        assert 'required' in error_schema
        for field in required_fields:
            assert field in error_schema['required']
        
        # Check properties exist
        assert 'properties' in error_schema
        for field in required_fields:
            assert field in error_schema['properties']
    
    def test_request_schemas_complete(self, openapi_spec):
        """Test that all request schemas are complete and valid."""
        schemas = openapi_spec['components']['schemas']
        
        # Key request schemas to validate
        request_schemas = [
            'GenerateRequest',
            'BatchGenerateRequest',
            'TrackCostRequest',
            'FeedbackRequest'
        ]
        
        for schema_name in request_schemas:
            assert schema_name in schemas, f"Request schema {schema_name} missing"
            schema = schemas[schema_name]
            
            # Should have properties
            assert 'properties' in schema, f"Schema {schema_name} missing properties"
            
            # Should have required fields (for most schemas)
            if schema_name != 'FeedbackRequest':  # FeedbackRequest has all optional fields
                assert 'required' in schema, f"Schema {schema_name} missing required fields"
    
    def test_response_schemas_complete(self, openapi_spec):
        """Test that all response schemas are complete and valid."""
        schemas = openapi_spec['components']['schemas']
        
        # Key response schemas to validate
        response_schemas = [
            'HealthResponse',
            'ModelsListResponse',
            'GenerateResponse',
            'TrackCostResponse',
            'FeedbackStatsResponse'
        ]
        
        for schema_name in response_schemas:
            assert schema_name in schemas, f"Response schema {schema_name} missing"
            schema = schemas[schema_name]
            
            # Should have properties
            assert 'properties' in schema, f"Schema {schema_name} missing properties"
            
            # Should have required fields
            assert 'required' in schema, f"Schema {schema_name} missing required fields"
    
    def test_security_scheme_defined(self, openapi_spec):
        """Test that security scheme is properly defined."""
        components = openapi_spec['components']
        
        assert 'securitySchemes' in components
        assert 'ApiKeyAuth' in components['securitySchemes']
        
        auth_scheme = components['securitySchemes']['ApiKeyAuth']
        assert auth_scheme['type'] == 'apiKey'
        assert auth_scheme['in'] == 'header'
        assert auth_scheme['name'] == 'Authorization'
    
    def test_tags_defined(self, openapi_spec):
        """Test that all tags are properly defined."""
        # Check that tags are defined at the root level
        assert 'tags' in openapi_spec
        
        tags = {tag['name'] for tag in openapi_spec['tags']}
        expected_tags = {
            'Health', 'Models', 'Generation', 'Commercial APIs',
            'Cost Tracking', 'Feedback', 'Monitoring'
        }
        
        assert expected_tags.issubset(tags), f"Missing tags: {expected_tags - tags}"
    
    def test_examples_provided(self, openapi_spec):
        """Test that examples are provided for key endpoints."""
        paths = openapi_spec['paths']
        
        # Check that generate endpoint has examples
        generate_path = paths['/generate']['post']
        request_body = generate_path['requestBody']['content']['application/json']
        assert 'example' in request_body, "Generate endpoint missing request example"
        
        # Check response examples
        responses = generate_path['responses']['200']['content']['application/json']
        assert 'example' in responses, "Generate endpoint missing response example"
    
    def test_parameter_validation(self, openapi_spec):
        """Test that parameters have proper validation constraints."""
        schemas = openapi_spec['components']['schemas']
        
        # Check GenerateRequest validation
        generate_req = schemas['GenerateRequest']['properties']
        
        # Prompt should have length constraints
        assert 'minLength' in generate_req['prompt']
        assert 'maxLength' in generate_req['prompt']
        
        # max_tokens should have range constraints
        assert 'minimum' in generate_req['max_tokens']
        assert 'maximum' in generate_req['max_tokens']
        
        # temperature should have range constraints
        assert 'minimum' in generate_req['temperature']
        assert 'maximum' in generate_req['temperature']
    
    def test_documentation_completeness(self, openapi_spec):
        """Test that all endpoints have proper descriptions."""
        paths = openapi_spec['paths']
        
        for path, methods in paths.items():
            for method, spec in methods.items():
                # Each endpoint should have summary and description
                assert 'summary' in spec, f"Missing summary for {method.upper()} {path}"
                assert 'description' in spec, f"Missing description for {method.upper()} {path}"
                
                # Should have tags
                assert 'tags' in spec, f"Missing tags for {method.upper()} {path}"
                
                # Should have responses
                assert 'responses' in spec, f"Missing responses for {method.upper()} {path}"
                
                # Should have at least 200 response
                assert '200' in spec['responses'], f"Missing 200 response for {method.upper()} {path}"


class TestAPIDocumentationIntegration:
    """Integration tests for API documentation."""
    
    def test_swagger_ui_html_valid(self):
        """Test that Swagger UI HTML file is valid."""
        html_path = Path(__file__).parent.parent / 'docs' / 'api_documentation.html'
        assert html_path.exists(), "API documentation HTML file missing"
        
        with open(html_path, 'r') as f:
            content = f.read()
        
        # Check for required elements
        assert 'swagger-ui-bundle.js' in content
        assert 'swagger-ui.css' in content
        assert 'api_specification.yaml' in content
        assert 'SwaggerUIBundle' in content
    
    def test_yaml_spec_valid_yaml(self):
        """Test that YAML specification is valid YAML."""
        spec_path = Path(__file__).parent.parent / 'docs' / 'api_specification.yaml'
        assert spec_path.exists(), "API specification YAML file missing"
        
        with open(spec_path, 'r') as f:
            try:
                yaml.safe_load(f)
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML in API specification: {e}")
    
    def test_spec_references_valid(self, openapi_spec):
        """Test that all $ref references in the spec are valid."""
        def check_refs(obj, path=""):
            if isinstance(obj, dict):
                if '$ref' in obj:
                    ref = obj['$ref']
                    if ref.startswith('#/'):
                        # Internal reference
                        ref_path = ref[2:].split('/')
                        current = openapi_spec
                        try:
                            for part in ref_path:
                                current = current[part]
                        except KeyError:
                            pytest.fail(f"Invalid reference {ref} at {path}")
                else:
                    for key, value in obj.items():
                        check_refs(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_refs(item, f"{path}[{i}]")
        
        check_refs(openapi_spec)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])