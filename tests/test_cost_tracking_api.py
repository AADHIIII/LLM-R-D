"""
Tests for cost tracking API endpoints.
"""

import pytest
import json
from unittest.mock import Mock, patch
from datetime import datetime

from api.app import create_app


class TestCostTrackingAPI:
    """Test cost tracking API endpoints"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.app = create_app('testing')
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
    
    def teardown_method(self):
        """Clean up test fixtures"""
        self.app_context.pop()
    
    @patch('api.blueprints.cost_tracking.cost_service')
    def test_track_api_call(self, mock_cost_service):
        """Test tracking an API call"""
        # Mock the service response
        mock_cost_service.track_api_call.return_value = {
            "cost_breakdown": {
                "model_name": "gpt-4",
                "total_cost": 0.045,
                "input_cost": 0.030,
                "output_cost": 0.015,
            },
            "budget_status": {
                "total_budget": 1000.0,
                "current_spend": 0.045,
                "remaining": 999.955,
                "utilization": 0.000045,
            },
            "alerts": [],
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        response = self.client.post('/api/v1/cost/track', 
                                  json={
                                      "model_name": "gpt-4",
                                      "input_tokens": 1000,
                                      "output_tokens": 500,
                                      "latency_ms": 1200
                                  })
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert "cost_breakdown" in data["data"]
        assert "budget_status" in data["data"]
        
        # Verify service was called correctly
        mock_cost_service.track_api_call.assert_called_once_with(
            model_name="gpt-4",
            input_tokens=1000,
            output_tokens=500,
            latency_ms=1200,
            timestamp=None
        )
    
    @patch('api.blueprints.cost_tracking.cost_service')
    def test_track_api_call_missing_fields(self, mock_cost_service):
        """Test tracking API call with missing required fields"""
        response = self.client.post('/api/v1/cost/track', 
                                  json={
                                      "model_name": "gpt-4",
                                      "input_tokens": 1000
                                      # Missing output_tokens
                                  })
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data["success"] is False
    
    @patch('api.blueprints.cost_tracking.cost_service')
    def test_estimate_cost(self, mock_cost_service):
        """Test cost estimation"""
        mock_cost_service.estimate_request_cost.return_value = 0.045
        
        response = self.client.post('/api/v1/cost/estimate', 
                                  json={
                                      "model_name": "gpt-4",
                                      "estimated_tokens": 1000
                                  })
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["data"]["estimated_cost_usd"] == 0.045
        
        mock_cost_service.estimate_request_cost.assert_called_once_with(
            model_name="gpt-4",
            estimated_tokens=1000
        )
    
    @patch('api.blueprints.cost_tracking.cost_service')
    def test_compare_model_costs(self, mock_cost_service):
        """Test model cost comparison"""
        mock_cost_service.compare_model_costs.return_value = {
            "gpt-4": {"total_cost": 0.045, "model_name": "gpt-4"},
            "claude-3": {"total_cost": 0.038, "model_name": "claude-3"},
        }
        
        response = self.client.post('/api/v1/cost/compare', 
                                  json={
                                      "models": ["gpt-4", "claude-3"],
                                      "input_tokens": 1000,
                                      "output_tokens": 500
                                  })
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert "comparisons" in data["data"]
        assert len(data["data"]["comparisons"]) == 2
    
    @patch('api.blueprints.cost_tracking.cost_service')
    def test_get_cheapest_model(self, mock_cost_service):
        """Test finding cheapest model"""
        mock_cost_service.get_cheapest_model.return_value = {
            "model": "fine-tuned-gpt2",
            "cost_breakdown": {"total_cost": 0.012, "model_name": "fine-tuned-gpt2"}
        }
        
        response = self.client.post('/api/v1/cost/cheapest', 
                                  json={
                                      "models": ["gpt-4", "claude-3", "fine-tuned-gpt2"],
                                      "input_tokens": 1000,
                                      "output_tokens": 500
                                  })
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["data"]["model"] == "fine-tuned-gpt2"
    
    @patch('api.blueprints.cost_tracking.cost_service')
    def test_get_budget_status(self, mock_cost_service):
        """Test getting budget status"""
        mock_cost_service.get_budget_status.return_value = {
            "total_budget": 1000.0,
            "current_spend": 125.50,
            "remaining": 874.50,
            "utilization": 0.1255,
            "is_over_budget": False,
            "is_near_budget": False,
        }
        
        response = self.client.get('/api/v1/cost/budget/status')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["data"]["total_budget"] == 1000.0
        assert data["data"]["current_spend"] == 125.50
    
    @patch('api.blueprints.cost_tracking.cost_service')
    def test_get_budget_alerts(self, mock_cost_service):
        """Test getting budget alerts"""
        mock_cost_service.get_budget_alerts.return_value = [
            {
                "type": "budget_warning",
                "message": "Budget utilization at 85.0%",
                "severity": "warning"
            }
        ]
        
        response = self.client.get('/api/v1/cost/budget/alerts')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["data"]["alert_count"] == 1
        assert len(data["data"]["alerts"]) == 1
    
    @patch('api.blueprints.cost_tracking.cost_service')
    def test_update_budget(self, mock_cost_service):
        """Test updating budget"""
        response = self.client.put('/api/v1/cost/budget/update', 
                                 json={
                                     "total_budget": 2000.0,
                                     "alert_threshold": 0.9
                                 })
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert "Budget updated to $2000.00" in data["message"]
        
        mock_cost_service.update_budget.assert_called_once_with(
            new_budget=2000.0,
            alert_threshold=0.9
        )
    
    @patch('api.blueprints.cost_tracking.cost_service')
    def test_reset_budget(self, mock_cost_service):
        """Test resetting budget"""
        response = self.client.post('/api/v1/cost/budget/reset')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert "reset successfully" in data["message"]
        
        mock_cost_service.reset_budget_tracking.assert_called_once()
    
    @patch('api.blueprints.cost_tracking.cost_service')
    def test_get_usage_analytics(self, mock_cost_service):
        """Test getting usage analytics"""
        mock_cost_service.get_usage_analytics.return_value = {
            "period_days": 30,
            "total_cost": 125.50,
            "total_requests": 450,
            "total_tokens": 125000,
            "model_breakdown": {
                "gpt-4": {"cost": 89.50, "requests": 200},
                "claude-3": {"cost": 36.00, "requests": 250}
            }
        }
        
        response = self.client.get('/api/v1/cost/analytics?days=30')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["data"]["period_days"] == 30
        assert data["data"]["total_cost"] == 125.50
        
        mock_cost_service.get_usage_analytics.assert_called_once_with(30)
    
    @patch('api.blueprints.cost_tracking.cost_service')
    def test_get_cost_trends(self, mock_cost_service):
        """Test getting cost trends"""
        mock_cost_service.get_cost_trends.return_value = {
            "daily_costs": [
                {"date": "2024-01-01", "total_cost": 25.50, "openai_cost": 20.00, "anthropic_cost": 5.50},
                {"date": "2024-01-02", "total_cost": 30.25, "openai_cost": 22.00, "anthropic_cost": 8.25},
            ],
            "provider_totals": {"openai": 42.00, "anthropic": 13.75, "fine_tuned": 0.00},
            "total_cost": 55.75,
            "period_days": 7
        }
        
        response = self.client.get('/api/v1/cost/trends?days=7')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["data"]["period_days"] == 7
        assert len(data["data"]["daily_costs"]) == 2
        
        mock_cost_service.get_cost_trends.assert_called_once_with(7)
    
    @patch('api.blueprints.cost_tracking.cost_service')
    def test_get_model_breakdown(self, mock_cost_service):
        """Test getting model breakdown"""
        mock_cost_service.get_model_cost_breakdown.return_value = [
            {
                "model_name": "gpt-4",
                "total_cost": 89.50,
                "request_count": 200,
                "avg_cost_per_request": 0.4475,
                "token_count": 75000,
                "cost_per_token": 0.001193
            }
        ]
        
        response = self.client.get('/api/v1/cost/breakdown')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["data"]["total_models"] == 1
        assert len(data["data"]["model_breakdown"]) == 1
    
    @patch('api.blueprints.cost_tracking.cost_service')
    def test_get_optimization_recommendations(self, mock_cost_service):
        """Test getting optimization recommendations"""
        mock_cost_service.get_cost_optimization_recommendations.return_value = [
            {
                "type": "expensive_model",
                "message": "Model 'gpt-4' accounts for $89.50 of spending",
                "suggestion": "Consider using a cheaper alternative model for non-critical tasks"
            }
        ]
        
        response = self.client.get('/api/v1/cost/recommendations')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["data"]["recommendation_count"] == 1
        assert len(data["data"]["recommendations"]) == 1
    
    @patch('api.blueprints.cost_tracking.cost_service')
    def test_export_cost_data(self, mock_cost_service):
        """Test exporting cost data"""
        mock_cost_service.export_cost_data.return_value = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "period_days": 30,
            "budget_status": {"total_budget": 1000.0, "current_spend": 125.50},
            "usage_analytics": {"total_cost": 125.50, "total_requests": 450}
        }
        
        response = self.client.get('/api/v1/cost/export?days=30&format=json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert "export_timestamp" in data["data"]
        
        mock_cost_service.export_cost_data.assert_called_once_with(30, "json")
    
    def test_export_cost_data_invalid_format(self):
        """Test exporting cost data with invalid format"""
        response = self.client.get('/api/v1/cost/export?format=xml')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data["success"] is False
        assert "Invalid format" in data["error"]
    
    @patch('api.blueprints.cost_tracking.cost_service')
    def test_cost_tracking_health(self, mock_cost_service):
        """Test cost tracking health check"""
        mock_cost_service.get_budget_status.return_value = {
            "utilization": 0.25
        }
        
        response = self.client.get('/api/v1/cost/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["service"] == "cost_tracking"
        assert data["status"] == "healthy"
        assert "budget_utilization" in data
    
    @patch('api.blueprints.cost_tracking.cost_service')
    def test_error_handling(self, mock_cost_service):
        """Test error handling in API endpoints"""
        # Mock service to raise an exception
        mock_cost_service.track_api_call.side_effect = Exception("Service error")
        
        response = self.client.post('/api/v1/cost/track', 
                                  json={
                                      "model_name": "gpt-4",
                                      "input_tokens": 1000,
                                      "output_tokens": 500
                                  })
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert data["success"] is False
        assert "error" in data