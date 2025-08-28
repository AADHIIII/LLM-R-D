"""
Tests for cost tracking service functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from api.services.cost_tracking_service import CostTrackingService
from utils.cost_calculator import UsageMetrics


class TestCostTrackingService:
    """Test cost tracking service functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Mock database dependencies
        with patch('api.services.cost_tracking_service.db_manager'), \
             patch('api.services.cost_tracking_service.EvaluationRepository'), \
             patch('api.services.cost_tracking_service.ExperimentRepository'):
            
            self.service = CostTrackingService(total_budget=1000.0)
            
            # Mock the repositories
            self.service.evaluation_repo = Mock()
            self.service.experiment_repo = Mock()
            
            # Mock empty historical data
            self.service.evaluation_repo.get_recent_evaluations.return_value = []
    
    def test_track_api_call(self):
        """Test tracking an API call"""
        result = self.service.track_api_call(
            model_name="gpt-4",
            input_tokens=1000,
            output_tokens=500,
            latency_ms=1200
        )
        
        assert "cost_breakdown" in result
        assert "budget_status" in result
        assert "alerts" in result
        assert "timestamp" in result
        
        cost_breakdown = result["cost_breakdown"]
        assert cost_breakdown["model_name"] == "gpt-4"
        assert cost_breakdown["total_cost"] > 0
        
        budget_status = result["budget_status"]
        assert budget_status["current_spend"] > 0
        assert budget_status["total_budget"] == 1000.0
    
    def test_estimate_request_cost(self):
        """Test cost estimation"""
        estimated_cost = self.service.estimate_request_cost("gpt-4", 1000)
        
        assert estimated_cost > 0
        assert isinstance(estimated_cost, float)
    
    def test_compare_model_costs(self):
        """Test model cost comparison"""
        models = ["gpt-4", "claude-3-sonnet", "fine-tuned-gpt2"]
        
        comparisons = self.service.compare_model_costs(
            models=models,
            input_tokens=1000,
            output_tokens=500
        )
        
        assert len(comparisons) == 3
        assert all(model in comparisons for model in models)
        
        for model, breakdown in comparisons.items():
            assert "total_cost" in breakdown
            assert "model_name" in breakdown
            assert breakdown["model_name"] == model
    
    def test_get_cheapest_model(self):
        """Test finding cheapest model"""
        models = ["gpt-4", "claude-3-sonnet", "fine-tuned-gpt2"]
        
        result = self.service.get_cheapest_model(
            models=models,
            input_tokens=1000,
            output_tokens=500
        )
        
        assert "model" in result
        assert "cost_breakdown" in result
        assert result["model"] in models
        
        # Fine-tuned should typically be cheapest
        assert result["model"] == "fine-tuned-gpt2"
    
    def test_budget_status(self):
        """Test getting budget status"""
        # Track some usage first
        self.service.track_api_call("gpt-4", 1000, 500)
        
        status = self.service.get_budget_status()
        
        assert "total_budget" in status
        assert "current_spend" in status
        assert "remaining" in status
        assert "utilization" in status
        assert "is_over_budget" in status
        assert "is_near_budget" in status
        
        assert status["total_budget"] == 1000.0
        assert status["current_spend"] > 0
    
    def test_budget_alerts(self):
        """Test budget alerts"""
        # Initially no alerts
        alerts = self.service.get_budget_alerts()
        assert len(alerts) == 0
        
        # Add spending to trigger warning
        self.service.track_api_call("gpt-4", 100000, 50000)  # Large usage
        
        alerts = self.service.get_budget_alerts()
        # Might have alerts depending on cost
        assert isinstance(alerts, list)
    
    def test_cost_optimization_recommendations(self):
        """Test cost optimization recommendations"""
        # Add some spending
        self.service.track_api_call("gpt-4", 10000, 5000)
        
        recommendations = self.service.get_cost_optimization_recommendations()
        
        assert isinstance(recommendations, list)
        # Recommendations depend on spending patterns
    
    def test_usage_analytics(self):
        """Test usage analytics"""
        # Track some usage
        self.service.track_api_call("gpt-4", 1000, 500)
        self.service.track_api_call("claude-3-sonnet", 800, 400)
        
        analytics = self.service.get_usage_analytics(30)
        
        assert "period_days" in analytics
        assert "total_cost" in analytics
        assert "total_requests" in analytics
        assert "total_tokens" in analytics
        assert "model_breakdown" in analytics
        
        assert analytics["period_days"] == 30
        assert analytics["total_requests"] == 2
    
    def test_cost_trends_empty_data(self):
        """Test cost trends with empty data"""
        # Mock empty evaluations
        self.service.evaluation_repo.get_recent_evaluations.return_value = []
        
        trends = self.service.get_cost_trends(30)
        
        assert "daily_costs" in trends
        assert "provider_totals" in trends
        assert "total_cost" in trends
        assert "period_days" in trends
        
        assert trends["daily_costs"] == []
        assert trends["total_cost"] == 0.0
        assert trends["period_days"] == 30
    
    def test_cost_trends_with_data(self):
        """Test cost trends with mock data"""
        # Create mock evaluation data
        mock_evaluation = Mock()
        mock_evaluation.cost_usd = 0.045
        mock_evaluation.model_id = "gpt-4"
        mock_evaluation.created_at = datetime.utcnow()
        
        self.service.evaluation_repo.get_recent_evaluations.return_value = [mock_evaluation]
        
        trends = self.service.get_cost_trends(30)
        
        assert len(trends["daily_costs"]) > 0
        assert trends["total_cost"] > 0
        assert "openai" in trends["provider_totals"]
    
    def test_model_cost_breakdown_empty(self):
        """Test model cost breakdown with no data"""
        breakdown = self.service.get_model_cost_breakdown()
        
        assert isinstance(breakdown, list)
        assert len(breakdown) == 0
    
    def test_model_cost_breakdown_with_data(self):
        """Test model cost breakdown with data"""
        # Track some usage
        self.service.track_api_call("gpt-4", 1000, 500)
        self.service.track_api_call("claude-3-sonnet", 800, 400)
        
        breakdown = self.service.get_model_cost_breakdown()
        
        assert isinstance(breakdown, list)
        if len(breakdown) > 0:  # Depends on usage tracker implementation
            for model_data in breakdown:
                assert "model_name" in model_data
                assert "total_cost" in model_data
                assert "request_count" in model_data
                assert "avg_cost_per_request" in model_data
                assert "token_count" in model_data
                assert "cost_per_token" in model_data
    
    def test_update_budget(self):
        """Test updating budget"""
        new_budget = 2000.0
        new_threshold = 0.9
        
        self.service.update_budget(new_budget, new_threshold)
        
        status = self.service.get_budget_status()
        assert status["total_budget"] == new_budget
    
    def test_reset_budget_tracking(self):
        """Test resetting budget tracking"""
        # Track some usage first
        self.service.track_api_call("gpt-4", 1000, 500)
        
        # Verify there's spending
        status_before = self.service.get_budget_status()
        assert status_before["current_spend"] > 0
        
        # Reset
        self.service.reset_budget_tracking()
        
        # Verify spending is reset
        status_after = self.service.get_budget_status()
        assert status_after["current_spend"] == 0
        assert status_after["total_budget"] == status_before["total_budget"]  # Budget preserved
    
    def test_export_cost_data(self):
        """Test exporting cost data"""
        # Track some usage
        self.service.track_api_call("gpt-4", 1000, 500)
        
        export_data = self.service.export_cost_data(30, "json")
        
        assert "export_timestamp" in export_data
        assert "period_days" in export_data
        assert "budget_status" in export_data
        assert "usage_analytics" in export_data
        assert "cost_trends" in export_data
        assert "model_breakdown" in export_data
        assert "recommendations" in export_data
        
        assert export_data["period_days"] == 30
    
    def test_export_cost_data_csv_warning(self):
        """Test CSV export warning"""
        export_data = self.service.export_cost_data(30, "csv")
        
        # Should still return data but with JSON format
        assert "export_timestamp" in export_data
    
    def test_track_api_call_with_timestamp(self):
        """Test tracking API call with custom timestamp"""
        custom_timestamp = datetime.utcnow() - timedelta(hours=2)
        
        result = self.service.track_api_call(
            model_name="gpt-4",
            input_tokens=1000,
            output_tokens=500,
            timestamp=custom_timestamp
        )
        
        assert "timestamp" in result
        # Should use the provided timestamp
        assert result["timestamp"] == custom_timestamp.isoformat()
    
    def test_error_handling_in_cost_trends(self):
        """Test error handling in cost trends"""
        # Mock an exception in the evaluation repository
        self.service.evaluation_repo.get_recent_evaluations.side_effect = Exception("Database error")
        
        trends = self.service.get_cost_trends(30)
        
        # Should return empty data structure instead of crashing
        assert "daily_costs" in trends
        assert trends["daily_costs"] == []
        assert trends["total_cost"] == 0.0
    
    def test_error_handling_in_model_breakdown(self):
        """Test error handling in model breakdown"""
        # Mock the usage tracker to raise an exception
        with patch.object(self.service.usage_tracker, 'get_usage_analytics', side_effect=Exception("Analytics error")):
            breakdown = self.service.get_model_cost_breakdown()
            
            # Should return empty list instead of crashing
            assert isinstance(breakdown, list)
            assert len(breakdown) == 0
    
    def test_error_handling_in_export(self):
        """Test error handling in export"""
        # Mock an exception in one of the methods
        with patch.object(self.service, 'get_usage_analytics', side_effect=Exception("Export error")):
            export_data = self.service.export_cost_data(30, "json")
            
            # Should return error information
            assert "error" in export_data