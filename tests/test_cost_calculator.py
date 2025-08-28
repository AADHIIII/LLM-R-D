"""
Tests for cost calculation functionality.
"""

import pytest
from datetime import datetime, timedelta
from utils.cost_calculator import (
    CostCalculator, BudgetTracker, UsageTracker, UsageMetrics, 
    CostConfig, ModelProvider
)


class TestCostCalculator:
    """Test cost calculation functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.calculator = CostCalculator()
    
    def test_calculate_cost_gpt4(self):
        """Test cost calculation for GPT-4"""
        usage = UsageMetrics(
            input_tokens=1000,
            output_tokens=500,
            request_count=1
        )
        
        breakdown = self.calculator.calculate_cost("gpt-4", usage)
        
        assert breakdown.model_name == "gpt-4"
        assert breakdown.provider == ModelProvider.OPENAI
        assert breakdown.input_cost == 1000 * 0.00003  # $0.03 per 1K tokens
        assert breakdown.output_cost == 500 * 0.00006  # $0.06 per 1K tokens
        assert breakdown.total_cost == breakdown.input_cost + breakdown.output_cost
        assert breakdown.cost_per_token > 0
        assert breakdown.cost_per_request == breakdown.total_cost
    
    def test_calculate_cost_claude(self):
        """Test cost calculation for Claude"""
        usage = UsageMetrics(
            input_tokens=1000,
            output_tokens=500,
            request_count=1
        )
        
        breakdown = self.calculator.calculate_cost("claude-3-sonnet", usage)
        
        assert breakdown.model_name == "claude-3-sonnet"
        assert breakdown.provider == ModelProvider.ANTHROPIC
        assert breakdown.input_cost == 1000 * 0.000003
        assert breakdown.output_cost == 500 * 0.000015
        assert breakdown.total_cost == breakdown.input_cost + breakdown.output_cost
    
    def test_calculate_cost_fine_tuned(self):
        """Test cost calculation for fine-tuned model"""
        usage = UsageMetrics(
            input_tokens=1000,
            output_tokens=500,
            request_count=1
        )
        
        breakdown = self.calculator.calculate_cost("fine-tuned-gpt2", usage)
        
        assert breakdown.model_name == "fine-tuned-gpt2"
        assert breakdown.provider == ModelProvider.FINE_TUNED
        assert breakdown.base_cost == 0.001  # Base compute cost
        assert breakdown.total_cost > breakdown.base_cost
    
    def test_calculate_cost_unknown_model(self):
        """Test cost calculation for unknown model"""
        usage = UsageMetrics(
            input_tokens=1000,
            output_tokens=500,
            request_count=1
        )
        
        breakdown = self.calculator.calculate_cost("unknown-model", usage)
        
        assert breakdown.model_name == "unknown-model"
        assert breakdown.total_cost > 0  # Should use default pricing
    
    def test_estimate_cost(self):
        """Test cost estimation"""
        estimated_cost = self.calculator.estimate_cost("gpt-4", 1000)
        
        assert estimated_cost > 0
        # Should be roughly half input, half output tokens
        expected_cost = (500 * 0.00003) + (500 * 0.00006)
        assert abs(estimated_cost - expected_cost) < 0.001
    
    def test_compare_model_costs(self):
        """Test model cost comparison"""
        usage = UsageMetrics(
            input_tokens=1000,
            output_tokens=500,
            request_count=1
        )
        
        models = ["gpt-4", "claude-3-sonnet", "fine-tuned-gpt2"]
        comparisons = self.calculator.compare_model_costs(models, usage)
        
        assert len(comparisons) == 3
        assert all(model in comparisons for model in models)
        assert all(breakdown.total_cost > 0 for breakdown in comparisons.values())
        
        # Fine-tuned should be cheapest
        assert comparisons["fine-tuned-gpt2"].total_cost < comparisons["gpt-4"].total_cost
    
    def test_get_cheapest_model(self):
        """Test finding cheapest model"""
        usage = UsageMetrics(
            input_tokens=1000,
            output_tokens=500,
            request_count=1
        )
        
        models = ["gpt-4", "claude-3-sonnet", "fine-tuned-gpt2"]
        cheapest_model, breakdown = self.calculator.get_cheapest_model(models, usage)
        
        assert cheapest_model == "fine-tuned-gpt2"
        assert breakdown.model_name == "fine-tuned-gpt2"
    
    def test_custom_pricing(self):
        """Test custom pricing configuration"""
        custom_pricing = {
            "custom-model": CostConfig(
                model_name="custom-model",
                provider=ModelProvider.OPENAI,
                input_cost_per_token=0.00001,
                output_cost_per_token=0.00002,
            )
        }
        
        calculator = CostCalculator(custom_pricing)
        
        usage = UsageMetrics(
            input_tokens=1000,
            output_tokens=500,
            request_count=1
        )
        
        breakdown = calculator.calculate_cost("custom-model", usage)
        
        assert breakdown.model_name == "custom-model"
        assert breakdown.input_cost == 1000 * 0.00001
        assert breakdown.output_cost == 500 * 0.00002


class TestBudgetTracker:
    """Test budget tracking functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.tracker = BudgetTracker(total_budget=100.0, alert_threshold=0.8)
    
    def test_initial_budget_status(self):
        """Test initial budget status"""
        status = self.tracker.get_budget_status()
        
        assert status["total_budget"] == 100.0
        assert status["current_spend"] == 0.0
        assert status["remaining"] == 100.0
        assert status["utilization"] == 0.0
        assert not status["is_over_budget"]
        assert not status["is_near_budget"]
    
    def test_add_cost(self):
        """Test adding costs to budget"""
        self.tracker.add_cost(25.0, "gpt-4")
        
        status = self.tracker.get_budget_status()
        
        assert status["current_spend"] == 25.0
        assert status["remaining"] == 75.0
        assert status["utilization"] == 0.25
        assert "gpt-4" in status["model_spend"]
        assert status["model_spend"]["gpt-4"] == 25.0
    
    def test_budget_warning(self):
        """Test budget warning alert"""
        self.tracker.add_cost(85.0, "gpt-4")  # 85% of budget
        
        alerts = self.tracker.check_budget_alerts()
        
        assert len(alerts) > 0
        assert any(alert["type"] == "budget_warning" for alert in alerts)
        
        status = self.tracker.get_budget_status()
        assert status["is_near_budget"]
        assert not status["is_over_budget"]
    
    def test_budget_exceeded(self):
        """Test budget exceeded alert"""
        self.tracker.add_cost(120.0, "gpt-4")  # Over budget
        
        alerts = self.tracker.check_budget_alerts()
        
        assert len(alerts) > 0
        assert any(alert["type"] == "budget_exceeded" for alert in alerts)
        
        status = self.tracker.get_budget_status()
        assert status["is_over_budget"]
        assert status["remaining"] == 0  # Should not go negative
    
    def test_daily_spending_tracking(self):
        """Test daily spending tracking"""
        today = datetime.utcnow()
        yesterday = today - timedelta(days=1)
        
        self.tracker.add_cost(30.0, "gpt-4", today)
        self.tracker.add_cost(20.0, "claude-3", yesterday)
        
        status = self.tracker.get_budget_status()
        
        assert len(status["daily_spend"]) == 2
        assert status["daily_spend"][today.date().isoformat()] == 30.0
        assert status["daily_spend"][yesterday.date().isoformat()] == 20.0
    
    def test_model_spending_tracking(self):
        """Test model spending tracking"""
        self.tracker.add_cost(30.0, "gpt-4")
        self.tracker.add_cost(20.0, "claude-3")
        self.tracker.add_cost(10.0, "gpt-4")  # Additional cost for same model
        
        status = self.tracker.get_budget_status()
        
        assert status["model_spend"]["gpt-4"] == 40.0
        assert status["model_spend"]["claude-3"] == 20.0
    
    def test_cost_optimization_recommendations(self):
        """Test cost optimization recommendations"""
        # Add significant spending on expensive model
        self.tracker.add_cost(60.0, "gpt-4")
        self.tracker.add_cost(5.0, "fine-tuned-gpt2")
        
        recommendations = self.tracker.get_cost_optimization_recommendations()
        
        assert len(recommendations) > 0
        assert any(rec["type"] == "expensive_model" for rec in recommendations)
    
    def test_zero_budget_handling(self):
        """Test handling of zero budget"""
        tracker = BudgetTracker(total_budget=0.0)
        tracker.add_cost(10.0, "gpt-4")
        
        status = tracker.get_budget_status()
        
        assert status["utilization"] == float('inf')
        assert status["is_over_budget"]


class TestUsageTracker:
    """Test usage tracking functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.tracker = UsageTracker()
    
    def test_record_usage(self):
        """Test recording usage"""
        usage = UsageMetrics(
            input_tokens=1000,
            output_tokens=500,
            request_count=1,
            latency_ms=1200
        )
        
        self.tracker.record_usage("gpt-4", usage, 0.045)
        
        assert len(self.tracker.usage_history) == 1
        assert "gpt-4" in self.tracker.model_stats
        
        stats = self.tracker.model_stats["gpt-4"]
        assert stats["total_requests"] == 1
        assert stats["total_tokens"] == 1500
        assert stats["total_cost"] == 0.045
        assert stats["avg_latency"] == 1200
    
    def test_multiple_usage_records(self):
        """Test multiple usage records"""
        usage1 = UsageMetrics(input_tokens=1000, output_tokens=500, latency_ms=1200)
        usage2 = UsageMetrics(input_tokens=800, output_tokens=400, latency_ms=800)
        
        self.tracker.record_usage("gpt-4", usage1, 0.045)
        self.tracker.record_usage("gpt-4", usage2, 0.036)
        
        stats = self.tracker.model_stats["gpt-4"]
        assert stats["total_requests"] == 2
        assert stats["total_tokens"] == 2700  # 1500 + 1200
        assert abs(stats["total_cost"] - 0.081) < 0.001  # 0.045 + 0.036
        assert stats["avg_latency"] == 1000  # (1200 + 800) / 2
    
    def test_get_usage_analytics(self):
        """Test getting usage analytics"""
        # Add some usage data
        usage = UsageMetrics(input_tokens=1000, output_tokens=500)
        self.tracker.record_usage("gpt-4", usage, 0.045)
        self.tracker.record_usage("claude-3", usage, 0.038)
        
        analytics = self.tracker.get_usage_analytics(30)
        
        assert "total_cost" in analytics
        assert "total_requests" in analytics
        assert "total_tokens" in analytics
        assert "model_breakdown" in analytics
        
        assert abs(analytics["total_cost"] - 0.083) < 0.001
        assert analytics["total_requests"] == 2
        assert analytics["total_tokens"] == 3000
        
        assert "gpt-4" in analytics["model_breakdown"]
        assert "claude-3" in analytics["model_breakdown"]
    
    def test_empty_analytics(self):
        """Test analytics with no data"""
        analytics = self.tracker.get_usage_analytics(30)
        
        assert "error" in analytics
        assert analytics["error"] == "No usage data available"
    
    def test_date_filtering(self):
        """Test date filtering in analytics"""
        # Add old usage (should be filtered out)
        old_usage = UsageMetrics(
            input_tokens=1000, 
            output_tokens=500,
            timestamp=datetime.utcnow() - timedelta(days=45)
        )
        
        # Add recent usage (should be included)
        recent_usage = UsageMetrics(
            input_tokens=800,
            output_tokens=400,
            timestamp=datetime.utcnow() - timedelta(days=15)
        )
        
        self.tracker.record_usage("gpt-4", old_usage, 0.045)
        self.tracker.record_usage("gpt-4", recent_usage, 0.036)
        
        analytics = self.tracker.get_usage_analytics(30)  # Last 30 days
        
        # Should only include recent usage
        assert analytics["total_requests"] == 1
        assert analytics["total_cost"] == 0.036