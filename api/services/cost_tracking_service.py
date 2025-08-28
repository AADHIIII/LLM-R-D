"""
Cost tracking service for API requests.
Integrates with the cost calculator and database to track usage and costs.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import asdict

from utils.cost_calculator import CostCalculator, BudgetTracker, UsageTracker, UsageMetrics
from database.repositories import EvaluationRepository, ExperimentRepository
from database.connection import db_manager

logger = logging.getLogger(__name__)


class CostTrackingService:
    """Service for tracking costs and usage across the platform"""
    
    def __init__(self, total_budget: float = 1000.0, alert_threshold: float = 0.8):
        """Initialize cost tracking service
        
        Args:
            total_budget: Total budget in USD
            alert_threshold: Budget alert threshold (0.0-1.0)
        """
        self.cost_calculator = CostCalculator()
        self.budget_tracker = BudgetTracker(total_budget, alert_threshold)
        self.usage_tracker = UsageTracker()
        
        # Initialize repositories
        self.db_manager = db_manager
        self.evaluation_repo = EvaluationRepository()
        self.experiment_repo = ExperimentRepository()
        
        # Load existing usage data
        self._load_historical_data()
    
    def _load_historical_data(self):
        """Load historical usage data from database"""
        try:
            # Load recent evaluations to rebuild cost tracking
            recent_evaluations = self.evaluation_repo.get_recent_evaluations(days=30)
            
            if not recent_evaluations:
                logger.info("No historical evaluations found for cost tracking")
                return
            
            for evaluation in recent_evaluations:
                if hasattr(evaluation, 'cost_usd') and evaluation.cost_usd and evaluation.cost_usd > 0:
                    # Reconstruct usage metrics from evaluation data
                    usage = UsageMetrics(
                        input_tokens=getattr(evaluation, 'input_tokens', 0),
                        output_tokens=getattr(evaluation, 'output_tokens', 0),
                        request_count=1,
                        latency_ms=getattr(evaluation, 'latency_ms', 0),
                        timestamp=evaluation.created_at,
                    )
                    
                    # Add to trackers
                    model_id = str(evaluation.model_id) if hasattr(evaluation, 'model_id') else 'unknown'
                    self.budget_tracker.add_cost(
                        evaluation.cost_usd, 
                        model_id, 
                        evaluation.created_at
                    )
                    self.usage_tracker.record_usage(
                        model_id, 
                        usage, 
                        evaluation.cost_usd
                    )
            
            logger.info(f"Loaded {len(recent_evaluations)} historical evaluations for cost tracking")
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    def track_api_call(self, 
                      model_name: str, 
                      input_tokens: int, 
                      output_tokens: int, 
                      latency_ms: int = 0,
                      timestamp: Optional[datetime] = None) -> Dict:
        """Track an API call and calculate costs
        
        Args:
            model_name: Name of the model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            latency_ms: Response latency in milliseconds
            timestamp: Timestamp of the call
            
        Returns:
            Dictionary with cost breakdown and tracking info
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Create usage metrics
        usage = UsageMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            request_count=1,
            latency_ms=latency_ms,
            timestamp=timestamp,
        )
        
        # Calculate cost
        cost_breakdown = self.cost_calculator.calculate_cost(model_name, usage)
        
        # Track in budget and usage trackers
        self.budget_tracker.add_cost(cost_breakdown.total_cost, model_name, timestamp)
        self.usage_tracker.record_usage(model_name, usage, cost_breakdown.total_cost)
        
        # Check for alerts
        alerts = self.budget_tracker.check_budget_alerts()
        
        return {
            "cost_breakdown": asdict(cost_breakdown),
            "budget_status": self.budget_tracker.get_budget_status(),
            "alerts": alerts,
            "timestamp": timestamp.isoformat(),
        }
    
    def estimate_request_cost(self, model_name: str, estimated_tokens: int) -> float:
        """Estimate cost for a request before making it"""
        return self.cost_calculator.estimate_cost(model_name, estimated_tokens)
    
    def compare_model_costs(self, models: List[str], input_tokens: int, output_tokens: int) -> Dict:
        """Compare costs across multiple models for the same usage"""
        usage = UsageMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        
        comparisons = self.cost_calculator.compare_model_costs(models, usage)
        
        # Convert to serializable format
        result = {}
        for model, breakdown in comparisons.items():
            result[model] = asdict(breakdown)
        
        return result
    
    def get_cheapest_model(self, models: List[str], input_tokens: int, output_tokens: int) -> Dict:
        """Find the cheapest model for given usage"""
        usage = UsageMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        
        cheapest_model, breakdown = self.cost_calculator.get_cheapest_model(models, usage)
        
        return {
            "model": cheapest_model,
            "cost_breakdown": asdict(breakdown),
        }
    
    def get_budget_status(self) -> Dict:
        """Get current budget status"""
        return self.budget_tracker.get_budget_status()
    
    def get_budget_alerts(self) -> List[Dict]:
        """Get current budget alerts"""
        return self.budget_tracker.check_budget_alerts()
    
    def get_cost_optimization_recommendations(self) -> List[Dict]:
        """Get cost optimization recommendations"""
        return self.budget_tracker.get_cost_optimization_recommendations()
    
    def get_usage_analytics(self, days: int = 30) -> Dict:
        """Get usage analytics for specified period"""
        return self.usage_tracker.get_usage_analytics(days)
    
    def get_cost_trends(self, days: int = 30) -> Dict:
        """Get cost trends over time"""
        try:
            # Get evaluations from database
            evaluations = self.evaluation_repo.get_recent_evaluations(days=days)
            
            # Group by date and provider
            daily_costs = {}
            provider_costs = {"openai": 0.0, "anthropic": 0.0, "fine_tuned": 0.0}
            
            if not evaluations:
                return {
                    "daily_costs": [],
                    "provider_totals": provider_costs,
                    "total_cost": 0.0,
                    "period_days": days,
                }
            
            for evaluation in evaluations:
                if not hasattr(evaluation, 'cost_usd') or not evaluation.cost_usd or evaluation.cost_usd <= 0:
                    continue
                
                date_key = evaluation.created_at.date().isoformat()
                if date_key not in daily_costs:
                    daily_costs[date_key] = {
                        "date": date_key,
                        "openai_cost": 0.0,
                        "anthropic_cost": 0.0,
                        "fine_tuned_cost": 0.0,
                        "total_cost": 0.0,
                        "request_count": 0,
                    }
                
                # Determine provider based on model name
                model_name = str(evaluation.model_id).lower() if hasattr(evaluation, 'model_id') else 'unknown'
                if "gpt" in model_name or "openai" in model_name:
                    daily_costs[date_key]["openai_cost"] += evaluation.cost_usd
                    provider_costs["openai"] += evaluation.cost_usd
                elif "claude" in model_name or "anthropic" in model_name:
                    daily_costs[date_key]["anthropic_cost"] += evaluation.cost_usd
                    provider_costs["anthropic"] += evaluation.cost_usd
                else:
                    daily_costs[date_key]["fine_tuned_cost"] += evaluation.cost_usd
                    provider_costs["fine_tuned"] += evaluation.cost_usd
                
                daily_costs[date_key]["total_cost"] += evaluation.cost_usd
                daily_costs[date_key]["request_count"] += 1
            
            # Convert to list and sort by date
            cost_trends = sorted(daily_costs.values(), key=lambda x: x["date"])
            
            return {
                "daily_costs": cost_trends,
                "provider_totals": provider_costs,
                "total_cost": sum(provider_costs.values()),
                "period_days": days,
            }
            
        except Exception as e:
            logger.error(f"Error getting cost trends: {e}")
            return {
                "daily_costs": [],
                "provider_totals": {"openai": 0.0, "anthropic": 0.0, "fine_tuned": 0.0},
                "total_cost": 0.0,
                "period_days": days,
            }
    
    def get_model_cost_breakdown(self) -> List[Dict]:
        """Get detailed cost breakdown by model"""
        try:
            analytics = self.usage_tracker.get_usage_analytics(30)
            
            if "model_breakdown" not in analytics:
                return []
            
            breakdown = []
            for model_name, stats in analytics["model_breakdown"].items():
                breakdown.append({
                    "model_name": model_name,
                    "total_cost": stats["cost"],
                    "request_count": stats["requests"],
                    "avg_cost_per_request": stats["avg_cost_per_request"],
                    "token_count": stats["tokens"],
                    "cost_per_token": stats["cost_per_token"],
                    "avg_latency_ms": stats["avg_latency"],
                })
            
            # Sort by total cost descending
            breakdown.sort(key=lambda x: x["total_cost"], reverse=True)
            return breakdown
            
        except Exception as e:
            logger.error(f"Error getting model cost breakdown: {e}")
            return []
    
    def update_budget(self, new_budget: float, alert_threshold: float = None):
        """Update budget settings"""
        self.budget_tracker.total_budget = new_budget
        if alert_threshold is not None:
            self.budget_tracker.alert_threshold = alert_threshold
        
        logger.info(f"Budget updated to ${new_budget:.2f}")
    
    def reset_budget_tracking(self):
        """Reset budget tracking (useful for new billing periods)"""
        current_budget = self.budget_tracker.total_budget
        current_threshold = self.budget_tracker.alert_threshold
        
        self.budget_tracker = BudgetTracker(current_budget, current_threshold)
        logger.info("Budget tracking reset")
    
    def export_cost_data(self, days: int = 30, format: str = "json") -> Dict:
        """Export cost data for external analysis
        
        Args:
            days: Number of days to export
            format: Export format ("json", "csv")
            
        Returns:
            Dictionary with exported data
        """
        try:
            # Get comprehensive data
            analytics = self.get_usage_analytics(days)
            trends = self.get_cost_trends(days)
            budget_status = self.get_budget_status()
            model_breakdown = self.get_model_cost_breakdown()
            
            export_data = {
                "export_timestamp": datetime.utcnow().isoformat(),
                "period_days": days,
                "budget_status": budget_status,
                "usage_analytics": analytics,
                "cost_trends": trends,
                "model_breakdown": model_breakdown,
                "recommendations": self.get_cost_optimization_recommendations(),
            }
            
            if format == "csv":
                # For CSV format, flatten the data structure
                # This would require additional processing
                logger.warning("CSV export not fully implemented, returning JSON")
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting cost data: {e}")
            return {"error": str(e)}