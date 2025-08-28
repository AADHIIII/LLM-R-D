"""
Cost calculation module for different API providers and models.
Handles token counting, cost estimation, and budget tracking.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    FINE_TUNED = "fine_tuned"


@dataclass
class CostConfig:
    """Configuration for model pricing"""
    model_name: str
    provider: ModelProvider
    input_cost_per_token: float  # Cost per input token in USD
    output_cost_per_token: float  # Cost per output token in USD
    base_cost: float = 0.0  # Base cost per request
    
    
@dataclass
class UsageMetrics:
    """Usage metrics for cost calculation"""
    input_tokens: int
    output_tokens: int
    request_count: int = 1
    latency_ms: int = 0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class CostBreakdown:
    """Detailed cost breakdown"""
    input_cost: float
    output_cost: float
    base_cost: float
    total_cost: float
    cost_per_token: float
    cost_per_request: float
    provider: ModelProvider
    model_name: str


class CostCalculator:
    """Main cost calculation engine"""
    
    # Default pricing configurations (as of 2024)
    DEFAULT_PRICING = {
        # OpenAI GPT-4 pricing
        "gpt-4": CostConfig(
            model_name="gpt-4",
            provider=ModelProvider.OPENAI,
            input_cost_per_token=0.00003,  # $0.03 per 1K tokens
            output_cost_per_token=0.00006,  # $0.06 per 1K tokens
        ),
        "gpt-4-turbo": CostConfig(
            model_name="gpt-4-turbo",
            provider=ModelProvider.OPENAI,
            input_cost_per_token=0.00001,  # $0.01 per 1K tokens
            output_cost_per_token=0.00003,  # $0.03 per 1K tokens
        ),
        "gpt-3.5-turbo": CostConfig(
            model_name="gpt-3.5-turbo",
            provider=ModelProvider.OPENAI,
            input_cost_per_token=0.0000015,  # $0.0015 per 1K tokens
            output_cost_per_token=0.000002,  # $0.002 per 1K tokens
        ),
        
        # Anthropic Claude pricing
        "claude-3-opus": CostConfig(
            model_name="claude-3-opus",
            provider=ModelProvider.ANTHROPIC,
            input_cost_per_token=0.000015,  # $0.015 per 1K tokens
            output_cost_per_token=0.000075,  # $0.075 per 1K tokens
        ),
        "claude-3-sonnet": CostConfig(
            model_name="claude-3-sonnet",
            provider=ModelProvider.ANTHROPIC,
            input_cost_per_token=0.000003,  # $0.003 per 1K tokens
            output_cost_per_token=0.000015,  # $0.015 per 1K tokens
        ),
        "claude-3-haiku": CostConfig(
            model_name="claude-3-haiku",
            provider=ModelProvider.ANTHROPIC,
            input_cost_per_token=0.00000025,  # $0.00025 per 1K tokens
            output_cost_per_token=0.00000125,  # $0.00125 per 1K tokens
        ),
        
        # Fine-tuned models (estimated costs)
        "fine-tuned-gpt2": CostConfig(
            model_name="fine-tuned-gpt2",
            provider=ModelProvider.FINE_TUNED,
            input_cost_per_token=0.0000001,  # Very low cost for self-hosted
            output_cost_per_token=0.0000001,
            base_cost=0.001,  # Small compute cost per request
        ),
    }
    
    def __init__(self, custom_pricing: Optional[Dict[str, CostConfig]] = None):
        """Initialize cost calculator with optional custom pricing"""
        self.pricing = self.DEFAULT_PRICING.copy()
        if custom_pricing:
            self.pricing.update(custom_pricing)
    
    def calculate_cost(self, model_name: str, usage: UsageMetrics) -> CostBreakdown:
        """Calculate cost for a specific model and usage"""
        if model_name not in self.pricing:
            logger.warning(f"Unknown model {model_name}, using default pricing")
            # Use a default configuration for unknown models
            config = CostConfig(
                model_name=model_name,
                provider=ModelProvider.OPENAI,
                input_cost_per_token=0.00001,
                output_cost_per_token=0.00002,
            )
        else:
            config = self.pricing[model_name]
        
        input_cost = usage.input_tokens * config.input_cost_per_token
        output_cost = usage.output_tokens * config.output_cost_per_token
        base_cost = config.base_cost * usage.request_count
        total_cost = input_cost + output_cost + base_cost
        
        total_tokens = usage.input_tokens + usage.output_tokens
        cost_per_token = total_cost / total_tokens if total_tokens > 0 else 0
        cost_per_request = total_cost / usage.request_count if usage.request_count > 0 else 0
        
        return CostBreakdown(
            input_cost=input_cost,
            output_cost=output_cost,
            base_cost=base_cost,
            total_cost=total_cost,
            cost_per_token=cost_per_token,
            cost_per_request=cost_per_request,
            provider=config.provider,
            model_name=model_name,
        )
    
    def estimate_cost(self, model_name: str, estimated_tokens: int) -> float:
        """Estimate cost for a given number of tokens (assuming 50/50 input/output split)"""
        input_tokens = estimated_tokens // 2
        output_tokens = estimated_tokens - input_tokens
        
        usage = UsageMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        
        breakdown = self.calculate_cost(model_name, usage)
        return breakdown.total_cost
    
    def compare_model_costs(self, models: List[str], usage: UsageMetrics) -> Dict[str, CostBreakdown]:
        """Compare costs across multiple models for the same usage"""
        comparisons = {}
        for model in models:
            comparisons[model] = self.calculate_cost(model, usage)
        return comparisons
    
    def get_cheapest_model(self, models: List[str], usage: UsageMetrics) -> Tuple[str, CostBreakdown]:
        """Find the cheapest model for given usage"""
        comparisons = self.compare_model_costs(models, usage)
        cheapest_model = min(comparisons.keys(), key=lambda m: comparisons[m].total_cost)
        return cheapest_model, comparisons[cheapest_model]


class BudgetTracker:
    """Budget tracking and alerting system"""
    
    def __init__(self, total_budget: float, alert_threshold: float = 0.8):
        """Initialize budget tracker
        
        Args:
            total_budget: Total budget in USD
            alert_threshold: Threshold for budget alerts (0.0-1.0)
        """
        self.total_budget = total_budget
        self.alert_threshold = alert_threshold
        self.current_spend = 0.0
        self.daily_spend = {}  # Track daily spending
        self.model_spend = {}  # Track spending by model
        
    def add_cost(self, cost: float, model_name: str, timestamp: Optional[datetime] = None):
        """Add cost to budget tracking"""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        self.current_spend += cost
        
        # Track daily spending
        date_key = timestamp.date().isoformat()
        if date_key not in self.daily_spend:
            self.daily_spend[date_key] = 0.0
        self.daily_spend[date_key] += cost
        
        # Track model spending
        if model_name not in self.model_spend:
            self.model_spend[model_name] = 0.0
        self.model_spend[model_name] += cost
    
    def get_budget_status(self) -> Dict:
        """Get current budget status"""
        utilization = self.current_spend / self.total_budget if self.total_budget > 0 else float('inf')
        remaining = max(0, self.total_budget - self.current_spend)
        
        return {
            "total_budget": self.total_budget,
            "current_spend": self.current_spend,
            "remaining": remaining,
            "utilization": utilization,
            "is_over_budget": utilization > 1.0,
            "is_near_budget": utilization > self.alert_threshold,
            "daily_spend": self.daily_spend,
            "model_spend": self.model_spend,
        }
    
    def check_budget_alerts(self) -> List[Dict]:
        """Check for budget alerts"""
        alerts = []
        status = self.get_budget_status()
        
        if status["is_over_budget"]:
            alerts.append({
                "type": "budget_exceeded",
                "message": f"Budget exceeded by ${status['current_spend'] - self.total_budget:.2f}",
                "severity": "critical",
                "current_spend": status["current_spend"],
                "total_budget": self.total_budget,
            })
        elif status["is_near_budget"]:
            alerts.append({
                "type": "budget_warning",
                "message": f"Budget utilization at {status['utilization']*100:.1f}%",
                "severity": "warning",
                "current_spend": status["current_spend"],
                "total_budget": self.total_budget,
            })
        
        # Check for unusual daily spending
        if self.daily_spend:
            recent_days = sorted(self.daily_spend.keys())[-7:]  # Last 7 days
            if len(recent_days) >= 2:
                avg_daily = sum(self.daily_spend[day] for day in recent_days) / len(recent_days)
                today = datetime.utcnow().date().isoformat()
                if today in self.daily_spend and self.daily_spend[today] > avg_daily * 2:
                    alerts.append({
                        "type": "unusual_spending",
                        "message": f"Today's spending (${self.daily_spend[today]:.2f}) is unusually high",
                        "severity": "info",
                        "daily_spend": self.daily_spend[today],
                        "avg_daily": avg_daily,
                    })
        
        return alerts
    
    def get_cost_optimization_recommendations(self) -> List[Dict]:
        """Generate cost optimization recommendations"""
        recommendations = []
        status = self.get_budget_status()
        
        if not self.model_spend:
            return recommendations
        
        # Find most expensive models
        sorted_models = sorted(self.model_spend.items(), key=lambda x: x[1], reverse=True)
        most_expensive = sorted_models[0]
        
        if most_expensive[1] > status["current_spend"] * 0.5:
            recommendations.append({
                "type": "expensive_model",
                "message": f"Model '{most_expensive[0]}' accounts for ${most_expensive[1]:.2f} of spending",
                "suggestion": "Consider using a cheaper alternative model for non-critical tasks",
                "model": most_expensive[0],
                "cost": most_expensive[1],
            })
        
        # Check for fine-tuning opportunities
        openai_spend = sum(cost for model, cost in self.model_spend.items() 
                          if any(openai_model in model.lower() for openai_model in ['gpt-4', 'gpt-3.5']))
        
        if openai_spend > 50:  # If spending more than $50 on OpenAI
            recommendations.append({
                "type": "fine_tuning_opportunity",
                "message": f"High OpenAI usage (${openai_spend:.2f}) detected",
                "suggestion": "Consider fine-tuning a smaller model for repetitive tasks",
                "potential_savings": openai_spend * 0.7,  # Estimated 70% savings
            })
        
        return recommendations


class UsageTracker:
    """Track API usage and generate analytics"""
    
    def __init__(self):
        self.usage_history = []
        self.model_stats = {}
    
    def record_usage(self, model_name: str, usage: UsageMetrics, cost: float):
        """Record usage for analytics"""
        record = {
            "model_name": model_name,
            "usage": usage,
            "cost": cost,
            "timestamp": usage.timestamp,
        }
        self.usage_history.append(record)
        
        # Update model statistics
        if model_name not in self.model_stats:
            self.model_stats[model_name] = {
                "total_requests": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "avg_latency": 0.0,
                "latency_samples": [],
            }
        
        stats = self.model_stats[model_name]
        stats["total_requests"] += usage.request_count
        stats["total_tokens"] += usage.input_tokens + usage.output_tokens
        stats["total_cost"] += cost
        
        if usage.latency_ms > 0:
            stats["latency_samples"].append(usage.latency_ms)
            stats["avg_latency"] = sum(stats["latency_samples"]) / len(stats["latency_samples"])
    
    def get_usage_analytics(self, days: int = 30) -> Dict:
        """Get usage analytics for the specified number of days"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_usage = [r for r in self.usage_history if r["timestamp"] >= cutoff_date]
        
        if not recent_usage:
            return {"error": "No usage data available"}
        
        # Calculate totals
        total_cost = sum(r["cost"] for r in recent_usage)
        total_requests = sum(r["usage"].request_count for r in recent_usage)
        total_tokens = sum(r["usage"].input_tokens + r["usage"].output_tokens for r in recent_usage)
        
        # Group by model
        model_breakdown = {}
        for record in recent_usage:
            model = record["model_name"]
            if model not in model_breakdown:
                model_breakdown[model] = {
                    "requests": 0,
                    "tokens": 0,
                    "cost": 0.0,
                    "avg_latency": 0.0,
                    "latencies": [],
                }
            
            breakdown = model_breakdown[model]
            breakdown["requests"] += record["usage"].request_count
            breakdown["tokens"] += record["usage"].input_tokens + record["usage"].output_tokens
            breakdown["cost"] += record["cost"]
            
            if record["usage"].latency_ms > 0:
                breakdown["latencies"].append(record["usage"].latency_ms)
        
        # Calculate averages
        for model_data in model_breakdown.values():
            if model_data["latencies"]:
                model_data["avg_latency"] = sum(model_data["latencies"]) / len(model_data["latencies"])
            model_data["avg_cost_per_request"] = model_data["cost"] / model_data["requests"] if model_data["requests"] > 0 else 0
            model_data["cost_per_token"] = model_data["cost"] / model_data["tokens"] if model_data["tokens"] > 0 else 0
            del model_data["latencies"]  # Remove raw latency data
        
        return {
            "period_days": days,
            "total_cost": total_cost,
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "avg_cost_per_request": total_cost / total_requests if total_requests > 0 else 0,
            "avg_cost_per_token": total_cost / total_tokens if total_tokens > 0 else 0,
            "model_breakdown": model_breakdown,
        }