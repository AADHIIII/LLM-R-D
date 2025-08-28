"""
API endpoints for cost tracking and budget management.
"""

import logging
from flask import Blueprint, request, jsonify
from datetime import datetime
from typing import Dict, Any

from api.services.cost_tracking_service import CostTrackingService
from api.middleware.validation_middleware import validate_json_schema, validate_query_params

logger = logging.getLogger(__name__)

# Create blueprint
cost_tracking_bp = Blueprint('cost_tracking', __name__, url_prefix='/api/v1/cost')

# Initialize cost tracking service (in production, this would be dependency injected)
cost_service = CostTrackingService()


@cost_tracking_bp.route('/track', methods=['POST'])
@validate_json_schema({'model_name': str, 'input_tokens': int, 'output_tokens': int})
def track_api_call():
    """Track an API call and calculate costs"""
    try:
        data = request.get_json()
        
        result = cost_service.track_api_call(
            model_name=data['model_name'],
            input_tokens=data['input_tokens'],
            output_tokens=data['output_tokens'],
            latency_ms=data.get('latency_ms', 0),
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else None
        )
        
        return jsonify({
            "success": True,
            "data": result
        }), 200
        
    except Exception as e:
        logger.error(f"Error tracking API call: {e}")
        return jsonify({
            "success": False,
            "error": "tracking_error",
            "message": str(e)
        }), 500


@cost_tracking_bp.route('/estimate', methods=['POST'])
@validate_json_schema({'model_name': str, 'estimated_tokens': int})
def estimate_cost():
    """Estimate cost for a request before making it"""
    try:
        data = request.get_json()
        
        estimated_cost = cost_service.estimate_request_cost(
            model_name=data['model_name'],
            estimated_tokens=data['estimated_tokens']
        )
        
        return jsonify({
            "success": True,
            "data": {
                "model_name": data['model_name'],
                "estimated_tokens": data['estimated_tokens'],
                "estimated_cost_usd": estimated_cost
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error estimating cost: {e}")
        return jsonify({
            "success": False,
            "error": "estimation_error",
            "message": str(e)
        }), 500


@cost_tracking_bp.route('/compare', methods=['POST'])
@validate_json_schema({'models': list, 'input_tokens': int, 'output_tokens': int})
def compare_model_costs():
    """Compare costs across multiple models"""
    try:
        data = request.get_json()
        
        comparisons = cost_service.compare_model_costs(
            models=data['models'],
            input_tokens=data['input_tokens'],
            output_tokens=data['output_tokens']
        )
        
        return jsonify({
            "success": True,
            "data": {
                "comparisons": comparisons,
                "input_tokens": data['input_tokens'],
                "output_tokens": data['output_tokens']
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error comparing model costs: {e}")
        return jsonify({"success": False, "error": "comparison_error", "message": str(e)}), 500


@cost_tracking_bp.route('/cheapest', methods=['POST'])
@validate_json_schema({'models': list, 'input_tokens': int, 'output_tokens': int})
def get_cheapest_model():
    """Find the cheapest model for given usage"""
    try:
        data = request.get_json()
        
        result = cost_service.get_cheapest_model(
            models=data['models'],
            input_tokens=data['input_tokens'],
            output_tokens=data['output_tokens']
        )
        
        return jsonify({
            "success": True,
            "data": result
        }), 200
        
    except Exception as e:
        logger.error(f"Error finding cheapest model: {e}")
        return jsonify({"success": False, "error": "cheapest_error", "message": str(e)}), 500


@cost_tracking_bp.route('/budget/status', methods=['GET'])
def get_budget_status():
    """Get current budget status"""
    try:
        status = cost_service.get_budget_status()
        
        return jsonify({
            "success": True,
            "data": status
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting budget status: {e}")
        return jsonify({"success": False, "error": "budget_status_error", "message": str(e)}), 500


@cost_tracking_bp.route('/budget/alerts', methods=['GET'])
def get_budget_alerts():
    """Get current budget alerts"""
    try:
        alerts = cost_service.get_budget_alerts()
        
        return jsonify({
            "success": True,
            "data": {
                "alerts": alerts,
                "alert_count": len(alerts)
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting budget alerts: {e}")
        return jsonify({"success": False, "error": "budget_alerts_error", "message": str(e)}), 500


@cost_tracking_bp.route('/budget/update', methods=['PUT'])
@validate_json_schema({'total_budget': (int, float)})
def update_budget():
    """Update budget settings"""
    try:
        data = request.get_json()
        
        cost_service.update_budget(
            new_budget=data['total_budget'],
            alert_threshold=data.get('alert_threshold')
        )
        
        return jsonify({
            "success": True,
            "message": f"Budget updated to ${data['total_budget']:.2f}"
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating budget: {e}")
        return jsonify({"success": False, "error": "budget_update_error", "message": str(e)}), 500


@cost_tracking_bp.route('/budget/reset', methods=['POST'])
def reset_budget():
    """Reset budget tracking"""
    try:
        cost_service.reset_budget_tracking()
        
        return jsonify({
            "success": True,
            "message": "Budget tracking reset successfully"
        }), 200
        
    except Exception as e:
        logger.error(f"Error resetting budget: {e}")
        return jsonify({"success": False, "error": "budget_reset_error", "message": str(e)}), 500


@cost_tracking_bp.route('/analytics', methods=['GET'])
@validate_query_params(optional_params=['days'])
def get_usage_analytics():
    """Get usage analytics for specified period"""
    try:
        days = request.args.get('days', 30, type=int)
        
        analytics = cost_service.get_usage_analytics(days)
        
        return jsonify({
            "success": True,
            "data": analytics
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting usage analytics: {e}")
        return jsonify({"success": False, "error": "analytics_error", "message": str(e)}), 500


@cost_tracking_bp.route('/trends', methods=['GET'])
@validate_query_params(optional_params=['days'])
def get_cost_trends():
    """Get cost trends over time"""
    try:
        days = request.args.get('days', 30, type=int)
        
        trends = cost_service.get_cost_trends(days)
        
        return jsonify({
            "success": True,
            "data": trends
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting cost trends: {e}")
        return jsonify({"success": False, "error": "trends_error", "message": str(e)}), 500


@cost_tracking_bp.route('/breakdown', methods=['GET'])
def get_model_breakdown():
    """Get detailed cost breakdown by model"""
    try:
        breakdown = cost_service.get_model_cost_breakdown()
        
        return jsonify({
            "success": True,
            "data": {
                "model_breakdown": breakdown,
                "total_models": len(breakdown)
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting model breakdown: {e}")
        return jsonify({"success": False, "error": "breakdown_error", "message": str(e)}), 500


@cost_tracking_bp.route('/recommendations', methods=['GET'])
def get_optimization_recommendations():
    """Get cost optimization recommendations"""
    try:
        recommendations = cost_service.get_cost_optimization_recommendations()
        
        return jsonify({
            "success": True,
            "data": {
                "recommendations": recommendations,
                "recommendation_count": len(recommendations)
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        return jsonify({"success": False, "error": "recommendations_error", "message": str(e)}), 500


@cost_tracking_bp.route('/export', methods=['GET'])
@validate_query_params(optional_params=['days', 'format'])
def export_cost_data():
    """Export cost data for external analysis"""
    try:
        days = request.args.get('days', 30, type=int)
        format_type = request.args.get('format', 'json', type=str)
        
        if format_type not in ['json', 'csv']:
            return jsonify({
                "success": False,
                "error": "Invalid format. Supported formats: json, csv"
            }), 400
        
        export_data = cost_service.export_cost_data(days, format_type)
        
        return jsonify({
            "success": True,
            "data": export_data
        }), 200
        
    except Exception as e:
        logger.error(f"Error exporting cost data: {e}")
        return jsonify({"success": False, "error": "export_error", "message": str(e)}), 500


@cost_tracking_bp.route('/health', methods=['GET'])
def cost_tracking_health():
    """Health check for cost tracking service"""
    try:
        status = cost_service.get_budget_status()
        
        return jsonify({
            "success": True,
            "service": "cost_tracking",
            "status": "healthy",
            "budget_utilization": status.get("utilization", 0),
            "timestamp": datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Cost tracking service health check failed: {e}")
        return jsonify({"success": False, "error": "health_check_error", "message": str(e)}), 500


# Error handlers for this blueprint
@cost_tracking_bp.errorhandler(400)
def bad_request(error):
    return jsonify({
        "success": False,
        "error": "Bad request",
        "message": str(error)
    }), 400


@cost_tracking_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500