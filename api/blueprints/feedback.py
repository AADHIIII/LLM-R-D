"""
Feedback API endpoints for human rating and feedback collection.
"""

import uuid
from flask import Blueprint, request, jsonify
from typing import Dict, Any, Optional

from database.connection import db_manager
from database.repositories import EvaluationRepository
from utils.exceptions import ValidationError, DatabaseError
from utils.error_handler import handle_api_error

import logging

logger = logging.getLogger(__name__)

feedback_bp = Blueprint('feedback', __name__, url_prefix='/api/v1/feedback')
evaluation_repo = EvaluationRepository()


@feedback_bp.route('/evaluation/<evaluation_id>', methods=['PUT'])
def update_evaluation_feedback(evaluation_id: str):
    """Update human feedback for a specific evaluation."""
    try:
        # Validate evaluation ID
        try:
            eval_uuid = uuid.UUID(evaluation_id)
        except ValueError:
            return jsonify({
                'error': 'Invalid evaluation ID format',
                'error_code': 'INVALID_UUID'
            }), 400
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Request body is required',
                'error_code': 'MISSING_BODY'
            }), 400
        
        # Validate feedback data
        feedback_data = {}
        
        # Validate thumbs rating
        if 'thumbs_rating' in data:
            thumbs_rating = data['thumbs_rating']
            if thumbs_rating not in ['up', 'down', None]:
                return jsonify({
                    'error': 'thumbs_rating must be "up", "down", or null',
                    'error_code': 'INVALID_THUMBS_RATING'
                }), 400
            feedback_data['thumbs_rating'] = thumbs_rating
        
        # Validate star rating
        if 'star_rating' in data:
            star_rating = data['star_rating']
            if star_rating is not None and (not isinstance(star_rating, int) or star_rating < 1 or star_rating > 5):
                return jsonify({
                    'error': 'star_rating must be an integer between 1 and 5, or null',
                    'error_code': 'INVALID_STAR_RATING'
                }), 400
            feedback_data['star_rating'] = star_rating
        
        # Validate qualitative feedback
        if 'qualitative_feedback' in data:
            qualitative_feedback = data['qualitative_feedback']
            if qualitative_feedback is not None:
                if not isinstance(qualitative_feedback, str):
                    return jsonify({
                        'error': 'qualitative_feedback must be a string or null',
                        'error_code': 'INVALID_FEEDBACK_TEXT'
                    }), 400
                if len(qualitative_feedback) > 1000:  # Reasonable limit
                    return jsonify({
                        'error': 'qualitative_feedback must be 1000 characters or less',
                        'error_code': 'FEEDBACK_TOO_LONG'
                    }), 400
            feedback_data['qualitative_feedback'] = qualitative_feedback
        
        # Add timestamp
        from datetime import datetime
        feedback_data['feedback_timestamp'] = datetime.utcnow().isoformat()
        
        # Update feedback in database
        with db_manager.get_session() as session:
            updated_evaluation = evaluation_repo.update_human_feedback(
                session, eval_uuid, feedback_data
            )
            
            if not updated_evaluation:
                return jsonify({
                    'error': 'Evaluation not found',
                    'error_code': 'EVALUATION_NOT_FOUND'
                }), 404
            
            session.commit()
            
            # Return updated evaluation data
            return jsonify({
                'id': str(updated_evaluation.id),
                'human_feedback': updated_evaluation.human_feedback_data,
                'human_rating': updated_evaluation.human_rating,
                'message': 'Feedback updated successfully'
            }), 200
    
    except ValidationError as e:
        return handle_api_error(e, 400)
    except DatabaseError as e:
        return handle_api_error(e, 500)
    except Exception as e:
        logger.error(f"Unexpected error updating feedback: {e}")
        return handle_api_error(e, 500)


@feedback_bp.route('/stats', methods=['GET'])
def get_feedback_stats():
    """Get aggregated feedback statistics."""
    try:
        # Get query parameters
        model_id = request.args.get('model_id')
        experiment_id = request.args.get('experiment_id')
        
        # Validate UUIDs if provided
        model_uuid = None
        experiment_uuid = None
        
        if model_id:
            try:
                model_uuid = uuid.UUID(model_id)
            except ValueError:
                return jsonify({
                    'error': 'Invalid model_id format',
                    'error_code': 'INVALID_UUID'
                }), 400
        
        if experiment_id:
            try:
                experiment_uuid = uuid.UUID(experiment_id)
            except ValueError:
                return jsonify({
                    'error': 'Invalid experiment_id format',
                    'error_code': 'INVALID_UUID'
                }), 400
        
        # Get feedback statistics
        with db_manager.get_session() as session:
            stats = evaluation_repo.get_feedback_stats(
                session, 
                model_id=model_uuid,
                experiment_id=experiment_uuid
            )
            
            return jsonify(stats), 200
    
    except DatabaseError as e:
        return handle_api_error(e, 500)
    except Exception as e:
        logger.error(f"Unexpected error getting feedback stats: {e}")
        return handle_api_error(e, 500)


@feedback_bp.route('/model/<model_id>/stats', methods=['GET'])
def get_model_feedback_stats(model_id: str):
    """Get feedback statistics for a specific model."""
    try:
        # Validate model ID
        try:
            model_uuid = uuid.UUID(model_id)
        except ValueError:
            return jsonify({
                'error': 'Invalid model ID format',
                'error_code': 'INVALID_UUID'
            }), 400
        
        # Get feedback statistics for the model
        with db_manager.get_session() as session:
            stats = evaluation_repo.get_feedback_stats(
                session, 
                model_id=model_uuid
            )
            
            return jsonify(stats), 200
    
    except DatabaseError as e:
        return handle_api_error(e, 500)
    except Exception as e:
        logger.error(f"Unexpected error getting model feedback stats: {e}")
        return handle_api_error(e, 500)


@feedback_bp.route('/experiment/<experiment_id>/stats', methods=['GET'])
def get_experiment_feedback_stats(experiment_id: str):
    """Get feedback statistics for a specific experiment."""
    try:
        # Validate experiment ID
        try:
            experiment_uuid = uuid.UUID(experiment_id)
        except ValueError:
            return jsonify({
                'error': 'Invalid experiment ID format',
                'error_code': 'INVALID_UUID'
            }), 400
        
        # Get feedback statistics for the experiment
        with db_manager.get_session() as session:
            stats = evaluation_repo.get_feedback_stats(
                session, 
                experiment_id=experiment_uuid
            )
            
            return jsonify(stats), 200
    
    except DatabaseError as e:
        return handle_api_error(e, 500)
    except Exception as e:
        logger.error(f"Unexpected error getting experiment feedback stats: {e}")
        return handle_api_error(e, 500)