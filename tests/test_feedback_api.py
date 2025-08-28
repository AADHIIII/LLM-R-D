"""
Tests for feedback API endpoints.
"""

import pytest
import uuid
import json
from unittest.mock import Mock, patch
from datetime import datetime

from api.app import create_app
from database.models import Evaluation
from utils.exceptions import DatabaseError


@pytest.fixture
def app():
    """Create test Flask app."""
    app = create_app('testing')
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture
def mock_evaluation():
    """Create mock evaluation object."""
    evaluation = Mock(spec=Evaluation)
    evaluation.id = uuid.uuid4()
    evaluation.human_feedback_data = {
        'star_rating': 4,
        'thumbs_rating': 'up',
        'qualitative_feedback': 'Great response!',
        'feedback_timestamp': datetime.utcnow().isoformat()
    }
    evaluation.human_rating = 4
    return evaluation


class TestFeedbackAPI:
    """Test cases for feedback API endpoints."""
    
    @patch('api.blueprints.feedback.db_manager')
    @patch('api.blueprints.feedback.evaluation_repo')
    def test_update_evaluation_feedback_success(self, mock_repo, mock_db, client, mock_evaluation):
        """Test successful feedback update."""
        # Setup mocks
        mock_session = Mock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_repo.update_human_feedback.return_value = mock_evaluation
        
        evaluation_id = str(mock_evaluation.id)
        feedback_data = {
            'star_rating': 5,
            'thumbs_rating': 'up',
            'qualitative_feedback': 'Excellent response!'
        }
        
        # Make request
        response = client.put(
            f'/api/v1/feedback/evaluation/{evaluation_id}',
            data=json.dumps(feedback_data),
            content_type='application/json'
        )
        
        # Assertions
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['id'] == evaluation_id
        assert data['message'] == 'Feedback updated successfully'
        
        # Verify repository was called correctly
        mock_repo.update_human_feedback.assert_called_once()
        args = mock_repo.update_human_feedback.call_args
        assert args[0][1] == mock_evaluation.id  # evaluation_id
        assert 'feedback_timestamp' in args[0][2]  # feedback_data contains timestamp
    
    def test_update_evaluation_feedback_invalid_uuid(self, client):
        """Test feedback update with invalid UUID."""
        response = client.put(
            '/api/v1/feedback/evaluation/invalid-uuid',
            data=json.dumps({'star_rating': 5}),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['error_code'] == 'INVALID_UUID'
    
    def test_update_evaluation_feedback_missing_body(self, client):
        """Test feedback update without request body."""
        evaluation_id = str(uuid.uuid4())
        response = client.put(f'/api/v1/feedback/evaluation/{evaluation_id}')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['error_code'] == 'MISSING_BODY'
    
    def test_update_evaluation_feedback_invalid_star_rating(self, client):
        """Test feedback update with invalid star rating."""
        evaluation_id = str(uuid.uuid4())
        feedback_data = {'star_rating': 6}  # Invalid: > 5
        
        response = client.put(
            f'/api/v1/feedback/evaluation/{evaluation_id}',
            data=json.dumps(feedback_data),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['error_code'] == 'INVALID_STAR_RATING'
    
    def test_update_evaluation_feedback_invalid_thumbs_rating(self, client):
        """Test feedback update with invalid thumbs rating."""
        evaluation_id = str(uuid.uuid4())
        feedback_data = {'thumbs_rating': 'maybe'}  # Invalid value
        
        response = client.put(
            f'/api/v1/feedback/evaluation/{evaluation_id}',
            data=json.dumps(feedback_data),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['error_code'] == 'INVALID_THUMBS_RATING'
    
    def test_update_evaluation_feedback_text_too_long(self, client):
        """Test feedback update with text that's too long."""
        evaluation_id = str(uuid.uuid4())
        feedback_data = {'qualitative_feedback': 'x' * 1001}  # Too long
        
        response = client.put(
            f'/api/v1/feedback/evaluation/{evaluation_id}',
            data=json.dumps(feedback_data),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['error_code'] == 'FEEDBACK_TOO_LONG'
    
    @patch('api.blueprints.feedback.db_manager')
    @patch('api.blueprints.feedback.evaluation_repo')
    def test_update_evaluation_feedback_not_found(self, mock_repo, mock_db, client):
        """Test feedback update for non-existent evaluation."""
        # Setup mocks
        mock_session = Mock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_repo.update_human_feedback.return_value = None  # Not found
        
        evaluation_id = str(uuid.uuid4())
        feedback_data = {'star_rating': 5}
        
        response = client.put(
            f'/api/v1/feedback/evaluation/{evaluation_id}',
            data=json.dumps(feedback_data),
            content_type='application/json'
        )
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert data['error_code'] == 'EVALUATION_NOT_FOUND'
    
    @patch('api.blueprints.feedback.db_manager')
    @patch('api.blueprints.feedback.evaluation_repo')
    def test_get_feedback_stats_success(self, mock_repo, mock_db, client):
        """Test successful feedback stats retrieval."""
        # Setup mocks
        mock_session = Mock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        
        mock_stats = {
            'total_ratings': 10,
            'average_star_rating': 4.2,
            'star_distribution': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
            'thumbs_up_count': 8,
            'thumbs_down_count': 2,
            'total_comments': 5,
            'recent_comments': []
        }
        mock_repo.get_feedback_stats.return_value = mock_stats
        
        response = client.get('/api/v1/feedback/stats')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data == mock_stats
    
    @patch('api.blueprints.feedback.db_manager')
    @patch('api.blueprints.feedback.evaluation_repo')
    def test_get_feedback_stats_with_filters(self, mock_repo, mock_db, client):
        """Test feedback stats with model and experiment filters."""
        # Setup mocks
        mock_session = Mock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_repo.get_feedback_stats.return_value = {}
        
        model_id = str(uuid.uuid4())
        experiment_id = str(uuid.uuid4())
        
        response = client.get(f'/api/v1/feedback/stats?model_id={model_id}&experiment_id={experiment_id}')
        
        assert response.status_code == 200
        
        # Verify repository was called with correct parameters
        mock_repo.get_feedback_stats.assert_called_once()
        args = mock_repo.get_feedback_stats.call_args
        assert str(args[1]['model_id']) == model_id
        assert str(args[1]['experiment_id']) == experiment_id
    
    def test_get_feedback_stats_invalid_model_id(self, client):
        """Test feedback stats with invalid model ID."""
        response = client.get('/api/v1/feedback/stats?model_id=invalid-uuid')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['error_code'] == 'INVALID_UUID'
    
    @patch('api.blueprints.feedback.db_manager')
    @patch('api.blueprints.feedback.evaluation_repo')
    def test_get_model_feedback_stats(self, mock_repo, mock_db, client):
        """Test model-specific feedback stats."""
        # Setup mocks
        mock_session = Mock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_repo.get_feedback_stats.return_value = {'total_ratings': 5}
        
        model_id = str(uuid.uuid4())
        response = client.get(f'/api/v1/feedback/model/{model_id}/stats')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['total_ratings'] == 5
    
    @patch('api.blueprints.feedback.db_manager')
    @patch('api.blueprints.feedback.evaluation_repo')
    def test_get_experiment_feedback_stats(self, mock_repo, mock_db, client):
        """Test experiment-specific feedback stats."""
        # Setup mocks
        mock_session = Mock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_repo.get_feedback_stats.return_value = {'total_ratings': 3}
        
        experiment_id = str(uuid.uuid4())
        response = client.get(f'/api/v1/feedback/experiment/{experiment_id}/stats')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['total_ratings'] == 3
    
    @patch('api.blueprints.feedback.db_manager')
    @patch('api.blueprints.feedback.evaluation_repo')
    def test_database_error_handling(self, mock_repo, mock_db, client):
        """Test database error handling."""
        # Setup mocks to raise DatabaseError
        mock_session = Mock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_repo.get_feedback_stats.side_effect = DatabaseError("Database connection failed")
        
        response = client.get('/api/v1/feedback/stats')
        
        assert response.status_code == 500