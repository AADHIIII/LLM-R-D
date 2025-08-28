"""
Tests for feedback-related repository methods.
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, patch

from database.repositories import EvaluationRepository
from database.models import Evaluation
from utils.exceptions import DatabaseError


@pytest.fixture
def evaluation_repo():
    """Create evaluation repository instance."""
    return EvaluationRepository()


@pytest.fixture
def mock_session():
    """Create mock database session."""
    return Mock()


@pytest.fixture
def sample_evaluation():
    """Create sample evaluation object."""
    evaluation = Evaluation(
        id=uuid.uuid4(),
        experiment_id=uuid.uuid4(),
        model_id=uuid.uuid4(),
        prompt="Test prompt",
        response="Test response",
        human_rating=4,
        human_feedback="Good response",
        human_feedback_data={
            'star_rating': 4,
            'thumbs_rating': 'up',
            'qualitative_feedback': 'Good response',
            'feedback_timestamp': datetime.utcnow().isoformat()
        },
        cost_usd=0.01,
        latency_ms=500,
        created_at=datetime.utcnow()
    )
    return evaluation


class TestFeedbackRepository:
    """Test cases for feedback repository methods."""
    
    def test_update_human_feedback_success(self, evaluation_repo, mock_session, sample_evaluation):
        """Test successful feedback update."""
        # Setup mock
        mock_session.query.return_value.filter.return_value.first.return_value = sample_evaluation
        
        feedback_data = {
            'star_rating': 5,
            'thumbs_rating': 'up',
            'qualitative_feedback': 'Excellent!',
            'feedback_timestamp': datetime.utcnow().isoformat()
        }
        
        # Call method
        result = evaluation_repo.update_human_feedback(
            mock_session, sample_evaluation.id, feedback_data
        )
        
        # Assertions
        assert result == sample_evaluation
        assert result.human_feedback_data == feedback_data
        assert result.human_rating == 5  # Updated from feedback_data
        assert result.human_feedback == 'Excellent!'  # Updated from feedback_data
        mock_session.flush.assert_called_once()
    
    def test_update_human_feedback_not_found(self, evaluation_repo, mock_session):
        """Test feedback update for non-existent evaluation."""
        # Setup mock to return None
        mock_session.query.return_value.filter.return_value.first.return_value = None
        
        feedback_data = {'star_rating': 5}
        evaluation_id = uuid.uuid4()
        
        # Call method
        result = evaluation_repo.update_human_feedback(
            mock_session, evaluation_id, feedback_data
        )
        
        # Assertions
        assert result is None
        mock_session.flush.assert_not_called()
    
    def test_update_human_feedback_partial_data(self, evaluation_repo, mock_session, sample_evaluation):
        """Test feedback update with partial data."""
        # Setup mock
        mock_session.query.return_value.filter.return_value.first.return_value = sample_evaluation
        
        feedback_data = {
            'thumbs_rating': 'down'  # Only thumbs rating, no star rating or comment
        }
        
        # Call method
        result = evaluation_repo.update_human_feedback(
            mock_session, sample_evaluation.id, feedback_data
        )
        
        # Assertions
        assert result == sample_evaluation
        assert result.human_feedback_data == feedback_data
        # Legacy fields should not be updated if not in feedback_data
        assert result.human_rating == 4  # Original value
        assert result.human_feedback == "Good response"  # Original value
    
    @patch('database.repositories.logger')
    def test_update_human_feedback_database_error(self, mock_logger, evaluation_repo, mock_session):
        """Test database error handling in feedback update."""
        # Setup mock to raise exception
        from sqlalchemy.exc import SQLAlchemyError
        mock_session.query.side_effect = SQLAlchemyError("Database error")
        
        feedback_data = {'star_rating': 5}
        evaluation_id = uuid.uuid4()
        
        # Call method and expect exception
        with pytest.raises(DatabaseError):
            evaluation_repo.update_human_feedback(
                mock_session, evaluation_id, feedback_data
            )
        
        # Verify error was logged
        mock_logger.error.assert_called_once()
    
    def test_get_feedback_stats_success(self, evaluation_repo, mock_session):
        """Test successful feedback stats calculation."""
        # Create sample evaluations with different feedback
        evaluations = [
            Mock(
                id=uuid.uuid4(),
                human_rating=5,
                human_feedback="Great!",
                human_feedback_data={
                    'star_rating': 5,
                    'thumbs_rating': 'up',
                    'qualitative_feedback': 'Great!'
                },
                created_at=datetime.utcnow()
            ),
            Mock(
                id=uuid.uuid4(),
                human_rating=3,
                human_feedback="Okay",
                human_feedback_data={
                    'star_rating': 3,
                    'thumbs_rating': 'down',
                    'qualitative_feedback': 'Okay'
                },
                created_at=datetime.utcnow()
            ),
            Mock(
                id=uuid.uuid4(),
                human_rating=None,
                human_feedback=None,
                human_feedback_data={
                    'thumbs_rating': 'up'  # Only thumbs, no star rating
                },
                created_at=datetime.utcnow()
            )
        ]
        
        # Setup mock
        mock_session.query.return_value.all.return_value = evaluations
        
        # Call method
        stats = evaluation_repo.get_feedback_stats(mock_session)
        
        # Assertions
        assert stats['total_ratings'] == 2  # Only 2 have star ratings
        assert stats['average_star_rating'] == 4.0  # (5 + 3) / 2
        assert stats['star_distribution'][5] == 1
        assert stats['star_distribution'][3] == 1
        assert stats['thumbs_up_count'] == 2
        assert stats['thumbs_down_count'] == 1
        assert stats['total_comments'] == 2
        assert len(stats['recent_comments']) == 2
    
    def test_get_feedback_stats_with_filters(self, evaluation_repo, mock_session):
        """Test feedback stats with model and experiment filters."""
        # Setup mock
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = []
        
        model_id = uuid.uuid4()
        experiment_id = uuid.uuid4()
        
        # Call method
        evaluation_repo.get_feedback_stats(
            mock_session, 
            model_id=model_id, 
            experiment_id=experiment_id
        )
        
        # Verify filters were applied
        assert mock_query.filter.call_count == 2  # One for model_id, one for experiment_id
    
    def test_get_feedback_stats_empty_data(self, evaluation_repo, mock_session):
        """Test feedback stats with no evaluations."""
        # Setup mock to return empty list
        mock_session.query.return_value.all.return_value = []
        
        # Call method
        stats = evaluation_repo.get_feedback_stats(mock_session)
        
        # Assertions
        assert stats['total_ratings'] == 0
        assert stats['average_star_rating'] == 0.0
        assert stats['thumbs_up_count'] == 0
        assert stats['thumbs_down_count'] == 0
        assert stats['total_comments'] == 0
        assert len(stats['recent_comments']) == 0
    
    def test_get_feedback_stats_legacy_data(self, evaluation_repo, mock_session):
        """Test feedback stats with legacy data (no human_feedback_data)."""
        # Create evaluation with only legacy fields
        evaluation = Mock(
            id=uuid.uuid4(),
            human_rating=4,
            human_feedback="Legacy feedback",
            human_feedback_data=None,  # No structured data
            created_at=datetime.utcnow()
        )
        
        # Setup mock
        mock_session.query.return_value.all.return_value = [evaluation]
        
        # Call method
        stats = evaluation_repo.get_feedback_stats(mock_session)
        
        # Assertions
        assert stats['total_ratings'] == 1
        assert stats['average_star_rating'] == 4.0
        assert stats['star_distribution'][4] == 1
        assert stats['thumbs_up_count'] == 0  # No thumbs data
        assert stats['thumbs_down_count'] == 0
        assert stats['total_comments'] == 1  # From legacy human_feedback
    
    @patch('database.repositories.logger')
    def test_get_feedback_stats_database_error(self, mock_logger, evaluation_repo, mock_session):
        """Test database error handling in feedback stats."""
        # Setup mock to raise exception
        from sqlalchemy.exc import SQLAlchemyError
        mock_session.query.side_effect = SQLAlchemyError("Database error")
        
        # Call method and expect exception
        with pytest.raises(DatabaseError):
            evaluation_repo.get_feedback_stats(mock_session)
        
        # Verify error was logged
        mock_logger.error.assert_called_once()
    
    def test_get_feedback_stats_recent_comments_limit(self, evaluation_repo, mock_session):
        """Test that recent comments are limited to 10."""
        # Create 15 evaluations with comments
        evaluations = []
        for i in range(15):
            evaluation = Mock(
                id=uuid.uuid4(),
                human_rating=5,
                human_feedback=f"Comment {i}",
                human_feedback_data={
                    'star_rating': 5,
                    'qualitative_feedback': f"Comment {i}"
                },
                created_at=datetime.utcnow()
            )
            evaluations.append(evaluation)
        
        # Setup mock
        mock_session.query.return_value.all.return_value = evaluations
        
        # Call method
        stats = evaluation_repo.get_feedback_stats(mock_session)
        
        # Assertions
        assert len(stats['recent_comments']) == 10  # Limited to 10
        assert stats['total_comments'] == 15  # But total count is correct