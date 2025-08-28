"""
Repository classes for data access operations.

Provides CRUD operations and query methods for experiments, models, evaluations, and datasets.
"""

import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func, text
from sqlalchemy.exc import SQLAlchemyError

from .models import Experiment, Model, Evaluation, Dataset, ComparisonResult
from .connection import db_manager
from utils.exceptions import DatabaseError, ValidationError
from api.models.user import User, APIKey

import logging

logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository with common CRUD operations."""
    
    def __init__(self, model_class):
        self.model_class = model_class
    
    def create(self, session: Session, **kwargs) -> Any:
        """Create a new record."""
        try:
            instance = self.model_class(**kwargs)
            session.add(instance)
            session.flush()  # Get ID without committing
            return instance
        except SQLAlchemyError as e:
            logger.error(f"Failed to create {self.model_class.__name__}: {e}")
            raise DatabaseError(f"Failed to create record: {e}")
    
    def get_by_id(self, session: Session, record_id: uuid.UUID) -> Optional[Any]:
        """Get record by ID."""
        try:
            return session.query(self.model_class).filter(
                self.model_class.id == record_id
            ).first()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get {self.model_class.__name__} by ID: {e}")
            raise DatabaseError(f"Failed to retrieve record: {e}")
    
    def update(self, session: Session, record_id: uuid.UUID, **kwargs) -> Optional[Any]:
        """Update record by ID."""
        try:
            instance = self.get_by_id(session, record_id)
            if not instance:
                return None
            
            for key, value in kwargs.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            
            if hasattr(instance, 'updated_at'):
                instance.updated_at = datetime.utcnow()
            
            session.flush()
            return instance
        except SQLAlchemyError as e:
            logger.error(f"Failed to update {self.model_class.__name__}: {e}")
            raise DatabaseError(f"Failed to update record: {e}")
    
    def delete(self, session: Session, record_id: uuid.UUID) -> bool:
        """Delete record by ID."""
        try:
            instance = self.get_by_id(session, record_id)
            if not instance:
                return False
            
            session.delete(instance)
            session.flush()
            return True
        except SQLAlchemyError as e:
            logger.error(f"Failed to delete {self.model_class.__name__}: {e}")
            raise DatabaseError(f"Failed to delete record: {e}")


class DatasetRepository(BaseRepository):
    """Repository for dataset operations."""
    
    def __init__(self):
        super().__init__(Dataset)
    
    def get_by_name(self, session: Session, name: str) -> Optional[Dataset]:
        """Get dataset by name."""
        try:
            return session.query(Dataset).filter(Dataset.name == name).first()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get dataset by name: {e}")
            raise DatabaseError(f"Failed to retrieve dataset: {e}")
    
    def list_datasets(
        self, 
        session: Session, 
        domain: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dataset]:
        """List datasets with optional filtering."""
        try:
            query = session.query(Dataset)
            
            if domain:
                query = query.filter(Dataset.domain == domain)
            
            return query.order_by(desc(Dataset.created_at)).offset(offset).limit(limit).all()
        except SQLAlchemyError as e:
            logger.error(f"Failed to list datasets: {e}")
            raise DatabaseError(f"Failed to list datasets: {e}")
    
    def get_dataset_stats(self, session: Session) -> Dict[str, Any]:
        """Get dataset statistics."""
        try:
            total_count = session.query(func.count(Dataset.id)).scalar()
            total_samples = session.query(func.sum(Dataset.size)).scalar() or 0
            domains = session.query(Dataset.domain, func.count(Dataset.id)).group_by(Dataset.domain).all()
            
            return {
                'total_datasets': total_count,
                'total_samples': total_samples,
                'domains': dict(domains)
            }
        except SQLAlchemyError as e:
            logger.error(f"Failed to get dataset stats: {e}")
            raise DatabaseError(f"Failed to get dataset statistics: {e}")


class ModelRepository(BaseRepository):
    """Repository for model operations."""
    
    def __init__(self):
        super().__init__(Model)
    
    def get_by_name_and_version(
        self, 
        session: Session, 
        name: str, 
        version: str = "1.0.0"
    ) -> Optional[Model]:
        """Get model by name and version."""
        try:
            return session.query(Model).filter(
                and_(Model.name == name, Model.version == version)
            ).first()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get model by name and version: {e}")
            raise DatabaseError(f"Failed to retrieve model: {e}")
    
    def list_models(
        self,
        session: Session,
        model_type: Optional[str] = None,
        status: str = 'active',
        limit: int = 100,
        offset: int = 0
    ) -> List[Model]:
        """List models with filtering."""
        try:
            query = session.query(Model).filter(Model.status == status)
            
            if model_type:
                query = query.filter(Model.type == model_type)
            
            return query.order_by(desc(Model.created_at)).offset(offset).limit(limit).all()
        except SQLAlchemyError as e:
            logger.error(f"Failed to list models: {e}")
            raise DatabaseError(f"Failed to list models: {e}")
    
    def get_commercial_models(self, session: Session) -> List[Model]:
        """Get all active commercial models."""
        try:
            return session.query(Model).filter(
                and_(Model.type == 'commercial', Model.status == 'active')
            ).all()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get commercial models: {e}")
            raise DatabaseError(f"Failed to get commercial models: {e}")
    
    def get_fine_tuned_models(self, session: Session) -> List[Model]:
        """Get all active fine-tuned models."""
        try:
            return session.query(Model).filter(
                and_(Model.type == 'fine-tuned', Model.status == 'active')
            ).all()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get fine-tuned models: {e}")
            raise DatabaseError(f"Failed to get fine-tuned models: {e}")


class ExperimentRepository(BaseRepository):
    """Repository for experiment operations."""
    
    def __init__(self):
        super().__init__(Experiment)
    
    def get_by_name(self, session: Session, name: str) -> Optional[Experiment]:
        """Get experiment by name."""
        try:
            return session.query(Experiment).filter(Experiment.name == name).first()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get experiment by name: {e}")
            raise DatabaseError(f"Failed to retrieve experiment: {e}")
    
    def list_experiments(
        self,
        session: Session,
        status: Optional[str] = None,
        experiment_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Experiment]:
        """List experiments with filtering."""
        try:
            query = session.query(Experiment)
            
            if status:
                query = query.filter(Experiment.status == status)
            
            if experiment_type:
                query = query.filter(Experiment.type == experiment_type)
            
            return query.order_by(desc(Experiment.created_at)).offset(offset).limit(limit).all()
        except SQLAlchemyError as e:
            logger.error(f"Failed to list experiments: {e}")
            raise DatabaseError(f"Failed to list experiments: {e}")
    
    def get_running_experiments(self, session: Session) -> List[Experiment]:
        """Get all running experiments."""
        try:
            return session.query(Experiment).filter(Experiment.status == 'running').all()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get running experiments: {e}")
            raise DatabaseError(f"Failed to get running experiments: {e}")
    
    def update_status(
        self, 
        session: Session, 
        experiment_id: uuid.UUID, 
        status: str,
        results: Optional[Dict[str, Any]] = None
    ) -> Optional[Experiment]:
        """Update experiment status and optionally results."""
        try:
            experiment = self.get_by_id(session, experiment_id)
            if not experiment:
                return None
            
            experiment.status = status
            experiment.updated_at = datetime.utcnow()
            
            if status == 'running' and not experiment.started_at:
                experiment.started_at = datetime.utcnow()
            elif status in ['completed', 'failed'] and not experiment.completed_at:
                experiment.completed_at = datetime.utcnow()
                if experiment.started_at:
                    duration = experiment.completed_at - experiment.started_at
                    experiment.duration_seconds = int(duration.total_seconds())
            
            if results:
                experiment.results = results
            
            session.flush()
            return experiment
        except SQLAlchemyError as e:
            logger.error(f"Failed to update experiment status: {e}")
            raise DatabaseError(f"Failed to update experiment: {e}")


class EvaluationRepository(BaseRepository):
    """Repository for evaluation operations."""
    
    def __init__(self):
        super().__init__(Evaluation)
    
    def create_batch(
        self, 
        session: Session, 
        evaluations: List[Dict[str, Any]]
    ) -> List[Evaluation]:
        """Create multiple evaluations in batch."""
        try:
            instances = []
            for eval_data in evaluations:
                instance = Evaluation(**eval_data)
                session.add(instance)
                instances.append(instance)
            
            session.flush()
            
            # Load all attributes before detaching
            for instance in instances:
                # Access all attributes to ensure they're loaded
                _ = instance.id
                _ = instance.prompt
                _ = instance.response
                _ = instance.cost_usd
                _ = instance.created_at
                # Expunge from session to make it detached but accessible
                session.expunge(instance)
            
            return instances
        except SQLAlchemyError as e:
            logger.error(f"Failed to create evaluation batch: {e}")
            raise DatabaseError(f"Failed to create evaluations: {e}")
    
    def get_by_experiment(
        self, 
        session: Session, 
        experiment_id: uuid.UUID,
        limit: int = 1000,
        offset: int = 0
    ) -> List[Evaluation]:
        """Get evaluations for an experiment."""
        try:
            return session.query(Evaluation).filter(
                Evaluation.experiment_id == experiment_id
            ).order_by(desc(Evaluation.created_at)).offset(offset).limit(limit).all()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get evaluations by experiment: {e}")
            raise DatabaseError(f"Failed to get evaluations: {e}")
    
    def get_by_model(
        self, 
        session: Session, 
        model_id: uuid.UUID,
        limit: int = 1000,
        offset: int = 0
    ) -> List[Evaluation]:
        """Get evaluations for a model."""
        try:
            return session.query(Evaluation).filter(
                Evaluation.model_id == model_id
            ).order_by(desc(Evaluation.created_at)).offset(offset).limit(limit).all()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get evaluations by model: {e}")
            raise DatabaseError(f"Failed to get evaluations: {e}")
    
    def get_recent_evaluations(
        self, 
        session: Session = None,
        days: int = 30
    ) -> List[Evaluation]:
        """Get recent evaluations within the specified number of days."""
        try:
            if session is None:
                with db_manager.get_session() as session:
                    return self._get_recent_evaluations_query(session, days)
            else:
                return self._get_recent_evaluations_query(session, days)
        except SQLAlchemyError as e:
            logger.error(f"Failed to get recent evaluations: {e}")
            return []  # Return empty list instead of raising exception
    
    def _get_recent_evaluations_query(self, session: Session, days: int) -> List[Evaluation]:
        """Internal method to execute the recent evaluations query."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        return session.query(Evaluation).filter(
            Evaluation.created_at >= cutoff_date
        ).order_by(desc(Evaluation.created_at)).all()

    def get_evaluation_stats(
        self, 
        session: Session, 
        experiment_id: Optional[uuid.UUID] = None,
        model_id: Optional[uuid.UUID] = None
    ) -> Dict[str, Any]:
        """Get evaluation statistics."""
        try:
            query = session.query(Evaluation)
            
            if experiment_id:
                query = query.filter(Evaluation.experiment_id == experiment_id)
            
            if model_id:
                query = query.filter(Evaluation.model_id == model_id)
            
            total_count = query.count()
            avg_cost = query.with_entities(func.avg(Evaluation.cost_usd)).scalar() or 0
            avg_latency = query.with_entities(func.avg(Evaluation.latency_ms)).scalar() or 0
            total_cost = query.with_entities(func.sum(Evaluation.cost_usd)).scalar() or 0
            
            # Human rating statistics
            rated_count = query.filter(Evaluation.human_rating.isnot(None)).count()
            avg_rating = query.with_entities(func.avg(Evaluation.human_rating)).scalar() or 0
            
            return {
                'total_evaluations': total_count,
                'average_cost_usd': float(avg_cost),
                'average_latency_ms': float(avg_latency),
                'total_cost_usd': float(total_cost),
                'human_rated_count': rated_count,
                'average_human_rating': float(avg_rating)
            }
        except SQLAlchemyError as e:
            logger.error(f"Failed to get evaluation stats: {e}")
            raise DatabaseError(f"Failed to get evaluation statistics: {e}")
    
    def update_human_feedback(
        self,
        session: Session,
        evaluation_id: uuid.UUID,
        feedback_data: Dict[str, Any]
    ) -> Optional[Evaluation]:
        """Update human feedback for an evaluation."""
        try:
            evaluation = session.query(Evaluation).filter(
                Evaluation.id == evaluation_id
            ).first()
            
            if not evaluation:
                return None
            
            # Update structured feedback data
            evaluation.human_feedback_data = feedback_data
            
            # Maintain backward compatibility with legacy fields
            if 'star_rating' in feedback_data:
                evaluation.human_rating = feedback_data['star_rating']
            
            if 'qualitative_feedback' in feedback_data:
                evaluation.human_feedback = feedback_data['qualitative_feedback']
            
            session.flush()
            return evaluation
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to update human feedback: {e}")
            raise DatabaseError(f"Failed to update human feedback: {e}")
    
    def get_feedback_stats(
        self,
        session: Session,
        model_id: Optional[uuid.UUID] = None,
        experiment_id: Optional[uuid.UUID] = None
    ) -> Dict[str, Any]:
        """Get comprehensive feedback statistics."""
        try:
            query = session.query(Evaluation)
            
            if model_id:
                query = query.filter(Evaluation.model_id == model_id)
            
            if experiment_id:
                query = query.filter(Evaluation.experiment_id == experiment_id)
            
            evaluations = query.all()
            
            # Initialize stats
            stats = {
                'total_ratings': 0,
                'average_star_rating': 0.0,
                'star_distribution': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                'thumbs_up_count': 0,
                'thumbs_down_count': 0,
                'total_comments': 0,
                'recent_comments': []
            }
            
            star_ratings = []
            recent_comments = []
            
            for evaluation in evaluations:
                feedback_data = evaluation.human_feedback_data or {}
                
                # Count star ratings
                star_rating = feedback_data.get('star_rating') or evaluation.human_rating
                if star_rating:
                    star_ratings.append(star_rating)
                    stats['star_distribution'][star_rating] += 1
                
                # Count thumbs ratings
                thumbs_rating = feedback_data.get('thumbs_rating')
                if thumbs_rating == 'up':
                    stats['thumbs_up_count'] += 1
                elif thumbs_rating == 'down':
                    stats['thumbs_down_count'] += 1
                
                # Count comments
                comment = feedback_data.get('qualitative_feedback') or evaluation.human_feedback
                if comment and comment.strip():
                    stats['total_comments'] += 1
                    recent_comments.append({
                        'id': str(evaluation.id),
                        'text': comment,
                        'star_rating': star_rating,
                        'thumbs_rating': thumbs_rating,
                        'timestamp': evaluation.created_at.isoformat()
                    })
            
            # Calculate averages
            if star_ratings:
                stats['total_ratings'] = len(star_ratings)
                stats['average_star_rating'] = sum(star_ratings) / len(star_ratings)
            
            # Sort recent comments by timestamp (most recent first)
            stats['recent_comments'] = sorted(
                recent_comments,
                key=lambda x: x['timestamp'],
                reverse=True
            )[:10]  # Limit to 10 most recent
            
            return stats
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to get feedback stats: {e}")
            raise DatabaseError(f"Failed to get feedback statistics: {e}")
    
    def search_evaluations(
        self,
        session: Session,
        query_text: Optional[str] = None,
        min_rating: Optional[int] = None,
        max_cost: Optional[float] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Evaluation], int]:
        """Search evaluations with filters."""
        try:
            query = session.query(Evaluation)
            
            # Text search in prompt or response
            if query_text:
                query = query.filter(
                    or_(
                        Evaluation.prompt.contains(query_text),
                        Evaluation.response.contains(query_text)
                    )
                )
            
            # Rating filter
            if min_rating is not None:
                query = query.filter(Evaluation.human_rating >= min_rating)
            
            # Cost filter
            if max_cost is not None:
                query = query.filter(Evaluation.cost_usd <= max_cost)
            
            # Date range filter
            if date_from:
                query = query.filter(Evaluation.created_at >= date_from)
            if date_to:
                query = query.filter(Evaluation.created_at <= date_to)
            
            # Get total count for pagination
            total_count = query.count()
            
            # Get results with pagination
            results = query.order_by(desc(Evaluation.created_at)).offset(offset).limit(limit).all()
            
            return results, total_count
        except SQLAlchemyError as e:
            logger.error(f"Failed to search evaluations: {e}")
            raise DatabaseError(f"Failed to search evaluations: {e}")


class ComparisonRepository(BaseRepository):
    """Repository for comparison result operations."""
    
    def __init__(self):
        super().__init__(ComparisonResult)
    
    def get_by_name(self, session: Session, name: str) -> Optional[ComparisonResult]:
        """Get comparison by name."""
        try:
            return session.query(ComparisonResult).filter(
                ComparisonResult.name == name
            ).first()
        except SQLAlchemyError as e:
            logger.error(f"Failed to get comparison by name: {e}")
            raise DatabaseError(f"Failed to retrieve comparison: {e}")
    
    def list_comparisons(
        self,
        session: Session,
        limit: int = 100,
        offset: int = 0
    ) -> List[ComparisonResult]:
        """List comparison results."""
        try:
            return session.query(ComparisonResult).order_by(
                desc(ComparisonResult.created_at)
            ).offset(offset).limit(limit).all()
        except SQLAlchemyError as e:
            logger.error(f"Failed to list comparisons: {e}")
            raise DatabaseError(f"Failed to list comparisons: {e}")


class UserRepository:
    """Repository for user operations."""
    
    def __init__(self):
        # For now, use in-memory storage (implement with SQLAlchemy later)
        self._users = {}
        self._users_by_username = {}
        self._users_by_email = {}
    
    def create(self, user: User) -> User:
        """Create a new user."""
        try:
            self._users[user.id] = user
            self._users_by_username[user.username] = user
            self._users_by_email[user.email] = user
            return user
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            raise DatabaseError(f"Failed to create user: {e}")
    
    def get_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self._users.get(user_id)
    
    def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self._users_by_username.get(username)
    
    def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return self._users_by_email.get(email)
    
    def update(self, user: User) -> User:
        """Update user."""
        try:
            if user.id in self._users:
                # Update indexes if username or email changed
                old_user = self._users[user.id]
                if old_user.username != user.username:
                    del self._users_by_username[old_user.username]
                    self._users_by_username[user.username] = user
                if old_user.email != user.email:
                    del self._users_by_email[old_user.email]
                    self._users_by_email[user.email] = user
                
                self._users[user.id] = user
            return user
        except Exception as e:
            logger.error(f"Failed to update user: {e}")
            raise DatabaseError(f"Failed to update user: {e}")
    
    def delete(self, user_id: str) -> bool:
        """Delete user by ID."""
        try:
            user = self._users.get(user_id)
            if user:
                del self._users[user_id]
                del self._users_by_username[user.username]
                del self._users_by_email[user.email]
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete user: {e}")
            raise DatabaseError(f"Failed to delete user: {e}")
    
    def list_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        """List users with pagination."""
        users = list(self._users.values())
        return users[offset:offset + limit]


class APIKeyRepository:
    """Repository for API key operations."""
    
    def __init__(self):
        # For now, use in-memory storage (implement with SQLAlchemy later)
        self._api_keys = {}
        self._api_keys_by_prefix = {}
        self._api_keys_by_user = {}
    
    def create(self, api_key: APIKey) -> APIKey:
        """Create a new API key."""
        try:
            self._api_keys[api_key.id] = api_key
            self._api_keys_by_prefix[api_key.key_prefix] = api_key
            
            if api_key.user_id not in self._api_keys_by_user:
                self._api_keys_by_user[api_key.user_id] = []
            self._api_keys_by_user[api_key.user_id].append(api_key)
            
            return api_key
        except Exception as e:
            logger.error(f"Failed to create API key: {e}")
            raise DatabaseError(f"Failed to create API key: {e}")
    
    def get_by_id(self, api_key_id: str) -> Optional[APIKey]:
        """Get API key by ID."""
        return self._api_keys.get(api_key_id)
    
    def get_by_prefix(self, prefix: str) -> Optional[APIKey]:
        """Get API key by prefix."""
        return self._api_keys_by_prefix.get(prefix)
    
    def get_by_user(self, user_id: str) -> List[APIKey]:
        """Get API keys for a user."""
        return self._api_keys_by_user.get(user_id, [])
    
    def update(self, api_key: APIKey) -> APIKey:
        """Update API key."""
        try:
            if api_key.id in self._api_keys:
                self._api_keys[api_key.id] = api_key
                self._api_keys_by_prefix[api_key.key_prefix] = api_key
                
                # Update user index
                for user_keys in self._api_keys_by_user.values():
                    for i, key in enumerate(user_keys):
                        if key.id == api_key.id:
                            user_keys[i] = api_key
                            break
            
            return api_key
        except Exception as e:
            logger.error(f"Failed to update API key: {e}")
            raise DatabaseError(f"Failed to update API key: {e}")
    
    def delete(self, api_key_id: str) -> bool:
        """Delete API key by ID."""
        try:
            api_key = self._api_keys.get(api_key_id)
            if api_key:
                del self._api_keys[api_key_id]
                del self._api_keys_by_prefix[api_key.key_prefix]
                
                # Remove from user index
                user_keys = self._api_keys_by_user.get(api_key.user_id, [])
                self._api_keys_by_user[api_key.user_id] = [
                    key for key in user_keys if key.id != api_key_id
                ]
                
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete API key: {e}")
            raise DatabaseError(f"Failed to delete API key: {e}")


# Repository instances
dataset_repository = DatasetRepository()
model_repository = ModelRepository()
experiment_repository = ExperimentRepository()
evaluation_repository = EvaluationRepository()
comparison_repository = ComparisonRepository()
user_repository = UserRepository()
api_key_repository = APIKeyRepository()