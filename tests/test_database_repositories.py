"""
Tests for database repositories.
"""

import pytest
import uuid
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base, Dataset, Model, Experiment, Evaluation, ComparisonResult
from database.repositories import (
    DatasetRepository, ModelRepository, ExperimentRepository, 
    EvaluationRepository, ComparisonRepository
)
from utils.exceptions import DatabaseError


@pytest.fixture
def db_session():
    """Create in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    yield session
    
    session.close()


@pytest.fixture
def sample_data(db_session):
    """Create sample data for testing."""
    # Create datasets
    datasets = [
        Dataset(name="dataset1", format="jsonl", size=1000, domain="nlp"),
        Dataset(name="dataset2", format="csv", size=500, domain="qa"),
    ]
    db_session.add_all(datasets)
    
    # Create models
    models = [
        Model(name="gpt2-fine", type="fine-tuned", base_model="gpt2", status="active"),
        Model(name="gpt-4", type="commercial", cost_per_token=0.00003, status="active"),
        Model(name="old-model", type="fine-tuned", status="deprecated"),
    ]
    db_session.add_all(models)
    
    db_session.commit()
    
    # Create experiments
    experiments = [
        Experiment(
            name="exp1", type="fine-tuning", status="completed",
            dataset_id=datasets[0].id, model_id=models[0].id
        ),
        Experiment(
            name="exp2", type="evaluation", status="running",
            dataset_id=datasets[1].id, model_id=models[1].id
        ),
    ]
    db_session.add_all(experiments)
    db_session.commit()
    
    # Create evaluations
    evaluations = [
        Evaluation(
            experiment_id=experiments[0].id, model_id=models[0].id,
            prompt="Test prompt 1", response="Test response 1",
            human_rating=4, cost_usd=0.001, latency_ms=200
        ),
        Evaluation(
            experiment_id=experiments[1].id, model_id=models[1].id,
            prompt="Test prompt 2", response="Test response 2",
            human_rating=5, cost_usd=0.002, latency_ms=150
        ),
    ]
    db_session.add_all(evaluations)
    db_session.commit()
    
    return {
        'datasets': datasets,
        'models': models,
        'experiments': experiments,
        'evaluations': evaluations
    }


class TestBaseRepository:
    """Test base repository functionality."""
    
    def test_create_record(self, db_session):
        """Test creating a record."""
        repo = DatasetRepository()
        
        dataset = repo.create(
            db_session,
            name="test_dataset",
            format="jsonl",
            size=100
        )
        
        assert dataset.id is not None
        assert dataset.name == "test_dataset"
        assert dataset.format == "jsonl"
        assert dataset.size == 100
    
    def test_get_by_id(self, db_session, sample_data):
        """Test getting record by ID."""
        repo = DatasetRepository()
        dataset = sample_data['datasets'][0]
        
        retrieved = repo.get_by_id(db_session, dataset.id)
        
        assert retrieved is not None
        assert retrieved.id == dataset.id
        assert retrieved.name == dataset.name
    
    def test_get_by_id_not_found(self, db_session):
        """Test getting non-existent record."""
        repo = DatasetRepository()
        
        retrieved = repo.get_by_id(db_session, uuid.uuid4())
        
        assert retrieved is None
    
    def test_update_record(self, db_session, sample_data):
        """Test updating a record."""
        repo = DatasetRepository()
        dataset = sample_data['datasets'][0]
        
        updated = repo.update(
            db_session,
            dataset.id,
            description="Updated description",
            size=2000
        )
        
        assert updated is not None
        assert updated.description == "Updated description"
        assert updated.size == 2000
        assert updated.name == dataset.name  # Unchanged
    
    def test_update_nonexistent_record(self, db_session):
        """Test updating non-existent record."""
        repo = DatasetRepository()
        
        updated = repo.update(db_session, uuid.uuid4(), name="new_name")
        
        assert updated is None
    
    def test_delete_record(self, db_session, sample_data):
        """Test deleting a record."""
        repo = DatasetRepository()
        dataset = sample_data['datasets'][0]
        
        success = repo.delete(db_session, dataset.id)
        
        assert success is True
        
        # Verify deletion
        retrieved = repo.get_by_id(db_session, dataset.id)
        assert retrieved is None
    
    def test_delete_nonexistent_record(self, db_session):
        """Test deleting non-existent record."""
        repo = DatasetRepository()
        
        success = repo.delete(db_session, uuid.uuid4())
        
        assert success is False


class TestDatasetRepository:
    """Test dataset repository."""
    
    def test_get_by_name(self, db_session, sample_data):
        """Test getting dataset by name."""
        repo = DatasetRepository()
        
        dataset = repo.get_by_name(db_session, "dataset1")
        
        assert dataset is not None
        assert dataset.name == "dataset1"
    
    def test_get_by_name_not_found(self, db_session):
        """Test getting non-existent dataset by name."""
        repo = DatasetRepository()
        
        dataset = repo.get_by_name(db_session, "nonexistent")
        
        assert dataset is None
    
    def test_list_datasets(self, db_session, sample_data):
        """Test listing datasets."""
        repo = DatasetRepository()
        
        datasets = repo.list_datasets(db_session)
        
        assert len(datasets) == 2
        assert datasets[0].name in ["dataset1", "dataset2"]
    
    def test_list_datasets_with_domain_filter(self, db_session, sample_data):
        """Test listing datasets with domain filter."""
        repo = DatasetRepository()
        
        datasets = repo.list_datasets(db_session, domain="nlp")
        
        assert len(datasets) == 1
        assert datasets[0].domain == "nlp"
    
    def test_list_datasets_with_pagination(self, db_session, sample_data):
        """Test listing datasets with pagination."""
        repo = DatasetRepository()
        
        datasets = repo.list_datasets(db_session, limit=1, offset=0)
        
        assert len(datasets) == 1
        
        datasets = repo.list_datasets(db_session, limit=1, offset=1)
        
        assert len(datasets) == 1
    
    def test_get_dataset_stats(self, db_session, sample_data):
        """Test getting dataset statistics."""
        repo = DatasetRepository()
        
        stats = repo.get_dataset_stats(db_session)
        
        assert stats['total_datasets'] == 2
        assert stats['total_samples'] == 1500  # 1000 + 500
        assert 'nlp' in stats['domains']
        assert 'qa' in stats['domains']


class TestModelRepository:
    """Test model repository."""
    
    def test_get_by_name_and_version(self, db_session, sample_data):
        """Test getting model by name and version."""
        repo = ModelRepository()
        
        model = repo.get_by_name_and_version(db_session, "gpt2-fine", "1.0.0")
        
        assert model is not None
        assert model.name == "gpt2-fine"
        assert model.version == "1.0.0"
    
    def test_list_models(self, db_session, sample_data):
        """Test listing models."""
        repo = ModelRepository()
        
        models = repo.list_models(db_session)
        
        # Should only return active models by default
        assert len(models) == 2
        for model in models:
            assert model.status == "active"
    
    def test_list_models_with_type_filter(self, db_session, sample_data):
        """Test listing models with type filter."""
        repo = ModelRepository()
        
        models = repo.list_models(db_session, model_type="commercial")
        
        assert len(models) == 1
        assert models[0].type == "commercial"
    
    def test_get_commercial_models(self, db_session, sample_data):
        """Test getting commercial models."""
        repo = ModelRepository()
        
        models = repo.get_commercial_models(db_session)
        
        assert len(models) == 1
        assert models[0].type == "commercial"
        assert models[0].status == "active"
    
    def test_get_fine_tuned_models(self, db_session, sample_data):
        """Test getting fine-tuned models."""
        repo = ModelRepository()
        
        models = repo.get_fine_tuned_models(db_session)
        
        assert len(models) == 1
        assert models[0].type == "fine-tuned"
        assert models[0].status == "active"


class TestExperimentRepository:
    """Test experiment repository."""
    
    def test_get_by_name(self, db_session, sample_data):
        """Test getting experiment by name."""
        repo = ExperimentRepository()
        
        experiment = repo.get_by_name(db_session, "exp1")
        
        assert experiment is not None
        assert experiment.name == "exp1"
    
    def test_list_experiments(self, db_session, sample_data):
        """Test listing experiments."""
        repo = ExperimentRepository()
        
        experiments = repo.list_experiments(db_session)
        
        assert len(experiments) == 2
    
    def test_list_experiments_with_status_filter(self, db_session, sample_data):
        """Test listing experiments with status filter."""
        repo = ExperimentRepository()
        
        experiments = repo.list_experiments(db_session, status="completed")
        
        assert len(experiments) == 1
        assert experiments[0].status == "completed"
    
    def test_list_experiments_with_type_filter(self, db_session, sample_data):
        """Test listing experiments with type filter."""
        repo = ExperimentRepository()
        
        experiments = repo.list_experiments(db_session, experiment_type="evaluation")
        
        assert len(experiments) == 1
        assert experiments[0].type == "evaluation"
    
    def test_get_running_experiments(self, db_session, sample_data):
        """Test getting running experiments."""
        repo = ExperimentRepository()
        
        experiments = repo.get_running_experiments(db_session)
        
        assert len(experiments) == 1
        assert experiments[0].status == "running"
    
    def test_update_status(self, db_session, sample_data):
        """Test updating experiment status."""
        repo = ExperimentRepository()
        experiment = sample_data['experiments'][0]
        
        updated = repo.update_status(
            db_session,
            experiment.id,
            "running",
            results={"accuracy": 0.85}
        )
        
        assert updated is not None
        assert updated.status == "running"
        assert updated.results == {"accuracy": 0.85}
        assert updated.started_at is not None
    
    def test_update_status_to_completed(self, db_session, sample_data):
        """Test updating experiment status to completed."""
        repo = ExperimentRepository()
        experiment = sample_data['experiments'][1]
        
        # First set to running
        repo.update_status(db_session, experiment.id, "running")
        db_session.commit()
        
        # Then complete
        updated = repo.update_status(db_session, experiment.id, "completed")
        
        assert updated.status == "completed"
        assert updated.completed_at is not None
        assert updated.duration_seconds is not None


class TestEvaluationRepository:
    """Test evaluation repository."""
    
    def test_create_batch(self, db_session, sample_data):
        """Test creating evaluations in batch."""
        repo = EvaluationRepository()
        experiment = sample_data['experiments'][0]
        model = sample_data['models'][0]
        
        evaluations_data = [
            {
                'experiment_id': experiment.id,
                'model_id': model.id,
                'prompt': f'Prompt {i}',
                'response': f'Response {i}',
                'cost_usd': 0.001
            }
            for i in range(3)
        ]
        
        evaluations = repo.create_batch(db_session, evaluations_data)
        
        assert len(evaluations) == 3
        for i, eval in enumerate(evaluations):
            assert eval.prompt == f'Prompt {i}'
            assert eval.response == f'Response {i}'
    
    def test_get_by_experiment(self, db_session, sample_data):
        """Test getting evaluations by experiment."""
        repo = EvaluationRepository()
        experiment = sample_data['experiments'][0]
        
        evaluations = repo.get_by_experiment(db_session, experiment.id)
        
        assert len(evaluations) == 1
        assert evaluations[0].experiment_id == experiment.id
    
    def test_get_by_model(self, db_session, sample_data):
        """Test getting evaluations by model."""
        repo = EvaluationRepository()
        model = sample_data['models'][0]
        
        evaluations = repo.get_by_model(db_session, model.id)
        
        assert len(evaluations) == 1
        assert evaluations[0].model_id == model.id
    
    def test_get_evaluation_stats(self, db_session, sample_data):
        """Test getting evaluation statistics."""
        repo = EvaluationRepository()
        
        stats = repo.get_evaluation_stats(db_session)
        
        assert stats['total_evaluations'] == 2
        assert stats['average_cost_usd'] == 0.0015  # (0.001 + 0.002) / 2
        assert stats['average_latency_ms'] == 175.0  # (200 + 150) / 2
        assert stats['total_cost_usd'] == 0.003
        assert stats['human_rated_count'] == 2
        assert stats['average_human_rating'] == 4.5  # (4 + 5) / 2
    
    def test_get_evaluation_stats_filtered(self, db_session, sample_data):
        """Test getting evaluation statistics with filters."""
        repo = EvaluationRepository()
        experiment = sample_data['experiments'][0]
        
        stats = repo.get_evaluation_stats(db_session, experiment_id=experiment.id)
        
        assert stats['total_evaluations'] == 1
        assert stats['average_cost_usd'] == 0.001
        assert stats['average_human_rating'] == 4.0
    
    def test_search_evaluations(self, db_session, sample_data):
        """Test searching evaluations."""
        repo = EvaluationRepository()
        
        # Search by text
        results, total = repo.search_evaluations(db_session, query_text="prompt 1")
        
        assert total == 1
        assert len(results) == 1
        assert "prompt 1" in results[0].prompt.lower()
    
    def test_search_evaluations_with_filters(self, db_session, sample_data):
        """Test searching evaluations with filters."""
        repo = EvaluationRepository()
        
        # Search with rating filter
        results, total = repo.search_evaluations(db_session, min_rating=5)
        
        assert total == 1
        assert len(results) == 1
        assert results[0].human_rating == 5
        
        # Search with cost filter
        results, total = repo.search_evaluations(db_session, max_cost=0.0015)
        
        assert total == 1
        assert len(results) == 1
        assert results[0].cost_usd <= 0.0015
    
    def test_search_evaluations_with_date_range(self, db_session, sample_data):
        """Test searching evaluations with date range."""
        repo = EvaluationRepository()
        
        # Search from yesterday
        yesterday = datetime.utcnow() - timedelta(days=1)
        results, total = repo.search_evaluations(db_session, date_from=yesterday)
        
        assert total == 2  # Both evaluations should be found
        assert len(results) == 2


class TestComparisonRepository:
    """Test comparison repository."""
    
    def test_create_and_get_comparison(self, db_session, sample_data):
        """Test creating and retrieving comparison."""
        repo = ComparisonRepository()
        models = sample_data['models']
        
        comparison = repo.create(
            db_session,
            name="test_comparison",
            description="Test comparison",
            model_ids=[str(models[0].id), str(models[1].id)],
            results={"model1": 0.85, "model2": 0.90},
            winner_model_id=models[1].id,
            sample_size=100
        )
        
        assert comparison.id is not None
        
        # Test get by name
        retrieved = repo.get_by_name(db_session, "test_comparison")
        
        assert retrieved is not None
        assert retrieved.name == "test_comparison"
        assert len(retrieved.model_ids) == 2
    
    def test_list_comparisons(self, db_session, sample_data):
        """Test listing comparisons."""
        repo = ComparisonRepository()
        models = sample_data['models']
        
        # Create test comparisons
        for i in range(3):
            repo.create(
                db_session,
                name=f"comparison_{i}",
                model_ids=[str(models[0].id)],
                results={"test": i}
            )
        
        db_session.commit()
        
        comparisons = repo.list_comparisons(db_session)
        
        assert len(comparisons) == 3


class TestRepositoryErrorHandling:
    """Test repository error handling."""
    
    def test_database_error_handling(self, db_session):
        """Test database error handling."""
        repo = DatasetRepository()
        
        # Force a database error by using invalid data
        with pytest.raises((DatabaseError, Exception)):
            # This should cause an error due to missing required fields
            repo.create(db_session, invalid_field="test")