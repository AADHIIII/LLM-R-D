"""
Tests for database models.
"""

import pytest
import uuid
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base, Dataset, Model, Experiment, Evaluation, ComparisonResult


@pytest.fixture
def db_session():
    """Create in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    yield session
    
    session.close()


class TestDatasetModel:
    """Test Dataset model."""
    
    def test_create_dataset(self, db_session):
        """Test creating a dataset."""
        dataset = Dataset(
            name="test_dataset",
            description="Test dataset for unit tests",
            format="jsonl",
            size=1000,
            domain="test",
            file_path="/path/to/dataset.jsonl",
            validation_split=0.2,
            extra_metadata={"source": "test"}
        )
        
        db_session.add(dataset)
        db_session.commit()
        
        # Verify dataset was created
        retrieved = db_session.query(Dataset).filter(Dataset.name == "test_dataset").first()
        assert retrieved is not None
        assert retrieved.name == "test_dataset"
        assert retrieved.format == "jsonl"
        assert retrieved.size == 1000
        assert retrieved.domain == "test"
        assert retrieved.validation_split == 0.2
        assert retrieved.extra_metadata == {"source": "test"}
        assert isinstance(retrieved.id, uuid.UUID)
        assert isinstance(retrieved.created_at, datetime)
    
    def test_dataset_required_fields(self, db_session):
        """Test dataset with only required fields."""
        dataset = Dataset(
            name="minimal_dataset",
            format="csv",
            size=500
        )
        
        db_session.add(dataset)
        db_session.commit()
        
        retrieved = db_session.query(Dataset).filter(Dataset.name == "minimal_dataset").first()
        assert retrieved is not None
        assert retrieved.validation_split == 0.2  # Default value
    
    def test_dataset_relationships(self, db_session):
        """Test dataset relationships."""
        dataset = Dataset(name="rel_dataset", format="jsonl", size=100)
        db_session.add(dataset)
        db_session.commit()
        
        # Create related experiment
        experiment = Experiment(
            name="test_experiment",
            type="evaluation",
            dataset_id=dataset.id
        )
        db_session.add(experiment)
        db_session.commit()
        
        # Test relationship
        assert len(dataset.experiments) == 1
        assert dataset.experiments[0].name == "test_experiment"


class TestModelModel:
    """Test Model model."""
    
    def test_create_fine_tuned_model(self, db_session):
        """Test creating a fine-tuned model."""
        model = Model(
            name="test_gpt2_fine_tuned",
            type="fine-tuned",
            base_model="gpt2",
            model_path="/path/to/model",
            training_config={"epochs": 3, "batch_size": 4},
            performance_metrics={"perplexity": 15.2},
            status="active",
            version="1.0.0"
        )
        
        db_session.add(model)
        db_session.commit()
        
        retrieved = db_session.query(Model).filter(Model.name == "test_gpt2_fine_tuned").first()
        assert retrieved is not None
        assert retrieved.type == "fine-tuned"
        assert retrieved.base_model == "gpt2"
        assert retrieved.training_config == {"epochs": 3, "batch_size": 4}
        assert retrieved.status == "active"
        assert retrieved.version == "1.0.0"
    
    def test_create_commercial_model(self, db_session):
        """Test creating a commercial model."""
        model = Model(
            name="gpt-4",
            type="commercial",
            api_endpoint="https://api.openai.com/v1/chat/completions",
            cost_per_token=0.00003,
            status="active"
        )
        
        db_session.add(model)
        db_session.commit()
        
        retrieved = db_session.query(Model).filter(Model.name == "gpt-4").first()
        assert retrieved is not None
        assert retrieved.type == "commercial"
        assert retrieved.cost_per_token == 0.00003
        assert retrieved.version == "1.0.0"  # Default value
    
    def test_model_unique_constraint(self, db_session):
        """Test model name-version unique constraint."""
        model1 = Model(name="test_model", type="fine-tuned", version="1.0.0")
        model2 = Model(name="test_model", type="fine-tuned", version="1.0.0")
        
        db_session.add(model1)
        db_session.commit()
        
        db_session.add(model2)
        with pytest.raises(Exception):  # Should raise integrity error
            db_session.commit()


class TestExperimentModel:
    """Test Experiment model."""
    
    def test_create_experiment(self, db_session):
        """Test creating an experiment."""
        # Create dataset and model first
        dataset = Dataset(name="exp_dataset", format="jsonl", size=100)
        model = Model(name="exp_model", type="fine-tuned")
        db_session.add_all([dataset, model])
        db_session.commit()
        
        experiment = Experiment(
            name="test_experiment",
            description="Test experiment",
            type="fine-tuning",
            status="created",
            dataset_id=dataset.id,
            model_id=model.id,
            config={"learning_rate": 0.001},
            tags=["test", "experiment"],
            notes="Test notes"
        )
        
        db_session.add(experiment)
        db_session.commit()
        
        retrieved = db_session.query(Experiment).filter(Experiment.name == "test_experiment").first()
        assert retrieved is not None
        assert retrieved.type == "fine-tuning"
        assert retrieved.status == "created"
        assert retrieved.config == {"learning_rate": 0.001}
        assert retrieved.tags == ["test", "experiment"]
        assert retrieved.total_cost_usd == 0.0  # Default value
    
    def test_experiment_relationships(self, db_session):
        """Test experiment relationships."""
        dataset = Dataset(name="rel_dataset", format="jsonl", size=100)
        model = Model(name="rel_model", type="fine-tuned")
        db_session.add_all([dataset, model])
        db_session.commit()
        
        experiment = Experiment(
            name="rel_experiment",
            type="evaluation",
            dataset_id=dataset.id,
            model_id=model.id
        )
        db_session.add(experiment)
        db_session.commit()
        
        # Test relationships
        assert experiment.dataset.name == "rel_dataset"
        assert experiment.model.name == "rel_model"
        assert len(dataset.experiments) == 1
        assert len(model.experiments) == 1


class TestEvaluationModel:
    """Test Evaluation model."""
    
    def test_create_evaluation(self, db_session):
        """Test creating an evaluation."""
        # Create dependencies
        dataset = Dataset(name="eval_dataset", format="jsonl", size=100)
        model = Model(name="eval_model", type="commercial")
        db_session.add_all([dataset, model])
        db_session.commit()
        
        experiment = Experiment(
            name="eval_experiment",
            type="evaluation",
            dataset_id=dataset.id,
            model_id=model.id
        )
        db_session.add(experiment)
        db_session.commit()
        
        evaluation = Evaluation(
            experiment_id=experiment.id,
            model_id=model.id,
            prompt="What is AI?",
            response="AI is artificial intelligence.",
            expected_response="AI stands for artificial intelligence.",
            metrics={"bleu": 0.85, "rouge": 0.78},
            llm_judge_scores={"helpfulness": 4, "clarity": 5},
            human_rating=4,
            human_feedback="Good response",
            latency_ms=250,
            token_count_input=10,
            token_count_output=15,
            cost_usd=0.001,
            evaluation_criteria=["accuracy", "helpfulness"],
            evaluator_version="1.0.0",
            extra_metadata={"temperature": 0.7}
        )
        
        db_session.add(evaluation)
        db_session.commit()
        
        retrieved = db_session.query(Evaluation).first()
        assert retrieved is not None
        assert retrieved.prompt == "What is AI?"
        assert retrieved.response == "AI is artificial intelligence."
        assert retrieved.metrics == {"bleu": 0.85, "rouge": 0.78}
        assert retrieved.human_rating == 4
        assert retrieved.latency_ms == 250
        assert retrieved.cost_usd == 0.001
    
    def test_evaluation_relationships(self, db_session):
        """Test evaluation relationships."""
        # Create dependencies
        dataset = Dataset(name="eval_dataset", format="jsonl", size=100)
        model = Model(name="eval_model", type="commercial")
        db_session.add_all([dataset, model])
        db_session.commit()
        
        experiment = Experiment(
            name="eval_experiment",
            type="evaluation",
            dataset_id=dataset.id,
            model_id=model.id
        )
        db_session.add(experiment)
        db_session.commit()
        
        evaluation = Evaluation(
            experiment_id=experiment.id,
            model_id=model.id,
            prompt="Test prompt",
            response="Test response"
        )
        db_session.add(evaluation)
        db_session.commit()
        
        # Test relationships
        assert evaluation.experiment.name == "eval_experiment"
        assert evaluation.model.name == "eval_model"
        assert len(experiment.evaluations) == 1
        assert len(model.evaluations) == 1


class TestComparisonResultModel:
    """Test ComparisonResult model."""
    
    def test_create_comparison_result(self, db_session):
        """Test creating a comparison result."""
        # Create models for comparison
        model1 = Model(name="model1", type="fine-tuned")
        model2 = Model(name="model2", type="commercial")
        db_session.add_all([model1, model2])
        db_session.commit()
        
        comparison = ComparisonResult(
            name="model_comparison_1",
            description="Comparison between model1 and model2",
            model_ids=[str(model1.id), str(model2.id)],
            prompt_variants=["Prompt 1", "Prompt 2"],
            evaluation_criteria=["accuracy", "speed"],
            results={
                "model1": {"accuracy": 0.85, "speed": 200},
                "model2": {"accuracy": 0.90, "speed": 150}
            },
            winner_model_id=model2.id,
            statistical_significance=0.05,
            sample_size=100
        )
        
        db_session.add(comparison)
        db_session.commit()
        
        retrieved = db_session.query(ComparisonResult).filter(
            ComparisonResult.name == "model_comparison_1"
        ).first()
        assert retrieved is not None
        assert len(retrieved.model_ids) == 2
        assert retrieved.winner_model_id == model2.id
        assert retrieved.statistical_significance == 0.05
        assert retrieved.sample_size == 100


class TestModelIndexes:
    """Test database indexes and constraints."""
    
    def test_dataset_indexes(self, db_session):
        """Test dataset indexes work correctly."""
        datasets = [
            Dataset(name=f"dataset_{i}", format="jsonl", size=100, domain="test")
            for i in range(5)
        ]
        db_session.add_all(datasets)
        db_session.commit()
        
        # Test name index
        result = db_session.query(Dataset).filter(Dataset.name == "dataset_1").first()
        assert result is not None
        
        # Test domain index
        results = db_session.query(Dataset).filter(Dataset.domain == "test").all()
        assert len(results) == 5
    
    def test_model_indexes(self, db_session):
        """Test model indexes work correctly."""
        models = [
            Model(name=f"model_{i}", type="fine-tuned", status="active")
            for i in range(3)
        ]
        models.append(Model(name="model_inactive", type="fine-tuned", status="inactive"))
        
        db_session.add_all(models)
        db_session.commit()
        
        # Test type index
        fine_tuned = db_session.query(Model).filter(Model.type == "fine-tuned").all()
        assert len(fine_tuned) == 4
        
        # Test status index
        active = db_session.query(Model).filter(Model.status == "active").all()
        assert len(active) == 3