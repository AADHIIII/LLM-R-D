"""
Tests for result storage and retrieval service.
"""

import pytest
import uuid
import json
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.models import Base, Dataset, Model, Experiment, Evaluation
from database.result_service import ResultStorageService, SearchFilters, ExportOptions
from database.connection import DatabaseManager
from utils.exceptions import DatabaseError, ValidationError


@pytest.fixture
def db_service():
    """Create in-memory database service for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    
    # Create a test database manager
    db_manager = DatabaseManager("sqlite:///:memory:")
    db_manager.engine = engine
    db_manager.session_factory = sessionmaker(bind=engine)
    
    service = ResultStorageService()
    service.db_manager = db_manager
    
    return service


@pytest.fixture
def sample_data(db_service):
    """Create sample data for testing."""
    with db_service.db_manager.get_session() as session:
        # Create dataset
        dataset = Dataset(
            name="test_dataset",
            format="jsonl",
            size=100,
            domain="test"
        )
        session.add(dataset)
        session.flush()
        
        # Create models
        model1 = Model(
            name="gpt2-fine",
            type="fine-tuned",
            base_model="gpt2",
            status="active"
        )
        model2 = Model(
            name="gpt-4",
            type="commercial",
            cost_per_token=0.00003,
            status="active"
        )
        session.add_all([model1, model2])
        session.flush()
        
        # Create experiments
        experiment1 = Experiment(
            name="test_experiment_1",
            type="evaluation",
            status="running",
            dataset_id=dataset.id,
            model_id=model1.id
        )
        experiment2 = Experiment(
            name="test_experiment_2",
            type="fine-tuning",
            status="completed",
            dataset_id=dataset.id,
            model_id=model2.id,
            results={"accuracy": 0.85},
            metrics={"loss": 0.15}
        )
        session.add_all([experiment1, experiment2])
        session.flush()
        
        # Create evaluations
        evaluations = []
        for i in range(5):
            evaluation = Evaluation(
                experiment_id=experiment1.id,
                model_id=model1.id,
                prompt=f"Test prompt {i}",
                response=f"Test response {i}",
                human_rating=4 + (i % 2),  # Ratings 4 or 5
                cost_usd=0.001 + (i * 0.0005),
                latency_ms=200 + (i * 10),
                metrics={"bleu": 0.8 + (i * 0.02)},
                llm_judge_scores={"helpfulness": 4, "clarity": 5}
            )
            evaluations.append(evaluation)
            session.add(evaluation)
        
        session.commit()
        
        # Store IDs to avoid detached instance issues
        return {
            'dataset': dataset,
            'dataset_id': dataset.id,
            'models': [model1, model2],
            'model_ids': [model1.id, model2.id],
            'experiments': [experiment1, experiment2],
            'experiment_ids': [experiment1.id, experiment2.id],
            'evaluations': evaluations
        }


class TestResultStorageService:
    """Test result storage service."""
    
    def test_store_experiment_results(self, db_service, sample_data):
        """Test storing experiment results."""
        experiment_id = sample_data['experiment_ids'][0]
        
        results = {"accuracy": 0.92, "f1_score": 0.89}
        metrics = {"precision": 0.91, "recall": 0.87}
        cost_data = {"total_cost": 0.05, "token_usage": {"input": 1000, "output": 500}}
        
        success = db_service.store_experiment_results(
            experiment_id, results, metrics, cost_data
        )
        
        assert success is True
        
        # Verify storage
        with db_service.db_manager.get_session() as session:
            updated_exp = session.query(Experiment).filter(
                Experiment.id == experiment_id
            ).first()
            
            assert updated_exp.results == results
            assert updated_exp.metrics == metrics
            assert updated_exp.total_cost_usd == 0.05
            assert updated_exp.token_usage == {"input": 1000, "output": 500}
            assert updated_exp.status == "completed"
    
    def test_store_experiment_results_not_found(self, db_service):
        """Test storing results for non-existent experiment."""
        fake_id = uuid.uuid4()
        
        with pytest.raises(DatabaseError, match="Result storage failed"):
            db_service.store_experiment_results(fake_id, {"test": "data"})
    
    def test_store_evaluation_batch(self, db_service, sample_data):
        """Test storing batch of evaluations."""
        experiment_id = sample_data['experiment_ids'][0]
        model_id = sample_data['model_ids'][0]
        
        evaluations_data = [
            {
                'experiment_id': experiment_id,
                'model_id': model_id,
                'prompt': f'Batch prompt {i}',
                'response': f'Batch response {i}',
                'cost_usd': 0.002,
                'human_rating': 4
            }
            for i in range(3)
        ]
        
        stored_evaluations = db_service.store_evaluation_batch(evaluations_data)
        
        assert len(stored_evaluations) == 3
        for i, evaluation in enumerate(stored_evaluations):
            assert evaluation.prompt == f'Batch prompt {i}'
            assert evaluation.cost_usd == 0.002
    
    def test_search_results_basic(self, db_service, sample_data):
        """Test basic result search."""
        filters = SearchFilters()
        
        results, total_count = db_service.search_results(filters)
        
        assert total_count == 5  # 5 evaluations created in sample data
        assert len(results) == 5
        
        # Check result structure
        result = results[0]
        assert 'id' in result
        assert 'prompt' in result
        assert 'response' in result
        assert 'model_name' in result
        assert 'experiment_name' in result
    
    def test_search_results_with_text_filter(self, db_service, sample_data):
        """Test search with text filter."""
        filters = SearchFilters(query_text="prompt 1")
        
        results, total_count = db_service.search_results(filters)
        
        assert total_count == 1
        assert "prompt 1" in results[0]['prompt']
    
    def test_search_results_with_rating_filter(self, db_service, sample_data):
        """Test search with rating filter."""
        filters = SearchFilters(min_rating=5)
        
        results, total_count = db_service.search_results(filters)
        
        # Should find evaluations with rating 5
        assert total_count > 0
        for result in results:
            assert result['human_rating'] >= 5
    
    def test_search_results_with_cost_filter(self, db_service, sample_data):
        """Test search with cost filter."""
        filters = SearchFilters(max_cost=0.002)
        
        results, total_count = db_service.search_results(filters)
        
        assert total_count > 0
        for result in results:
            assert result['cost_usd'] <= 0.002
    
    def test_search_results_with_date_filter(self, db_service, sample_data):
        """Test search with date filter."""
        yesterday = datetime.utcnow() - timedelta(days=1)
        tomorrow = datetime.utcnow() + timedelta(days=1)
        
        filters = SearchFilters(date_from=yesterday, date_to=tomorrow)
        
        results, total_count = db_service.search_results(filters)
        
        assert total_count == 5  # All evaluations should be within range
    
    def test_search_results_with_experiment_filter(self, db_service, sample_data):
        """Test search with experiment filter."""
        experiment_id = sample_data['experiment_ids'][0]
        filters = SearchFilters(experiment_ids=[experiment_id])
        
        results, total_count = db_service.search_results(filters)
        
        assert total_count == 5  # All evaluations belong to this experiment
        for result in results:
            assert result['experiment_id'] == str(experiment_id)
    
    def test_search_results_with_pagination(self, db_service, sample_data):
        """Test search with pagination."""
        filters = SearchFilters()
        
        # First page
        results1, total_count = db_service.search_results(filters, limit=2, offset=0)
        assert len(results1) == 2
        assert total_count == 5
        
        # Second page
        results2, _ = db_service.search_results(filters, limit=2, offset=2)
        assert len(results2) == 2
        
        # Results should be different
        assert results1[0]['id'] != results2[0]['id']
    
    def test_search_results_with_sorting(self, db_service, sample_data):
        """Test search with sorting."""
        filters = SearchFilters()
        
        # Sort by cost ascending
        results_asc, _ = db_service.search_results(
            filters, sort_by='cost_usd', sort_order='asc'
        )
        
        # Sort by cost descending
        results_desc, _ = db_service.search_results(
            filters, sort_by='cost_usd', sort_order='desc'
        )
        
        # First result in ascending should have lower cost than first in descending
        assert results_asc[0]['cost_usd'] <= results_desc[0]['cost_usd']
    
    def test_get_experiment_summary(self, db_service, sample_data):
        """Test getting experiment summary."""
        experiment_id = sample_data['experiment_ids'][0]
        
        summary = db_service.get_experiment_summary(experiment_id)
        
        assert 'experiment' in summary
        assert 'dataset' in summary
        assert 'model' in summary
        assert 'evaluation_stats' in summary
        assert 'rating_distribution' in summary
        
        # Check experiment data
        exp_data = summary['experiment']
        assert exp_data['name'] == "test_experiment_1"
        assert exp_data['type'] == "evaluation"
        
        # Check evaluation stats
        eval_stats = summary['evaluation_stats']
        assert eval_stats['total_evaluations'] == 5
        assert eval_stats['human_rated_count'] == 5
    
    def test_get_experiment_summary_not_found(self, db_service):
        """Test getting summary for non-existent experiment."""
        fake_id = uuid.uuid4()
        
        with pytest.raises(DatabaseError, match="Summary retrieval failed"):
            db_service.get_experiment_summary(fake_id)
    
    def test_export_csv(self, db_service, sample_data):
        """Test CSV export."""
        filters = SearchFilters()
        options = ExportOptions(format='csv', include_metadata=True)
        
        csv_data = db_service.export_results(filters, options)
        
        assert isinstance(csv_data, str)
        assert 'id,experiment_name,model_name' in csv_data
        assert 'Test prompt' in csv_data
        
        # Check that all rows are present
        lines = csv_data.strip().split('\n')
        assert len(lines) == 6  # Header + 5 data rows
    
    def test_export_json(self, db_service, sample_data):
        """Test JSON export."""
        filters = SearchFilters()
        options = ExportOptions(format='json')
        
        json_data = db_service.export_results(filters, options)
        
        assert isinstance(json_data, str)
        
        # Parse and validate JSON
        data = json.loads(json_data)
        assert 'export_timestamp' in data
        assert 'total_records' in data
        assert 'results' in data
        assert data['total_records'] == 5
        assert len(data['results']) == 5
    
    def test_export_pdf(self, db_service, sample_data):
        """Test PDF export."""
        filters = SearchFilters()
        options = ExportOptions(format='pdf')
        
        pdf_data = db_service.export_results(filters, options)
        
        assert isinstance(pdf_data, bytes)
        
        # Convert to string to check content
        content = pdf_data.decode('utf-8')
        assert 'LLM Optimization Platform' in content
        assert 'Results Export' in content
        assert 'Total Records: 5' in content
    
    def test_export_unsupported_format(self, db_service, sample_data):
        """Test export with unsupported format."""
        filters = SearchFilters()
        options = ExportOptions(format='xml')
        
        with pytest.raises(DatabaseError, match="Export failed"):
            db_service.export_results(filters, options)
    
    def test_export_with_max_records(self, db_service, sample_data):
        """Test export with record limit."""
        filters = SearchFilters()
        options = ExportOptions(format='json', max_records=2)
        
        json_data = db_service.export_results(filters, options)
        data = json.loads(json_data)
        
        assert len(data['results']) == 2
    
    def test_get_analytics_data(self, db_service, sample_data):
        """Test getting analytics data."""
        analytics = db_service.get_analytics_data()
        
        assert 'overview' in analytics
        assert 'model_performance' in analytics
        assert 'recent_experiments' in analytics
        
        # Check overview
        overview = analytics['overview']
        assert overview['total_experiments'] == 2
        assert overview['total_evaluations'] == 5
        assert overview['active_models'] == 2
        assert overview['total_cost_usd'] > 0
        
        # Check model performance
        model_perf = analytics['model_performance']
        assert len(model_perf) == 1  # Only one model has evaluations
        assert model_perf[0]['model_name'] == 'gpt2-fine'
        assert model_perf[0]['evaluation_count'] == 5
        
        # Check recent experiments
        recent = analytics['recent_experiments']
        assert len(recent) == 2
        assert all('id' in exp for exp in recent)
        assert all('name' in exp for exp in recent)


class TestSearchFilters:
    """Test search filters functionality."""
    
    def test_search_filters_creation(self):
        """Test creating search filters."""
        filters = SearchFilters(
            query_text="test",
            min_rating=4,
            max_cost=0.01
        )
        
        assert filters.query_text == "test"
        assert filters.min_rating == 4
        assert filters.max_cost == 0.01
        assert filters.experiment_ids is None  # Default value
    
    def test_search_filters_with_lists(self):
        """Test search filters with list values."""
        exp_ids = [uuid.uuid4(), uuid.uuid4()]
        model_ids = [uuid.uuid4()]
        
        filters = SearchFilters(
            experiment_ids=exp_ids,
            model_ids=model_ids
        )
        
        assert filters.experiment_ids == exp_ids
        assert filters.model_ids == model_ids


class TestExportOptions:
    """Test export options functionality."""
    
    def test_export_options_defaults(self):
        """Test default export options."""
        options = ExportOptions()
        
        assert options.format == 'csv'
        assert options.include_metadata is True
        assert options.include_human_feedback is True
        assert options.include_metrics is True
        assert options.max_records is None
    
    def test_export_options_custom(self):
        """Test custom export options."""
        options = ExportOptions(
            format='json',
            include_metadata=False,
            max_records=100
        )
        
        assert options.format == 'json'
        assert options.include_metadata is False
        assert options.max_records == 100


class TestResultServiceErrorHandling:
    """Test error handling in result service."""
    
    def test_database_error_handling(self, db_service):
        """Test database error handling."""
        # Close the database connection to force an error
        db_service.db_manager.engine.dispose()
        
        filters = SearchFilters()
        
        with pytest.raises(DatabaseError):
            db_service.search_results(filters)