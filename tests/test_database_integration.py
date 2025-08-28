"""
Integration tests for the complete database layer.
"""

import pytest
import tempfile
import os
from datetime import datetime

from database import (
    DatabaseManager, Dataset, Model, Experiment, Evaluation,
    dataset_repository, model_repository, experiment_repository, evaluation_repository,
    ResultStorageService, SearchFilters, ExportOptions
)


@pytest.fixture
def integrated_db():
    """Create integrated database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_url = f"sqlite:///{tmp.name}"
        db_manager = DatabaseManager(db_url)
        db_manager.initialize()
        
        yield db_manager
        
        db_manager.close()
        os.unlink(tmp.name)


class TestDatabaseIntegration:
    """Test complete database layer integration."""
    
    def test_complete_workflow(self, integrated_db):
        """Test complete workflow from dataset to results export."""
        db_manager = integrated_db
        result_service = ResultStorageService()
        result_service.db_manager = db_manager
        
        with db_manager.get_session() as session:
            # 1. Create dataset
            dataset = dataset_repository.create(
                session,
                name="integration_dataset",
                format="jsonl",
                size=1000,
                domain="nlp",
                extra_metadata={"source": "test"}
            )
            
            # 2. Create models
            fine_tuned_model = model_repository.create(
                session,
                name="test-gpt2-fine",
                type="fine-tuned",
                base_model="gpt2",
                status="active",
                training_config={"epochs": 3, "batch_size": 4}
            )
            
            commercial_model = model_repository.create(
                session,
                name="gpt-4",
                type="commercial",
                cost_per_token=0.00003,
                status="active"
            )
            
            # 3. Create experiment
            experiment = experiment_repository.create(
                session,
                name="integration_experiment",
                type="evaluation",
                status="running",
                dataset_id=dataset.id,
                model_id=fine_tuned_model.id,
                config={"temperature": 0.7}
            )
            
            session.commit()
            
            # 4. Store evaluation results
            evaluations_data = [
                {
                    'experiment_id': experiment.id,
                    'model_id': fine_tuned_model.id,
                    'prompt': f'Integration test prompt {i}',
                    'response': f'Integration test response {i}',
                    'human_rating': 4 + (i % 2),
                    'cost_usd': 0.001 + (i * 0.0005),
                    'latency_ms': 200 + (i * 10),
                    'metrics': {'bleu': 0.8 + (i * 0.02)},
                    'llm_judge_scores': {'helpfulness': 4, 'clarity': 5}
                }
                for i in range(10)
            ]
            
            stored_evaluations = evaluation_repository.create_batch(
                session, evaluations_data
            )
            
            session.commit()
            
            # 5. Update experiment with results
            experiment_results = {
                "total_evaluations": len(stored_evaluations),
                "avg_rating": sum(e.human_rating for e in stored_evaluations) / len(stored_evaluations),
                "completion_status": "success"
            }
            
            success = result_service.store_experiment_results(
                experiment.id,
                experiment_results,
                metrics={"avg_bleu": 0.85},
                cost_data={"total_cost": 0.05, "token_usage": {"input": 1000, "output": 500}}
            )
            
            assert success is True
            
            # 6. Search and filter results
            filters = SearchFilters(
                query_text="Integration test",
                min_rating=4,
                experiment_ids=[experiment.id]
            )
            
            search_results, total_count = result_service.search_results(filters)
            
            assert total_count == 10
            assert len(search_results) == 10
            assert all("Integration test" in r['prompt'] for r in search_results)
            
            # 7. Get experiment summary
            summary = result_service.get_experiment_summary(experiment.id)
            
            assert summary['experiment']['name'] == "integration_experiment"
            assert summary['experiment']['status'] == "completed"
            assert summary['evaluation_stats']['total_evaluations'] == 10
            assert summary['dataset']['name'] == "integration_dataset"
            assert summary['model']['name'] == "test-gpt2-fine"
            
            # 8. Export results
            export_options = ExportOptions(format='csv', include_metadata=True)
            csv_data = result_service.export_results(filters, export_options)
            
            assert isinstance(csv_data, str)
            assert "Integration test prompt" in csv_data
            assert len(csv_data.split('\n')) == 12  # Header + 10 data rows + empty line
            
            # 9. Get analytics
            analytics = result_service.get_analytics_data()
            
            assert analytics['overview']['total_experiments'] == 1
            assert analytics['overview']['total_evaluations'] == 10
            assert analytics['overview']['active_models'] == 2
            assert len(analytics['model_performance']) == 1
            assert analytics['model_performance'][0]['model_name'] == "test-gpt2-fine"
            
            # 10. Test repository queries
            # Test dataset queries
            datasets = dataset_repository.list_datasets(session, domain="nlp")
            assert len(datasets) == 1
            assert datasets[0].name == "integration_dataset"
            
            # Test model queries
            fine_tuned_models = model_repository.get_fine_tuned_models(session)
            commercial_models = model_repository.get_commercial_models(session)
            assert len(fine_tuned_models) == 1
            assert len(commercial_models) == 1
            
            # Test experiment queries
            running_experiments = experiment_repository.get_running_experiments(session)
            assert len(running_experiments) == 0  # Should be completed now
            
            completed_experiments = experiment_repository.list_experiments(
                session, status="completed"
            )
            assert len(completed_experiments) == 1
            
            # Test evaluation queries
            experiment_evaluations = evaluation_repository.get_by_experiment(
                session, experiment.id
            )
            assert len(experiment_evaluations) == 10
            
            eval_stats = evaluation_repository.get_evaluation_stats(
                session, experiment_id=experiment.id
            )
            assert eval_stats['total_evaluations'] == 10
            assert eval_stats['human_rated_count'] == 10
            assert eval_stats['average_human_rating'] == 4.5
    
    def test_database_health_and_migration(self, integrated_db):
        """Test database health check and migration."""
        db_manager = integrated_db
        
        # Test health check
        assert db_manager.health_check() is True
        
        # Test migration
        db_manager.migrate_schema()
        
        # Verify tables exist after migration
        with db_manager.get_session() as session:
            from sqlalchemy import text
            tables = session.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).fetchall()
            table_names = [table[0] for table in tables]
            
            expected_tables = ['datasets', 'models', 'experiments', 'evaluations', 'comparison_results']
            for table in expected_tables:
                assert table in table_names
    
    def test_error_handling_integration(self, integrated_db):
        """Test error handling across the database layer."""
        db_manager = integrated_db
        result_service = ResultStorageService()
        result_service.db_manager = db_manager
        
        # Test invalid experiment ID
        import uuid
        fake_id = uuid.uuid4()
        
        from utils.exceptions import DatabaseError
        with pytest.raises(DatabaseError):
            result_service.get_experiment_summary(fake_id)
        
        # Test invalid search
        filters = SearchFilters(query_text="nonexistent")
        results, count = result_service.search_results(filters)
        assert count == 0
        assert len(results) == 0
    
    def test_performance_with_large_dataset(self, integrated_db):
        """Test performance with larger dataset."""
        db_manager = integrated_db
        
        with db_manager.get_session() as session:
            # Create dataset and model
            dataset = dataset_repository.create(
                session,
                name="large_dataset",
                format="jsonl",
                size=10000,
                domain="performance_test"
            )
            
            model = model_repository.create(
                session,
                name="performance_model",
                type="fine-tuned",
                status="active"
            )
            
            experiment = experiment_repository.create(
                session,
                name="performance_experiment",
                type="evaluation",
                dataset_id=dataset.id,
                model_id=model.id
            )
            
            session.commit()
            
            # Create batch of evaluations
            batch_size = 100
            evaluations_data = [
                {
                    'experiment_id': experiment.id,
                    'model_id': model.id,
                    'prompt': f'Performance test prompt {i}',
                    'response': f'Performance test response {i}',
                    'human_rating': (i % 5) + 1,
                    'cost_usd': 0.001,
                    'latency_ms': 200
                }
                for i in range(batch_size)
            ]
            
            # Measure batch creation time
            import time
            start_time = time.time()
            
            stored_evaluations = evaluation_repository.create_batch(
                session, evaluations_data
            )
            session.commit()
            
            end_time = time.time()
            creation_time = end_time - start_time
            
            assert len(stored_evaluations) == batch_size
            assert creation_time < 5.0  # Should complete within 5 seconds
            
            # Test search performance
            start_time = time.time()
            
            filters = SearchFilters(experiment_ids=[experiment.id])
            result_service = ResultStorageService()
            result_service.db_manager = db_manager
            
            results, total_count = result_service.search_results(filters)
            
            end_time = time.time()
            search_time = end_time - start_time
            
            assert total_count == batch_size
            assert search_time < 2.0  # Search should be fast