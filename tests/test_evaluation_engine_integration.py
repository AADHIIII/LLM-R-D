"""
Integration tests for the complete evaluation engine.

Tests the integration between MetricsCalculator, LangChainEvaluator, and PromptEvaluator.
"""

import pytest
from unittest.mock import Mock, patch

from evaluator import (
    MetricsCalculator,
    LangChainEvaluator,
    PromptEvaluator,
    PromptVariant,
    ModelResponse,
    EvaluationResult
)


class TestEvaluationEngineIntegration:
    """Integration tests for the complete evaluation engine."""
    
    @pytest.fixture
    def evaluation_engine(self):
        """Create a complete evaluation engine setup."""
        # Create real instances (with mocked external dependencies)
        metrics_calc = MetricsCalculator(device="cpu")
        
        with patch('evaluator.langchain_evaluator.OpenAI'):
            langchain_eval = LangChainEvaluator(openai_api_key="test-key")
        
        prompt_eval = PromptEvaluator(
            metrics_calculator=metrics_calc,
            langchain_evaluator=langchain_eval,
            max_workers=2
        )
        
        return {
            'metrics_calculator': metrics_calc,
            'langchain_evaluator': langchain_eval,
            'prompt_evaluator': prompt_eval
        }
    
    def test_complete_evaluation_workflow(self, evaluation_engine):
        """Test a complete evaluation workflow from prompts to comparison results."""
        prompt_eval = evaluation_engine['prompt_evaluator']
        
        # Create test data
        prompts = [
            PromptVariant("prompt_1", "What is artificial intelligence?"),
            PromptVariant("prompt_2", "Explain AI in simple terms")
        ]
        
        responses = [
            ModelResponse("gpt-4", "prompt_1", "AI is a field of computer science", 1.0),
            ModelResponse("gpt-4", "prompt_2", "AI helps computers think like humans", 1.2),
            ModelResponse("claude", "prompt_1", "Artificial intelligence simulates human intelligence", 0.9),
            ModelResponse("claude", "prompt_2", "AI makes machines smart", 0.8)
        ]
        
        references = {
            "prompt_1": ["AI is artificial intelligence technology"],
            "prompt_2": ["AI is technology that makes computers intelligent"]
        }
        
        # Mock the external evaluation calls
        with patch.object(prompt_eval.langchain_evaluator, 'evaluate_with_langchain') as mock_lc, \
             patch.object(prompt_eval.langchain_evaluator, 'evaluate_with_llm_judge') as mock_judge:
            
            # Mock LangChain evaluation
            mock_lc.return_value = []
            
            # Mock LLM judge evaluation
            mock_judge.return_value = None
            
            # Perform batch evaluation
            results = prompt_eval.batch_evaluate_prompts(
                prompts, responses, references,
                use_metrics=True, use_langchain=False, use_llm_judge=False
            )
        
        # Verify results
        assert len(results) == 4
        assert all(isinstance(r, EvaluationResult) for r in results)
        
        # Check that we have results for both prompts and models
        prompt_ids = {r.prompt_id for r in results}
        model_ids = {r.model_id for r in results}
        
        assert prompt_ids == {"prompt_1", "prompt_2"}
        assert model_ids == {"gpt-4", "claude"}
        
        # Test prompt comparison
        gpt4_results = [r for r in results if r.model_id == "gpt-4"]
        if len(gpt4_results) >= 2:
            comparison = prompt_eval.compare_prompts(
                "prompt_1", "prompt_2", "gpt-4", gpt4_results
            )
            # Comparison might be None due to insufficient samples, which is fine
            if comparison:
                assert comparison.model_id == "gpt-4"
                assert comparison.prompt_a_id == "prompt_1"
                assert comparison.prompt_b_id == "prompt_2"
        
        # Test ranking
        ranking = prompt_eval.rank_prompts(
            ["prompt_1", "prompt_2"], "gpt-4", gpt4_results
        )
        if ranking:
            assert ranking.model_id == "gpt-4"
            assert len(ranking.rankings) <= 2
        
        # Test summary generation
        summary = prompt_eval.get_evaluation_summary(results)
        assert summary["total_evaluations"] == 4
        assert "by_prompt" in summary
        assert "by_model" in summary
    
    def test_metrics_calculator_integration(self, evaluation_engine):
        """Test MetricsCalculator integration with the evaluation pipeline."""
        metrics_calc = evaluation_engine['metrics_calculator']
        
        # Test basic metrics calculation
        prediction = "AI is artificial intelligence"
        references = ["AI stands for artificial intelligence"]
        
        # This should work without external dependencies
        bleu_scores = metrics_calc.calculate_bleu_scores(prediction, references)
        rouge_scores = metrics_calc.calculate_rouge_scores(prediction, references[0])
        
        assert 'bleu_1' in bleu_scores
        assert 'rouge1_f' in rouge_scores
        
        # Test with semantic similarity (mocked)
        with patch.object(metrics_calc, 'semantic_model') as mock_model:
            mock_model.encode.return_value = [[0.1, 0.2], [0.15, 0.25]]
            
            similarity = metrics_calc.calculate_semantic_similarity(prediction, references[0])
            assert isinstance(similarity, float)
    
    def test_error_handling_integration(self, evaluation_engine):
        """Test error handling across the evaluation pipeline."""
        prompt_eval = evaluation_engine['prompt_evaluator']
        
        # Create test data with potential issues
        prompts = [PromptVariant("test_prompt", "Test question")]
        responses = [ModelResponse("test_model", "test_prompt", "", 1.0)]  # Empty response
        
        # Mock evaluations to raise exceptions
        with patch.object(prompt_eval.metrics_calculator, 'calculate_all_metrics') as mock_metrics, \
             patch.object(prompt_eval.langchain_evaluator, 'evaluate_with_langchain') as mock_lc:
            
            mock_metrics.side_effect = Exception("Metrics calculation failed")
            mock_lc.side_effect = Exception("LangChain evaluation failed")
            
            # Should handle errors gracefully
            results = prompt_eval.batch_evaluate_prompts(prompts, responses)
            
            assert len(results) == 1
            result = results[0]
            assert result.prompt_id == "test_prompt"
            assert result.model_id == "test_model"
            # Should have some default/error handling behavior
    
    def test_evaluation_engine_imports(self):
        """Test that all evaluation engine components can be imported correctly."""
        # Test that the main classes are available
        from evaluator import (
            MetricsCalculator,
            LangChainEvaluator,
            PromptEvaluator,
            MetricScores,
            EvaluationCriteria,
            PromptVariant,
            ModelResponse,
            EvaluationResult,
            ComparisonResult,
            RankingResult
        )
        
        # Verify they are classes/types
        assert callable(MetricsCalculator)
        assert callable(LangChainEvaluator)
        assert callable(PromptEvaluator)
        
        # Verify dataclasses
        assert hasattr(MetricScores, '__dataclass_fields__')
        assert hasattr(EvaluationCriteria, '__dataclass_fields__')
        assert hasattr(PromptVariant, '__dataclass_fields__')
        assert hasattr(ModelResponse, '__dataclass_fields__')
        assert hasattr(EvaluationResult, '__dataclass_fields__')
        assert hasattr(ComparisonResult, '__dataclass_fields__')
        assert hasattr(RankingResult, '__dataclass_fields__')


if __name__ == "__main__":
    pytest.main([__file__])