"""
Tests for PromptEvaluator class.

Comprehensive tests for prompt comparison system, statistical significance testing,
and result aggregation.
"""

import pytest
import statistics
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from evaluator.prompt_evaluator import (
    PromptEvaluator,
    PromptVariant,
    ModelResponse,
    EvaluationResult,
    ComparisonResult,
    RankingResult
)
from evaluator.metrics_calculator import MetricScores
from evaluator.langchain_evaluator import LangChainEvaluationResult, LLMJudgeResult


class TestPromptVariant:
    """Test suite for PromptVariant dataclass."""
    
    def test_prompt_variant_creation(self):
        """Test PromptVariant creation."""
        variant = PromptVariant(
            id="prompt_1",
            prompt="What is AI?",
            description="Basic AI question",
            metadata={"category": "general"}
        )
        
        assert variant.id == "prompt_1"
        assert variant.prompt == "What is AI?"
        assert variant.description == "Basic AI question"
        assert variant.metadata["category"] == "general"
    
    def test_prompt_variant_defaults(self):
        """Test PromptVariant default values."""
        variant = PromptVariant(
            id="prompt_2",
            prompt="Explain machine learning"
        )
        
        assert variant.description is None
        assert variant.metadata == {}


class TestModelResponse:
    """Test suite for ModelResponse dataclass."""
    
    def test_model_response_creation(self):
        """Test ModelResponse creation."""
        response = ModelResponse(
            model_id="gpt-4",
            prompt_id="prompt_1",
            response="AI is artificial intelligence",
            generation_time=1.5,
            metadata={"tokens": 100}
        )
        
        assert response.model_id == "gpt-4"
        assert response.prompt_id == "prompt_1"
        assert response.response == "AI is artificial intelligence"
        assert response.generation_time == 1.5
        assert response.metadata["tokens"] == 100


class TestEvaluationResult:
    """Test suite for EvaluationResult dataclass."""
    
    def test_evaluation_result_creation(self):
        """Test EvaluationResult creation."""
        result = EvaluationResult(
            prompt_id="prompt_1",
            model_id="gpt-4",
            response="AI is artificial intelligence",
            combined_score=0.85,
            confidence=0.9
        )
        
        assert result.prompt_id == "prompt_1"
        assert result.model_id == "gpt-4"
        assert result.combined_score == 0.85
        assert result.confidence == 0.9
    
    def test_evaluation_result_optional_fields(self):
        """Test EvaluationResult with optional fields."""
        result = EvaluationResult(
            prompt_id="prompt_1",
            model_id="gpt-4",
            response="AI is artificial intelligence"
        )
        
        assert result.metric_scores is None
        assert result.langchain_results is None
        assert result.llm_judge_result is None


class TestComparisonResult:
    """Test suite for ComparisonResult dataclass."""
    
    def test_comparison_result_creation(self):
        """Test ComparisonResult creation."""
        result = ComparisonResult(
            prompt_a_id="prompt_1",
            prompt_b_id="prompt_2",
            model_id="gpt-4",
            win_rate_a=0.6,
            win_rate_b=0.3,
            tie_rate=0.1,
            p_value=0.02,
            is_significant=True,
            effect_size=0.5,
            mean_score_a=0.8,
            mean_score_b=0.7,
            score_difference=0.1,
            confidence_interval=(0.05, 0.15),
            n_comparisons=100
        )
        
        assert result.prompt_a_id == "prompt_1"
        assert result.win_rate_a == 0.6
        assert result.is_significant is True
        assert result.confidence_interval == (0.05, 0.15)


class TestRankingResult:
    """Test suite for RankingResult dataclass."""
    
    def test_ranking_result_creation(self):
        """Test RankingResult creation."""
        rankings = [
            {"prompt_id": "prompt_1", "rank": 1, "score": 0.9},
            {"prompt_id": "prompt_2", "rank": 2, "score": 0.8}
        ]
        
        result = RankingResult(
            model_id="gpt-4",
            rankings=rankings,
            ranking_confidence=0.85
        )
        
        assert result.model_id == "gpt-4"
        assert len(result.rankings) == 2
        assert result.rankings[0]["rank"] == 1
        assert result.ranking_confidence == 0.85


class TestPromptEvaluator:
    """Test suite for PromptEvaluator class."""
    
    @pytest.fixture
    def mock_metrics_calculator(self):
        """Create a mock MetricsCalculator."""
        mock_calc = Mock()
        mock_calc.calculate_all_metrics.return_value = MetricScores(
            bleu_1=0.8, bleu_2=0.7, bleu_3=0.6, bleu_4=0.5,
            rouge_1_f=0.75, rouge_1_p=0.8, rouge_1_r=0.7,
            rouge_2_f=0.65, rouge_2_p=0.7, rouge_2_r=0.6,
            rouge_l_f=0.7, rouge_l_p=0.75, rouge_l_r=0.65,
            semantic_similarity=0.85
        )
        return mock_calc
    
    @pytest.fixture
    def mock_langchain_evaluator(self):
        """Create a mock LangChainEvaluator."""
        mock_eval = Mock()
        mock_eval.evaluate_with_langchain.return_value = [
            LangChainEvaluationResult("helpfulness", 4.0, "Good response"),
            LangChainEvaluationResult("clarity", 4.5, "Very clear")
        ]
        mock_eval.evaluate_with_llm_judge.return_value = LLMJudgeResult(
            overall_score=4.2,
            criteria_scores={"helpfulness": 4.0, "clarity": 4.5},
            reasoning="Well-structured response",
            confidence=0.9,
            evaluation_time=2.0
        )
        return mock_eval
    
    @pytest.fixture
    def evaluator(self, mock_metrics_calculator, mock_langchain_evaluator):
        """Create a PromptEvaluator instance for testing."""
        return PromptEvaluator(
            metrics_calculator=mock_metrics_calculator,
            langchain_evaluator=mock_langchain_evaluator,
            max_workers=2,
            significance_threshold=0.05,
            min_samples=5
        )
    
    def test_initialization(self, evaluator):
        """Test PromptEvaluator initialization."""
        assert evaluator.max_workers == 2
        assert evaluator.significance_threshold == 0.05
        assert evaluator.min_samples == 5
        assert evaluator.metrics_calculator is not None
        assert evaluator.langchain_evaluator is not None
    
    def test_calculate_combined_score(self, evaluator):
        """Test combined score calculation."""
        # Create a result with all evaluation types
        result = EvaluationResult(
            prompt_id="test",
            model_id="test",
            response="test response"
        )
        
        # Add metric scores
        result.metric_scores = MetricScores(
            bleu_1=0.8, bleu_2=0.7, bleu_3=0.6, bleu_4=0.5,
            rouge_1_f=0.75, rouge_1_p=0.8, rouge_1_r=0.7,
            rouge_2_f=0.65, rouge_2_p=0.7, rouge_2_r=0.6,
            rouge_l_f=0.7, rouge_l_p=0.75, rouge_l_r=0.65,
            semantic_similarity=0.85
        )
        
        # Add LangChain results
        result.langchain_results = [
            LangChainEvaluationResult("helpfulness", 4.0, "Good"),
            LangChainEvaluationResult("clarity", 4.5, "Clear")
        ]
        
        # Add LLM judge result
        result.llm_judge_result = LLMJudgeResult(
            overall_score=4.2,
            criteria_scores={},
            reasoning="Good",
            confidence=0.9,
            evaluation_time=1.0
        )
        
        combined_score = evaluator._calculate_combined_score(result)
        
        assert combined_score is not None
        assert 0.0 <= combined_score <= 1.0
    
    def test_calculate_combined_score_no_data(self, evaluator):
        """Test combined score calculation with no evaluation data."""
        result = EvaluationResult(
            prompt_id="test",
            model_id="test",
            response="test response"
        )
        
        combined_score = evaluator._calculate_combined_score(result)
        
        assert combined_score is None
    
    def test_calculate_confidence(self, evaluator):
        """Test confidence calculation."""
        result = EvaluationResult(
            prompt_id="test",
            model_id="test",
            response="test response"
        )
        
        # Add LLM judge result with confidence
        result.llm_judge_result = LLMJudgeResult(
            overall_score=4.0,
            criteria_scores={},
            reasoning="Good",
            confidence=4.5,  # Out of 5
            evaluation_time=1.0
        )
        
        # Add other evaluation results
        result.metric_scores = MetricScores(
            bleu_1=0.8, bleu_2=0.7, bleu_3=0.6, bleu_4=0.5,
            rouge_1_f=0.75, rouge_1_p=0.8, rouge_1_r=0.7,
            rouge_2_f=0.65, rouge_2_p=0.7, rouge_2_r=0.6,
            rouge_l_f=0.7, rouge_l_p=0.75, rouge_l_r=0.65,
            semantic_similarity=0.85
        )
        result.langchain_results = [LangChainEvaluationResult("test", 4.0, "Good")]
        
        confidence = evaluator._calculate_confidence(result)
        
        assert confidence is not None
        assert 0.0 <= confidence <= 1.0
    
    def test_evaluate_single_response(self, evaluator):
        """Test single response evaluation."""
        prompt = PromptVariant("prompt_1", "What is AI?")
        response = ModelResponse("gpt-4", "prompt_1", "AI is artificial intelligence", 1.0)
        references = ["AI stands for artificial intelligence"]
        
        result = evaluator._evaluate_single_response(
            prompt, response, references
        )
        
        assert isinstance(result, EvaluationResult)
        assert result.prompt_id == "prompt_1"
        assert result.model_id == "gpt-4"
        assert result.response == "AI is artificial intelligence"
        assert result.combined_score is not None
        assert result.evaluation_time is not None
    
    def test_batch_evaluate_prompts(self, evaluator):
        """Test batch prompt evaluation."""
        prompts = [
            PromptVariant("prompt_1", "What is AI?"),
            PromptVariant("prompt_2", "Explain machine learning")
        ]
        
        responses = [
            ModelResponse("gpt-4", "prompt_1", "AI is artificial intelligence", 1.0),
            ModelResponse("gpt-4", "prompt_2", "ML is machine learning", 1.2)
        ]
        
        references = {
            "prompt_1": ["AI stands for artificial intelligence"],
            "prompt_2": ["Machine learning is a subset of AI"]
        }
        
        results = evaluator.batch_evaluate_prompts(
            prompts, responses, references
        )
        
        assert len(results) == 2
        assert all(isinstance(r, EvaluationResult) for r in results)
        assert {r.prompt_id for r in results} == {"prompt_1", "prompt_2"}
    
    def test_compare_prompts_sufficient_data(self, evaluator):
        """Test prompt comparison with sufficient data."""
        # Create evaluation results
        results = []
        
        # Results for prompt_1 (higher scores)
        for i in range(10):
            results.append(EvaluationResult(
                prompt_id="prompt_1",
                model_id="gpt-4",
                response=f"Response {i}",
                combined_score=0.8 + (i % 3) * 0.05  # Scores around 0.8-0.9
            ))
        
        # Results for prompt_2 (lower scores)
        for i in range(10):
            results.append(EvaluationResult(
                prompt_id="prompt_2",
                model_id="gpt-4",
                response=f"Response {i}",
                combined_score=0.6 + (i % 3) * 0.05  # Scores around 0.6-0.7
            ))
        
        comparison = evaluator.compare_prompts(
            "prompt_1", "prompt_2", "gpt-4", results
        )
        
        assert comparison is not None
        assert isinstance(comparison, ComparisonResult)
        assert comparison.prompt_a_id == "prompt_1"
        assert comparison.prompt_b_id == "prompt_2"
        assert comparison.model_id == "gpt-4"
        assert comparison.mean_score_a > comparison.mean_score_b
        assert comparison.win_rate_a > 0.5
        assert comparison.n_comparisons == 10
    
    def test_compare_prompts_insufficient_data(self, evaluator):
        """Test prompt comparison with insufficient data."""
        # Create only a few results (less than min_samples)
        results = [
            EvaluationResult("prompt_1", "gpt-4", "Response 1", combined_score=0.8),
            EvaluationResult("prompt_2", "gpt-4", "Response 2", combined_score=0.7)
        ]
        
        comparison = evaluator.compare_prompts(
            "prompt_1", "prompt_2", "gpt-4", results
        )
        
        assert comparison is None
    
    def test_rank_prompts(self, evaluator):
        """Test prompt ranking."""
        # Create evaluation results for multiple prompts
        results = []
        
        # prompt_1: high scores
        for i in range(5):
            results.append(EvaluationResult(
                "prompt_1", "gpt-4", f"Response {i}", combined_score=0.9
            ))
        
        # prompt_2: medium scores
        for i in range(5):
            results.append(EvaluationResult(
                "prompt_2", "gpt-4", f"Response {i}", combined_score=0.7
            ))
        
        # prompt_3: low scores
        for i in range(5):
            results.append(EvaluationResult(
                "prompt_3", "gpt-4", f"Response {i}", combined_score=0.5
            ))
        
        ranking = evaluator.rank_prompts(
            ["prompt_1", "prompt_2", "prompt_3"], "gpt-4", results
        )
        
        assert ranking is not None
        assert isinstance(ranking, RankingResult)
        assert ranking.model_id == "gpt-4"
        assert len(ranking.rankings) == 3
        
        # Check ranking order
        ranks = [r["prompt_id"] for r in ranking.rankings]
        assert ranks == ["prompt_1", "prompt_2", "prompt_3"]
        
        # Check scores are in descending order
        scores = [r["score"] for r in ranking.rankings]
        assert scores == sorted(scores, reverse=True)
    
    def test_rank_prompts_no_data(self, evaluator):
        """Test prompt ranking with no data."""
        ranking = evaluator.rank_prompts(
            ["prompt_1", "prompt_2"], "gpt-4", []
        )
        
        assert ranking is None
    
    def test_calculate_ranking_confidence(self, evaluator):
        """Test ranking confidence calculation."""
        rankings = [
            {"prompt_id": "prompt_1", "score": 0.9, "score_std": 0.1},
            {"prompt_id": "prompt_2", "score": 0.7, "score_std": 0.1},
            {"prompt_id": "prompt_3", "score": 0.5, "score_std": 0.1}
        ]
        
        confidence = evaluator._calculate_ranking_confidence(rankings)
        
        assert 0.0 <= confidence <= 1.0
    
    def test_calculate_ranking_confidence_single_item(self, evaluator):
        """Test ranking confidence with single item."""
        rankings = [{"prompt_id": "prompt_1", "score": 0.9, "score_std": 0.1}]
        
        confidence = evaluator._calculate_ranking_confidence(rankings)
        
        assert confidence == 1.0
    
    def test_generate_comparison_matrix(self, evaluator):
        """Test comparison matrix generation."""
        # Create evaluation results
        results = []
        
        for prompt_id in ["prompt_1", "prompt_2", "prompt_3"]:
            for i in range(10):
                score = 0.8 if prompt_id == "prompt_1" else (0.7 if prompt_id == "prompt_2" else 0.6)
                results.append(EvaluationResult(
                    prompt_id, "gpt-4", f"Response {i}", combined_score=score
                ))
        
        matrix = evaluator.generate_comparison_matrix(
            ["prompt_1", "prompt_2", "prompt_3"], "gpt-4", results
        )
        
        assert len(matrix) == 3
        assert all(len(row) == 3 for row in matrix.values())
        
        # Diagonal should be None
        for prompt_id in ["prompt_1", "prompt_2", "prompt_3"]:
            assert matrix[prompt_id][prompt_id] is None
        
        # Off-diagonal should have comparison results
        assert matrix["prompt_1"]["prompt_2"] is not None
        assert matrix["prompt_2"]["prompt_1"] is not None
    
    def test_get_evaluation_summary(self, evaluator):
        """Test evaluation summary generation."""
        results = [
            EvaluationResult("prompt_1", "gpt-4", "Response 1", combined_score=0.8, confidence=0.9),
            EvaluationResult("prompt_1", "gpt-4", "Response 2", combined_score=0.7, confidence=0.8),
            EvaluationResult("prompt_2", "claude", "Response 3", combined_score=0.9, confidence=0.95),
            EvaluationResult("prompt_2", "claude", "Response 4", combined_score=None)  # Invalid result
        ]
        
        summary = evaluator.get_evaluation_summary(results)
        
        assert summary["total_evaluations"] == 4
        assert summary["valid_evaluations"] == 3
        assert "score_statistics" in summary
        assert "confidence_statistics" in summary
        assert "by_prompt" in summary
        assert "by_model" in summary
        
        # Check score statistics
        assert summary["score_statistics"]["mean"] == pytest.approx(0.8, rel=1e-2)
        assert summary["score_statistics"]["min"] == 0.7
        assert summary["score_statistics"]["max"] == 0.9
        
        # Check groupings
        assert "prompt_1" in summary["by_prompt"]
        assert "prompt_2" in summary["by_prompt"]
        assert "gpt-4" in summary["by_model"]
        assert "claude" in summary["by_model"]
    
    def test_get_evaluation_summary_empty(self, evaluator):
        """Test evaluation summary with empty results."""
        summary = evaluator.get_evaluation_summary([])
        
        assert summary == {}
    
    def test_get_evaluation_summary_no_valid_results(self, evaluator):
        """Test evaluation summary with no valid results."""
        results = [
            EvaluationResult("prompt_1", "gpt-4", "Response 1", combined_score=None),
            EvaluationResult("prompt_2", "gpt-4", "Response 2", combined_score=None)
        ]
        
        summary = evaluator.get_evaluation_summary(results)
        
        assert summary["total_evaluations"] == 2
        assert summary["valid_evaluations"] == 0


class TestStatisticalFunctions:
    """Test suite for statistical functions in PromptEvaluator."""
    
    @pytest.fixture
    def evaluator(self):
        """Create a basic PromptEvaluator for testing."""
        return PromptEvaluator()
    
    def test_statistical_significance_detection(self, evaluator):
        """Test that statistical significance is properly detected."""
        # Create results with clear difference
        results = []
        
        # Group A: consistently high scores
        for i in range(20):
            results.append(EvaluationResult(
                "prompt_a", "model", f"Response {i}", combined_score=0.9
            ))
        
        # Group B: consistently low scores
        for i in range(20):
            results.append(EvaluationResult(
                "prompt_b", "model", f"Response {i}", combined_score=0.5
            ))
        
        comparison = evaluator.compare_prompts("prompt_a", "prompt_b", "model", results)
        
        assert comparison is not None
        assert comparison.is_significant == True
        assert comparison.p_value < 0.05
        assert abs(comparison.effect_size) > 1.0  # Large effect size
    
    def test_no_statistical_significance(self, evaluator):
        """Test that no significance is detected when scores are similar."""
        # Create results with similar scores
        results = []
        
        # Both groups have similar scores with some noise
        import random
        random.seed(42)  # For reproducible tests
        
        for i in range(20):
            results.append(EvaluationResult(
                "prompt_a", "model", f"Response {i}", 
                combined_score=0.7 + random.uniform(-0.1, 0.1)
            ))
            results.append(EvaluationResult(
                "prompt_b", "model", f"Response {i}", 
                combined_score=0.7 + random.uniform(-0.1, 0.1)
            ))
        
        comparison = evaluator.compare_prompts("prompt_a", "prompt_b", "model", results)
        
        assert comparison is not None
        assert comparison.p_value > 0.05  # Not significant
        assert abs(comparison.effect_size) < 1.0  # Moderate effect size acceptable


if __name__ == "__main__":
    pytest.main([__file__])