"""
Tests for LangChainEvaluator class.

Comprehensive tests for LangChain-based evaluation pipeline and LLM-as-judge evaluation.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from evaluator.langchain_evaluator import (
    LangChainEvaluator,
    EvaluationCriteria,
    LangChainEvaluationResult,
    LLMJudgeResult
)


class TestEvaluationCriteria:
    """Test suite for EvaluationCriteria dataclass."""
    
    def test_criteria_creation(self):
        """Test EvaluationCriteria creation."""
        criteria = EvaluationCriteria(
            name="helpfulness",
            description="How helpful is the response?",
            scale="1-5",
            weight=1.5
        )
        
        assert criteria.name == "helpfulness"
        assert criteria.description == "How helpful is the response?"
        assert criteria.scale == "1-5"
        assert criteria.weight == 1.5
    
    def test_criteria_defaults(self):
        """Test EvaluationCriteria default values."""
        criteria = EvaluationCriteria(
            name="clarity",
            description="How clear is the response?"
        )
        
        assert criteria.scale == "1-5"
        assert criteria.weight == 1.0


class TestLangChainEvaluationResult:
    """Test suite for LangChainEvaluationResult dataclass."""
    
    def test_result_creation(self):
        """Test LangChainEvaluationResult creation."""
        result = LangChainEvaluationResult(
            criterion="helpfulness",
            score=4.5,
            reasoning="The response is very helpful",
            confidence=0.9,
            evaluation_time=1.2
        )
        
        assert result.criterion == "helpfulness"
        assert result.score == 4.5
        assert result.reasoning == "The response is very helpful"
        assert result.confidence == 0.9
        assert result.evaluation_time == 1.2
    
    def test_result_optional_fields(self):
        """Test LangChainEvaluationResult with optional fields."""
        result = LangChainEvaluationResult(
            criterion="clarity",
            score=3.0,
            reasoning="Average clarity"
        )
        
        assert result.confidence is None
        assert result.evaluation_time is None


class TestLLMJudgeResult:
    """Test suite for LLMJudgeResult dataclass."""
    
    def test_judge_result_creation(self):
        """Test LLMJudgeResult creation."""
        result = LLMJudgeResult(
            overall_score=4.2,
            criteria_scores={"helpfulness": 4.0, "clarity": 4.5},
            reasoning="Good overall response",
            confidence=0.85,
            evaluation_time=2.3
        )
        
        assert result.overall_score == 4.2
        assert result.criteria_scores["helpfulness"] == 4.0
        assert result.criteria_scores["clarity"] == 4.5
        assert result.reasoning == "Good overall response"
        assert result.confidence == 0.85
        assert result.evaluation_time == 2.3


class TestLangChainEvaluator:
    """Test suite for LangChainEvaluator class."""
    
    @pytest.fixture
    def evaluator(self):
        """Create a LangChainEvaluator instance for testing."""
        with patch('evaluator.langchain_evaluator.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            evaluator = LangChainEvaluator(
                openai_api_key="test-key",
                judge_model="gpt-4",
                max_workers=2,
                timeout=10.0
            )
            evaluator.openai_client = mock_client
            return evaluator
    
    @pytest.fixture
    def evaluator_no_openai(self):
        """Create a LangChainEvaluator without OpenAI for testing."""
        return LangChainEvaluator(openai_api_key=None)
    
    def test_initialization_with_openai(self, evaluator):
        """Test LangChainEvaluator initialization with OpenAI."""
        assert evaluator.judge_model == "gpt-4"
        assert evaluator.max_workers == 2
        assert evaluator.timeout == 10.0
        assert evaluator.openai_client is not None
        assert len(evaluator.criteria) == 5  # Default criteria
    
    def test_initialization_without_openai(self, evaluator_no_openai):
        """Test LangChainEvaluator initialization without OpenAI."""
        assert evaluator_no_openai.openai_client is None
        assert len(evaluator_no_openai.criteria) == 5  # Default criteria
    
    def test_add_custom_criterion(self, evaluator):
        """Test adding custom evaluation criterion."""
        custom_criterion = EvaluationCriteria(
            name="creativity",
            description="How creative is the response?",
            weight=1.3
        )
        
        initial_count = len(evaluator.criteria)
        evaluator.add_custom_criterion(custom_criterion)
        
        assert len(evaluator.criteria) == initial_count + 1
        assert any(c.name == "creativity" for c in evaluator.criteria)
    
    def test_remove_criterion(self, evaluator):
        """Test removing evaluation criterion."""
        initial_count = len(evaluator.criteria)
        evaluator.remove_criterion("helpfulness")
        
        assert len(evaluator.criteria) == initial_count - 1
        assert not any(c.name == "helpfulness" for c in evaluator.criteria)
    
    @patch('evaluator.langchain_evaluator.load_evaluator')
    def test_evaluate_single_criterion_success(self, mock_load_evaluator, evaluator):
        """Test successful single criterion evaluation."""
        # Mock evaluator
        mock_evaluator = Mock()
        mock_evaluator.evaluate_strings.return_value = {
            "score": 4.5,
            "reasoning": "Excellent response"
        }
        mock_load_evaluator.return_value = mock_evaluator
        
        criterion = EvaluationCriteria("helpfulness", "How helpful?")
        
        result = evaluator._evaluate_single_criterion(
            "What is AI?", "AI is artificial intelligence", None, criterion
        )
        
        assert isinstance(result, LangChainEvaluationResult)
        assert result.criterion == "helpfulness"
        assert result.score == 4.5
        assert result.reasoning == "Excellent response"
        assert result.evaluation_time is not None
    
    @patch('evaluator.langchain_evaluator.load_evaluator')
    def test_evaluate_single_criterion_failure(self, mock_load_evaluator, evaluator):
        """Test single criterion evaluation failure."""
        # Mock evaluator to raise exception
        mock_load_evaluator.side_effect = Exception("Evaluation failed")
        
        criterion = EvaluationCriteria("helpfulness", "How helpful?")
        
        result = evaluator._evaluate_single_criterion(
            "What is AI?", "AI is artificial intelligence", None, criterion
        )
        
        assert isinstance(result, LangChainEvaluationResult)
        assert result.criterion == "helpfulness"
        assert result.score == 0.0
        assert "Evaluation failed" in result.reasoning
    
    @patch('evaluator.langchain_evaluator.load_evaluator')
    def test_evaluate_with_langchain(self, mock_load_evaluator, evaluator):
        """Test LangChain evaluation with multiple criteria."""
        # Mock evaluator
        mock_evaluator = Mock()
        mock_evaluator.evaluate_strings.return_value = {
            "score": 4.0,
            "reasoning": "Good response"
        }
        mock_load_evaluator.return_value = mock_evaluator
        
        results = evaluator.evaluate_with_langchain(
            "What is AI?", 
            "AI is artificial intelligence",
            criteria_subset=["helpfulness", "clarity"]
        )
        
        assert len(results) == 2
        assert all(isinstance(r, LangChainEvaluationResult) for r in results)
        assert {r.criterion for r in results} == {"helpfulness", "clarity"}
    
    def test_create_judge_prompt(self, evaluator):
        """Test LLM-as-judge prompt creation."""
        prompt = evaluator._create_judge_prompt(
            "What is AI?",
            "AI is artificial intelligence",
            "AI stands for artificial intelligence"
        )
        
        assert "What is AI?" in prompt
        assert "AI is artificial intelligence" in prompt
        assert "AI stands for artificial intelligence" in prompt
        assert "helpfulness" in prompt.lower()
        assert "clarity" in prompt.lower()
        assert "JSON" in prompt
    
    def test_evaluate_with_llm_judge_success(self, evaluator):
        """Test successful LLM-as-judge evaluation."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        
        judge_result = {
            "criteria_scores": {
                "helpfulness": 4.5,
                "clarity": 4.0,
                "accuracy": 4.2
            },
            "reasoning": "Well-structured and accurate response",
            "overall_score": 4.2,
            "confidence": 0.9
        }
        
        mock_message.content = json.dumps(judge_result)
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        evaluator.openai_client.chat.completions.create.return_value = mock_response
        
        result = evaluator.evaluate_with_llm_judge(
            "What is AI?",
            "AI is artificial intelligence"
        )
        
        assert isinstance(result, LLMJudgeResult)
        assert result.overall_score == 4.2
        assert result.criteria_scores["helpfulness"] == 4.5
        assert result.confidence == 0.9
        assert "Well-structured" in result.reasoning
    
    def test_evaluate_with_llm_judge_no_client(self, evaluator_no_openai):
        """Test LLM-as-judge evaluation without OpenAI client."""
        result = evaluator_no_openai.evaluate_with_llm_judge(
            "What is AI?",
            "AI is artificial intelligence"
        )
        
        assert result is None
    
    def test_evaluate_with_llm_judge_json_parsing_error(self, evaluator):
        """Test LLM-as-judge evaluation with JSON parsing error."""
        # Mock OpenAI response with invalid JSON
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Invalid JSON response"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        evaluator.openai_client.chat.completions.create.return_value = mock_response
        
        result = evaluator.evaluate_with_llm_judge(
            "What is AI?",
            "AI is artificial intelligence"
        )
        
        assert result is None
    
    def test_evaluate_with_llm_judge_json_extraction(self, evaluator):
        """Test LLM-as-judge evaluation with JSON extraction from text."""
        # Mock OpenAI response with JSON wrapped in text
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        
        judge_result = {
            "criteria_scores": {"helpfulness": 4.0},
            "reasoning": "Good response",
            "overall_score": 4.0,
            "confidence": 0.8
        }
        
        mock_message.content = f"Here is the evaluation: {json.dumps(judge_result)} End of evaluation."
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        evaluator.openai_client.chat.completions.create.return_value = mock_response
        
        result = evaluator.evaluate_with_llm_judge(
            "What is AI?",
            "AI is artificial intelligence"
        )
        
        assert isinstance(result, LLMJudgeResult)
        assert result.overall_score == 4.0
    
    @patch('evaluator.langchain_evaluator.load_evaluator')
    def test_evaluate_comprehensive(self, mock_load_evaluator, evaluator):
        """Test comprehensive evaluation with both methods."""
        # Mock LangChain evaluator
        mock_lc_evaluator = Mock()
        mock_lc_evaluator.evaluate_strings.return_value = {
            "score": 4.0,
            "reasoning": "Good response"
        }
        mock_load_evaluator.return_value = mock_lc_evaluator
        
        # Mock OpenAI response
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        
        judge_result = {
            "criteria_scores": {"helpfulness": 4.5},
            "reasoning": "Excellent response",
            "overall_score": 4.5,
            "confidence": 0.9
        }
        
        mock_message.content = json.dumps(judge_result)
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        evaluator.openai_client.chat.completions.create.return_value = mock_response
        
        result = evaluator.evaluate_comprehensive(
            "What is AI?",
            "AI is artificial intelligence"
        )
        
        assert "langchain_results" in result
        assert "llm_judge_result" in result
        assert "combined_score" in result
        assert "evaluation_summary" in result
        
        assert result["langchain_results"] is not None
        assert result["llm_judge_result"] is not None
        assert result["combined_score"] is not None
    
    @patch('evaluator.langchain_evaluator.load_evaluator')
    def test_batch_evaluate(self, mock_load_evaluator, evaluator):
        """Test batch evaluation."""
        # Mock LangChain evaluator
        mock_lc_evaluator = Mock()
        mock_lc_evaluator.evaluate_strings.return_value = {
            "score": 4.0,
            "reasoning": "Good response"
        }
        mock_load_evaluator.return_value = mock_lc_evaluator
        
        # Mock OpenAI response
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        
        judge_result = {
            "criteria_scores": {"helpfulness": 4.0},
            "reasoning": "Good response",
            "overall_score": 4.0,
            "confidence": 0.8
        }
        
        mock_message.content = json.dumps(judge_result)
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        evaluator.openai_client.chat.completions.create.return_value = mock_response
        
        evaluations = [
            {"input": "What is AI?", "prediction": "AI is artificial intelligence"},
            {"input": "What is ML?", "prediction": "ML is machine learning", "reference": "Machine learning"}
        ]
        
        results = evaluator.batch_evaluate(evaluations)
        
        assert len(results) == 2
        assert all("combined_score" in r for r in results)
    
    def test_get_evaluation_statistics(self, evaluator):
        """Test evaluation statistics calculation."""
        results = [
            {
                "combined_score": 4.0,
                "evaluation_summary": {
                    "langchain_weighted_score": 3.8,
                    "llm_judge_score": 4.2
                }
            },
            {
                "combined_score": 3.5,
                "evaluation_summary": {
                    "langchain_weighted_score": 3.2,
                    "llm_judge_score": 3.8
                }
            }
        ]
        
        stats = evaluator.get_evaluation_statistics(results)
        
        assert stats["total_evaluations"] == 2
        assert stats["successful_evaluations"] == 2
        assert "combined_score_stats" in stats
        assert "langchain_score_stats" in stats
        assert "llm_judge_score_stats" in stats
        
        assert stats["combined_score_stats"]["mean"] == 3.75
        assert stats["combined_score_stats"]["min"] == 3.5
        assert stats["combined_score_stats"]["max"] == 4.0
    
    def test_get_evaluation_statistics_empty(self, evaluator):
        """Test evaluation statistics with empty results."""
        stats = evaluator.get_evaluation_statistics([])
        
        assert stats == {}
    
    def test_get_evaluation_statistics_with_errors(self, evaluator):
        """Test evaluation statistics with some failed evaluations."""
        results = [
            {"combined_score": 4.0},
            {"error": "Evaluation failed"},
            {"combined_score": 3.5}
        ]
        
        stats = evaluator.get_evaluation_statistics(results)
        
        assert stats["total_evaluations"] == 3
        assert stats["successful_evaluations"] == 2
        assert stats["combined_score_stats"]["count"] == 2


class TestDefaultCriteria:
    """Test suite for default evaluation criteria."""
    
    def test_default_criteria_exist(self):
        """Test that default criteria are properly defined."""
        evaluator = LangChainEvaluator()
        
        expected_criteria = {"helpfulness", "clarity", "accuracy", "relevance", "completeness"}
        actual_criteria = {c.name for c in evaluator.criteria}
        
        assert actual_criteria == expected_criteria
    
    def test_default_criteria_weights(self):
        """Test that default criteria have appropriate weights."""
        evaluator = LangChainEvaluator()
        
        # Find accuracy criterion (should have highest weight)
        accuracy_criterion = next(c for c in evaluator.criteria if c.name == "accuracy")
        assert accuracy_criterion.weight == 1.8
        
        # Find helpfulness criterion
        helpfulness_criterion = next(c for c in evaluator.criteria if c.name == "helpfulness")
        assert helpfulness_criterion.weight == 1.5


if __name__ == "__main__":
    pytest.main([__file__])