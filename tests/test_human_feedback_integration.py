"""
Tests for human feedback integration in the evaluation pipeline.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from evaluator.prompt_evaluator import (
    PromptEvaluator, EvaluationResult, HumanFeedback, 
    PromptVariant, ModelResponse
)
from evaluator.metrics_calculator import MetricScores
from evaluator.langchain_evaluator import LangChainEvaluationResult, LLMJudgeResult


@pytest.fixture
def prompt_evaluator():
    """Create PromptEvaluator instance."""
    return PromptEvaluator()


@pytest.fixture
def sample_evaluation_result():
    """Create sample evaluation result."""
    return EvaluationResult(
        prompt_id="test_prompt",
        model_id="test_model",
        response="This is a test response",
        metric_scores=MetricScores(
            bleu_1=0.8,
            bleu_2=0.7,
            bleu_3=0.6,
            bleu_4=0.5,
            rouge_1_f=0.75,
            rouge_1_p=0.73,
            rouge_1_r=0.77,
            rouge_2_f=0.65,
            rouge_2_p=0.63,
            rouge_2_r=0.67,
            rouge_l_f=0.7,
            rouge_l_p=0.68,
            rouge_l_r=0.72,
            semantic_similarity=0.85
        ),
        combined_score=0.7
    )


@pytest.fixture
def human_feedback_star():
    """Create human feedback with star rating."""
    return HumanFeedback(
        star_rating=4,
        qualitative_feedback="Good response, but could be more detailed",
        feedback_timestamp=datetime.utcnow().isoformat()
    )


@pytest.fixture
def human_feedback_thumbs():
    """Create human feedback with thumbs rating."""
    return HumanFeedback(
        thumbs_rating='up',
        qualitative_feedback="Helpful response",
        feedback_timestamp=datetime.utcnow().isoformat()
    )


class TestHumanFeedback:
    """Test cases for HumanFeedback class."""
    
    def test_star_rating_to_score(self):
        """Test conversion of star rating to normalized score."""
        feedback = HumanFeedback(star_rating=5)
        assert feedback.to_score() == 1.0
        
        feedback = HumanFeedback(star_rating=1)
        assert feedback.to_score() == 0.0
        
        feedback = HumanFeedback(star_rating=3)
        assert feedback.to_score() == 0.5
    
    def test_thumbs_rating_to_score(self):
        """Test conversion of thumbs rating to normalized score."""
        feedback = HumanFeedback(thumbs_rating='up')
        assert feedback.to_score() == 1.0
        
        feedback = HumanFeedback(thumbs_rating='down')
        assert feedback.to_score() == 0.0
    
    def test_no_rating_to_score(self):
        """Test that no rating returns None."""
        feedback = HumanFeedback(qualitative_feedback="Just text")
        assert feedback.to_score() is None
    
    def test_star_rating_priority(self):
        """Test that star rating takes priority over thumbs rating."""
        feedback = HumanFeedback(
            star_rating=2,
            thumbs_rating='up'
        )
        assert feedback.to_score() == 0.25  # Star rating used


class TestHumanFeedbackIntegration:
    """Test cases for human feedback integration in evaluation pipeline."""
    
    def test_update_human_feedback(self, prompt_evaluator, sample_evaluation_result, human_feedback_star):
        """Test updating evaluation result with human feedback."""
        original_score = sample_evaluation_result.combined_score
        
        updated_result = prompt_evaluator.update_human_feedback(
            sample_evaluation_result, 
            human_feedback_star
        )
        
        assert updated_result.human_feedback == human_feedback_star
        assert updated_result.combined_score != original_score
        assert updated_result.human_feedback.star_rating == 4
    
    def test_combined_score_with_human_feedback(self, prompt_evaluator, sample_evaluation_result, human_feedback_star):
        """Test that human feedback is incorporated into combined score."""
        # Add human feedback
        sample_evaluation_result.human_feedback = human_feedback_star
        
        # Calculate combined score with human feedback
        combined_score = prompt_evaluator._calculate_combined_score(
            sample_evaluation_result, 
            human_feedback_weight=0.6
        )
        
        # Human feedback score should influence the result
        human_score = human_feedback_star.to_score()  # 4 stars = 0.75
        assert combined_score is not None
        assert 0.0 <= combined_score <= 1.0
        
        # With high human feedback weight, score should be closer to human score
        assert abs(combined_score - human_score) < abs(combined_score - 0.7)  # Original score
    
    def test_combined_score_without_human_feedback(self, prompt_evaluator, sample_evaluation_result):
        """Test combined score calculation without human feedback."""
        combined_score = prompt_evaluator._calculate_combined_score(
            sample_evaluation_result,
            human_feedback_weight=0.5
        )
        
        # Should still calculate score based on automated metrics
        assert combined_score is not None
        assert 0.0 <= combined_score <= 1.0
    
    def test_confidence_with_human_feedback(self, prompt_evaluator, sample_evaluation_result, human_feedback_star):
        """Test confidence calculation with human feedback."""
        # Without human feedback
        confidence_without = prompt_evaluator._calculate_confidence_with_human_feedback(sample_evaluation_result)
        
        # With human feedback
        sample_evaluation_result.human_feedback = human_feedback_star
        confidence_with = prompt_evaluator._calculate_confidence_with_human_feedback(sample_evaluation_result)
        
        # Confidence should be higher with human feedback
        assert confidence_with > confidence_without
    
    def test_feedback_based_ranking(self, prompt_evaluator):
        """Test ranking of results based on human feedback."""
        # Create multiple evaluation results
        results = [
            EvaluationResult(
                prompt_id="prompt1",
                model_id="model1",
                response="Response 1",
                combined_score=0.6,
                human_feedback=HumanFeedback(star_rating=5)  # High human rating
            ),
            EvaluationResult(
                prompt_id="prompt2",
                model_id="model2",
                response="Response 2",
                combined_score=0.8,  # High automated score
                human_feedback=HumanFeedback(star_rating=2)  # Low human rating
            ),
            EvaluationResult(
                prompt_id="prompt3",
                model_id="model3",
                response="Response 3",
                combined_score=0.7,
                human_feedback=HumanFeedback(star_rating=4)  # Medium human rating
            )
        ]
        
        # Rank with high human feedback weight
        ranked_results = prompt_evaluator.create_feedback_based_ranking(
            results, 
            human_feedback_weight=0.8
        )
        
        # Result with highest human rating should be first
        assert ranked_results[0].human_feedback.star_rating == 5
        assert ranked_results[1].human_feedback.star_rating == 4
        assert ranked_results[2].human_feedback.star_rating == 2
    
    def test_ranking_without_human_feedback(self, prompt_evaluator):
        """Test ranking when some results don't have human feedback."""
        results = [
            EvaluationResult(
                prompt_id="prompt1",
                model_id="model1",
                response="Response 1",
                combined_score=0.6,
                human_feedback=HumanFeedback(star_rating=5)
            ),
            EvaluationResult(
                prompt_id="prompt2",
                model_id="model2",
                response="Response 2",
                combined_score=0.9  # No human feedback, high automated score
            ),
            EvaluationResult(
                prompt_id="prompt3",
                model_id="model3",
                response="Response 3",
                combined_score=0.5  # No human feedback, low automated score
            )
        ]
        
        ranked_results = prompt_evaluator.create_feedback_based_ranking(
            results,
            human_feedback_weight=0.5
        )
        
        # Should still be able to rank all results
        assert len(ranked_results) == 3
        assert all(r.combined_score is not None for r in ranked_results)
    
    def test_human_feedback_weight_impact(self, prompt_evaluator, sample_evaluation_result, human_feedback_star):
        """Test impact of different human feedback weights."""
        sample_evaluation_result.human_feedback = human_feedback_star
        
        # Low human feedback weight
        score_low_weight = prompt_evaluator._calculate_combined_score(
            sample_evaluation_result,
            human_feedback_weight=0.1
        )
        
        # High human feedback weight
        score_high_weight = prompt_evaluator._calculate_combined_score(
            sample_evaluation_result,
            human_feedback_weight=0.9
        )
        
        human_score = human_feedback_star.to_score()  # 0.75 for 4 stars
        
        # High weight should be closer to human score
        assert abs(score_high_weight - human_score) < abs(score_low_weight - human_score)
    
    def test_thumbs_vs_star_rating_priority(self, prompt_evaluator, sample_evaluation_result):
        """Test that star rating takes priority over thumbs rating."""
        feedback = HumanFeedback(
            star_rating=2,  # Low star rating (0.25)
            thumbs_rating='up'  # Positive thumbs (1.0)
        )
        
        sample_evaluation_result.human_feedback = feedback
        
        combined_score = prompt_evaluator._calculate_combined_score(
            sample_evaluation_result,
            human_feedback_weight=1.0  # Only human feedback
        )
        
        # Should use star rating (0.25), not thumbs rating (1.0)
        assert abs(combined_score - 0.25) < abs(combined_score - 1.0)