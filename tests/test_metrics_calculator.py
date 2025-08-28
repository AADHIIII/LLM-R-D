"""
Tests for MetricsCalculator class.

Comprehensive tests for BLEU, ROUGE, semantic similarity, perplexity,
and custom metrics calculation.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from evaluator.metrics_calculator import (
    MetricsCalculator, 
    MetricScores,
    calculate_length_ratio,
    calculate_keyword_coverage,
    calculate_repetition_penalty
)


class TestMetricsCalculator:
    """Test suite for MetricsCalculator class."""
    
    @pytest.fixture
    def calculator(self):
        """Create a MetricsCalculator instance for testing."""
        with patch('evaluator.metrics_calculator.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_st.return_value = mock_model
            calc = MetricsCalculator(device="cpu")
            calc.semantic_model = mock_model
            return calc
    
    def test_initialization(self):
        """Test MetricsCalculator initialization."""
        with patch('evaluator.metrics_calculator.SentenceTransformer'):
            calc = MetricsCalculator(
                semantic_model_name="test-model",
                device="cpu"
            )
            assert calc.semantic_model_name == "test-model"
            assert calc.device == "cpu"
    
    def test_bleu_scores_calculation(self, calculator):
        """Test BLEU scores calculation."""
        prediction = "The cat sat on the mat"
        references = ["A cat was sitting on the mat", "The cat is on the mat"]
        
        scores = calculator.calculate_bleu_scores(prediction, references)
        
        assert 'bleu_1' in scores
        assert 'bleu_2' in scores
        assert 'bleu_3' in scores
        assert 'bleu_4' in scores
        
        # BLEU scores should be between 0 and 1
        for score in scores.values():
            assert 0.0 <= score <= 1.0
        
        # BLEU-1 should generally be higher than BLEU-4
        assert scores['bleu_1'] >= scores['bleu_4']
    
    def test_bleu_scores_identical_texts(self, calculator):
        """Test BLEU scores with identical prediction and reference."""
        text = "The quick brown fox jumps over the lazy dog"
        
        scores = calculator.calculate_bleu_scores(text, [text])
        
        # All BLEU scores should be 1.0 for identical texts
        for score in scores.values():
            assert score == pytest.approx(1.0, rel=1e-3)
    
    def test_bleu_scores_empty_input(self, calculator):
        """Test BLEU scores with empty input."""
        scores = calculator.calculate_bleu_scores("", ["reference"])
        
        # Should return zero scores for empty prediction
        for score in scores.values():
            assert score == 0.0
    
    def test_rouge_scores_calculation(self, calculator):
        """Test ROUGE scores calculation."""
        prediction = "The cat sat on the mat"
        reference = "A cat was sitting on the mat"
        
        scores = calculator.calculate_rouge_scores(prediction, reference)
        
        # Check all ROUGE metrics are present
        expected_keys = [
            'rouge1_f', 'rouge1_p', 'rouge1_r',
            'rouge2_f', 'rouge2_p', 'rouge2_r',
            'rougeL_f', 'rougeL_p', 'rougeL_r'
        ]
        
        for key in expected_keys:
            assert key in scores
            assert 0.0 <= scores[key] <= 1.0
    
    def test_rouge_scores_identical_texts(self, calculator):
        """Test ROUGE scores with identical texts."""
        text = "The quick brown fox jumps over the lazy dog"
        
        scores = calculator.calculate_rouge_scores(text, text)
        
        # F1, precision, and recall should all be 1.0 for identical texts
        for key, score in scores.items():
            assert score == pytest.approx(1.0, rel=1e-3)
    
    def test_semantic_similarity_calculation(self, calculator):
        """Test semantic similarity calculation."""
        prediction = "The cat sat on the mat"
        reference = "A cat was sitting on the mat"
        
        # Mock the semantic model to return embeddings
        calculator.semantic_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3],  # prediction embedding
            [0.15, 0.25, 0.35]  # reference embedding
        ])
        
        similarity = calculator.calculate_semantic_similarity(prediction, reference)
        
        assert isinstance(similarity, float)
        assert -1.0 <= similarity <= 1.0
    
    def test_semantic_similarity_no_model(self, calculator):
        """Test semantic similarity when model is not loaded."""
        calculator.semantic_model = None
        
        similarity = calculator.calculate_semantic_similarity("test", "test")
        
        assert similarity == 0.0
    
    def test_perplexity_calculation(self, calculator):
        """Test perplexity calculation."""
        # Mock perplexity model and tokenizer
        mock_tokenizer = Mock()
        mock_model = Mock()
        
        calculator.perplexity_tokenizer = mock_tokenizer
        calculator.perplexity_model = mock_model
        
        # Mock tokenizer output - simulate the dictionary-like behavior
        mock_inputs = {
            "input_ids": Mock(),
            "attention_mask": Mock()
        }
        mock_inputs_obj = Mock()
        mock_inputs_obj.__getitem__ = lambda self, key: mock_inputs[key]
        mock_inputs_obj.keys = lambda: mock_inputs.keys()
        mock_inputs_obj.to = Mock(return_value=mock_inputs_obj)
        mock_tokenizer.return_value = mock_inputs_obj
        
        # Mock model output
        mock_output = Mock()
        mock_loss = Mock()
        mock_loss.item.return_value = 2.0  # ln(perplexity)
        mock_output.loss = mock_loss
        mock_model.return_value = mock_output
        
        with patch('torch.no_grad'), patch('torch.exp') as mock_exp:
            mock_exp.return_value.item.return_value = 7.389  # e^2
            
            perplexity = calculator.calculate_perplexity("test text")
            
            assert perplexity == 7.389
    
    def test_perplexity_no_model(self, calculator):
        """Test perplexity calculation when model is not loaded."""
        perplexity = calculator.calculate_perplexity("test text")
        
        assert perplexity is None
    
    def test_custom_metrics_calculation(self, calculator):
        """Test custom metrics calculation."""
        prediction = "The cat sat on the mat"
        reference = "A cat was sitting on the mat"
        
        def dummy_metric(pred, ref):
            return 0.75
        
        def another_metric(pred, ref):
            return 0.5
        
        custom_functions = {
            "dummy": dummy_metric,
            "another": another_metric
        }
        
        scores = calculator.calculate_custom_metrics(
            prediction, reference, custom_functions
        )
        
        assert scores["dummy"] == 0.75
        assert scores["another"] == 0.5
    
    def test_custom_metrics_error_handling(self, calculator):
        """Test custom metrics error handling."""
        def failing_metric(pred, ref):
            raise ValueError("Test error")
        
        custom_functions = {"failing": failing_metric}
        
        scores = calculator.calculate_custom_metrics(
            "test", "test", custom_functions
        )
        
        assert scores["failing"] == 0.0
    
    def test_calculate_all_metrics(self, calculator):
        """Test comprehensive metrics calculation."""
        prediction = "The cat sat on the mat"
        references = ["A cat was sitting on the mat"]
        
        # Mock semantic model
        calculator.semantic_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.15, 0.25, 0.35]
        ])
        
        custom_functions = {"length_ratio": calculate_length_ratio}
        
        metrics = calculator.calculate_all_metrics(
            prediction, references, custom_functions
        )
        
        assert isinstance(metrics, MetricScores)
        assert hasattr(metrics, 'bleu_1')
        assert hasattr(metrics, 'rouge_1_f')
        assert hasattr(metrics, 'semantic_similarity')
        assert metrics.custom_metrics is not None
        assert "length_ratio" in metrics.custom_metrics
    
    def test_calculate_all_metrics_no_references(self, calculator):
        """Test metrics calculation with no references."""
        with pytest.raises(ValueError, match="At least one reference text is required"):
            calculator.calculate_all_metrics("test", [])
    
    def test_batch_calculate_metrics(self, calculator):
        """Test batch metrics calculation."""
        predictions = ["The cat sat", "The dog ran"]
        references_list = [["A cat was sitting"], ["A dog was running"]]
        
        # Mock semantic model
        calculator.semantic_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.15, 0.25, 0.35]
        ])
        
        results = calculator.batch_calculate_metrics(predictions, references_list)
        
        assert len(results) == 2
        assert all(isinstance(result, MetricScores) for result in results)
    
    def test_batch_calculate_metrics_length_mismatch(self, calculator):
        """Test batch calculation with mismatched lengths."""
        predictions = ["test1", "test2"]
        references_list = [["ref1"]]  # Only one reference list
        
        with pytest.raises(ValueError, match="Number of predictions must match"):
            calculator.batch_calculate_metrics(predictions, references_list)


class TestCustomMetrics:
    """Test suite for custom metric functions."""
    
    def test_length_ratio(self):
        """Test length ratio calculation."""
        prediction = "The cat sat on the mat"  # 6 words
        reference = "A cat was sitting"  # 4 words
        
        ratio = calculate_length_ratio(prediction, reference)
        
        assert ratio == 1.5  # 6/4
    
    def test_length_ratio_empty_reference(self):
        """Test length ratio with empty reference."""
        ratio = calculate_length_ratio("test", "")
        
        assert ratio == 0.0
    
    def test_keyword_coverage(self):
        """Test keyword coverage calculation."""
        prediction = "The cat sat on the mat"
        reference = "cat mat"  # 2 words, both in prediction
        
        coverage = calculate_keyword_coverage(prediction, reference)
        
        assert coverage == 1.0  # 100% coverage
    
    def test_keyword_coverage_partial(self):
        """Test partial keyword coverage."""
        prediction = "The cat sat"
        reference = "cat dog mat"  # 3 words, only 1 in prediction
        
        coverage = calculate_keyword_coverage(prediction, reference)
        
        assert coverage == pytest.approx(1/3, rel=1e-3)
    
    def test_keyword_coverage_empty_reference(self):
        """Test keyword coverage with empty reference."""
        coverage = calculate_keyword_coverage("test", "")
        
        assert coverage == 0.0
    
    def test_repetition_penalty(self):
        """Test repetition penalty calculation."""
        # No repetition
        penalty1 = calculate_repetition_penalty("the cat sat on mat")
        
        # High repetition
        penalty2 = calculate_repetition_penalty("cat cat cat cat cat")
        
        # penalty2 should be higher (more repetitive)
        assert penalty2 > penalty1
    
    def test_repetition_penalty_single_word(self):
        """Test repetition penalty with single word."""
        penalty = calculate_repetition_penalty("cat")
        
        assert penalty == 0.0
    
    def test_repetition_penalty_empty(self):
        """Test repetition penalty with empty text."""
        penalty = calculate_repetition_penalty("")
        
        assert penalty == 0.0


class TestMetricScores:
    """Test suite for MetricScores dataclass."""
    
    def test_metric_scores_creation(self):
        """Test MetricScores object creation."""
        scores = MetricScores(
            bleu_1=0.8,
            bleu_2=0.7,
            bleu_3=0.6,
            bleu_4=0.5,
            rouge_1_f=0.75,
            rouge_1_p=0.8,
            rouge_1_r=0.7,
            rouge_2_f=0.65,
            rouge_2_p=0.7,
            rouge_2_r=0.6,
            rouge_l_f=0.7,
            rouge_l_p=0.75,
            rouge_l_r=0.65,
            semantic_similarity=0.85,
            perplexity=15.2,
            custom_metrics={"length_ratio": 1.2}
        )
        
        assert scores.bleu_1 == 0.8
        assert scores.semantic_similarity == 0.85
        assert scores.perplexity == 15.2
        assert scores.custom_metrics["length_ratio"] == 1.2
    
    def test_metric_scores_optional_fields(self):
        """Test MetricScores with optional fields."""
        scores = MetricScores(
            bleu_1=0.8, bleu_2=0.7, bleu_3=0.6, bleu_4=0.5,
            rouge_1_f=0.75, rouge_1_p=0.8, rouge_1_r=0.7,
            rouge_2_f=0.65, rouge_2_p=0.7, rouge_2_r=0.6,
            rouge_l_f=0.7, rouge_l_p=0.75, rouge_l_r=0.65,
            semantic_similarity=0.85
        )
        
        assert scores.perplexity is None
        assert scores.custom_metrics is None


if __name__ == "__main__":
    pytest.main([__file__])