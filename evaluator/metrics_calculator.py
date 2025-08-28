"""
Automated metrics calculation for LLM evaluation.

This module provides comprehensive metrics calculation including BLEU, ROUGE,
perplexity, and semantic similarity scoring for evaluating LLM outputs.
"""

import logging
import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

# NLTK imports for BLEU
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

# ROUGE imports
from rouge_score import rouge_scorer

# Sentence transformers for semantic similarity
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Transformers for perplexity calculation
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


@dataclass
class MetricScores:
    """Container for all calculated metrics."""
    bleu_1: float
    bleu_2: float
    bleu_3: float
    bleu_4: float
    rouge_1_f: float
    rouge_1_p: float
    rouge_1_r: float
    rouge_2_f: float
    rouge_2_p: float
    rouge_2_r: float
    rouge_l_f: float
    rouge_l_p: float
    rouge_l_r: float
    semantic_similarity: float
    perplexity: Optional[float] = None
    custom_metrics: Optional[Dict[str, float]] = None


class MetricsCalculator:
    """
    Comprehensive metrics calculator for LLM evaluation.
    
    Supports BLEU, ROUGE, perplexity, semantic similarity, and custom metrics.
    """
    
    def __init__(self, 
                 semantic_model_name: str = "all-MiniLM-L6-v2",
                 perplexity_model_name: Optional[str] = None,
                 device: str = "cpu"):
        """
        Initialize the metrics calculator.
        
        Args:
            semantic_model_name: Name of sentence transformer model for semantic similarity
            perplexity_model_name: Name of model for perplexity calculation (optional)
            device: Device to run models on ('cpu' or 'cuda')
        """
        self.device = device
        self.semantic_model_name = semantic_model_name
        self.perplexity_model_name = perplexity_model_name
        
        # Initialize NLTK data
        self._download_nltk_data()
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        
        # Initialize semantic similarity model
        self._load_semantic_model()
        
        # Initialize perplexity model if specified
        self.perplexity_model = None
        self.perplexity_tokenizer = None
        if perplexity_model_name:
            self._load_perplexity_model()
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
    
    def _load_semantic_model(self):
        """Load sentence transformer model for semantic similarity."""
        try:
            self.semantic_model = SentenceTransformer(
                self.semantic_model_name,
                device=self.device
            )
            logger.info(f"Loaded semantic model: {self.semantic_model_name}")
        except Exception as e:
            logger.error(f"Failed to load semantic model: {e}")
            self.semantic_model = None
    
    def _load_perplexity_model(self):
        """Load model for perplexity calculation."""
        try:
            self.perplexity_tokenizer = AutoTokenizer.from_pretrained(
                self.perplexity_model_name
            )
            self.perplexity_model = AutoModelForCausalLM.from_pretrained(
                self.perplexity_model_name
            ).to(self.device)
            self.perplexity_model.eval()
            logger.info(f"Loaded perplexity model: {self.perplexity_model_name}")
        except Exception as e:
            logger.error(f"Failed to load perplexity model: {e}")
            self.perplexity_model = None
            self.perplexity_tokenizer = None
    
    def calculate_bleu_scores(self, 
                            prediction: str, 
                            references: List[str]) -> Dict[str, float]:
        """
        Calculate BLEU scores (1-gram to 4-gram).
        
        Args:
            prediction: Generated text
            references: List of reference texts
            
        Returns:
            Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
        """
        try:
            # Tokenize prediction and references
            pred_tokens = word_tokenize(prediction.lower())
            ref_tokens_list = [word_tokenize(ref.lower()) for ref in references]
            
            # Smoothing function for BLEU
            smoothing = SmoothingFunction().method1
            
            # Calculate BLEU scores for different n-grams
            bleu_scores = {}
            for n in range(1, 5):
                weights = tuple([1.0/n] * n + [0.0] * (4-n))
                score = sentence_bleu(
                    ref_tokens_list, 
                    pred_tokens, 
                    weights=weights,
                    smoothing_function=smoothing
                )
                bleu_scores[f'bleu_{n}'] = score
            
            return bleu_scores
            
        except Exception as e:
            logger.error(f"Error calculating BLEU scores: {e}")
            return {f'bleu_{n}': 0.0 for n in range(1, 5)}
    
    def calculate_rouge_scores(self, 
                             prediction: str, 
                             reference: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
        
        Args:
            prediction: Generated text
            reference: Reference text
            
        Returns:
            Dictionary with ROUGE scores (precision, recall, f1)
        """
        try:
            scores = self.rouge_scorer.score(reference, prediction)
            
            rouge_scores = {}
            for rouge_type, score in scores.items():
                rouge_scores[f'{rouge_type}_f'] = score.fmeasure
                rouge_scores[f'{rouge_type}_p'] = score.precision
                rouge_scores[f'{rouge_type}_r'] = score.recall
            
            return rouge_scores
            
        except Exception as e:
            logger.error(f"Error calculating ROUGE scores: {e}")
            return {
                'rouge1_f': 0.0, 'rouge1_p': 0.0, 'rouge1_r': 0.0,
                'rouge2_f': 0.0, 'rouge2_p': 0.0, 'rouge2_r': 0.0,
                'rougeL_f': 0.0, 'rougeL_p': 0.0, 'rougeL_r': 0.0
            }
    
    def calculate_semantic_similarity(self, 
                                    prediction: str, 
                                    reference: str) -> float:
        """
        Calculate semantic similarity using sentence transformers.
        
        Args:
            prediction: Generated text
            reference: Reference text
            
        Returns:
            Cosine similarity score between embeddings
        """
        if self.semantic_model is None:
            logger.warning("Semantic model not loaded, returning 0.0")
            return 0.0
        
        try:
            # Generate embeddings
            embeddings = self.semantic_model.encode([prediction, reference])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def calculate_perplexity(self, text: str) -> Optional[float]:
        """
        Calculate perplexity of text using a language model.
        
        Args:
            text: Text to calculate perplexity for
            
        Returns:
            Perplexity score or None if model not available
        """
        if self.perplexity_model is None or self.perplexity_tokenizer is None:
            logger.warning("Perplexity model not loaded")
            return None
        
        try:
            # Tokenize text
            inputs = self.perplexity_tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.perplexity_model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
            
            return perplexity
            
        except Exception as e:
            logger.error(f"Error calculating perplexity: {e}")
            return None
    
    def calculate_custom_metrics(self, 
                               prediction: str, 
                               reference: str,
                               custom_functions: Dict[str, callable]) -> Dict[str, float]:
        """
        Calculate custom domain-specific metrics.
        
        Args:
            prediction: Generated text
            reference: Reference text
            custom_functions: Dictionary of metric name -> function mappings
            
        Returns:
            Dictionary of custom metric scores
        """
        custom_scores = {}
        
        for metric_name, metric_func in custom_functions.items():
            try:
                score = metric_func(prediction, reference)
                custom_scores[metric_name] = float(score)
            except Exception as e:
                logger.error(f"Error calculating custom metric {metric_name}: {e}")
                custom_scores[metric_name] = 0.0
        
        return custom_scores
    
    def calculate_all_metrics(self, 
                            prediction: str, 
                            references: List[str],
                            custom_functions: Optional[Dict[str, callable]] = None) -> MetricScores:
        """
        Calculate all available metrics for a prediction.
        
        Args:
            prediction: Generated text
            references: List of reference texts (first used for ROUGE and semantic similarity)
            custom_functions: Optional custom metric functions
            
        Returns:
            MetricScores object with all calculated metrics
        """
        if not references:
            raise ValueError("At least one reference text is required")
        
        primary_reference = references[0]
        
        # Calculate BLEU scores
        bleu_scores = self.calculate_bleu_scores(prediction, references)
        
        # Calculate ROUGE scores
        rouge_scores = self.calculate_rouge_scores(prediction, primary_reference)
        
        # Calculate semantic similarity
        semantic_similarity = self.calculate_semantic_similarity(
            prediction, primary_reference
        )
        
        # Calculate perplexity
        perplexity = self.calculate_perplexity(prediction)
        
        # Calculate custom metrics if provided
        custom_metrics = None
        if custom_functions:
            custom_metrics = self.calculate_custom_metrics(
                prediction, primary_reference, custom_functions
            )
        
        return MetricScores(
            bleu_1=bleu_scores['bleu_1'],
            bleu_2=bleu_scores['bleu_2'],
            bleu_3=bleu_scores['bleu_3'],
            bleu_4=bleu_scores['bleu_4'],
            rouge_1_f=rouge_scores['rouge1_f'],
            rouge_1_p=rouge_scores['rouge1_p'],
            rouge_1_r=rouge_scores['rouge1_r'],
            rouge_2_f=rouge_scores['rouge2_f'],
            rouge_2_p=rouge_scores['rouge2_p'],
            rouge_2_r=rouge_scores['rouge2_r'],
            rouge_l_f=rouge_scores['rougeL_f'],
            rouge_l_p=rouge_scores['rougeL_p'],
            rouge_l_r=rouge_scores['rougeL_r'],
            semantic_similarity=semantic_similarity,
            perplexity=perplexity,
            custom_metrics=custom_metrics
        )
    
    def batch_calculate_metrics(self, 
                              predictions: List[str], 
                              references_list: List[List[str]],
                              custom_functions: Optional[Dict[str, callable]] = None) -> List[MetricScores]:
        """
        Calculate metrics for multiple predictions in batch.
        
        Args:
            predictions: List of generated texts
            references_list: List of reference lists for each prediction
            custom_functions: Optional custom metric functions
            
        Returns:
            List of MetricScores objects
        """
        if len(predictions) != len(references_list):
            raise ValueError("Number of predictions must match number of reference lists")
        
        results = []
        for pred, refs in zip(predictions, references_list):
            metrics = self.calculate_all_metrics(pred, refs, custom_functions)
            results.append(metrics)
        
        return results


# Domain-specific custom metrics examples
def calculate_length_ratio(prediction: str, reference: str) -> float:
    """Calculate ratio of prediction length to reference length."""
    pred_len = len(prediction.split())
    ref_len = len(reference.split())
    if ref_len == 0:
        return 0.0
    return pred_len / ref_len


def calculate_keyword_coverage(prediction: str, reference: str) -> float:
    """Calculate what fraction of reference keywords appear in prediction."""
    pred_words = set(prediction.lower().split())
    ref_words = set(reference.lower().split())
    
    if not ref_words:
        return 0.0
    
    coverage = len(pred_words.intersection(ref_words)) / len(ref_words)
    return coverage


def calculate_repetition_penalty(prediction: str, reference: str = None) -> float:
    """Calculate penalty for repetitive text (lower is better)."""
    words = prediction.lower().split()
    if len(words) <= 1:
        return 0.0
    
    unique_words = len(set(words))
    total_words = len(words)
    
    # Return 1 - (unique/total) so higher repetition gives higher penalty
    return 1.0 - (unique_words / total_words)