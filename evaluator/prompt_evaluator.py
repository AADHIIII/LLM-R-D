"""
Prompt comparison and evaluation system.

This module provides comprehensive prompt evaluation capabilities including
batch evaluation, statistical significance testing, and result aggregation.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools

# Statistical testing
from scipy import stats
import numpy as np

# Import our evaluation modules
from .metrics_calculator import MetricsCalculator, MetricScores
from .langchain_evaluator import LangChainEvaluator, LangChainEvaluationResult, LLMJudgeResult

logger = logging.getLogger(__name__)


@dataclass
class PromptVariant:
    """Represents a prompt variant for evaluation."""
    id: str
    prompt: str
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    """Represents a model's response to a prompt."""
    model_id: str
    prompt_id: str
    response: str
    generation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HumanFeedback:
    """Human feedback for an evaluation result."""
    thumbs_rating: Optional[str] = None  # 'up' or 'down'
    star_rating: Optional[int] = None    # 1-5 stars
    qualitative_feedback: Optional[str] = None
    feedback_timestamp: Optional[str] = None
    
    def to_score(self) -> Optional[float]:
        """Convert human feedback to a normalized score (0-1)."""
        if self.star_rating is not None:
            return (self.star_rating - 1) / 4.0  # Convert 1-5 to 0-1
        elif self.thumbs_rating is not None:
            return 1.0 if self.thumbs_rating == 'up' else 0.0
        return None


@dataclass
class EvaluationResult:
    """Comprehensive evaluation result for a prompt-model combination."""
    prompt_id: str
    model_id: str
    response: str
    
    # Automated metrics
    metric_scores: Optional[MetricScores] = None
    
    # LangChain evaluation results
    langchain_results: Optional[List[LangChainEvaluationResult]] = None
    
    # LLM-as-judge result
    llm_judge_result: Optional[LLMJudgeResult] = None
    
    # Human feedback
    human_feedback: Optional[HumanFeedback] = None
    
    # Combined scores
    combined_score: Optional[float] = None
    confidence: Optional[float] = None
    
    # Timing and metadata
    evaluation_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Result of comparing two prompt variants."""
    prompt_a_id: str
    prompt_b_id: str
    model_id: str
    
    # Win rates
    win_rate_a: float
    win_rate_b: float
    tie_rate: float
    
    # Statistical significance
    p_value: float
    is_significant: bool
    effect_size: float
    
    # Score differences
    mean_score_a: float
    mean_score_b: float
    score_difference: float
    
    # Confidence intervals
    confidence_interval: Tuple[float, float]
    
    # Sample sizes
    n_comparisons: int
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RankingResult:
    """Result of ranking multiple prompt variants."""
    model_id: str
    rankings: List[Dict[str, Any]]  # List of {prompt_id, rank, score, confidence}
    
    # Statistical measures
    kendall_tau: Optional[float] = None  # Rank correlation if multiple models
    spearman_rho: Optional[float] = None
    
    # Confidence in ranking
    ranking_confidence: float = 0.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class PromptEvaluator:
    """
    Comprehensive prompt evaluation and comparison system.
    
    Supports batch evaluation, statistical significance testing, and ranking.
    """
    
    def __init__(self,
                 metrics_calculator: Optional[MetricsCalculator] = None,
                 langchain_evaluator: Optional[LangChainEvaluator] = None,
                 max_workers: int = 4,
                 significance_threshold: float = 0.05,
                 min_samples: int = 10):
        """
        Initialize the prompt evaluator.
        
        Args:
            metrics_calculator: MetricsCalculator instance for automated metrics
            langchain_evaluator: LangChainEvaluator instance for LLM evaluation
            max_workers: Maximum number of concurrent evaluation workers
            significance_threshold: P-value threshold for statistical significance
            min_samples: Minimum number of samples required for statistical tests
        """
        self.metrics_calculator = metrics_calculator or MetricsCalculator()
        self.langchain_evaluator = langchain_evaluator or LangChainEvaluator()
        self.max_workers = max_workers
        self.significance_threshold = significance_threshold
        self.min_samples = min_samples
        
        # Thread pool for concurrent evaluations
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def _evaluate_single_response(self,
                                prompt_variant: PromptVariant,
                                model_response: ModelResponse,
                                reference_responses: Optional[List[str]] = None,
                                use_metrics: bool = True,
                                use_langchain: bool = True,
                                use_llm_judge: bool = True) -> EvaluationResult:
        """
        Evaluate a single model response to a prompt variant.
        
        Args:
            prompt_variant: The prompt variant used
            model_response: The model's response
            reference_responses: Optional reference responses for comparison
            use_metrics: Whether to calculate automated metrics
            use_langchain: Whether to use LangChain evaluation
            use_llm_judge: Whether to use LLM-as-judge evaluation
            
        Returns:
            EvaluationResult with comprehensive scores
        """
        start_time = time.time()
        
        result = EvaluationResult(
            prompt_id=prompt_variant.id,
            model_id=model_response.model_id,
            response=model_response.response
        )
        
        # Automated metrics
        if use_metrics and reference_responses:
            try:
                metric_scores = self.metrics_calculator.calculate_all_metrics(
                    model_response.response,
                    reference_responses
                )
                result.metric_scores = metric_scores
            except Exception as e:
                logger.error(f"Automated metrics calculation failed: {e}")
        
        # LangChain evaluation
        if use_langchain:
            try:
                primary_reference = reference_responses[0] if reference_responses else None
                langchain_results = self.langchain_evaluator.evaluate_with_langchain(
                    prompt_variant.prompt,
                    model_response.response,
                    primary_reference
                )
                result.langchain_results = langchain_results
            except Exception as e:
                logger.error(f"LangChain evaluation failed: {e}")
        
        # LLM-as-judge evaluation
        if use_llm_judge:
            try:
                primary_reference = reference_responses[0] if reference_responses else None
                llm_judge_result = self.langchain_evaluator.evaluate_with_llm_judge(
                    prompt_variant.prompt,
                    model_response.response,
                    primary_reference
                )
                result.llm_judge_result = llm_judge_result
            except Exception as e:
                logger.error(f"LLM-as-judge evaluation failed: {e}")
        
        # Calculate combined score
        result.combined_score = self._calculate_combined_score(result)
        result.confidence = self._calculate_confidence(result)
        
        result.evaluation_time = time.time() - start_time
        
        return result
    
    def _calculate_combined_score(self, result: EvaluationResult, 
                                 human_feedback_weight: float = 0.5) -> Optional[float]:
        """
        Calculate a combined score from all available evaluation results.
        
        Args:
            result: EvaluationResult to calculate score for
            human_feedback_weight: Weight for human feedback (0-1)
        
        Returns:
            Combined score (0-1) or None if no scores available
        """
        scores = []
        weights = []
        
        # Human feedback (highest priority when available)
        if result.human_feedback:
            human_score = result.human_feedback.to_score()
            if human_score is not None:
                scores.append(human_score)
                weights.append(human_feedback_weight)
        
        # Adjust remaining weights based on human feedback presence
        remaining_weight = 1.0 - (human_feedback_weight if result.human_feedback and result.human_feedback.to_score() is not None else 0.0)
        
        # Automated metrics (if available)
        if result.metric_scores:
            # Use semantic similarity as primary automated metric
            if result.metric_scores.semantic_similarity is not None:
                scores.append(result.metric_scores.semantic_similarity)
                weights.append(0.2 * remaining_weight)
            
            # Use BLEU-4 as secondary metric
            scores.append(result.metric_scores.bleu_4)
            weights.append(0.1 * remaining_weight)
        
        # LangChain evaluation (weighted average)
        if result.langchain_results:
            langchain_scores = [r.score for r in result.langchain_results if r.score is not None]
            if langchain_scores:
                langchain_avg = sum(langchain_scores) / len(langchain_scores)
                scores.append(langchain_avg / 5.0)  # Normalize to 0-1 scale
                weights.append(0.3 * remaining_weight)
        
        # LLM-as-judge (high weight for automated evaluation)
        if result.llm_judge_result and result.llm_judge_result.overall_score is not None:
            scores.append(result.llm_judge_result.overall_score / 5.0)  # Normalize to 0-1 scale
            weights.append(0.4 * remaining_weight)
        
        if not scores:
            return None
        
        # Calculate weighted average
        if len(scores) == len(weights) and sum(weights) > 0:
            combined = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        else:
            combined = sum(scores) / len(scores)
        
        return combined
    
    def update_human_feedback(self, result: EvaluationResult, 
                            human_feedback: HumanFeedback,
                            human_feedback_weight: float = 0.5) -> EvaluationResult:
        """
        Update an evaluation result with human feedback and recalculate scores.
        
        Args:
            result: Existing EvaluationResult to update
            human_feedback: HumanFeedback to add
            human_feedback_weight: Weight for human feedback in combined score
        
        Returns:
            Updated EvaluationResult with new combined score
        """
        # Update human feedback
        result.human_feedback = human_feedback
        
        # Recalculate combined score with human feedback
        result.combined_score = self._calculate_combined_score(result, human_feedback_weight)
        
        # Update confidence based on human feedback
        result.confidence = self._calculate_confidence_with_human_feedback(result)
        
        return result
    
    def create_feedback_based_ranking(self, results: List[EvaluationResult],
                                    human_feedback_weight: float = 0.5) -> List[EvaluationResult]:
        """
        Create a ranking of evaluation results that incorporates human feedback.
        
        Args:
            results: List of EvaluationResult objects
            human_feedback_weight: Weight for human feedback in ranking
        
        Returns:
            Sorted list of results (best first)
        """
        # Recalculate scores for all results with human feedback weight
        updated_results = []
        for result in results:
            updated_result = result
            updated_result.combined_score = self._calculate_combined_score(result, human_feedback_weight)
            updated_results.append(updated_result)
        
        # Sort by combined score (descending)
        ranked_results = sorted(
            updated_results,
            key=lambda r: r.combined_score if r.combined_score is not None else 0.0,
            reverse=True
        )
        
        return ranked_results
    
    def _calculate_confidence(self, result: EvaluationResult) -> Optional[float]:
        """Calculate confidence in the evaluation result."""
        return self._calculate_confidence_with_human_feedback(result)
    
    def _calculate_confidence_with_human_feedback(self, result: EvaluationResult) -> Optional[float]:
        """Calculate confidence in the evaluation result including human feedback."""
        confidence_factors = []
        
        # Human feedback confidence (highest when available)
        if result.human_feedback and result.human_feedback.to_score() is not None:
            # Human feedback is considered highly reliable
            if result.human_feedback.star_rating is not None:
                # Star ratings are more detailed, higher confidence
                confidence_factors.append(0.9)
            elif result.human_feedback.thumbs_rating is not None:
                # Thumbs ratings are binary, moderate confidence
                confidence_factors.append(0.7)
        
        # LLM-as-judge confidence
        if result.llm_judge_result and result.llm_judge_result.confidence is not None:
            confidence_factors.append(result.llm_judge_result.confidence / 5.0)
        
        # Number of evaluation methods used (including human feedback)
        methods_used = sum([
            result.metric_scores is not None,
            result.langchain_results is not None,
            result.llm_judge_result is not None,
            result.human_feedback is not None and result.human_feedback.to_score() is not None
        ])
        
        method_confidence = methods_used / 4.0  # Now out of 4 possible methods
        confidence_factors.append(method_confidence)
        
        if not confidence_factors:
            return None
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def batch_evaluate_prompts(self,
                             prompt_variants: List[PromptVariant],
                             model_responses: List[ModelResponse],
                             reference_responses: Optional[Dict[str, List[str]]] = None,
                             use_metrics: bool = True,
                             use_langchain: bool = True,
                             use_llm_judge: bool = True) -> List[EvaluationResult]:
        """
        Evaluate multiple prompt-response pairs in batch.
        
        Args:
            prompt_variants: List of prompt variants
            model_responses: List of model responses
            reference_responses: Optional dict mapping prompt_id to reference responses
            use_metrics: Whether to calculate automated metrics
            use_langchain: Whether to use LangChain evaluation
            use_llm_judge: Whether to use LLM-as-judge evaluation
            
        Returns:
            List of EvaluationResult objects
        """
        # Create prompt lookup
        prompt_lookup = {p.id: p for p in prompt_variants}
        
        # Prepare evaluation tasks
        evaluation_tasks = []
        for response in model_responses:
            if response.prompt_id in prompt_lookup:
                prompt = prompt_lookup[response.prompt_id]
                refs = reference_responses.get(response.prompt_id) if reference_responses else None
                
                evaluation_tasks.append((prompt, response, refs))
        
        # Execute evaluations concurrently
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(
                    self._evaluate_single_response,
                    prompt, response, refs, use_metrics, use_langchain, use_llm_judge
                ): (prompt, response)
                for prompt, response, refs in evaluation_tasks
            }
            
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    prompt, response = future_to_task[future]
                    logger.error(f"Evaluation failed for prompt {prompt.id}, model {response.model_id}: {e}")
                    
                    # Create error result
                    error_result = EvaluationResult(
                        prompt_id=prompt.id,
                        model_id=response.model_id,
                        response=response.response,
                        combined_score=0.0,
                        metadata={"error": str(e)}
                    )
                    results.append(error_result)
        
        return results
    
    def compare_prompts(self,
                       prompt_a_id: str,
                       prompt_b_id: str,
                       model_id: str,
                       evaluation_results: List[EvaluationResult],
                       confidence_level: float = 0.95) -> Optional[ComparisonResult]:
        """
        Compare two prompt variants statistically.
        
        Args:
            prompt_a_id: ID of first prompt variant
            prompt_b_id: ID of second prompt variant
            model_id: ID of the model used
            evaluation_results: List of evaluation results
            confidence_level: Confidence level for statistical tests
            
        Returns:
            ComparisonResult or None if insufficient data
        """
        # Filter results for the specific prompts and model
        results_a = [r for r in evaluation_results 
                    if r.prompt_id == prompt_a_id and r.model_id == model_id 
                    and r.combined_score is not None]
        results_b = [r for r in evaluation_results 
                    if r.prompt_id == prompt_b_id and r.model_id == model_id 
                    and r.combined_score is not None]
        
        if len(results_a) < self.min_samples or len(results_b) < self.min_samples:
            logger.warning(f"Insufficient samples for comparison: {len(results_a)} vs {len(results_b)}")
            return None
        
        # Extract scores
        scores_a = [r.combined_score for r in results_a]
        scores_b = [r.combined_score for r in results_b]
        
        # Calculate basic statistics
        mean_a = statistics.mean(scores_a)
        mean_b = statistics.mean(scores_b)
        
        # Perform statistical test (Welch's t-test for unequal variances)
        t_stat, p_value = stats.ttest_ind(scores_a, scores_b, equal_var=False)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(scores_a) - 1) * np.var(scores_a, ddof=1) + 
                             (len(scores_b) - 1) * np.var(scores_b, ddof=1)) / 
                            (len(scores_a) + len(scores_b) - 2))
        effect_size = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0
        
        # Calculate win rates
        wins_a = sum(1 for sa, sb in zip(scores_a, scores_b) if sa > sb)
        wins_b = sum(1 for sa, sb in zip(scores_a, scores_b) if sb > sa)
        ties = len(scores_a) - wins_a - wins_b
        
        total_comparisons = len(scores_a)
        win_rate_a = wins_a / total_comparisons
        win_rate_b = wins_b / total_comparisons
        tie_rate = ties / total_comparisons
        
        # Calculate confidence interval for the difference
        se_diff = np.sqrt(np.var(scores_a, ddof=1) / len(scores_a) + 
                         np.var(scores_b, ddof=1) / len(scores_b))
        
        alpha = 1 - confidence_level
        df = len(scores_a) + len(scores_b) - 2
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        margin_error = t_critical * se_diff
        diff = mean_a - mean_b
        confidence_interval = (diff - margin_error, diff + margin_error)
        
        return ComparisonResult(
            prompt_a_id=prompt_a_id,
            prompt_b_id=prompt_b_id,
            model_id=model_id,
            win_rate_a=win_rate_a,
            win_rate_b=win_rate_b,
            tie_rate=tie_rate,
            p_value=p_value,
            is_significant=p_value < self.significance_threshold,
            effect_size=effect_size,
            mean_score_a=mean_a,
            mean_score_b=mean_b,
            score_difference=diff,
            confidence_interval=confidence_interval,
            n_comparisons=total_comparisons
        )
    
    def rank_prompts(self,
                    prompt_ids: List[str],
                    model_id: str,
                    evaluation_results: List[EvaluationResult]) -> Optional[RankingResult]:
        """
        Rank multiple prompt variants by performance.
        
        Args:
            prompt_ids: List of prompt IDs to rank
            model_id: ID of the model used
            evaluation_results: List of evaluation results
            
        Returns:
            RankingResult with ranked prompts
        """
        # Filter and aggregate results by prompt
        prompt_scores = {}
        prompt_confidences = {}
        
        for prompt_id in prompt_ids:
            results = [r for r in evaluation_results 
                      if r.prompt_id == prompt_id and r.model_id == model_id 
                      and r.combined_score is not None]
            
            if results:
                scores = [r.combined_score for r in results]
                confidences = [r.confidence for r in results if r.confidence is not None]
                
                prompt_scores[prompt_id] = {
                    'mean_score': statistics.mean(scores),
                    'std_score': statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    'n_samples': len(scores)
                }
                
                if confidences:
                    prompt_confidences[prompt_id] = statistics.mean(confidences)
                else:
                    prompt_confidences[prompt_id] = 0.5  # Default confidence
        
        if not prompt_scores:
            return None
        
        # Sort prompts by mean score (descending)
        sorted_prompts = sorted(prompt_scores.items(), 
                              key=lambda x: x[1]['mean_score'], 
                              reverse=True)
        
        # Create ranking list
        rankings = []
        for rank, (prompt_id, scores) in enumerate(sorted_prompts, 1):
            rankings.append({
                'prompt_id': prompt_id,
                'rank': rank,
                'score': scores['mean_score'],
                'score_std': scores['std_score'],
                'n_samples': scores['n_samples'],
                'confidence': prompt_confidences.get(prompt_id, 0.5)
            })
        
        # Calculate ranking confidence (based on score differences and sample sizes)
        ranking_confidence = self._calculate_ranking_confidence(rankings)
        
        return RankingResult(
            model_id=model_id,
            rankings=rankings,
            ranking_confidence=ranking_confidence
        )
    
    def _calculate_ranking_confidence(self, rankings: List[Dict[str, Any]]) -> float:
        """Calculate confidence in the ranking based on score differences and sample sizes."""
        if len(rankings) < 2:
            return 1.0
        
        confidence_factors = []
        
        # Check score differences between adjacent ranks
        for i in range(len(rankings) - 1):
            current = rankings[i]
            next_rank = rankings[i + 1]
            
            score_diff = current['score'] - next_rank['score']
            combined_std = np.sqrt(current['score_std']**2 + next_rank['score_std']**2)
            
            if combined_std > 0:
                # Normalized difference (higher is more confident)
                normalized_diff = score_diff / combined_std
                confidence_factors.append(min(1.0, normalized_diff))
            else:
                confidence_factors.append(0.5)
        
        # Average confidence across all adjacent pairs
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    def generate_comparison_matrix(self,
                                 prompt_ids: List[str],
                                 model_id: str,
                                 evaluation_results: List[EvaluationResult]) -> Dict[str, Dict[str, Optional[ComparisonResult]]]:
        """
        Generate a matrix of pairwise prompt comparisons.
        
        Args:
            prompt_ids: List of prompt IDs to compare
            model_id: ID of the model used
            evaluation_results: List of evaluation results
            
        Returns:
            Dictionary matrix of comparison results
        """
        comparison_matrix = {}
        
        # Initialize matrix
        for prompt_a in prompt_ids:
            comparison_matrix[prompt_a] = {}
            for prompt_b in prompt_ids:
                if prompt_a == prompt_b:
                    comparison_matrix[prompt_a][prompt_b] = None
                else:
                    comparison_matrix[prompt_a][prompt_b] = None
        
        # Perform pairwise comparisons
        for prompt_a, prompt_b in itertools.combinations(prompt_ids, 2):
            comparison = self.compare_prompts(
                prompt_a, prompt_b, model_id, evaluation_results
            )
            
            comparison_matrix[prompt_a][prompt_b] = comparison
            
            # Create reverse comparison
            if comparison:
                reverse_comparison = ComparisonResult(
                    prompt_a_id=prompt_b,
                    prompt_b_id=prompt_a,
                    model_id=comparison.model_id,
                    win_rate_a=comparison.win_rate_b,
                    win_rate_b=comparison.win_rate_a,
                    tie_rate=comparison.tie_rate,
                    p_value=comparison.p_value,
                    is_significant=comparison.is_significant,
                    effect_size=-comparison.effect_size,
                    mean_score_a=comparison.mean_score_b,
                    mean_score_b=comparison.mean_score_a,
                    score_difference=-comparison.score_difference,
                    confidence_interval=(-comparison.confidence_interval[1], 
                                       -comparison.confidence_interval[0]),
                    n_comparisons=comparison.n_comparisons
                )
                comparison_matrix[prompt_b][prompt_a] = reverse_comparison
        
        return comparison_matrix
    
    def get_evaluation_summary(self, evaluation_results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Generate a summary of evaluation results.
        
        Args:
            evaluation_results: List of evaluation results
            
        Returns:
            Dictionary with summary statistics
        """
        if not evaluation_results:
            return {}
        
        # Filter valid results
        valid_results = [r for r in evaluation_results if r.combined_score is not None]
        
        if not valid_results:
            return {"total_evaluations": len(evaluation_results), "valid_evaluations": 0}
        
        # Basic statistics
        scores = [r.combined_score for r in valid_results]
        confidences = [r.confidence for r in valid_results if r.confidence is not None]
        
        summary = {
            "total_evaluations": len(evaluation_results),
            "valid_evaluations": len(valid_results),
            "score_statistics": {
                "mean": statistics.mean(scores),
                "median": statistics.median(scores),
                "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                "min": min(scores),
                "max": max(scores)
            }
        }
        
        if confidences:
            summary["confidence_statistics"] = {
                "mean": statistics.mean(confidences),
                "median": statistics.median(confidences),
                "std": statistics.stdev(confidences) if len(confidences) > 1 else 0.0
            }
        
        # Group by prompt and model
        by_prompt = {}
        by_model = {}
        
        for result in valid_results:
            # By prompt
            if result.prompt_id not in by_prompt:
                by_prompt[result.prompt_id] = []
            by_prompt[result.prompt_id].append(result.combined_score)
            
            # By model
            if result.model_id not in by_model:
                by_model[result.model_id] = []
            by_model[result.model_id].append(result.combined_score)
        
        # Calculate averages
        summary["by_prompt"] = {
            prompt_id: {
                "mean_score": statistics.mean(scores),
                "n_samples": len(scores)
            }
            for prompt_id, scores in by_prompt.items()
        }
        
        summary["by_model"] = {
            model_id: {
                "mean_score": statistics.mean(scores),
                "n_samples": len(scores)
            }
            for model_id, scores in by_model.items()
        }
        
        return summary
    
    def __del__(self):
        """Cleanup thread pool executor."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)