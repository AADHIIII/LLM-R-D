"""
Evaluation engine for LLM optimization platform.

This package provides comprehensive evaluation capabilities including:
- Automated metrics calculation (BLEU, ROUGE, semantic similarity, perplexity)
- LangChain-based evaluation with multiple criteria
- LLM-as-judge evaluation using GPT-4
- Prompt comparison and statistical significance testing
- Batch evaluation and result aggregation
"""

from .metrics_calculator import (
    MetricsCalculator,
    MetricScores,
    calculate_length_ratio,
    calculate_keyword_coverage,
    calculate_repetition_penalty
)

from .langchain_evaluator import (
    LangChainEvaluator,
    EvaluationCriteria,
    LangChainEvaluationResult,
    LLMJudgeResult
)

from .prompt_evaluator import (
    PromptEvaluator,
    PromptVariant,
    ModelResponse,
    EvaluationResult,
    ComparisonResult,
    RankingResult
)

__all__ = [
    # Metrics Calculator
    'MetricsCalculator',
    'MetricScores',
    'calculate_length_ratio',
    'calculate_keyword_coverage',
    'calculate_repetition_penalty',
    
    # LangChain Evaluator
    'LangChainEvaluator',
    'EvaluationCriteria',
    'LangChainEvaluationResult',
    'LLMJudgeResult',
    
    # Prompt Evaluator
    'PromptEvaluator',
    'PromptVariant',
    'ModelResponse',
    'EvaluationResult',
    'ComparisonResult',
    'RankingResult'
]