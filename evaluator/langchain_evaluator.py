"""
LangChain-based evaluation pipeline for LLM outputs.

This module provides comprehensive evaluation using LangChain evaluators
and LLM-as-judge evaluation with GPT-4 for multiple criteria assessment.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

# LangChain imports
from langchain.evaluation import (
    EvaluatorType,
    load_evaluator,
    Criteria
)
from langchain.schema import BaseOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.base import BaseLLM

# OpenAI for LLM-as-judge
from openai import OpenAI
import os

logger = logging.getLogger(__name__)


@dataclass
class EvaluationCriteria:
    """Evaluation criteria configuration."""
    name: str
    description: str
    scale: str = "1-5"
    weight: float = 1.0


@dataclass
class LangChainEvaluationResult:
    """Result from LangChain evaluation."""
    criterion: str
    score: float
    reasoning: str
    confidence: Optional[float] = None
    evaluation_time: Optional[float] = None


@dataclass
class LLMJudgeResult:
    """Result from LLM-as-judge evaluation."""
    overall_score: float
    criteria_scores: Dict[str, float]
    reasoning: str
    confidence: float
    evaluation_time: float


class LangChainEvaluator:
    """
    LangChain-based evaluation pipeline for comprehensive LLM output assessment.
    
    Supports multiple evaluation criteria and LLM-as-judge evaluation.
    """
    
    # Default evaluation criteria
    DEFAULT_CRITERIA = [
        EvaluationCriteria(
            name="helpfulness",
            description="How helpful is the response in addressing the user's question or request?",
            weight=1.5
        ),
        EvaluationCriteria(
            name="clarity",
            description="How clear and well-structured is the response?",
            weight=1.2
        ),
        EvaluationCriteria(
            name="accuracy",
            description="How accurate and factually correct is the information provided?",
            weight=1.8
        ),
        EvaluationCriteria(
            name="relevance",
            description="How relevant is the response to the input prompt?",
            weight=1.3
        ),
        EvaluationCriteria(
            name="completeness",
            description="How complete is the response in addressing all aspects of the prompt?",
            weight=1.1
        )
    ]
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 judge_model: str = "gpt-4",
                 max_workers: int = 3,
                 timeout: float = 30.0):
        """
        Initialize the LangChain evaluator.
        
        Args:
            openai_api_key: OpenAI API key for LLM-as-judge evaluation
            judge_model: Model to use for LLM-as-judge evaluation
            max_workers: Maximum number of concurrent evaluation workers
            timeout: Timeout for individual evaluations in seconds
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.judge_model = judge_model
        self.max_workers = max_workers
        self.timeout = timeout
        
        # Initialize OpenAI client for LLM-as-judge
        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None
            logger.warning("OpenAI API key not provided. LLM-as-judge evaluation will be disabled.")
        
        # Initialize evaluation criteria
        self.criteria = self.DEFAULT_CRITERIA.copy()
        
        # Thread pool for concurrent evaluations
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def add_custom_criterion(self, criterion: EvaluationCriteria):
        """Add a custom evaluation criterion."""
        self.criteria.append(criterion)
        logger.info(f"Added custom criterion: {criterion.name}")
    
    def remove_criterion(self, criterion_name: str):
        """Remove an evaluation criterion by name."""
        self.criteria = [c for c in self.criteria if c.name != criterion_name]
        logger.info(f"Removed criterion: {criterion_name}")
    
    def _evaluate_single_criterion(self, 
                                 input_text: str, 
                                 prediction: str, 
                                 reference: Optional[str],
                                 criterion: EvaluationCriteria) -> LangChainEvaluationResult:
        """
        Evaluate a single criterion using LangChain evaluators.
        
        Args:
            input_text: Original input/prompt
            prediction: Generated response
            reference: Reference response (optional)
            criterion: Evaluation criterion
            
        Returns:
            LangChainEvaluationResult with score and reasoning
        """
        start_time = time.time()
        
        try:
            # Create custom criteria for LangChain
            custom_criteria = {
                criterion.name: criterion.description
            }
            
            # Load the criteria evaluator
            evaluator = load_evaluator(
                EvaluatorType.CRITERIA,
                criteria=custom_criteria,
                llm=None  # Will use default LLM
            )
            
            # Prepare evaluation input
            eval_input = {
                "input": input_text,
                "prediction": prediction
            }
            
            if reference:
                eval_input["reference"] = reference
            
            # Run evaluation
            result = evaluator.evaluate_strings(**eval_input)
            
            # Extract score and reasoning
            score = result.get("score", 0.0)
            reasoning = result.get("reasoning", "No reasoning provided")
            
            evaluation_time = time.time() - start_time
            
            return LangChainEvaluationResult(
                criterion=criterion.name,
                score=float(score),
                reasoning=reasoning,
                evaluation_time=evaluation_time
            )
            
        except Exception as e:
            logger.error(f"Error evaluating criterion {criterion.name}: {e}")
            evaluation_time = time.time() - start_time
            
            return LangChainEvaluationResult(
                criterion=criterion.name,
                score=0.0,
                reasoning=f"Evaluation failed: {str(e)}",
                evaluation_time=evaluation_time
            )
    
    def evaluate_with_langchain(self, 
                              input_text: str, 
                              prediction: str, 
                              reference: Optional[str] = None,
                              criteria_subset: Optional[List[str]] = None) -> List[LangChainEvaluationResult]:
        """
        Evaluate using LangChain evaluators for multiple criteria.
        
        Args:
            input_text: Original input/prompt
            prediction: Generated response
            reference: Reference response (optional)
            criteria_subset: Subset of criteria to evaluate (None for all)
            
        Returns:
            List of LangChainEvaluationResult objects
        """
        # Filter criteria if subset specified
        if criteria_subset:
            eval_criteria = [c for c in self.criteria if c.name in criteria_subset]
        else:
            eval_criteria = self.criteria
        
        results = []
        
        # Evaluate each criterion
        for criterion in eval_criteria:
            try:
                result = self._evaluate_single_criterion(
                    input_text, prediction, reference, criterion
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate criterion {criterion.name}: {e}")
                results.append(LangChainEvaluationResult(
                    criterion=criterion.name,
                    score=0.0,
                    reasoning=f"Evaluation failed: {str(e)}"
                ))
        
        return results
    
    def _create_judge_prompt(self, 
                           input_text: str, 
                           prediction: str, 
                           reference: Optional[str] = None) -> str:
        """Create a comprehensive prompt for LLM-as-judge evaluation."""
        
        criteria_descriptions = "\n".join([
            f"- {c.name.title()}: {c.description} (Weight: {c.weight})"
            for c in self.criteria
        ])
        
        prompt = f"""You are an expert evaluator tasked with assessing the quality of an AI-generated response.

EVALUATION CRITERIA:
{criteria_descriptions}

INPUT PROMPT:
{input_text}

AI RESPONSE TO EVALUATE:
{prediction}"""
        
        if reference:
            prompt += f"""

REFERENCE RESPONSE:
{reference}"""
        
        prompt += """

INSTRUCTIONS:
1. Evaluate the AI response against each criterion on a scale of 1-5 (1=Poor, 2=Below Average, 3=Average, 4=Good, 5=Excellent)
2. Consider the reference response (if provided) as a benchmark
3. Provide specific reasoning for each score
4. Calculate a weighted overall score based on the criterion weights
5. Indicate your confidence level in the evaluation (1-5 scale)

RESPONSE FORMAT:
{
    "criteria_scores": {
        "helpfulness": <score>,
        "clarity": <score>,
        "accuracy": <score>,
        "relevance": <score>,
        "completeness": <score>
    },
    "reasoning": "<detailed explanation of scores>",
    "overall_score": <weighted_average>,
    "confidence": <confidence_level>
}

Provide only the JSON response without additional text."""
        
        return prompt
    
    def evaluate_with_llm_judge(self, 
                              input_text: str, 
                              prediction: str, 
                              reference: Optional[str] = None,
                              temperature: float = 0.1) -> Optional[LLMJudgeResult]:
        """
        Evaluate using LLM-as-judge with GPT-4.
        
        Args:
            input_text: Original input/prompt
            prediction: Generated response
            reference: Reference response (optional)
            temperature: Temperature for LLM generation
            
        Returns:
            LLMJudgeResult or None if evaluation fails
        """
        if not self.openai_client:
            logger.error("OpenAI client not initialized. Cannot perform LLM-as-judge evaluation.")
            return None
        
        start_time = time.time()
        
        try:
            # Create evaluation prompt
            prompt = self._create_judge_prompt(input_text, prediction, reference)
            
            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": "You are an expert AI response evaluator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=1000,
                timeout=self.timeout
            )
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            import json
            try:
                result_data = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback: extract JSON from response if wrapped in text
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result_data = json.loads(json_match.group())
                else:
                    raise ValueError("Could not parse JSON from LLM response")
            
            evaluation_time = time.time() - start_time
            
            return LLMJudgeResult(
                overall_score=float(result_data.get("overall_score", 0.0)),
                criteria_scores=result_data.get("criteria_scores", {}),
                reasoning=result_data.get("reasoning", "No reasoning provided"),
                confidence=float(result_data.get("confidence", 0.0)),
                evaluation_time=evaluation_time
            )
            
        except Exception as e:
            logger.error(f"LLM-as-judge evaluation failed: {e}")
            return None
    
    def evaluate_comprehensive(self, 
                             input_text: str, 
                             prediction: str, 
                             reference: Optional[str] = None,
                             use_langchain: bool = True,
                             use_llm_judge: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation using both LangChain and LLM-as-judge.
        
        Args:
            input_text: Original input/prompt
            prediction: Generated response
            reference: Reference response (optional)
            use_langchain: Whether to use LangChain evaluators
            use_llm_judge: Whether to use LLM-as-judge evaluation
            
        Returns:
            Dictionary containing all evaluation results
        """
        results = {
            "input": input_text,
            "prediction": prediction,
            "reference": reference,
            "langchain_results": None,
            "llm_judge_result": None,
            "combined_score": None,
            "evaluation_summary": {}
        }
        
        # LangChain evaluation
        if use_langchain:
            try:
                langchain_results = self.evaluate_with_langchain(
                    input_text, prediction, reference
                )
                results["langchain_results"] = langchain_results
                
                # Calculate average LangChain score
                if langchain_results:
                    weighted_sum = sum(r.score * next(c.weight for c in self.criteria if c.name == r.criterion) 
                                     for r in langchain_results)
                    total_weight = sum(c.weight for c in self.criteria)
                    results["evaluation_summary"]["langchain_weighted_score"] = weighted_sum / total_weight
                
            except Exception as e:
                logger.error(f"LangChain evaluation failed: {e}")
        
        # LLM-as-judge evaluation
        if use_llm_judge:
            try:
                llm_judge_result = self.evaluate_with_llm_judge(
                    input_text, prediction, reference
                )
                results["llm_judge_result"] = llm_judge_result
                
                if llm_judge_result:
                    results["evaluation_summary"]["llm_judge_score"] = llm_judge_result.overall_score
                    results["evaluation_summary"]["llm_judge_confidence"] = llm_judge_result.confidence
                
            except Exception as e:
                logger.error(f"LLM-as-judge evaluation failed: {e}")
        
        # Calculate combined score if both evaluations succeeded
        langchain_score = results["evaluation_summary"].get("langchain_weighted_score")
        llm_judge_score = results["evaluation_summary"].get("llm_judge_score")
        
        if langchain_score is not None and llm_judge_score is not None:
            # Weight LLM-as-judge more heavily due to its sophistication
            combined_score = (langchain_score * 0.3 + llm_judge_score * 0.7)
            results["combined_score"] = combined_score
            results["evaluation_summary"]["combined_score"] = combined_score
        elif llm_judge_score is not None:
            results["combined_score"] = llm_judge_score
        elif langchain_score is not None:
            results["combined_score"] = langchain_score
        
        return results
    
    def batch_evaluate(self, 
                      evaluations: List[Dict[str, str]],
                      use_langchain: bool = True,
                      use_llm_judge: bool = True) -> List[Dict[str, Any]]:
        """
        Perform batch evaluation on multiple input-prediction pairs.
        
        Args:
            evaluations: List of dicts with 'input', 'prediction', and optional 'reference'
            use_langchain: Whether to use LangChain evaluators
            use_llm_judge: Whether to use LLM-as-judge evaluation
            
        Returns:
            List of comprehensive evaluation results
        """
        results = []
        
        for eval_data in evaluations:
            try:
                result = self.evaluate_comprehensive(
                    input_text=eval_data["input"],
                    prediction=eval_data["prediction"],
                    reference=eval_data.get("reference"),
                    use_langchain=use_langchain,
                    use_llm_judge=use_llm_judge
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Batch evaluation failed for item: {e}")
                results.append({
                    "input": eval_data.get("input", ""),
                    "prediction": eval_data.get("prediction", ""),
                    "reference": eval_data.get("reference"),
                    "error": str(e),
                    "combined_score": 0.0
                })
        
        return results
    
    def get_evaluation_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics from batch evaluation results.
        
        Args:
            results: List of evaluation results from batch_evaluate
            
        Returns:
            Dictionary with evaluation statistics
        """
        if not results:
            return {}
        
        # Extract scores
        combined_scores = [r.get("combined_score", 0.0) for r in results if r.get("combined_score") is not None]
        langchain_scores = [r.get("evaluation_summary", {}).get("langchain_weighted_score", 0.0) 
                          for r in results if r.get("evaluation_summary", {}).get("langchain_weighted_score") is not None]
        llm_judge_scores = [r.get("evaluation_summary", {}).get("llm_judge_score", 0.0) 
                          for r in results if r.get("evaluation_summary", {}).get("llm_judge_score") is not None]
        
        stats = {
            "total_evaluations": len(results),
            "successful_evaluations": len([r for r in results if "error" not in r])
        }
        
        if combined_scores:
            stats["combined_score_stats"] = {
                "mean": sum(combined_scores) / len(combined_scores),
                "min": min(combined_scores),
                "max": max(combined_scores),
                "count": len(combined_scores)
            }
        
        if langchain_scores:
            stats["langchain_score_stats"] = {
                "mean": sum(langchain_scores) / len(langchain_scores),
                "min": min(langchain_scores),
                "max": max(langchain_scores),
                "count": len(langchain_scores)
            }
        
        if llm_judge_scores:
            stats["llm_judge_score_stats"] = {
                "mean": sum(llm_judge_scores) / len(llm_judge_scores),
                "min": min(llm_judge_scores),
                "max": max(llm_judge_scores),
                "count": len(llm_judge_scores)
            }
        
        return stats
    
    def __del__(self):
        """Cleanup thread pool executor."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)