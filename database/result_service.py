"""
Result storage and retrieval service with export functionality.

Provides comprehensive result management including search, filtering, and export
capabilities for experiments, evaluations, and comparisons.
"""

import csv
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from io import StringIO, BytesIO
from dataclasses import dataclass, asdict

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func

from .connection import db_manager
from .models import Experiment, Model, Evaluation, Dataset, ComparisonResult
from .repositories import (
    experiment_repository, evaluation_repository, model_repository,
    dataset_repository, comparison_repository
)
from utils.exceptions import DatabaseError, ValidationError

import logging

logger = logging.getLogger(__name__)


@dataclass
class SearchFilters:
    """Search filters for result queries."""
    query_text: Optional[str] = None
    experiment_ids: Optional[List[uuid.UUID]] = None
    model_ids: Optional[List[uuid.UUID]] = None
    dataset_ids: Optional[List[uuid.UUID]] = None
    min_rating: Optional[int] = None
    max_rating: Optional[int] = None
    min_cost: Optional[float] = None
    max_cost: Optional[float] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    status: Optional[str] = None
    experiment_type: Optional[str] = None


@dataclass
class ExportOptions:
    """Export configuration options."""
    format: str = 'csv'  # 'csv', 'json', 'pdf'
    include_metadata: bool = True
    include_human_feedback: bool = True
    include_metrics: bool = True
    max_records: Optional[int] = None


class ResultStorageService:
    """
    Service for storing, retrieving, and exporting experiment results.
    
    Provides comprehensive result management with search, filtering,
    and export capabilities.
    """
    
    def __init__(self):
        self.db_manager = db_manager
    
    def store_experiment_results(
        self,
        experiment_id: uuid.UUID,
        results: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        cost_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store experiment results with metadata.
        
        Args:
            experiment_id: Experiment identifier
            results: Experiment results data
            metrics: Performance metrics
            cost_data: Cost tracking data
            
        Returns:
            bool: True if storage successful
            
        Raises:
            DatabaseError: If storage fails
        """
        try:
            with self.db_manager.get_session() as session:
                experiment = experiment_repository.get_by_id(session, experiment_id)
                if not experiment:
                    raise ValidationError(f"Experiment {experiment_id} not found")
                
                # Update experiment with results
                update_data = {
                    'results': results,
                    'status': 'completed',
                    'completed_at': datetime.utcnow()
                }
                
                if metrics:
                    update_data['metrics'] = metrics
                
                if cost_data:
                    update_data['total_cost_usd'] = cost_data.get('total_cost', 0.0)
                    update_data['token_usage'] = cost_data.get('token_usage', {})
                
                experiment_repository.update(session, experiment_id, **update_data)
                
                logger.info(f"Stored results for experiment {experiment_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store experiment results: {e}")
            raise DatabaseError(f"Result storage failed: {e}")
    
    def store_evaluation_batch(
        self,
        evaluations: List[Dict[str, Any]]
    ) -> List[Evaluation]:
        """
        Store batch of evaluation results.
        
        Args:
            evaluations: List of evaluation data
            
        Returns:
            List[Evaluation]: Created evaluation records
            
        Raises:
            DatabaseError: If storage fails
        """
        try:
            with self.db_manager.get_session() as session:
                stored_evaluations = evaluation_repository.create_batch(
                    session, evaluations
                )
                
                logger.info(f"Stored {len(stored_evaluations)} evaluations")
                return stored_evaluations
                
        except Exception as e:
            logger.error(f"Failed to store evaluation batch: {e}")
            raise DatabaseError(f"Evaluation storage failed: {e}")
    
    def search_results(
        self,
        filters: SearchFilters,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = 'created_at',
        sort_order: str = 'desc'
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Search results with comprehensive filtering.
        
        Args:
            filters: Search filters
            limit: Maximum results to return
            offset: Results offset for pagination
            sort_by: Field to sort by
            sort_order: Sort order ('asc' or 'desc')
            
        Returns:
            Tuple[List[Dict], int]: Results and total count
            
        Raises:
            DatabaseError: If search fails
        """
        try:
            with self.db_manager.get_session() as session:
                # Build query based on filters
                query = session.query(Evaluation)
                
                # Apply filters
                if filters.query_text:
                    query = query.filter(
                        or_(
                            Evaluation.prompt.contains(filters.query_text),
                            Evaluation.response.contains(filters.query_text)
                        )
                    )
                
                if filters.experiment_ids:
                    query = query.filter(
                        Evaluation.experiment_id.in_(filters.experiment_ids)
                    )
                
                if filters.model_ids:
                    query = query.filter(
                        Evaluation.model_id.in_(filters.model_ids)
                    )
                
                if filters.min_rating is not None:
                    query = query.filter(Evaluation.human_rating >= filters.min_rating)
                
                if filters.max_rating is not None:
                    query = query.filter(Evaluation.human_rating <= filters.max_rating)
                
                if filters.min_cost is not None:
                    query = query.filter(Evaluation.cost_usd >= filters.min_cost)
                
                if filters.max_cost is not None:
                    query = query.filter(Evaluation.cost_usd <= filters.max_cost)
                
                if filters.date_from:
                    query = query.filter(Evaluation.created_at >= filters.date_from)
                
                if filters.date_to:
                    query = query.filter(Evaluation.created_at <= filters.date_to)
                
                # Get total count
                total_count = query.count()
                
                # Apply sorting
                sort_column = getattr(Evaluation, sort_by, Evaluation.created_at)
                if sort_order.lower() == 'desc':
                    query = query.order_by(desc(sort_column))
                else:
                    query = query.order_by(asc(sort_column))
                
                # Apply pagination
                results = query.offset(offset).limit(limit).all()
                
                # Convert to dictionaries with related data
                result_dicts = []
                for evaluation in results:
                    result_dict = {
                        'id': str(evaluation.id),
                        'experiment_id': str(evaluation.experiment_id),
                        'model_id': str(evaluation.model_id),
                        'prompt': evaluation.prompt,
                        'response': evaluation.response,
                        'expected_response': evaluation.expected_response,
                        'metrics': evaluation.metrics,
                        'llm_judge_scores': evaluation.llm_judge_scores,
                        'human_rating': evaluation.human_rating,
                        'human_feedback': evaluation.human_feedback,
                        'latency_ms': evaluation.latency_ms,
                        'token_count_input': evaluation.token_count_input,
                        'token_count_output': evaluation.token_count_output,
                        'cost_usd': evaluation.cost_usd,
                        'created_at': evaluation.created_at.isoformat(),
                        'model_name': evaluation.model.name if evaluation.model else None,
                        'experiment_name': evaluation.experiment.name if evaluation.experiment else None
                    }
                    result_dicts.append(result_dict)
                
                return result_dicts, total_count
                
        except Exception as e:
            logger.error(f"Failed to search results: {e}")
            raise DatabaseError(f"Result search failed: {e}")
    
    def get_experiment_summary(
        self,
        experiment_id: uuid.UUID
    ) -> Dict[str, Any]:
        """
        Get comprehensive experiment summary.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Dict[str, Any]: Experiment summary with statistics
            
        Raises:
            DatabaseError: If retrieval fails
        """
        try:
            with self.db_manager.get_session() as session:
                experiment = experiment_repository.get_by_id(session, experiment_id)
                if not experiment:
                    raise ValidationError(f"Experiment {experiment_id} not found")
                
                # Get evaluation statistics
                eval_stats = evaluation_repository.get_evaluation_stats(
                    session, experiment_id=experiment_id
                )
                
                # Get evaluations for detailed analysis
                evaluations = evaluation_repository.get_by_experiment(
                    session, experiment_id
                )
                
                # Calculate additional metrics
                if evaluations:
                    ratings = [e.human_rating for e in evaluations if e.human_rating]
                    rating_distribution = {}
                    for rating in range(1, 6):
                        rating_distribution[rating] = ratings.count(rating)
                else:
                    rating_distribution = {}
                
                summary = {
                    'experiment': {
                        'id': str(experiment.id),
                        'name': experiment.name,
                        'description': experiment.description,
                        'type': experiment.type,
                        'status': experiment.status,
                        'created_at': experiment.created_at.isoformat(),
                        'completed_at': experiment.completed_at.isoformat() if experiment.completed_at else None,
                        'duration_seconds': experiment.duration_seconds,
                        'results': experiment.results,
                        'metrics': experiment.metrics,
                        'total_cost_usd': experiment.total_cost_usd,
                        'token_usage': experiment.token_usage
                    },
                    'dataset': {
                        'id': str(experiment.dataset.id) if experiment.dataset else None,
                        'name': experiment.dataset.name if experiment.dataset else None,
                        'size': experiment.dataset.size if experiment.dataset else None,
                        'domain': experiment.dataset.domain if experiment.dataset else None
                    },
                    'model': {
                        'id': str(experiment.model.id) if experiment.model else None,
                        'name': experiment.model.name if experiment.model else None,
                        'type': experiment.model.type if experiment.model else None
                    },
                    'evaluation_stats': eval_stats,
                    'rating_distribution': rating_distribution
                }
                
                return summary
                
        except Exception as e:
            logger.error(f"Failed to get experiment summary: {e}")
            raise DatabaseError(f"Summary retrieval failed: {e}")
    
    def export_results(
        self,
        filters: SearchFilters,
        options: ExportOptions
    ) -> Union[str, bytes]:
        """
        Export results in specified format.
        
        Args:
            filters: Search filters for results to export
            options: Export configuration
            
        Returns:
            Union[str, bytes]: Exported data
            
        Raises:
            DatabaseError: If export fails
        """
        try:
            # Get results to export
            limit = options.max_records or 10000  # Default limit
            results, total_count = self.search_results(filters, limit=limit)
            
            if options.format.lower() == 'csv':
                return self._export_csv(results, options)
            elif options.format.lower() == 'json':
                return self._export_json(results, options)
            elif options.format.lower() == 'pdf':
                return self._export_pdf(results, options)
            else:
                raise ValidationError(f"Unsupported export format: {options.format}")
                
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            raise DatabaseError(f"Export failed: {e}")
    
    def _export_csv(
        self,
        results: List[Dict[str, Any]],
        options: ExportOptions
    ) -> str:
        """Export results as CSV."""
        output = StringIO()
        
        if not results:
            return ""
        
        # Define columns based on options
        columns = [
            'id', 'experiment_name', 'model_name', 'prompt', 'response',
            'latency_ms', 'cost_usd', 'created_at'
        ]
        
        if options.include_human_feedback:
            columns.extend(['human_rating', 'human_feedback'])
        
        if options.include_metrics:
            columns.append('metrics')
        
        if options.include_metadata:
            columns.extend([
                'token_count_input', 'token_count_output',
                'llm_judge_scores', 'expected_response'
            ])
        
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        
        for result in results:
            # Flatten complex fields for CSV
            row = {col: result.get(col, '') for col in columns}
            
            # Convert complex objects to strings
            if 'metrics' in row and isinstance(row['metrics'], dict):
                row['metrics'] = json.dumps(row['metrics'])
            
            if 'llm_judge_scores' in row and isinstance(row['llm_judge_scores'], dict):
                row['llm_judge_scores'] = json.dumps(row['llm_judge_scores'])
            
            writer.writerow(row)
        
        return output.getvalue()
    
    def _export_json(
        self,
        results: List[Dict[str, Any]],
        options: ExportOptions
    ) -> str:
        """Export results as JSON."""
        export_data = {
            'export_timestamp': datetime.utcnow().isoformat(),
            'total_records': len(results),
            'export_options': asdict(options),
            'results': results
        }
        
        return json.dumps(export_data, indent=2, default=str)
    
    def _export_pdf(
        self,
        results: List[Dict[str, Any]],
        options: ExportOptions
    ) -> bytes:
        """Export results as PDF."""
        # For now, return a simple text-based PDF content
        # In a real implementation, you'd use a library like reportlab
        
        content = f"""
LLM Optimization Platform - Results Export
==========================================

Export Date: {datetime.utcnow().isoformat()}
Total Records: {len(results)}

Results Summary:
"""
        
        for i, result in enumerate(results[:10]):  # Limit for PDF
            content += f"""
Result {i+1}:
- Experiment: {result.get('experiment_name', 'N/A')}
- Model: {result.get('model_name', 'N/A')}
- Rating: {result.get('human_rating', 'N/A')}
- Cost: ${result.get('cost_usd', 0):.4f}
- Latency: {result.get('latency_ms', 'N/A')}ms

"""
        
        if len(results) > 10:
            content += f"\n... and {len(results) - 10} more results"
        
        return content.encode('utf-8')
    
    def get_analytics_data(
        self,
        filters: Optional[SearchFilters] = None
    ) -> Dict[str, Any]:
        """
        Get analytics data for dashboard.
        
        Args:
            filters: Optional filters for analytics
            
        Returns:
            Dict[str, Any]: Analytics data
        """
        try:
            with self.db_manager.get_session() as session:
                # Get overall statistics
                total_experiments = session.query(func.count(Experiment.id)).scalar()
                total_evaluations = session.query(func.count(Evaluation.id)).scalar()
                total_cost = session.query(func.sum(Evaluation.cost_usd)).scalar() or 0
                
                # Get model performance comparison
                model_stats = session.query(
                    Model.name,
                    func.count(Evaluation.id).label('evaluation_count'),
                    func.avg(Evaluation.human_rating).label('avg_rating'),
                    func.avg(Evaluation.cost_usd).label('avg_cost'),
                    func.avg(Evaluation.latency_ms).label('avg_latency')
                ).join(Evaluation).group_by(Model.id, Model.name).all()
                
                # Get recent activity
                recent_experiments = session.query(Experiment).order_by(
                    desc(Experiment.created_at)
                ).limit(10).all()
                
                analytics = {
                    'overview': {
                        'total_experiments': total_experiments,
                        'total_evaluations': total_evaluations,
                        'total_cost_usd': float(total_cost),
                        'active_models': session.query(func.count(Model.id)).filter(
                            Model.status == 'active'
                        ).scalar()
                    },
                    'model_performance': [
                        {
                            'model_name': stat.name,
                            'evaluation_count': stat.evaluation_count,
                            'avg_rating': float(stat.avg_rating) if stat.avg_rating else 0,
                            'avg_cost': float(stat.avg_cost) if stat.avg_cost else 0,
                            'avg_latency': float(stat.avg_latency) if stat.avg_latency else 0
                        }
                        for stat in model_stats
                    ],
                    'recent_experiments': [
                        {
                            'id': str(exp.id),
                            'name': exp.name,
                            'type': exp.type,
                            'status': exp.status,
                            'created_at': exp.created_at.isoformat()
                        }
                        for exp in recent_experiments
                    ]
                }
                
                return analytics
                
        except Exception as e:
            logger.error(f"Failed to get analytics data: {e}")
            raise DatabaseError(f"Analytics retrieval failed: {e}")


# Service instance
result_service = ResultStorageService()