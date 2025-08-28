"""
SQLAlchemy models for experiments, models, evaluations, and datasets.
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    Column, String, Text, Integer, Float, DateTime, Boolean, 
    ForeignKey, JSON, Index, UniqueConstraint
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.types import TypeDecorator, CHAR
from sqlalchemy import String as SQLString

Base = declarative_base()


class GUID(TypeDecorator):
    """Platform-independent GUID type."""
    
    impl = CHAR
    cache_ok = True
    
    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(UUID())
        else:
            return dialect.type_descriptor(CHAR(32))
    
    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return str(value)
        else:
            if not isinstance(value, uuid.UUID):
                return "%.32x" % uuid.UUID(value).int
            else:
                return "%.32x" % value.int
    
    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                return uuid.UUID(value)
            else:
                return value


class Dataset(Base):
    """Dataset model for storing training and evaluation datasets."""
    
    __tablename__ = 'datasets'
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    format = Column(String(50), nullable=False)  # 'jsonl', 'csv'
    size = Column(Integer, nullable=False)  # Number of samples
    domain = Column(String(100))  # Domain/category
    file_path = Column(String(500))  # Path to dataset file
    validation_split = Column(Float, default=0.2)
    extra_metadata = Column(JSON)  # Additional metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    experiments = relationship("Experiment", back_populates="dataset")
    
    # Indexes
    __table_args__ = (
        Index('idx_dataset_name', 'name'),
        Index('idx_dataset_domain', 'domain'),
        Index('idx_dataset_created', 'created_at'),
    )


class Model(Base):
    """Model registry for fine-tuned and commercial models."""
    
    __tablename__ = 'models'
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    type = Column(String(50), nullable=False)  # 'fine-tuned', 'commercial'
    base_model = Column(String(100))  # Base model name (e.g., 'gpt2', 'gpt-4')
    model_path = Column(String(500))  # Path to model files (for fine-tuned)
    api_endpoint = Column(String(500))  # API endpoint (for commercial)
    training_config = Column(JSON)  # Training configuration
    performance_metrics = Column(JSON)  # Performance benchmarks
    cost_per_token = Column(Float)  # Cost per token for commercial models
    status = Column(String(50), default='active')  # 'active', 'deprecated', 'training'
    version = Column(String(50), default='1.0.0')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    experiments = relationship("Experiment", back_populates="model")
    evaluations = relationship("Evaluation", back_populates="model")
    
    # Indexes
    __table_args__ = (
        Index('idx_model_name', 'name'),
        Index('idx_model_type', 'type'),
        Index('idx_model_status', 'status'),
        UniqueConstraint('name', 'version', name='uq_model_name_version'),
    )


class Experiment(Base):
    """Experiment tracking for fine-tuning and evaluation runs."""
    
    __tablename__ = 'experiments'
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    type = Column(String(50), nullable=False)  # 'fine-tuning', 'evaluation', 'comparison'
    status = Column(String(50), default='created')  # 'created', 'running', 'completed', 'failed'
    
    # Foreign keys
    dataset_id = Column(GUID(), ForeignKey('datasets.id'))
    model_id = Column(GUID(), ForeignKey('models.id'))
    
    # Configuration and results
    config = Column(JSON)  # Experiment configuration
    results = Column(JSON)  # Experiment results
    metrics = Column(JSON)  # Performance metrics
    
    # Tracking fields
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    duration_seconds = Column(Integer)
    
    # Cost tracking
    total_cost_usd = Column(Float, default=0.0)
    token_usage = Column(JSON)  # Token usage statistics
    
    # Metadata
    tags = Column(JSON)  # Experiment tags for organization
    notes = Column(Text)  # User notes
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="experiments")
    model = relationship("Model", back_populates="experiments")
    evaluations = relationship("Evaluation", back_populates="experiment")
    
    # Indexes
    __table_args__ = (
        Index('idx_experiment_name', 'name'),
        Index('idx_experiment_status', 'status'),
        Index('idx_experiment_type', 'type'),
        Index('idx_experiment_created', 'created_at'),
    )


class Evaluation(Base):
    """Individual evaluation results for prompt-response pairs."""
    
    __tablename__ = 'evaluations'
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    
    # Foreign keys
    experiment_id = Column(GUID(), ForeignKey('experiments.id'), nullable=False)
    model_id = Column(GUID(), ForeignKey('models.id'), nullable=False)
    
    # Input and output
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    expected_response = Column(Text)  # For supervised evaluation
    
    # Evaluation metrics
    metrics = Column(JSON)  # Automated metrics (BLEU, ROUGE, etc.)
    llm_judge_scores = Column(JSON)  # LLM-as-judge evaluation scores
    human_rating = Column(Integer)  # Human rating (1-5) - deprecated, use human_feedback_data
    human_feedback = Column(Text)  # Qualitative human feedback - deprecated, use human_feedback_data
    human_feedback_data = Column(JSON)  # Structured human feedback (thumbs, stars, comments)
    
    # Performance metrics
    latency_ms = Column(Integer)  # Response latency
    token_count_input = Column(Integer)
    token_count_output = Column(Integer)
    cost_usd = Column(Float)  # Cost for this evaluation
    
    # Metadata
    evaluation_criteria = Column(JSON)  # Criteria used for evaluation
    evaluator_version = Column(String(50))  # Version of evaluation system
    extra_metadata = Column(JSON)  # Additional metadata
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    experiment = relationship("Experiment", back_populates="evaluations")
    model = relationship("Model", back_populates="evaluations")
    
    # Indexes
    __table_args__ = (
        Index('idx_evaluation_experiment', 'experiment_id'),
        Index('idx_evaluation_model', 'model_id'),
        Index('idx_evaluation_created', 'created_at'),
        Index('idx_evaluation_human_rating', 'human_rating'),
    )


class ComparisonResult(Base):
    """Results from comparing multiple models or prompts."""
    
    __tablename__ = 'comparison_results'
    
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Comparison configuration
    model_ids = Column(JSON, nullable=False)  # List of model IDs being compared
    prompt_variants = Column(JSON)  # List of prompt variants
    evaluation_criteria = Column(JSON)  # Criteria for comparison
    
    # Results
    results = Column(JSON, nullable=False)  # Detailed comparison results
    winner_model_id = Column(GUID(), ForeignKey('models.id'))
    statistical_significance = Column(Float)  # P-value or confidence score
    
    # Metadata
    sample_size = Column(Integer)  # Number of evaluations compared
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_comparison_name', 'name'),
        Index('idx_comparison_created', 'created_at'),
    )