"""
Configuration management for LLM optimization platform.
Handles environment variables and configuration settings.
"""

import os
from typing import Optional
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field, ConfigDict
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings, Field
    ConfigDict = None
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API Key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API Key")
    gemini_api_key: Optional[str] = Field(default=None, description="Gemini API Key")
    
    # Database Configuration
    database_url: str = Field(default="sqlite:///llm_optimization.db", description="Database URL")
    postgres_db: Optional[str] = Field(default=None, description="PostgreSQL Database Name")
    postgres_user: Optional[str] = Field(default=None, description="PostgreSQL User")
    postgres_password: Optional[str] = Field(default=None, description="PostgreSQL Password")
    
    # Redis Configuration
    redis_password: Optional[str] = Field(default=None, description="Redis Password")
    
    # Flask Configuration
    flask_env: str = Field(default="development", description="Flask Environment")
    flask_debug: bool = Field(default=True, description="Flask Debug Mode")
    secret_key: str = Field(default="dev-secret-key", description="Flask Secret Key")
    jwt_secret_key: str = Field(default="dev-jwt-secret", description="JWT Secret Key")
    
    # Storage Paths
    model_storage_path: str = Field(default="./models", description="Model Storage Path")
    dataset_storage_path: str = Field(default="./datasets", description="Dataset Storage Path")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Log Level")
    log_file: str = Field(default="logs/app.log", description="Log File Path")
    
    # Training Configuration
    default_batch_size: int = Field(default=4, description="Default Batch Size")
    default_learning_rate: float = Field(default=5e-5, description="Default Learning Rate")
    default_epochs: int = Field(default=3, description="Default Epochs")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API Host")
    api_port: int = Field(default=5000, description="API Port")
    
    # CORS Configuration
    cors_origins: Optional[str] = Field(default=None, description="CORS Origins")
    
    # Monitoring Configuration
    grafana_password: Optional[str] = Field(default=None, description="Grafana Password")
    
    # Streamlit Configuration
    streamlit_port: int = Field(default=8501, description="Streamlit Port")
    
    if ConfigDict:
        model_config = ConfigDict(
            env_file=".env",
            case_sensitive=False,
            env_prefix="",
            extra="ignore",  # Ignore extra fields
            protected_namespaces=()  # Allow model_ prefix
        )
    else:
        class Config:
            env_file = ".env"
            case_sensitive = False
            extra = "ignore"


class TrainingConfig:
    """Configuration for model training parameters."""
    
    def __init__(
        self,
        base_model: str = "gpt2",
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        warmup_steps: int = 100,
        use_lora: bool = True,
        lora_rank: int = 16,
        max_length: int = 512,
        gradient_accumulation_steps: int = 1,
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 100,
    ):
        self.base_model = base_model
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.max_length = max_length
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "base_model": self.base_model,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "use_lora": self.use_lora,
            "lora_rank": self.lora_rank,
            "max_length": self.max_length,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "logging_steps": self.logging_steps,
        }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance."""
    return settings


def create_directories():
    """Create necessary directories for the application."""
    directories = [
        settings.model_storage_path,
        settings.dataset_storage_path,
        os.path.dirname(settings.log_file),
    ]
    
    for directory in directories:
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)


# Create directories on import
create_directories()