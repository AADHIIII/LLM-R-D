// Common types used throughout the application

export interface User {
  id: string;
  email: string;
  name: string;
  role: 'admin' | 'user';
  created_at: string;
}

export interface Dataset {
  id: string;
  name: string;
  format: 'jsonl' | 'csv';
  size: number;
  domain: string;
  created_at: string;
  validation_split: number;
}

export interface Model {
  id: string;
  name: string;
  type: 'fine-tuned' | 'commercial';
  base_model?: string;
  model_path?: string;
  training_config?: TrainingConfig;
  created_at: string;
}

export interface TrainingConfig {
  base_model: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  warmup_steps: number;
  use_lora: boolean;
  lora_rank: number;
}

export interface Experiment {
  id: string;
  name: string;
  description: string;
  dataset_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  training_job?: TrainingJob;
  results?: EvaluationResult[];
  created_at: string;
  updated_at: string;
}

export interface TrainingJob {
  id: string;
  experiment_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  current_epoch: number;
  loss: number;
  created_at: string;
  updated_at: string;
  config: TrainingConfig;
}

export interface EvaluationResult {
  id: string;
  experiment_id: string;
  prompt: string;
  model_id: string;
  response: string;
  metrics: {
    bleu?: number;
    rouge?: number;
    perplexity?: number;
    semantic_similarity?: number;
    llm_judge_score?: number;
  };
  human_rating?: number;
  human_feedback?: HumanFeedback;
  cost_usd: number;
  latency_ms: number;
  created_at: string;
}

export interface HumanFeedback {
  thumbs_rating?: 'up' | 'down';
  star_rating?: number;
  qualitative_feedback?: string;
  feedback_timestamp?: string;
}

export interface ComparisonReport {
  model_a: string;
  model_b: string;
  win_rate_a: number;
  statistical_significance: number;
  metric_differences: Record<string, number>;
  cost_comparison: Record<string, number>;
}

export interface ApiError {
  error_code: string;
  message: string;
  details?: Record<string, any>;
  suggested_actions?: string[];
  retry_after?: number;
  documentation_link?: string;
}

// Form types
export interface ExperimentFormData {
  name: string;
  description: string;
  dataset_id: string;
  training_config: TrainingConfig;
}

export interface PromptTestFormData {
  prompts: string[];
  models: string[];
  evaluation_criteria: string[];
}

// UI State types
export interface LoadingState {
  isLoading: boolean;
  message?: string;
}

export interface NotificationState {
  type: 'success' | 'error' | 'warning' | 'info';
  message: string;
  id: string;
}