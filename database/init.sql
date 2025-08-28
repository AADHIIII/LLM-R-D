-- Database initialization script for LLM Optimization Platform
-- This script creates the necessary tables and indexes

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create experiments table
CREATE TABLE IF NOT EXISTS experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    dataset_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status VARCHAR(50) DEFAULT 'created',
    config JSONB,
    CONSTRAINT experiments_name_unique UNIQUE (name)
);

-- Create datasets table
CREATE TABLE IF NOT EXISTS datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    file_path VARCHAR(500),
    format VARCHAR(20) NOT NULL,
    size_bytes BIGINT,
    row_count INTEGER,
    domain VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    validation_split DECIMAL(3,2) DEFAULT 0.2,
    metadata JSONB
);

-- Create models table
CREATE TABLE IF NOT EXISTS models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    type VARCHAR(50) NOT NULL, -- 'fine-tuned' or 'commercial'
    base_model VARCHAR(100),
    model_path VARCHAR(500),
    training_config JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status VARCHAR(50) DEFAULT 'created',
    size_bytes BIGINT,
    version VARCHAR(20) DEFAULT '1.0.0',
    metadata JSONB
);

-- Create evaluations table
CREATE TABLE IF NOT EXISTS evaluations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID REFERENCES experiments(id) ON DELETE CASCADE,
    prompt TEXT NOT NULL,
    model_id UUID REFERENCES models(id) ON DELETE CASCADE,
    response TEXT,
    metrics JSONB,
    human_rating INTEGER CHECK (human_rating >= 1 AND human_rating <= 5),
    human_feedback TEXT,
    cost_usd DECIMAL(10,4),
    latency_ms INTEGER,
    tokens_input INTEGER,
    tokens_output INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    evaluation_type VARCHAR(50) DEFAULT 'automated'
);

-- Create training_jobs table
CREATE TABLE IF NOT EXISTS training_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID REFERENCES experiments(id) ON DELETE CASCADE,
    dataset_id UUID REFERENCES datasets(id) ON DELETE CASCADE,
    model_id UUID REFERENCES models(id) ON DELETE SET NULL,
    status VARCHAR(50) DEFAULT 'pending',
    config JSONB,
    progress DECIMAL(5,2) DEFAULT 0.0,
    loss_history JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT
);

-- Create feedback table for human ratings
CREATE TABLE IF NOT EXISTS feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    evaluation_id UUID REFERENCES evaluations(id) ON DELETE CASCADE,
    user_id VARCHAR(100),
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    feedback_type VARCHAR(50) DEFAULT 'thumbs', -- 'thumbs', 'stars', 'text'
    feedback_value VARCHAR(20), -- 'up', 'down', or star rating
    comments TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create cost_tracking table
CREATE TABLE IF NOT EXISTS cost_tracking (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    api_provider VARCHAR(50) NOT NULL,
    tokens_input INTEGER NOT NULL DEFAULT 0,
    tokens_output INTEGER NOT NULL DEFAULT 0,
    cost_usd DECIMAL(10,6) NOT NULL,
    request_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    experiment_id UUID REFERENCES experiments(id) ON DELETE SET NULL,
    evaluation_id UUID REFERENCES evaluations(id) ON DELETE SET NULL
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_datasets_created_at ON datasets(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_models_type ON models(type);
CREATE INDEX IF NOT EXISTS idx_models_created_at ON models(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_evaluations_experiment_id ON evaluations(experiment_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_model_id ON evaluations(model_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_created_at ON evaluations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status);
CREATE INDEX IF NOT EXISTS idx_training_jobs_created_at ON training_jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_evaluation_id ON feedback(evaluation_id);
CREATE INDEX IF NOT EXISTS idx_cost_tracking_timestamp ON cost_tracking(request_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_cost_tracking_model ON cost_tracking(model_name);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_experiments_updated_at BEFORE UPDATE ON experiments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_models_updated_at BEFORE UPDATE ON models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default commercial models
INSERT INTO models (name, type, base_model, status, metadata) VALUES
    ('gpt-4', 'commercial', 'gpt-4', 'available', '{"provider": "openai", "context_length": 8192, "cost_per_1k_tokens": {"input": 0.03, "output": 0.06}}'),
    ('gpt-3.5-turbo', 'commercial', 'gpt-3.5-turbo', 'available', '{"provider": "openai", "context_length": 4096, "cost_per_1k_tokens": {"input": 0.0015, "output": 0.002}}'),
    ('claude-3-opus', 'commercial', 'claude-3-opus', 'available', '{"provider": "anthropic", "context_length": 200000, "cost_per_1k_tokens": {"input": 0.015, "output": 0.075}}'),
    ('claude-3-sonnet', 'commercial', 'claude-3-sonnet', 'available', '{"provider": "anthropic", "context_length": 200000, "cost_per_1k_tokens": {"input": 0.003, "output": 0.015}}')
ON CONFLICT (name) DO NOTHING;

-- Create a view for evaluation summaries
CREATE OR REPLACE VIEW evaluation_summary AS
SELECT 
    e.id,
    e.experiment_id,
    exp.name as experiment_name,
    e.model_id,
    m.name as model_name,
    m.type as model_type,
    COUNT(*) as total_evaluations,
    AVG(e.human_rating) as avg_human_rating,
    AVG(e.cost_usd) as avg_cost_usd,
    AVG(e.latency_ms) as avg_latency_ms,
    SUM(e.cost_usd) as total_cost_usd,
    MIN(e.created_at) as first_evaluation,
    MAX(e.created_at) as last_evaluation
FROM evaluations e
JOIN experiments exp ON e.experiment_id = exp.id
JOIN models m ON e.model_id = m.id
GROUP BY e.experiment_id, exp.name, e.model_id, m.name, m.type, e.id;

-- Grant permissions (adjust as needed for your security requirements)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO llm_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO llm_user;