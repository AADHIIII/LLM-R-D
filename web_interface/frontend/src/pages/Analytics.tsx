import React, { useState, useEffect } from 'react';
import PerformanceMetricsChart from '../components/PerformanceMetricsChart';
import CostAnalysisDashboard from '../components/CostAnalysisDashboard';
import PerformanceComparison from '../components/PerformanceComparison';

// Mock data - in real implementation, this would come from API
const mockTrainingMetrics = [
  { epoch: 1, training_loss: 2.5, validation_loss: 2.3, learning_rate: 5e-5, timestamp: '2024-01-01T10:00:00Z' },
  { epoch: 2, training_loss: 2.1, validation_loss: 2.0, learning_rate: 4.5e-5, timestamp: '2024-01-01T11:00:00Z' },
  { epoch: 3, training_loss: 1.8, validation_loss: 1.7, learning_rate: 4e-5, timestamp: '2024-01-01T12:00:00Z' },
  { epoch: 4, training_loss: 1.5, validation_loss: 1.4, learning_rate: 3.5e-5, timestamp: '2024-01-01T13:00:00Z' },
  { epoch: 5, training_loss: 1.3, validation_loss: 1.2, learning_rate: 3e-5, timestamp: '2024-01-01T14:00:00Z' },
];

const mockEvaluationMetrics = [
  {
    model_name: 'GPT-4',
    bleu_score: 0.85,
    rouge_score: 0.78,
    perplexity: 15.2,
    semantic_similarity: 0.92,
    llm_judge_score: 4.2,
    cost_usd: 0.045,
    latency_ms: 1200,
  },
  {
    model_name: 'Claude-3',
    bleu_score: 0.82,
    rouge_score: 0.75,
    perplexity: 18.5,
    semantic_similarity: 0.89,
    llm_judge_score: 4.0,
    cost_usd: 0.038,
    latency_ms: 950,
  },
  {
    model_name: 'Fine-tuned GPT-2',
    bleu_score: 0.76,
    rouge_score: 0.71,
    perplexity: 22.1,
    semantic_similarity: 0.84,
    llm_judge_score: 3.8,
    cost_usd: 0.012,
    latency_ms: 450,
  },
];

const mockCostData = [
  { date: '2024-01-01', openai_cost: 12.50, anthropic_cost: 8.30, fine_tuned_cost: 2.10, total_cost: 22.90, token_count: 45000, request_count: 120 },
  { date: '2024-01-02', openai_cost: 15.20, anthropic_cost: 9.80, fine_tuned_cost: 2.80, total_cost: 27.80, token_count: 52000, request_count: 145 },
  { date: '2024-01-03', openai_cost: 18.70, anthropic_cost: 11.20, fine_tuned_cost: 3.20, total_cost: 33.10, token_count: 58000, request_count: 167 },
  { date: '2024-01-04', openai_cost: 14.30, anthropic_cost: 7.90, fine_tuned_cost: 2.50, total_cost: 24.70, token_count: 48000, request_count: 132 },
  { date: '2024-01-05', openai_cost: 16.80, anthropic_cost: 10.50, fine_tuned_cost: 3.10, total_cost: 30.40, token_count: 55000, request_count: 158 },
];

const mockModelBreakdown = [
  {
    model_name: 'GPT-4',
    total_cost: 89.50,
    request_count: 450,
    avg_cost_per_request: 0.199,
    token_count: 125000,
    cost_per_token: 0.000716,
  },
  {
    model_name: 'Claude-3',
    total_cost: 67.20,
    request_count: 380,
    avg_cost_per_request: 0.177,
    token_count: 98000,
    cost_per_token: 0.000686,
  },
  {
    model_name: 'Fine-tuned GPT-2',
    total_cost: 18.90,
    request_count: 720,
    avg_cost_per_request: 0.026,
    token_count: 156000,
    cost_per_token: 0.000121,
  },
];

const mockExperiments = [
  {
    id: '1',
    name: 'Medical Q&A Fine-tuning',
    model_name: 'Fine-tuned GPT-2',
    metrics: {
      bleu_score: 0.76,
      rouge_score: 0.71,
      perplexity: 22.1,
      semantic_similarity: 0.84,
      llm_judge_score: 3.8,
      human_rating: 4.1,
    },
    performance: {
      avg_latency_ms: 450,
      total_cost_usd: 18.90,
      tokens_per_second: 85,
      success_rate: 0.95,
    },
    created_at: '2024-01-01T10:00:00Z',
  },
  {
    id: '2',
    name: 'GPT-4 Baseline',
    model_name: 'GPT-4',
    metrics: {
      bleu_score: 0.85,
      rouge_score: 0.78,
      perplexity: 15.2,
      semantic_similarity: 0.92,
      llm_judge_score: 4.2,
      human_rating: 4.5,
    },
    performance: {
      avg_latency_ms: 1200,
      total_cost_usd: 89.50,
      tokens_per_second: 45,
      success_rate: 0.98,
    },
    created_at: '2024-01-02T10:00:00Z',
  },
];

const Analytics: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'metrics' | 'costs' | 'comparison'>('metrics');
  const [selectedExperiments, setSelectedExperiments] = useState<string[]>(['1', '2']);
  const [totalBudget] = useState(1000);
  const [currentSpend] = useState(175.60);

  const tabs = [
    { id: 'metrics', name: 'Performance Metrics', icon: '📊' },
    { id: 'costs', name: 'Cost Analysis', icon: '💰' },
    { id: 'comparison', name: 'Experiment Comparison', icon: '⚖️' },
  ];

  return (
    <div className="space-y-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Analytics Dashboard</h1>
        <p className="mt-2 text-gray-600">
          Performance metrics, cost tracking, and experiment comparison
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <span className="mr-2">{tab.icon}</span>
              {tab.name}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'metrics' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <h2 className="text-xl font-semibold mb-4">Training Progress</h2>
              <PerformanceMetricsChart
                trainingMetrics={mockTrainingMetrics}
                type="training"
              />
            </div>
            <div>
              <h2 className="text-xl font-semibold mb-4">Model Evaluation</h2>
              <PerformanceMetricsChart
                evaluationMetrics={mockEvaluationMetrics}
                type="evaluation"
              />
            </div>
          </div>
          
          <div>
            <h2 className="text-xl font-semibold mb-4">Performance vs Cost</h2>
            <PerformanceMetricsChart
              evaluationMetrics={mockEvaluationMetrics}
              type="comparison"
            />
          </div>
        </div>
      )}

      {activeTab === 'costs' && (
        <div>
          <h2 className="text-xl font-semibold mb-4">Cost Analysis</h2>
          <CostAnalysisDashboard
            costData={mockCostData}
            modelBreakdown={mockModelBreakdown}
            totalBudget={totalBudget}
            currentSpend={currentSpend}
          />
        </div>
      )}

      {activeTab === 'comparison' && (
        <div>
          <h2 className="text-xl font-semibold mb-4">Experiment Comparison</h2>
          <PerformanceComparison
            experiments={mockExperiments}
            selectedExperiments={selectedExperiments}
            onExperimentSelect={setSelectedExperiments}
          />
        </div>
      )}

      {/* Quick Stats Summary */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Quick Summary</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">
              {mockExperiments.length}
            </div>
            <div className="text-sm text-gray-500">Active Experiments</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">
              ${currentSpend.toFixed(2)}
            </div>
            <div className="text-sm text-gray-500">Total Spend</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">
              {mockEvaluationMetrics.reduce((sum, m) => sum + m.latency_ms, 0) / mockEvaluationMetrics.length}ms
            </div>
            <div className="text-sm text-gray-500">Avg Latency</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-600">
              {((currentSpend / totalBudget) * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-500">Budget Used</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analytics;