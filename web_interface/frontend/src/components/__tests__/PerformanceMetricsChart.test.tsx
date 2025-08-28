import React from 'react';
import { render, screen } from '@testing-library/react';
import PerformanceMetricsChart from '../PerformanceMetricsChart';

const mockTrainingMetrics = [
  { epoch: 1, training_loss: 2.5, validation_loss: 2.3, learning_rate: 5e-5, timestamp: '2024-01-01T10:00:00Z' },
  { epoch: 2, training_loss: 2.1, validation_loss: 2.0, learning_rate: 4.5e-5, timestamp: '2024-01-01T11:00:00Z' },
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
];

// Mock recharts components
jest.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: any) => <div data-testid="responsive-container">{children}</div>,
  LineChart: ({ children }: any) => <div data-testid="line-chart">{children}</div>,
  BarChart: ({ children }: any) => <div data-testid="bar-chart">{children}</div>,
  PieChart: ({ children }: any) => <div data-testid="pie-chart">{children}</div>,
  Line: () => <div data-testid="line" />,
  Bar: () => <div data-testid="bar" />,
  Pie: () => <div data-testid="pie" />,
  Cell: () => <div data-testid="cell" />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Tooltip: () => <div data-testid="tooltip" />,
  Legend: () => <div data-testid="legend" />,
}));

describe('PerformanceMetricsChart', () => {
  it('renders training chart correctly', () => {
    render(
      <PerformanceMetricsChart
        trainingMetrics={mockTrainingMetrics}
        type="training"
      />
    );

    expect(screen.getByText('Training Loss Over Time')).toBeInTheDocument();
    expect(screen.getByText('Learning Rate Schedule')).toBeInTheDocument();
    expect(screen.getAllByTestId('line-chart')).toHaveLength(2);
  });

  it('renders evaluation chart correctly', () => {
    render(
      <PerformanceMetricsChart
        evaluationMetrics={mockEvaluationMetrics}
        type="evaluation"
      />
    );

    expect(screen.getByText('Model Performance Comparison')).toBeInTheDocument();
    expect(screen.getByText('Cost Distribution')).toBeInTheDocument();
    expect(screen.getByText('Response Latency')).toBeInTheDocument();
    expect(screen.getAllByTestId('bar-chart')).toHaveLength(2); // One for performance comparison, one for latency
    expect(screen.getByTestId('pie-chart')).toBeInTheDocument();
  });

  it('renders comparison chart correctly', () => {
    render(
      <PerformanceMetricsChart
        evaluationMetrics={mockEvaluationMetrics}
        type="comparison"
      />
    );

    expect(screen.getByText('Performance vs Cost Analysis')).toBeInTheDocument();
    expect(screen.getByTestId('line-chart')).toBeInTheDocument();
  });

  it('handles empty data gracefully', () => {
    render(
      <PerformanceMetricsChart
        trainingMetrics={[]}
        evaluationMetrics={[]}
        type="training"
      />
    );

    // Should still render the chart structure
    expect(screen.getByText('Training Loss Over Time')).toBeInTheDocument();
  });

  it('displays correct chart titles and labels', () => {
    render(
      <PerformanceMetricsChart
        evaluationMetrics={mockEvaluationMetrics}
        type="evaluation"
      />
    );

    expect(screen.getByText('Model Performance Comparison')).toBeInTheDocument();
    expect(screen.getByText('Cost Distribution')).toBeInTheDocument();
    expect(screen.getByText('Response Latency')).toBeInTheDocument();
  });
});