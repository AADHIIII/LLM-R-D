import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import PerformanceComparison from '../PerformanceComparison';

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

// Mock recharts components
jest.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: any) => <div data-testid="responsive-container">{children}</div>,
  RadarChart: ({ children }: any) => <div data-testid="radar-chart">{children}</div>,
  ScatterChart: ({ children }: any) => <div data-testid="scatter-chart">{children}</div>,
  BarChart: ({ children }: any) => <div data-testid="bar-chart">{children}</div>,
  Radar: () => <div data-testid="radar" />,
  Scatter: () => <div data-testid="scatter" />,
  Bar: () => <div data-testid="bar" />,
  Cell: () => <div data-testid="cell" />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  PolarGrid: () => <div data-testid="polar-grid" />,
  PolarAngleAxis: () => <div data-testid="polar-angle-axis" />,
  PolarRadiusAxis: () => <div data-testid="polar-radius-axis" />,
  Tooltip: () => <div data-testid="tooltip" />,
  Legend: () => <div data-testid="legend" />,
}));

describe('PerformanceComparison', () => {
  const mockOnExperimentSelect = jest.fn();

  const defaultProps = {
    experiments: mockExperiments,
    selectedExperiments: ['1', '2'],
    onExperimentSelect: mockOnExperimentSelect,
  };

  beforeEach(() => {
    mockOnExperimentSelect.mockClear();
  });

  it('renders experiment selector', () => {
    render(<PerformanceComparison {...defaultProps} />);

    expect(screen.getByText('Select Experiments to Compare')).toBeInTheDocument();
    expect(screen.getByText('Medical Q&A Fine-tuning (Fine-tuned GPT-2)')).toBeInTheDocument();
    expect(screen.getByText('GPT-4 Baseline (GPT-4)')).toBeInTheDocument();
  });

  it('handles experiment selection', () => {
    render(<PerformanceComparison {...defaultProps} selectedExperiments={['1']} />);

    const checkbox = screen.getByRole('checkbox', { name: /GPT-4 Baseline/ });
    fireEvent.click(checkbox);

    expect(mockOnExperimentSelect).toHaveBeenCalledWith(['1', '2']);
  });

  it('handles experiment deselection', () => {
    render(<PerformanceComparison {...defaultProps} />);

    const checkbox = screen.getByRole('checkbox', { name: /Medical Q&A Fine-tuning/ });
    fireEvent.click(checkbox);

    expect(mockOnExperimentSelect).toHaveBeenCalledWith(['2']);
  });

  it('renders view selector buttons', () => {
    render(<PerformanceComparison {...defaultProps} />);

    expect(screen.getByText('Radar')).toBeInTheDocument();
    expect(screen.getByText('Scatter')).toBeInTheDocument();
    expect(screen.getByText('Bar')).toBeInTheDocument();
  });

  it('switches between different chart views', () => {
    render(<PerformanceComparison {...defaultProps} />);

    // Default should be radar
    expect(screen.getByTestId('radar-chart')).toBeInTheDocument();

    // Switch to scatter
    fireEvent.click(screen.getByText('Scatter'));
    expect(screen.getByTestId('scatter-chart')).toBeInTheDocument();

    // Switch to bar
    fireEvent.click(screen.getByText('Bar'));
    expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
  });

  it('shows focus selector for bar chart', () => {
    render(<PerformanceComparison {...defaultProps} />);

    fireEvent.click(screen.getByText('Bar'));

    expect(screen.getByText('Quality')).toBeInTheDocument();
    expect(screen.getByText('Performance')).toBeInTheDocument();
    expect(screen.getByText('Cost')).toBeInTheDocument();
  });

  it('renders statistical summary table', () => {
    render(<PerformanceComparison {...defaultProps} />);

    expect(screen.getByText('Statistical Summary')).toBeInTheDocument();
    expect(screen.getByText('Medical Q&A Fine-tuning')).toBeInTheDocument();
    expect(screen.getByText('GPT-4 Baseline')).toBeInTheDocument();
    expect(screen.getByText('Fine-tuned GPT-2')).toBeInTheDocument();
    expect(screen.getByText('GPT-4')).toBeInTheDocument();
  });

  it('calculates average quality score correctly', () => {
    render(<PerformanceComparison {...defaultProps} />);

    // For Medical Q&A: (0.76 + 0.71 + 0.84 + 3.8 + 4.1) / 5 = 2.042
    expect(screen.getByText('2.042')).toBeInTheDocument();
    
    // For GPT-4: (0.85 + 0.78 + 0.92 + 4.2 + 4.5) / 5 = 2.250
    expect(screen.getByText('2.250')).toBeInTheDocument();
  });

  it('displays cost efficiency correctly', () => {
    render(<PerformanceComparison {...defaultProps} />);

    // Cost efficiency = avg_quality / total_cost
    // Medical Q&A: 2.042 / 18.90 = 0.11
    expect(screen.getByText('0.11')).toBeInTheDocument();
    
    // GPT-4: 2.250 / 89.50 = 0.03
    expect(screen.getByText('0.03')).toBeInTheDocument();
  });

  it('shows empty state when no experiments selected', () => {
    render(
      <PerformanceComparison
        {...defaultProps}
        selectedExperiments={[]}
      />
    );

    expect(screen.getByText('Select experiments to compare their performance')).toBeInTheDocument();
  });

  it('renders radar chart with correct titles', () => {
    render(<PerformanceComparison {...defaultProps} />);

    expect(screen.getByText('Quality Metrics Comparison')).toBeInTheDocument();
    expect(screen.getByTestId('radar-chart')).toBeInTheDocument();
  });

  it('renders scatter chart with correct titles', () => {
    render(<PerformanceComparison {...defaultProps} />);

    fireEvent.click(screen.getByText('Scatter'));
    expect(screen.getByText('Cost vs Quality Analysis')).toBeInTheDocument();
    expect(screen.getByTestId('scatter-chart')).toBeInTheDocument();
  });

  it('changes bar chart focus correctly', () => {
    render(<PerformanceComparison {...defaultProps} />);

    fireEvent.click(screen.getByText('Bar'));
    
    // Default should be quality
    expect(screen.getByText('Quality Metrics Comparison')).toBeInTheDocument();

    // Switch to performance
    fireEvent.click(screen.getByText('Performance'));
    expect(screen.getByText('Performance Metrics Comparison')).toBeInTheDocument();

    // Switch to cost
    fireEvent.click(screen.getByText('Cost'));
    expect(screen.getByText('Cost Metrics Comparison')).toBeInTheDocument();
  });
});