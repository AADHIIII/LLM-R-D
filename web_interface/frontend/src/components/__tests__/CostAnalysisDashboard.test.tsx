import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import CostAnalysisDashboard from '../CostAnalysisDashboard';

const mockCostData = [
  { date: '2024-01-01', openai_cost: 12.50, anthropic_cost: 8.30, fine_tuned_cost: 2.10, total_cost: 22.90, token_count: 45000, request_count: 120 },
  { date: '2024-01-02', openai_cost: 15.20, anthropic_cost: 9.80, fine_tuned_cost: 2.80, total_cost: 27.80, token_count: 52000, request_count: 145 },
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
];

// Mock recharts components
jest.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: any) => <div data-testid="responsive-container">{children}</div>,
  AreaChart: ({ children }: any) => <div data-testid="area-chart">{children}</div>,
  BarChart: ({ children }: any) => <div data-testid="bar-chart">{children}</div>,
  PieChart: ({ children }: any) => <div data-testid="pie-chart">{children}</div>,
  Area: () => <div data-testid="area" />,
  Bar: () => <div data-testid="bar" />,
  Pie: () => <div data-testid="pie" />,
  Cell: () => <div data-testid="cell" />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Tooltip: () => <div data-testid="tooltip" />,
  Legend: () => <div data-testid="legend" />,
}));

describe('CostAnalysisDashboard', () => {
  const defaultProps = {
    costData: mockCostData,
    modelBreakdown: mockModelBreakdown,
    totalBudget: 1000,
    currentSpend: 156.70,
  };

  it('renders budget overview correctly', () => {
    render(<CostAnalysisDashboard {...defaultProps} />);

    expect(screen.getByText('Total Budget')).toBeInTheDocument();
    expect(screen.getByText('$1000.00')).toBeInTheDocument();
    expect(screen.getByText('Current Spend')).toBeInTheDocument();
    expect(screen.getByText('$156.70')).toBeInTheDocument();
    expect(screen.getByText('Remaining')).toBeInTheDocument();
    expect(screen.getByText('Utilization')).toBeInTheDocument();
  });

  it('shows budget warning when near limit', () => {
    render(
      <CostAnalysisDashboard
        {...defaultProps}
        currentSpend={850} // 85% of budget
      />
    );

    expect(screen.getByText('Budget Warning')).toBeInTheDocument();
    expect(screen.getByText(/You have used 85.0% of your budget/)).toBeInTheDocument();
  });

  it('shows budget exceeded alert when over limit', () => {
    render(
      <CostAnalysisDashboard
        {...defaultProps}
        currentSpend={1200} // Over budget
      />
    );

    expect(screen.getByText('Budget Exceeded')).toBeInTheDocument();
    expect(screen.getByText(/You have exceeded your budget by \$200.00/)).toBeInTheDocument();
  });

  it('renders cost trends chart', () => {
    render(<CostAnalysisDashboard {...defaultProps} />);

    expect(screen.getByText('Cost Trends')).toBeInTheDocument();
    expect(screen.getByTestId('area-chart')).toBeInTheDocument();
  });

  it('allows time range selection', () => {
    render(<CostAnalysisDashboard {...defaultProps} />);

    const sevenDayButton = screen.getByText('7d');
    const thirtyDayButton = screen.getByText('30d');
    const ninetyDayButton = screen.getByText('90d');

    expect(sevenDayButton).toBeInTheDocument();
    expect(thirtyDayButton).toBeInTheDocument();
    expect(ninetyDayButton).toBeInTheDocument();

    // Test clicking different time ranges
    fireEvent.click(sevenDayButton);
    expect(sevenDayButton).toHaveClass('bg-blue-100');

    fireEvent.click(ninetyDayButton);
    expect(ninetyDayButton).toHaveClass('bg-blue-100');
  });

  it('renders cost breakdown charts', () => {
    render(<CostAnalysisDashboard {...defaultProps} />);

    expect(screen.getByText('Cost by Provider')).toBeInTheDocument();
    expect(screen.getByText('Model Cost Efficiency')).toBeInTheDocument();
    expect(screen.getByTestId('pie-chart')).toBeInTheDocument();
    expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
  });

  it('renders detailed model breakdown table', () => {
    render(<CostAnalysisDashboard {...defaultProps} />);

    expect(screen.getByText('Detailed Model Breakdown')).toBeInTheDocument();
    expect(screen.getByText('GPT-4')).toBeInTheDocument();
    expect(screen.getByText('Claude-3')).toBeInTheDocument();
    expect(screen.getByText('$89.500')).toBeInTheDocument();
    expect(screen.getByText('$67.200')).toBeInTheDocument();
  });

  it('calculates budget utilization correctly', () => {
    render(
      <CostAnalysisDashboard
        {...defaultProps}
        currentSpend={250}
        totalBudget={1000}
      />
    );

    expect(screen.getByText('25.0%')).toBeInTheDocument();
  });

  it('handles zero budget gracefully', () => {
    render(
      <CostAnalysisDashboard
        {...defaultProps}
        totalBudget={0}
        currentSpend={100}
      />
    );

    // Should not crash and should handle division by zero
    expect(screen.getAllByText('$0.00')).toHaveLength(2); // Total budget and remaining budget both show $0.00
  });

  it('displays correct remaining budget', () => {
    render(
      <CostAnalysisDashboard
        {...defaultProps}
        totalBudget={1000}
        currentSpend={300}
      />
    );

    expect(screen.getByText('$700.00')).toBeInTheDocument();
  });
});