import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import ComparisonView from '../ComparisonView';
import { EvaluationResult } from '../../types';

describe('ComparisonView', () => {
  const mockResults: EvaluationResult[] = [
    {
      id: 'result-1',
      experiment_id: 'test-exp',
      prompt: 'Test prompt',
      model_id: 'gpt-4',
      response: 'Test response from GPT-4',
      metrics: {
        bleu: 0.85,
        rouge: 0.92,
        semantic_similarity: 0.88,
        llm_judge_score: 0.90,
      },
      cost_usd: 0.0045,
      latency_ms: 1200,
      created_at: '2024-01-01T00:00:00Z',
    },
    {
      id: 'result-2',
      experiment_id: 'test-exp',
      prompt: 'Test prompt',
      model_id: 'claude-3',
      response: 'Test response from Claude',
      metrics: {
        bleu: 0.78,
        rouge: 0.89,
        semantic_similarity: 0.91,
        llm_judge_score: 0.87,
      },
      cost_usd: 0.0032,
      latency_ms: 950,
      created_at: '2024-01-01T00:00:00Z',
    },
  ];

  const mockResultsMultiplePrompts: EvaluationResult[] = [
    {
      id: 'result-1',
      experiment_id: 'test-exp',
      prompt: 'First test prompt',
      model_id: 'gpt-4',
      response: 'Response to first prompt',
      metrics: { bleu: 0.85 },
      cost_usd: 0.0045,
      latency_ms: 1200,
      created_at: '2024-01-01T00:00:00Z',
    },
    {
      id: 'result-2',
      experiment_id: 'test-exp',
      prompt: 'Second test prompt',
      model_id: 'gpt-4',
      response: 'Response to second prompt',
      metrics: { bleu: 0.78 },
      cost_usd: 0.0032,
      latency_ms: 950,
      created_at: '2024-01-01T00:00:00Z',
    },
  ];

  test('renders comparison view component', () => {
    render(<ComparisonView results={mockResults} />);
    
    expect(screen.getByText(/prompt 0/i)).toBeInTheDocument();
  });

  test('displays prompt text', () => {
    render(<ComparisonView results={mockResults} />);
    
    expect(screen.getByText('Test prompt')).toBeInTheDocument();
  });

  test('displays multiple prompts separately', () => {
    render(<ComparisonView results={mockResultsMultiplePrompts} />);
    
    expect(screen.getByText('First test prompt')).toBeInTheDocument();
    expect(screen.getByText('Second test prompt')).toBeInTheDocument();
  });

  test('displays model responses', () => {
    render(<ComparisonView results={mockResults} />);
    
    expect(screen.getByText('Test response from GPT-4')).toBeInTheDocument();
    expect(screen.getByText('Test response from Claude')).toBeInTheDocument();
  });

  test('displays evaluation metrics', () => {
    render(<ComparisonView results={mockResults} />);
    
    expect(screen.getAllByText('BLEU')).toHaveLength(2); // One for each model result
    expect(screen.getAllByText('ROUGE')).toHaveLength(2);
    expect(screen.getAllByText('Similarity')).toHaveLength(2);
    expect(screen.getAllByText('LLM Judge')).toHaveLength(2);
  });

  test('displays performance metrics', () => {
    render(<ComparisonView results={mockResults} />);
    
    // Should show latency and cost
    expect(screen.getByText('1.2s')).toBeInTheDocument();
    expect(screen.getByText('950ms')).toBeInTheDocument();
  });

  test('shows loading state', () => {
    render(<ComparisonView results={[]} loading={true} />);
    
    expect(screen.getByText(/generating responses and evaluating/i)).toBeInTheDocument();
  });

  test('shows empty state when no results', () => {
    render(<ComparisonView results={[]} />);
    
    expect(screen.getByText(/no results yet/i)).toBeInTheDocument();
    expect(screen.getByText(/run your prompts to see comparison results/i)).toBeInTheDocument();
  });

  test('displays summary statistics', () => {
    render(<ComparisonView results={mockResults} />);
    
    expect(screen.getByText(/summary/i)).toBeInTheDocument();
    expect(screen.getByText(/total cost/i)).toBeInTheDocument();
    expect(screen.getByText(/avg latency/i)).toBeInTheDocument();
    expect(screen.getByText(/models tested/i)).toBeInTheDocument();
  });

  test('allows human rating of results', () => {
    const mockOnResultUpdate = jest.fn();
    
    render(<ComparisonView results={mockResults} onResultUpdate={mockOnResultUpdate} />);
    
    // Click on 4-star rating for first result
    const starButtons = screen.getAllByTitle(/rate 4 stars/i);
    fireEvent.click(starButtons[0]);
    
    expect(mockOnResultUpdate).toHaveBeenCalledWith({
      ...mockResults[0],
      human_rating: 4,
    });
  });

  test('shows real-time progress during loading with partial results', () => {
    const partialResults = [mockResults[0]]; // Only one result completed
    
    render(<ComparisonView results={partialResults} loading={true} />);
    
    expect(screen.getByText(/evaluation in progress/i)).toBeInTheDocument();
    expect(screen.getByText(/1 results completed/i)).toBeInTheDocument();
    expect(screen.getByText(/completed results/i)).toBeInTheDocument();
  });

  test('displays human ratings when present', () => {
    const resultsWithRating = [
      {
        ...mockResults[0],
        human_rating: 5,
      },
    ];
    
    render(<ComparisonView results={resultsWithRating} />);
    
    expect(screen.getByText('5/5')).toBeInTheDocument();
  });
});