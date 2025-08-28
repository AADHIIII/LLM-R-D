import React from 'react';
import { render, screen } from '@testing-library/react';
import ModelSelector from '../ModelSelector';
import { Model } from '../../types';

describe('ModelSelector', () => {
  const mockOnSelectionChange = jest.fn();
  
  const mockModels: Model[] = [
    {
      id: 'gpt-4',
      name: 'GPT-4',
      type: 'commercial',
      created_at: '2024-01-01T00:00:00Z',
    },
    {
      id: 'fine-tuned-1',
      name: 'Custom Model',
      type: 'fine-tuned',
      base_model: 'gpt2',
      created_at: '2024-01-15T10:00:00Z',
    },
  ];

  beforeEach(() => {
    mockOnSelectionChange.mockClear();
  });

  test('renders model selector component', () => {
    render(
      <ModelSelector
        models={mockModels}
        selectedModels={[]}
        onSelectionChange={mockOnSelectionChange}
      />
    );
    
    expect(screen.getByText(/model selection/i)).toBeInTheDocument();
    expect(screen.getByText(/choose models to test/i)).toBeInTheDocument();
  });

  test('displays available models', () => {
    render(
      <ModelSelector
        models={mockModels}
        selectedModels={[]}
        onSelectionChange={mockOnSelectionChange}
      />
    );
    
    expect(screen.getByText('GPT-4')).toBeInTheDocument();
    expect(screen.getByText('Custom Model')).toBeInTheDocument();
    expect(screen.getByText('commercial')).toBeInTheDocument();
    expect(screen.getByText('fine-tuned')).toBeInTheDocument();
  });

  test('shows selection count', () => {
    render(
      <ModelSelector
        models={mockModels}
        selectedModels={['gpt-4']}
        onSelectionChange={mockOnSelectionChange}
      />
    );
    
    expect(screen.getByText(/1 of 2 models selected/i)).toBeInTheDocument();
  });

  test('shows select all and clear all buttons', () => {
    render(
      <ModelSelector
        models={mockModels}
        selectedModels={[]}
        onSelectionChange={mockOnSelectionChange}
      />
    );
    
    expect(screen.getByText(/select all/i)).toBeInTheDocument();
    expect(screen.getByText(/clear all/i)).toBeInTheDocument();
  });

  test('shows model type legend', () => {
    render(
      <ModelSelector
        models={mockModels}
        selectedModels={[]}
        onSelectionChange={mockOnSelectionChange}
      />
    );
    
    expect(screen.getByText(/model types/i)).toBeInTheDocument();
    expect(screen.getByText(/fine-tuned models/i)).toBeInTheDocument();
    expect(screen.getByText(/commercial apis/i)).toBeInTheDocument();
  });

  test('shows loading state', () => {
    render(
      <ModelSelector
        models={[]}
        selectedModels={[]}
        onSelectionChange={mockOnSelectionChange}
        loading={true}
      />
    );
    
    expect(screen.getByText(/loading available models/i)).toBeInTheDocument();
  });

  test('shows empty state when no models available', () => {
    render(
      <ModelSelector
        models={[]}
        selectedModels={[]}
        onSelectionChange={mockOnSelectionChange}
      />
    );
    
    expect(screen.getByText(/no models available/i)).toBeInTheDocument();
  });
});