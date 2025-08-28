import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { AppProvider } from '../../context/AppContext';
import PromptTesting from '../PromptTesting';

// Mock the child components
jest.mock('../../components/PromptInput', () => {
  return function MockPromptInput({ prompts, onPromptsChange }: any) {
    return (
      <div data-testid="prompt-input">
        <button onClick={() => onPromptsChange(['Test prompt'])}>
          Add Test Prompt
        </button>
        <div>Prompts: {prompts.length}</div>
      </div>
    );
  };
});

jest.mock('../../components/ModelSelector', () => {
  return function MockModelSelector({ selectedModels, onSelectionChange }: any) {
    return (
      <div data-testid="model-selector">
        <button onClick={() => onSelectionChange(['gpt-4'])}>
          Select GPT-4
        </button>
        <div>Selected: {selectedModels.length}</div>
      </div>
    );
  };
});

jest.mock('../../components/ComparisonView', () => {
  return function MockComparisonView({ results, loading }: any) {
    return (
      <div data-testid="comparison-view">
        {loading ? 'Loading...' : `Results: ${results.length}`}
      </div>
    );
  };
});

const renderWithProvider = (component: React.ReactElement) => {
  return render(
    <AppProvider>
      {component}
    </AppProvider>
  );
};

describe('PromptTesting', () => {
  test('renders prompt testing page', () => {
    renderWithProvider(<PromptTesting />);
    
    expect(screen.getByText('Prompt Testing')).toBeInTheDocument();
    expect(screen.getByText('Test and compare prompts across different models')).toBeInTheDocument();
    expect(screen.getByTestId('prompt-input')).toBeInTheDocument();
    expect(screen.getByTestId('model-selector')).toBeInTheDocument();
    expect(screen.getByTestId('comparison-view')).toBeInTheDocument();
  });

  test('shows run evaluation button when prompts and models are selected', async () => {
    renderWithProvider(<PromptTesting />);
    
    // Add a prompt
    fireEvent.click(screen.getByText('Add Test Prompt'));
    
    // Select a model
    fireEvent.click(screen.getByText('Select GPT-4'));
    
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /run evaluation/i })).toBeInTheDocument();
    });
  });

  test('disables run evaluation button when no prompts', () => {
    renderWithProvider(<PromptTesting />);
    
    const runButton = screen.getByRole('button', { name: /run evaluation/i });
    expect(runButton).toBeDisabled();
  });

  test('shows stop button during evaluation', async () => {
    renderWithProvider(<PromptTesting />);
    
    // Add prompt and select model
    fireEvent.click(screen.getByText('Add Test Prompt'));
    fireEvent.click(screen.getByText('Select GPT-4'));
    
    await waitFor(() => {
      const runButton = screen.getByRole('button', { name: /run evaluation/i });
      fireEvent.click(runButton);
    });

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /stop evaluation/i })).toBeInTheDocument();
    });
  });

  test('shows clear results button when results exist', async () => {
    renderWithProvider(<PromptTesting />);
    
    // Add prompt and select model
    fireEvent.click(screen.getByText('Add Test Prompt'));
    fireEvent.click(screen.getByText('Select GPT-4'));
    
    // Run evaluation
    await waitFor(() => {
      const runButton = screen.getByRole('button', { name: /run evaluation/i });
      fireEvent.click(runButton);
    });

    // Wait for evaluation to complete
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /clear results/i })).toBeInTheDocument();
    }, { timeout: 5000 });
  });

  test('clears results when clear button is clicked', async () => {
    renderWithProvider(<PromptTesting />);
    
    // Add prompt and select model
    fireEvent.click(screen.getByText('Add Test Prompt'));
    fireEvent.click(screen.getByText('Select GPT-4'));
    
    // Run evaluation
    await waitFor(() => {
      const runButton = screen.getByRole('button', { name: /run evaluation/i });
      fireEvent.click(runButton);
    });

    // Wait for results and clear them
    await waitFor(() => {
      const clearButton = screen.getByRole('button', { name: /clear results/i });
      fireEvent.click(clearButton);
    }, { timeout: 5000 });

    // Check that results are cleared
    expect(screen.getByTestId('comparison-view')).toHaveTextContent('Results: 0');
  });
});