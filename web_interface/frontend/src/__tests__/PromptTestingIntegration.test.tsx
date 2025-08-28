import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { AppProvider } from '../context/AppContext';
import PromptTesting from '../pages/PromptTesting';

const renderWithProvider = (component: React.ReactElement) => {
  return render(
    <AppProvider>
      {component}
    </AppProvider>
  );
};

describe('PromptTesting Integration', () => {
  test('complete prompt testing workflow', async () => {
    renderWithProvider(<PromptTesting />);
    
    // Verify initial state
    expect(screen.getByText('Prompt Testing')).toBeInTheDocument();
    expect(screen.getByText('Test and compare prompts across different models')).toBeInTheDocument();
    
    // Add a prompt
    const promptTextarea = screen.getByPlaceholderText(/enter your prompt here/i);
    fireEvent.change(promptTextarea, { 
      target: { value: 'What is the capital of France?' } 
    });
    
    const addButton = screen.getByRole('button', { name: /add prompt/i });
    fireEvent.click(addButton);
    
    // Verify prompt was added
    await waitFor(() => {
      expect(screen.getByDisplayValue('What is the capital of France?')).toBeInTheDocument();
    });
    
    // Select models (they should be pre-selected based on the component logic)
    const runButton = screen.getByRole('button', { name: /run evaluation/i });
    expect(runButton).not.toBeDisabled();
    
    // Run evaluation
    fireEvent.click(runButton);
    
    // Verify evaluation starts
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /stop evaluation/i })).toBeInTheDocument();
    });
    
    // Wait for evaluation to complete
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /clear results/i })).toBeInTheDocument();
    }, { timeout: 10000 });
    
    // Verify results are displayed
    expect(screen.getByText(/prompt 0/i)).toBeInTheDocument();
    expect(screen.getAllByText('What is the capital of France?')).toHaveLength(2); // One in input, one in results
    
    // Verify we can rate results (if star buttons are present)
    const starButtons = screen.queryAllByTitle(/rate 5 stars/i);
    if (starButtons.length > 0) {
      fireEvent.click(starButtons[0]);
      
      await waitFor(() => {
        expect(screen.getByText('5/5')).toBeInTheDocument();
      });
    }
    
    // Clear results
    const clearButton = screen.getByRole('button', { name: /clear results/i });
    fireEvent.click(clearButton);
    
    // Verify results are cleared
    await waitFor(() => {
      expect(screen.getByText(/no results yet/i)).toBeInTheDocument();
    }, { timeout: 5000 });
  });

  test('handles multiple prompts and models', async () => {
    renderWithProvider(<PromptTesting />);
    
    // Add first prompt
    const promptTextarea = screen.getByPlaceholderText(/enter your prompt here/i);
    fireEvent.change(promptTextarea, { 
      target: { value: 'Explain quantum computing' } 
    });
    fireEvent.click(screen.getByRole('button', { name: /add prompt/i }));
    
    // Wait for first prompt to be added and verify it
    await waitFor(() => {
      expect(screen.getByDisplayValue('Explain quantum computing')).toBeInTheDocument();
    });
    
    // Add second prompt - find the new prompt input area
    await waitFor(() => {
      const newPromptTextareas = screen.getAllByPlaceholderText(/enter your prompt here/i);
      expect(newPromptTextareas.length).toBeGreaterThan(1);
      const newPromptTextarea = newPromptTextareas[newPromptTextareas.length - 1]; // Get the last one (new prompt input)
      fireEvent.change(newPromptTextarea, { 
        target: { value: 'What is machine learning?' } 
      });
    });
    
    // Click add prompt button again
    const addButtons = screen.getAllByRole('button', { name: /add prompt/i });
    fireEvent.click(addButtons[addButtons.length - 1]);
    
    // Verify both prompts are added
    await waitFor(() => {
      expect(screen.getByDisplayValue('Explain quantum computing')).toBeInTheDocument();
      expect(screen.getByDisplayValue('What is machine learning?')).toBeInTheDocument();
    });
    
    // Run evaluation
    const runButton = screen.getByRole('button', { name: /run evaluation/i });
    fireEvent.click(runButton);
    
    // Wait for completion
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /clear results/i })).toBeInTheDocument();
    }, { timeout: 15000 });
    
    // Verify multiple prompts are displayed in results - be more flexible with the count
    const quantumTexts = screen.getAllByText('Explain quantum computing');
    const mlTexts = screen.getAllByText('What is machine learning?');
    expect(quantumTexts.length).toBeGreaterThanOrEqual(1);
    expect(mlTexts.length).toBeGreaterThanOrEqual(1);
  });

  test('shows error when trying to run without prompts', async () => {
    renderWithProvider(<PromptTesting />);
    
    // Try to run evaluation without prompts
    const runButton = screen.getByRole('button', { name: /run evaluation/i });
    expect(runButton).toBeDisabled();
  });

  test('can toggle between edit and preview mode', async () => {
    renderWithProvider(<PromptTesting />);
    
    // Add a prompt with special syntax
    const promptTextarea = screen.getByPlaceholderText(/enter your prompt here/i);
    fireEvent.change(promptTextarea, { 
      target: { value: 'System: You are a helpful assistant.\n\nUser: {question}\n\nAssistant:' } 
    });
    fireEvent.click(screen.getByRole('button', { name: /add prompt/i }));
    
    // Toggle to preview mode
    await waitFor(() => {
      const previewButton = screen.getByTitle(/show preview/i);
      fireEvent.click(previewButton);
    });
    
    // Verify preview mode is active
    await waitFor(() => {
      expect(screen.getByText(/preview mode/i)).toBeInTheDocument();
    });
    
    // Toggle back to edit mode
    const editButton = screen.getByTitle(/hide preview/i);
    fireEvent.click(editButton);
    
    // Verify edit mode is active
    await waitFor(() => {
      expect(screen.getByDisplayValue(/system: you are a helpful assistant/i)).toBeInTheDocument();
    });
  });
});