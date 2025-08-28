import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import PromptInput from '../PromptInput';

describe('PromptInput', () => {
  const mockOnPromptsChange = jest.fn();

  beforeEach(() => {
    mockOnPromptsChange.mockClear();
  });

  test('renders prompt input component', () => {
    render(<PromptInput prompts={[]} onPromptsChange={mockOnPromptsChange} />);
    
    expect(screen.getByText(/prompt variants/i)).toBeInTheDocument();
    expect(screen.getByText(/add multiple prompt variations/i)).toBeInTheDocument();
    expect(screen.getByText(/prompt tips/i)).toBeInTheDocument();
  });

  test('displays existing prompts', () => {
    const prompts = ['Test prompt 1', 'Test prompt 2'];
    render(<PromptInput prompts={prompts} onPromptsChange={mockOnPromptsChange} />);
    
    expect(screen.getByDisplayValue('Test prompt 1')).toBeInTheDocument();
    expect(screen.getByDisplayValue('Test prompt 2')).toBeInTheDocument();
  });

  test('shows add prompt interface when under max limit', () => {
    render(<PromptInput prompts={[]} onPromptsChange={mockOnPromptsChange} maxPrompts={3} />);
    
    expect(screen.getByRole('button', { name: /add prompt/i })).toBeInTheDocument();
  });

  test('shows character count', () => {
    const prompts = ['Hello world'];
    render(<PromptInput prompts={prompts} onPromptsChange={mockOnPromptsChange} />);
    
    expect(screen.getByText(/11 characters/i)).toBeInTheDocument();
  });

  test('shows prompt tips and syntax guide', () => {
    render(<PromptInput prompts={[]} onPromptsChange={mockOnPromptsChange} />);
    
    expect(screen.getByText(/be specific and clear/i)).toBeInTheDocument();
    expect(screen.getByText(/include examples when possible/i)).toBeInTheDocument();
    expect(screen.getByText(/syntax highlighting/i)).toBeInTheDocument();
    expect(screen.getByText(/variables/i)).toBeInTheDocument();
  });

  test('toggles between edit and preview mode', () => {
    const prompts = ['Test prompt with {variable}'];
    render(<PromptInput prompts={prompts} onPromptsChange={mockOnPromptsChange} />);
    
    // Should show textarea by default
    expect(screen.getByDisplayValue('Test prompt with {variable}')).toBeInTheDocument();
    
    // Click preview button
    const previewButton = screen.getByTitle(/show preview/i);
    fireEvent.click(previewButton);
    
    // Should show preview mode
    expect(screen.getByText(/preview mode/i)).toBeInTheDocument();
    expect(screen.queryByDisplayValue('Test prompt with {variable}')).not.toBeInTheDocument();
  });

  test('adds prompt with keyboard shortcut', () => {
    render(<PromptInput prompts={[]} onPromptsChange={mockOnPromptsChange} />);
    
    const textarea = screen.getByPlaceholderText(/enter your prompt here/i);
    fireEvent.change(textarea, { target: { value: 'New prompt' } });
    fireEvent.keyDown(textarea, { key: 'Enter', ctrlKey: true });
    
    expect(mockOnPromptsChange).toHaveBeenCalledWith(['New prompt']);
  });
});