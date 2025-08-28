import React from 'react';
import { render, screen } from '@testing-library/react';
import ExperimentForm from '../ExperimentForm';

describe('ExperimentForm', () => {
  const mockOnSubmit = jest.fn();

  test('renders experiment form', () => {
    render(<ExperimentForm onSubmit={mockOnSubmit} />);
    
    expect(screen.getByText(/experiment details/i)).toBeInTheDocument();
    expect(screen.getByText(/training configuration/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /create experiment/i })).toBeInTheDocument();
  });

  test('renders form fields', () => {
    render(<ExperimentForm onSubmit={mockOnSubmit} />);
    
    expect(screen.getByLabelText(/experiment name/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/epochs/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/batch size/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/learning rate/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/warmup steps/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/lora rank/i)).toBeInTheDocument();
  });

  test('shows loading state when loading prop is true', () => {
    render(<ExperimentForm onSubmit={mockOnSubmit} loading={true} />);
    
    const submitButton = screen.getByRole('button', { name: /create experiment/i });
    expect(submitButton).toBeDisabled();
  });
});