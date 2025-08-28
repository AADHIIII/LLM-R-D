import React from 'react';
import { render, screen } from '@testing-library/react';
import DatasetUpload from '../DatasetUpload';

describe('DatasetUpload', () => {
  const mockOnUpload = jest.fn();

  test('renders dataset upload component', () => {
    render(<DatasetUpload onUpload={mockOnUpload} />);
    
    expect(screen.getByText(/dataset upload/i)).toBeInTheDocument();
    expect(screen.getByText(/drop your dataset here/i)).toBeInTheDocument();
    expect(screen.getByText(/jsonl or csv files up to 100mb/i)).toBeInTheDocument();
  });

  test('shows format requirements', () => {
    render(<DatasetUpload onUpload={mockOnUpload} />);
    
    expect(screen.getByText(/dataset format requirements/i)).toBeInTheDocument();
    expect(screen.getByText(/JSONL:/)).toBeInTheDocument();
    expect(screen.getByText(/CSV:/)).toBeInTheDocument();
    expect(screen.getByText(/Example:/)).toBeInTheDocument();
  });

  test('renders file input', () => {
    render(<DatasetUpload onUpload={mockOnUpload} />);
    
    const fileInput = screen.getByLabelText(/drop your dataset here/i);
    expect(fileInput).toBeInTheDocument();
    expect(fileInput).toHaveAttribute('type', 'file');
    expect(fileInput).toHaveAttribute('accept', '.jsonl,.csv');
  });
});