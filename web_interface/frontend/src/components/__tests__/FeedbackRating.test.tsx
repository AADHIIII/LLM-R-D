import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import FeedbackRating, { FeedbackData } from '../FeedbackRating';

describe('FeedbackRating', () => {
  const mockOnFeedbackChange = jest.fn();

  beforeEach(() => {
    mockOnFeedbackChange.mockClear();
  });

  it('renders all feedback components by default', () => {
    render(
      <FeedbackRating onFeedbackChange={mockOnFeedbackChange} />
    );

    expect(screen.getByText('Quick Rating:')).toBeInTheDocument();
    expect(screen.getByText('Detailed Rating:')).toBeInTheDocument();
    expect(screen.getByText('Additional Comments:')).toBeInTheDocument();
    expect(screen.getByPlaceholderText('Share your thoughts about this response...')).toBeInTheDocument();
  });

  it('handles thumbs up/down clicks correctly', async () => {
    render(
      <FeedbackRating onFeedbackChange={mockOnFeedbackChange} />
    );

    const thumbsUpButton = screen.getByTitle('Good response');
    fireEvent.click(thumbsUpButton);

    expect(mockOnFeedbackChange).toHaveBeenCalledWith({
      thumbsRating: 'up'
    });

    // Click again to deselect
    fireEvent.click(thumbsUpButton);
    expect(mockOnFeedbackChange).toHaveBeenCalledWith({
      thumbsRating: undefined
    });
  });

  it('handles star rating clicks correctly', async () => {
    render(
      <FeedbackRating onFeedbackChange={mockOnFeedbackChange} />
    );

    const fourStarButton = screen.getByTitle('Rate 4 stars');
    fireEvent.click(fourStarButton);

    expect(mockOnFeedbackChange).toHaveBeenCalledWith({
      starRating: 4
    });
  });

  it('handles text input correctly', async () => {
    render(
      <FeedbackRating onFeedbackChange={mockOnFeedbackChange} />
    );

    const textArea = screen.getByPlaceholderText('Share your thoughts about this response...');
    fireEvent.change(textArea, { target: { value: 'Great response!' } });

    expect(mockOnFeedbackChange).toHaveBeenCalledWith({
      qualitativeFeedback: 'Great response!'
    });
  });

  it('respects character limit for text input', async () => {
    render(
      <FeedbackRating 
        onFeedbackChange={mockOnFeedbackChange} 
        maxTextLength={10}
      />
    );

    const textArea = screen.getByPlaceholderText('Share your thoughts about this response...');
    fireEvent.change(textArea, { target: { value: 'This is a very long text that exceeds the limit' } });

    // Should only contain first 10 characters
    expect(textArea).toHaveValue('This is a ');
    expect(mockOnFeedbackChange).toHaveBeenCalledWith({
      qualitativeFeedback: 'This is a '
    });
  });

  it('displays initial feedback correctly', () => {
    const initialFeedback: FeedbackData = {
      thumbsRating: 'up',
      starRating: 4,
      qualitativeFeedback: 'Good response'
    };

    render(
      <FeedbackRating 
        initialFeedback={initialFeedback}
        onFeedbackChange={mockOnFeedbackChange} 
      />
    );

    expect(screen.getByDisplayValue('Good response')).toBeInTheDocument();
    expect(screen.getByText('4/5')).toBeInTheDocument();
  });

  it('can be disabled', () => {
    render(
      <FeedbackRating 
        onFeedbackChange={mockOnFeedbackChange}
        disabled={true}
      />
    );

    const thumbsUpButton = screen.getByTitle('Good response');
    const textArea = screen.getByPlaceholderText('Share your thoughts about this response...');

    expect(thumbsUpButton).toBeDisabled();
    expect(textArea).toBeDisabled();
  });

  it('can hide specific components', () => {
    render(
      <FeedbackRating 
        onFeedbackChange={mockOnFeedbackChange}
        showThumbs={false}
        showStars={false}
        showTextInput={false}
      />
    );

    expect(screen.queryByText('Quick Rating:')).not.toBeInTheDocument();
    expect(screen.queryByText('Detailed Rating:')).not.toBeInTheDocument();
    expect(screen.queryByText('Additional Comments:')).not.toBeInTheDocument();
  });

  it('shows feedback summary when feedback is provided', async () => {
    render(
      <FeedbackRating onFeedbackChange={mockOnFeedbackChange} />
    );

    const thumbsUpButton = screen.getByTitle('Good response');
    fireEvent.click(thumbsUpButton);

    expect(screen.getByText('Feedback Summary:')).toBeInTheDocument();
    expect(screen.getByText('Quick rating: 👍 Positive')).toBeInTheDocument();
  });

  it('shows character count and warning', async () => {
    render(
      <FeedbackRating 
        onFeedbackChange={mockOnFeedbackChange}
        maxTextLength={20}
      />
    );

    const textArea = screen.getByPlaceholderText('Share your thoughts about this response...');
    fireEvent.change(textArea, { target: { value: 'This is getting lon' } }); // 18 characters

    expect(screen.getByText('18/20')).toBeInTheDocument();
    expect(screen.getByText(/Approaching character limit/)).toBeInTheDocument();
  });
});