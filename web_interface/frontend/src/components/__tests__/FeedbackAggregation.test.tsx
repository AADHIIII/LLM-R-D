import React from 'react';
import { render, screen } from '@testing-library/react';
import FeedbackAggregation, { FeedbackStats } from '../FeedbackAggregation';

describe('FeedbackAggregation', () => {
  const mockStats: FeedbackStats = {
    totalRatings: 10,
    averageStarRating: 4.2,
    starDistribution: {
      5: 4,
      4: 3,
      3: 2,
      2: 1,
      1: 0
    },
    thumbsUpCount: 8,
    thumbsDownCount: 2,
    totalComments: 5,
    recentComments: [
      {
        id: '1',
        text: 'Great response, very helpful!',
        starRating: 5,
        thumbsRating: 'up',
        timestamp: new Date(Date.now() - 1000 * 60 * 30).toISOString() // 30 minutes ago
      },
      {
        id: '2',
        text: 'Could be better, lacks detail.',
        starRating: 2,
        thumbsRating: 'down',
        timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2).toISOString() // 2 hours ago
      }
    ]
  };

  it('renders feedback summary correctly', () => {
    render(<FeedbackAggregation stats={mockStats} />);

    expect(screen.getByText('Feedback Summary')).toBeInTheDocument();
    expect(screen.getByText('4.2')).toBeInTheDocument();
    expect(screen.getByText('Average Rating')).toBeInTheDocument();
    expect(screen.getByText('8')).toBeInTheDocument();
    expect(screen.getByText('2')).toBeInTheDocument();
    expect(screen.getByText('5')).toBeInTheDocument();
  });

  it('displays star distribution correctly', () => {
    render(<FeedbackAggregation stats={mockStats} />);

    expect(screen.getByText('Rating Distribution')).toBeInTheDocument();
    
    // Check that star counts are displayed
    expect(screen.getByText('4 (40%)')).toBeInTheDocument(); // 5 stars
    expect(screen.getByText('3 (30%)')).toBeInTheDocument(); // 4 stars
    expect(screen.getByText('2 (20%)')).toBeInTheDocument(); // 3 stars
    expect(screen.getByText('1 (10%)')).toBeInTheDocument(); // 2 stars
    expect(screen.getByText('0 (0%)')).toBeInTheDocument();  // 1 star
  });

  it('shows recent comments', () => {
    render(<FeedbackAggregation stats={mockStats} />);

    expect(screen.getByText('Recent Comments')).toBeInTheDocument();
    expect(screen.getByText('Great response, very helpful!')).toBeInTheDocument();
    expect(screen.getByText('Could be better, lacks detail.')).toBeInTheDocument();
  });

  it('formats timestamps correctly', () => {
    render(<FeedbackAggregation stats={mockStats} />);

    expect(screen.getByText('30m ago')).toBeInTheDocument();
    expect(screen.getByText('2h ago')).toBeInTheDocument();
  });

  it('shows model name when provided', () => {
    render(<FeedbackAggregation stats={mockStats} modelName="GPT-4" />);

    expect(screen.getByText('GPT-4 Feedback')).toBeInTheDocument();
  });

  it('handles empty stats correctly', () => {
    const emptyStats: FeedbackStats = {
      totalRatings: 0,
      averageStarRating: 0,
      starDistribution: {},
      thumbsUpCount: 0,
      thumbsDownCount: 0,
      totalComments: 0,
      recentComments: []
    };

    render(<FeedbackAggregation stats={emptyStats} />);

    expect(screen.getByText('No feedback yet')).toBeInTheDocument();
    expect(screen.getByText('Be the first to rate this model\'s responses')).toBeInTheDocument();
  });

  it('calculates percentages correctly', () => {
    render(<FeedbackAggregation stats={mockStats} />);

    // Thumbs up percentage: 8/(8+2) = 80%
    expect(screen.getByText('80% positive')).toBeInTheDocument();
    
    // Comments percentage: 5/10 = 50%
    expect(screen.getByText('50% with feedback')).toBeInTheDocument();
  });

  it('limits number of comments shown', () => {
    const statsWithManyComments: FeedbackStats = {
      ...mockStats,
      recentComments: [
        ...mockStats.recentComments,
        {
          id: '3',
          text: 'Another comment',
          starRating: 3,
          timestamp: new Date().toISOString()
        },
        {
          id: '4',
          text: 'Yet another comment',
          starRating: 4,
          timestamp: new Date().toISOString()
        }
      ]
    };

    render(<FeedbackAggregation stats={statsWithManyComments} maxCommentsToShow={2} />);

    expect(screen.getByText('Great response, very helpful!')).toBeInTheDocument();
    expect(screen.getByText('Could be better, lacks detail.')).toBeInTheDocument();
    expect(screen.queryByText('Another comment')).not.toBeInTheDocument();
    expect(screen.getByText('View 2 more comments')).toBeInTheDocument();
  });

  it('can hide recent comments', () => {
    render(<FeedbackAggregation stats={mockStats} showRecentComments={false} />);

    expect(screen.queryByText('Recent Comments')).not.toBeInTheDocument();
    expect(screen.queryByText('Great response, very helpful!')).not.toBeInTheDocument();
  });
});