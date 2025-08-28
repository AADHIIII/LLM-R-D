import React from 'react';
import { 
  StarIcon, 
  HandThumbUpIcon, 
  HandThumbDownIcon,
  ChatBubbleLeftRightIcon,
  ChartBarIcon
} from '@heroicons/react/24/outline';
import { StarIcon as StarSolidIcon } from '@heroicons/react/24/solid';

export interface FeedbackStats {
  totalRatings: number;
  averageStarRating: number;
  starDistribution: Record<number, number>;
  thumbsUpCount: number;
  thumbsDownCount: number;
  totalComments: number;
  recentComments: Array<{
    id: string;
    text: string;
    starRating?: number;
    thumbsRating?: 'up' | 'down';
    timestamp: string;
  }>;
}

interface FeedbackAggregationProps {
  stats: FeedbackStats;
  modelName?: string;
  className?: string;
  showRecentComments?: boolean;
  maxCommentsToShow?: number;
}

const FeedbackAggregation: React.FC<FeedbackAggregationProps> = ({
  stats,
  modelName,
  className = '',
  showRecentComments = true,
  maxCommentsToShow = 3,
}) => {
  const formatPercentage = (value: number, total: number) => {
    if (total === 0) return '0%';
    return `${Math.round((value / total) * 100)}%`;
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${diffDays}d ago`;
  };

  const renderStarDistribution = () => {
    const maxCount = Math.max(...Object.values(stats.starDistribution));
    
    return (
      <div className="space-y-2">
        {[5, 4, 3, 2, 1].map((star) => {
          const count = stats.starDistribution[star] || 0;
          const percentage = stats.totalRatings > 0 ? (count / stats.totalRatings) * 100 : 0;
          
          return (
            <div key={star} className="flex items-center space-x-2 text-sm">
              <div className="flex items-center space-x-1 w-12">
                <span className="text-gray-600">{star}</span>
                <StarSolidIcon className="h-3 w-3 text-yellow-500" />
              </div>
              <div className="flex-1 bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-yellow-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${percentage}%` }}
                />
              </div>
              <div className="text-xs text-gray-500 w-12 text-right">
                {count} ({formatPercentage(count, stats.totalRatings)})
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className={`bg-white rounded-lg border p-4 space-y-4 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-medium text-gray-900">
          {modelName ? `${modelName} Feedback` : 'Feedback Summary'}
        </h3>
        <div className="flex items-center space-x-1 text-sm text-gray-500">
          <ChartBarIcon className="h-4 w-4" />
          <span>{stats.totalRatings} ratings</span>
        </div>
      </div>  
    {stats.totalRatings === 0 ? (
        <div className="text-center py-8">
          <ChatBubbleLeftRightIcon className="mx-auto h-12 w-12 text-gray-400" />
          <h4 className="mt-2 text-sm font-medium text-gray-900">No feedback yet</h4>
          <p className="mt-1 text-sm text-gray-500">
            Be the first to rate this model's responses
          </p>
        </div>
      ) : (
        <div className="space-y-6">
          {/* Overall rating summary */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Average star rating */}
            <div className="text-center">
              <div className="flex items-center justify-center space-x-1 mb-1">
                <span className="text-2xl font-bold text-gray-900">
                  {stats.averageStarRating.toFixed(1)}
                </span>
                <StarSolidIcon className="h-6 w-6 text-yellow-500" />
              </div>
              <div className="text-sm text-gray-500">Average Rating</div>
              <div className="flex justify-center mt-1">
                {[1, 2, 3, 4, 5].map((star) => (
                  <StarSolidIcon
                    key={star}
                    className={`h-4 w-4 ${
                      star <= Math.round(stats.averageStarRating)
                        ? 'text-yellow-500'
                        : 'text-gray-300'
                    }`}
                  />
                ))}
              </div>
            </div>

            {/* Thumbs up/down ratio */}
            <div className="text-center">
              <div className="flex items-center justify-center space-x-4 mb-1">
                <div className="flex items-center space-x-1 text-green-600">
                  <HandThumbUpIcon className="h-5 w-5" />
                  <span className="font-semibold">{stats.thumbsUpCount}</span>
                </div>
                <div className="flex items-center space-x-1 text-red-600">
                  <HandThumbDownIcon className="h-5 w-5" />
                  <span className="font-semibold">{stats.thumbsDownCount}</span>
                </div>
              </div>
              <div className="text-sm text-gray-500">Quick Ratings</div>
              {(stats.thumbsUpCount + stats.thumbsDownCount) > 0 && (
                <div className="text-xs text-gray-600 mt-1">
                  {formatPercentage(
                    stats.thumbsUpCount, 
                    stats.thumbsUpCount + stats.thumbsDownCount
                  )} positive
                </div>
              )}
            </div>

            {/* Comments count */}
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900 mb-1">
                {stats.totalComments}
              </div>
              <div className="text-sm text-gray-500">Comments</div>
              <div className="text-xs text-gray-600 mt-1">
                {formatPercentage(stats.totalComments, stats.totalRatings)} with feedback
              </div>
            </div>
          </div>

          {/* Star rating distribution */}
          <div>
            <h4 className="text-sm font-medium text-gray-900 mb-3">Rating Distribution</h4>
            {renderStarDistribution()}
          </div>

          {/* Recent comments */}
          {showRecentComments && stats.recentComments.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-gray-900 mb-3">Recent Comments</h4>
              <div className="space-y-3">
                {stats.recentComments.slice(0, maxCommentsToShow).map((comment) => (
                  <div key={comment.id} className="bg-gray-50 rounded-lg p-3">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        {comment.starRating && (
                          <div className="flex items-center space-x-1">
                            {[1, 2, 3, 4, 5].map((star) => (
                              <StarSolidIcon
                                key={star}
                                className={`h-3 w-3 ${
                                  star <= comment.starRating!
                                    ? 'text-yellow-500'
                                    : 'text-gray-300'
                                }`}
                              />
                            ))}
                          </div>
                        )}
                        {comment.thumbsRating && (
                          <div className={`${
                            comment.thumbsRating === 'up' ? 'text-green-600' : 'text-red-600'
                          }`}>
                            {comment.thumbsRating === 'up' ? (
                              <HandThumbUpIcon className="h-3 w-3" />
                            ) : (
                              <HandThumbDownIcon className="h-3 w-3" />
                            )}
                          </div>
                        )}
                      </div>
                      <span className="text-xs text-gray-500">
                        {formatTimestamp(comment.timestamp)}
                      </span>
                    </div>
                    <p className="text-sm text-gray-700">{comment.text}</p>
                  </div>
                ))}
                {stats.recentComments.length > maxCommentsToShow && (
                  <div className="text-center">
                    <button className="text-sm text-blue-600 hover:text-blue-800">
                      View {stats.recentComments.length - maxCommentsToShow} more comments
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default FeedbackAggregation;