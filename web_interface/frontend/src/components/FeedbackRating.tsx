import React, { useState } from 'react';
import { 
  HandThumbUpIcon, 
  HandThumbDownIcon, 
  StarIcon 
} from '@heroicons/react/24/outline';
import { 
  HandThumbUpIcon as HandThumbUpSolidIcon, 
  HandThumbDownIcon as HandThumbDownSolidIcon,
  StarIcon as StarSolidIcon 
} from '@heroicons/react/24/solid';

export interface FeedbackData {
  thumbsRating?: 'up' | 'down';
  starRating?: number;
  qualitativeFeedback?: string;
}

interface FeedbackRatingProps {
  initialFeedback?: FeedbackData;
  onFeedbackChange: (feedback: FeedbackData) => void;
  disabled?: boolean;
  showThumbs?: boolean;
  showStars?: boolean;
  showTextInput?: boolean;
  maxTextLength?: number;
  className?: string;
}

const FeedbackRating: React.FC<FeedbackRatingProps> = ({
  initialFeedback = {},
  onFeedbackChange,
  disabled = false,
  showThumbs = true,
  showStars = true,
  showTextInput = true,
  maxTextLength = 500,
  className = '',
}) => {
  const [feedback, setFeedback] = useState<FeedbackData>(initialFeedback);
  const [hoveredStar, setHoveredStar] = useState<number | null>(null);
  const [textLength, setTextLength] = useState(initialFeedback.qualitativeFeedback?.length || 0);

  const updateFeedback = (updates: Partial<FeedbackData>) => {
    const newFeedback = { ...feedback, ...updates };
    setFeedback(newFeedback);
    onFeedbackChange(newFeedback);
  };

  const handleThumbsClick = (rating: 'up' | 'down') => {
    const newRating = feedback.thumbsRating === rating ? undefined : rating;
    updateFeedback({ thumbsRating: newRating });
  };

  const handleStarClick = (rating: number) => {
    const newRating = feedback.starRating === rating ? undefined : rating;
    updateFeedback({ starRating: newRating });
  };

  const handleTextChange = (text: string) => {
    const truncatedText = text.length > maxTextLength ? text.substring(0, maxTextLength) : text;
    setTextLength(truncatedText.length);
    updateFeedback({ qualitativeFeedback: truncatedText });
  };

  return (
    <div className={`space-y-3 ${className}`}>
      {/* Thumbs up/down rating */}
      {showThumbs && (
        <div className="flex items-center space-x-2">
          <span className="text-sm font-medium text-gray-700">Quick Rating:</span>
          <div className="flex items-center space-x-1">
            <button
              onClick={() => handleThumbsClick('up')}
              disabled={disabled}
              className={`p-2 rounded-full transition-colors ${
                disabled 
                  ? 'cursor-not-allowed opacity-50' 
                  : 'hover:bg-green-50'
              } ${
                feedback.thumbsRating === 'up' 
                  ? 'bg-green-100 text-green-600' 
                  : 'text-gray-400 hover:text-green-500'
              }`}
              title="Good response"
            >
              {feedback.thumbsRating === 'up' ? (
                <HandThumbUpSolidIcon className="h-5 w-5" />
              ) : (
                <HandThumbUpIcon className="h-5 w-5" />
              )}
            </button>
            <button
              onClick={() => handleThumbsClick('down')}
              disabled={disabled}
              className={`p-2 rounded-full transition-colors ${
                disabled 
                  ? 'cursor-not-allowed opacity-50' 
                  : 'hover:bg-red-50'
              } ${
                feedback.thumbsRating === 'down' 
                  ? 'bg-red-100 text-red-600' 
                  : 'text-gray-400 hover:text-red-500'
              }`}
              title="Poor response"
            >
              {feedback.thumbsRating === 'down' ? (
                <HandThumbDownSolidIcon className="h-5 w-5" />
              ) : (
                <HandThumbDownIcon className="h-5 w-5" />
              )}
            </button>
          </div>
        </div>
      )}

      {/* 5-star rating */}
      {showStars && (
        <div className="flex items-center space-x-2">
          <span className="text-sm font-medium text-gray-700">Detailed Rating:</span>
          <div className="flex items-center space-x-1">
            {[1, 2, 3, 4, 5].map((star) => {
              const isActive = feedback.starRating && star <= feedback.starRating;
              const isHovered = hoveredStar && star <= hoveredStar;
              
              return (
                <button
                  key={star}
                  onClick={() => handleStarClick(star)}
                  onMouseEnter={() => !disabled && setHoveredStar(star)}
                  onMouseLeave={() => !disabled && setHoveredStar(null)}
                  disabled={disabled}
                  className={`transition-colors ${
                    disabled 
                      ? 'cursor-not-allowed opacity-50' 
                      : 'hover:scale-110'
                  } ${
                    isActive || isHovered
                      ? 'text-yellow-500' 
                      : 'text-gray-300 hover:text-yellow-400'
                  }`}
                  title={`Rate ${star} star${star > 1 ? 's' : ''}`}
                >
                  {isActive ? (
                    <StarSolidIcon className="h-5 w-5" />
                  ) : (
                    <StarIcon className="h-5 w-5" />
                  )}
                </button>
              );
            })}
            {feedback.starRating && (
              <span className="text-sm text-gray-600 ml-2">
                {feedback.starRating}/5
              </span>
            )}
          </div>
        </div>
      )}

      {/* Qualitative feedback text input */}
      {showTextInput && (
        <div className="space-y-2">
          <label className="text-sm font-medium text-gray-700">
            Additional Comments:
          </label>
          <div className="relative">
            <textarea
              value={feedback.qualitativeFeedback || ''}
              onChange={(e) => handleTextChange(e.target.value)}
              disabled={disabled}
              placeholder="Share your thoughts about this response..."
              className={`w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 ${
                disabled ? 'bg-gray-50 cursor-not-allowed' : 'bg-white'
              }`}
              rows={3}
              maxLength={maxTextLength}
            />
            <div className="absolute bottom-2 right-2 text-xs text-gray-500">
              {textLength}/{maxTextLength}
            </div>
          </div>
          {textLength > maxTextLength * 0.9 && (
            <p className="text-xs text-amber-600">
              Approaching character limit ({maxTextLength - textLength} remaining)
            </p>
          )}
        </div>
      )}

      {/* Feedback summary */}
      {(feedback.thumbsRating || feedback.starRating || feedback.qualitativeFeedback) && (
        <div className="bg-blue-50 rounded-lg p-3">
          <div className="text-xs font-medium text-blue-900 mb-1">Feedback Summary:</div>
          <div className="text-xs text-blue-800 space-y-1">
            {feedback.thumbsRating && (
              <div>Quick rating: {feedback.thumbsRating === 'up' ? '👍 Positive' : '👎 Negative'}</div>
            )}
            {feedback.starRating && (
              <div>Detailed rating: {feedback.starRating}/5 stars</div>
            )}
            {feedback.qualitativeFeedback && (
              <div>Comments: {feedback.qualitativeFeedback.length} characters</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default FeedbackRating;