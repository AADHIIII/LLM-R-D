import React from 'react';
import { 
  ClockIcon, 
  CurrencyDollarIcon, 
  StarIcon,
  ChartBarIcon,
  DocumentTextIcon,
} from '@heroicons/react/24/outline';
import { EvaluationResult } from '../types';
import Card from './Card';
import FeedbackRating, { FeedbackData } from './FeedbackRating';

interface ComparisonViewProps {
  results: EvaluationResult[];
  loading?: boolean;
  onResultUpdate?: (result: EvaluationResult) => void;
}

interface GroupedResults {
  [promptIndex: string]: {
    prompt: string;
    results: EvaluationResult[];
  };
}

const ComparisonView: React.FC<ComparisonViewProps> = ({ 
  results, 
  loading = false, 
  onResultUpdate 
}) => {
  // Group results by prompt text
  const groupedResults: GroupedResults = results.reduce((acc, result) => {
    const promptKey = result.prompt;
    if (!acc[promptKey]) {
      acc[promptKey] = {
        prompt: result.prompt,
        results: [],
      };
    }
    acc[promptKey].results.push(result);
    return acc;
  }, {} as GroupedResults);

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 4,
    }).format(amount);
  };

  const formatLatency = (ms: number) => {
    if (ms < 1000) {
      return `${ms}ms`;
    }
    return `${(ms / 1000).toFixed(1)}s`;
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreBackground = (score: number) => {
    if (score >= 0.8) return 'bg-green-100';
    if (score >= 0.6) return 'bg-yellow-100';
    return 'bg-red-100';
  };

  if (loading) {
    return (
      <Card title="Comparison Results" subtitle="Generating responses and evaluating...">
        <div className="space-y-6">
          {/* Progress indicator */}
          <div className="bg-blue-50 rounded-lg p-4">
            <div className="flex items-center space-x-3">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
              <div>
                <div className="text-sm font-medium text-blue-900">
                  Evaluation in progress...
                </div>
                <div className="text-xs text-blue-700">
                  {results.length > 0 ? `${results.length} results completed` : 'Starting evaluation'}
                </div>
              </div>
            </div>
          </div>

          {/* Show completed results while loading */}
          {results.length > 0 && (
            <div className="space-y-4">
              <h3 className="text-sm font-medium text-gray-900">Completed Results</h3>
              {Object.entries(groupedResults).map(([promptKey, group], index) => (
                <div key={promptKey} className="border rounded-lg p-4 bg-green-50">
                  <div className="text-sm font-medium text-green-900 mb-2">
                    Prompt {index} - {group.results.length} models completed
                  </div>
                  <div className="text-xs text-green-700 truncate">
                    {group.prompt}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Loading placeholders for remaining work */}
          <div className="space-y-4">
            {[1, 2].map((i) => (
              <div key={i} className="animate-pulse">
                <div className="h-4 bg-gray-300 rounded w-3/4 mb-4"></div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {[1, 2].map((j) => (
                    <div key={j} className="border rounded-lg p-4">
                      <div className="h-4 bg-gray-300 rounded w-1/2 mb-2"></div>
                      <div className="space-y-2">
                        <div className="h-3 bg-gray-300 rounded"></div>
                        <div className="h-3 bg-gray-300 rounded w-5/6"></div>
                        <div className="h-3 bg-gray-300 rounded w-4/6"></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </Card>
    );
  }

  if (results.length === 0) {
    return (
      <Card title="Comparison Results">
        <div className="text-center py-12">
          <ChartBarIcon className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">No results yet</h3>
          <p className="mt-1 text-sm text-gray-500">
            Run your prompts to see comparison results here
          </p>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {Object.entries(groupedResults).map(([promptKey, group], index) => (
        <Card key={promptKey} title={`Prompt ${index}`}>
          <div className="space-y-4">
            {/* Prompt display */}
            <div className="bg-gray-50 rounded-lg p-3">
              <div className="flex items-center space-x-2 mb-2">
                <DocumentTextIcon className="h-4 w-4 text-gray-500" />
                <span className="text-sm font-medium text-gray-700">Prompt</span>
              </div>
              <p className="text-sm text-gray-900 font-mono whitespace-pre-wrap">
                {group.prompt}
              </p>
            </div>

            {/* Results grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {group.results.map((result) => (
                <div key={result.id} className="border rounded-lg p-4 space-y-3">
                  {/* Model header */}
                  <div className="flex items-center justify-between">
                    <h4 className="font-medium text-gray-900">{result.model_id}</h4>
                  </div>

                  {/* Response */}
                  <div className="bg-gray-50 rounded p-3">
                    <p className="text-sm text-gray-900 whitespace-pre-wrap">
                      {result.response}
                    </p>
                  </div>

                  {/* Human Feedback */}
                  <FeedbackRating
                    initialFeedback={{
                      thumbsRating: result.human_feedback?.thumbs_rating,
                      starRating: result.human_feedback?.star_rating || result.human_rating,
                      qualitativeFeedback: result.human_feedback?.qualitative_feedback,
                    }}
                    onFeedbackChange={(feedback: FeedbackData) => {
                      if (onResultUpdate) {
                        onResultUpdate({
                          ...result,
                          human_rating: feedback.starRating,
                          human_feedback: {
                            thumbs_rating: feedback.thumbsRating,
                            star_rating: feedback.starRating,
                            qualitative_feedback: feedback.qualitativeFeedback,
                            feedback_timestamp: new Date().toISOString(),
                          }
                        });
                      }
                    }}
                    className="border-t pt-3"
                  />

                  {/* Metrics */}
                  <div className="grid grid-cols-2 gap-3 text-xs">
                    {result.metrics.bleu && (
                      <div className={`p-2 rounded ${getScoreBackground(result.metrics.bleu)}`}>
                        <div className="font-medium">BLEU</div>
                        <div className={`font-mono ${getScoreColor(result.metrics.bleu)}`}>
                          {result.metrics.bleu.toFixed(3)}
                        </div>
                      </div>
                    )}
                    
                    {result.metrics.rouge && (
                      <div className={`p-2 rounded ${getScoreBackground(result.metrics.rouge)}`}>
                        <div className="font-medium">ROUGE</div>
                        <div className={`font-mono ${getScoreColor(result.metrics.rouge)}`}>
                          {result.metrics.rouge.toFixed(3)}
                        </div>
                      </div>
                    )}
                    
                    {result.metrics.semantic_similarity && (
                      <div className={`p-2 rounded ${getScoreBackground(result.metrics.semantic_similarity)}`}>
                        <div className="font-medium">Similarity</div>
                        <div className={`font-mono ${getScoreColor(result.metrics.semantic_similarity)}`}>
                          {result.metrics.semantic_similarity.toFixed(3)}
                        </div>
                      </div>
                    )}
                    
                    {result.metrics.llm_judge_score && (
                      <div className={`p-2 rounded ${getScoreBackground(result.metrics.llm_judge_score)}`}>
                        <div className="font-medium">LLM Judge</div>
                        <div className={`font-mono ${getScoreColor(result.metrics.llm_judge_score)}`}>
                          {result.metrics.llm_judge_score.toFixed(3)}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Performance metrics */}
                  <div className="flex items-center justify-between text-xs text-gray-500 pt-2 border-t">
                    <div className="flex items-center space-x-1">
                      <ClockIcon className="h-3 w-3" />
                      <span>{formatLatency(result.latency_ms)}</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <CurrencyDollarIcon className="h-3 w-3" />
                      <span>{formatCurrency(result.cost_usd)}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Summary stats */}
            {group.results.length > 1 && (
              <div className="bg-blue-50 rounded-lg p-3">
                <h5 className="text-sm font-medium text-blue-900 mb-2">Summary</h5>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                  <div>
                    <div className="text-blue-700">Total Cost</div>
                    <div className="font-mono text-blue-900">
                      {formatCurrency(group.results.reduce((sum, r) => sum + r.cost_usd, 0))}
                    </div>
                  </div>
                  <div>
                    <div className="text-blue-700">Avg Latency</div>
                    <div className="font-mono text-blue-900">
                      {formatLatency(
                        group.results.reduce((sum, r) => sum + r.latency_ms, 0) / group.results.length
                      )}
                    </div>
                  </div>
                  <div>
                    <div className="text-blue-700">Best BLEU</div>
                    <div className="font-mono text-blue-900">
                      {Math.max(...group.results.map(r => r.metrics.bleu || 0)).toFixed(3)}
                    </div>
                  </div>
                  <div>
                    <div className="text-blue-700">Models Tested</div>
                    <div className="font-mono text-blue-900">
                      {group.results.length}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </Card>
      ))}
    </div>
  );
};

export default ComparisonView;