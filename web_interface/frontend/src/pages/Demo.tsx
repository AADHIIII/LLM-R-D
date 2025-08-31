import React, { useState } from 'react';
import { PlayIcon } from '@heroicons/react/24/outline';
import { getGeminiService } from '../services/geminiService';

const Demo: React.FC = () => {
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const geminiService = getGeminiService();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    setLoading(true);
    setError('');
    setResponse('');

    try {
      if (geminiService) {
        const result = await geminiService.generateText(prompt);
        setResponse(result);
      } else {
        setError('Gemini API key not configured. This is a demo version.');
        setResponse('This is a demo response. In the full version with your API key, you would see the actual Gemini AI response here.');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-6">
          🤖 LLM Optimization Platform Demo
        </h1>
        
        <div className="mb-6">
          <p className="text-gray-600 mb-4">
            Test your prompts with Gemini AI. This is a live demo of the LLM optimization platform.
          </p>
          
          {!geminiService && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4 mb-4">
              <p className="text-yellow-800">
                <strong>Demo Mode:</strong> API key not configured. Responses will be simulated.
              </p>
            </div>
          )}
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="prompt" className="block text-sm font-medium text-gray-700 mb-2">
              Enter your prompt:
            </label>
            <textarea
              id="prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              rows={4}
              placeholder="e.g., Write a short story about a robot learning to paint..."
            />
          </div>

          <button
            type="submit"
            disabled={loading || !prompt.trim()}
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <div className="animate-spin -ml-1 mr-3 h-5 w-5 text-white">
                  <div className="h-5 w-5 border-2 border-white border-t-transparent rounded-full"></div>
                </div>
                Generating...
              </>
            ) : (
              <>
                <PlayIcon className="-ml-1 mr-2 h-5 w-5" />
                Generate Response
              </>
            )}
          </button>
        </form>

        {error && (
          <div className="mt-4 bg-red-50 border border-red-200 rounded-md p-4">
            <p className="text-red-800">{error}</p>
          </div>
        )}

        {response && (
          <div className="mt-6">
            <h3 className="text-lg font-medium text-gray-900 mb-3">Response:</h3>
            <div className="bg-gray-50 border border-gray-200 rounded-md p-4">
              <p className="text-gray-800 whitespace-pre-wrap">{response}</p>
            </div>
          </div>
        )}

        <div className="mt-8 border-t pt-6">
          <h3 className="text-lg font-medium text-gray-900 mb-3">Platform Features:</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-blue-50 p-4 rounded-lg">
              <h4 className="font-medium text-blue-900">🎯 Prompt Testing</h4>
              <p className="text-blue-700 text-sm mt-1">Test and optimize your prompts with multiple AI models</p>
            </div>
            <div className="bg-green-50 p-4 rounded-lg">
              <h4 className="font-medium text-green-900">📊 Performance Analytics</h4>
              <p className="text-green-700 text-sm mt-1">Track costs, latency, and quality metrics</p>
            </div>
            <div className="bg-purple-50 p-4 rounded-lg">
              <h4 className="font-medium text-purple-900">🔧 Model Comparison</h4>
              <p className="text-purple-700 text-sm mt-1">Compare responses across different AI models</p>
            </div>
            <div className="bg-orange-50 p-4 rounded-lg">
              <h4 className="font-medium text-orange-900">🚀 Fine-tuning</h4>
              <p className="text-orange-700 text-sm mt-1">Upload datasets and fine-tune custom models</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Demo;