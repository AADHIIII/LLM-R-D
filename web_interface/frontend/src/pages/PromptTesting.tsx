import React, { useState, useEffect } from 'react';
import { PlayIcon, StopIcon } from '@heroicons/react/24/outline';
import { Model, EvaluationResult } from '../types';
import { useNotifications, useLoading } from '../context/AppContext';
import Button from '../components/Button';
import PromptInput from '../components/PromptInput';
import ModelSelector from '../components/ModelSelector';
import ComparisonView from '../components/ComparisonView';

const PromptTesting: React.FC = () => {
  const [prompts, setPrompts] = useState<string[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [results, setResults] = useState<EvaluationResult[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  
  const { addNotification } = useNotifications();
  const { loading, setLoading } = useLoading();

  // Mock data for demonstration
  useEffect(() => {
    const mockModels: Model[] = [
      {
        id: 'gpt-4',
        name: 'GPT-4',
        type: 'commercial',
        created_at: '2024-01-01T00:00:00Z',
      },
      {
        id: 'gpt-3.5-turbo',
        name: 'GPT-3.5 Turbo',
        type: 'commercial',
        created_at: '2024-01-01T00:00:00Z',
      },
      {
        id: 'claude-3-sonnet',
        name: 'Claude 3 Sonnet',
        type: 'commercial',
        created_at: '2024-01-01T00:00:00Z',
      },
      {
        id: 'fine-tuned-gpt2-1',
        name: 'Customer Support GPT-2',
        type: 'fine-tuned',
        base_model: 'gpt2',
        created_at: '2024-01-15T10:00:00Z',
      },
      {
        id: 'fine-tuned-gpt2-2',
        name: 'Medical Q&A GPT-2',
        type: 'fine-tuned',
        base_model: 'gpt2',
        created_at: '2024-01-16T09:00:00Z',
      },
    ];

    setModels(mockModels);
    setSelectedModels([mockModels[0].id, mockModels[3].id]); // Pre-select some models
  }, []);

  const runEvaluation = async () => {
    if (prompts.length === 0) {
      addNotification({
        type: 'error',
        message: 'Please add at least one prompt to test',
      });
      return;
    }

    if (selectedModels.length === 0) {
      addNotification({
        type: 'error',
        message: 'Please select at least one model to test',
      });
      return;
    }

    setIsRunning(true);
    setLoading({ isLoading: true, message: 'Running prompt evaluation...' });
    setResults([]);

    try {
      // Simulate API calls for each prompt-model combination
      const mockResults: EvaluationResult[] = [];
      
      for (let promptIndex = 0; promptIndex < prompts.length; promptIndex++) {
        const prompt = prompts[promptIndex];
        
        for (const modelId of selectedModels) {
          // Simulate delay
          await new Promise(resolve => setTimeout(resolve, 1000));
          
          const mockResult: EvaluationResult = {
            id: `result-${promptIndex}-${modelId}`,
            experiment_id: 'test-experiment',
            prompt,
            model_id: modelId,
            response: generateMockResponse(prompt, modelId),
            metrics: {
              bleu: Math.random() * 0.4 + 0.6, // 0.6-1.0
              rouge: Math.random() * 0.3 + 0.7, // 0.7-1.0
              semantic_similarity: Math.random() * 0.2 + 0.8, // 0.8-1.0
              llm_judge_score: Math.random() * 0.3 + 0.7, // 0.7-1.0
            },
            cost_usd: modelId.includes('gpt-4') ? Math.random() * 0.01 + 0.005 : Math.random() * 0.005 + 0.001,
            latency_ms: Math.random() * 2000 + 500,
            created_at: new Date().toISOString(),
          };
          
          mockResults.push(mockResult);
          setResults([...mockResults]); // Update results incrementally
        }
      }

      addNotification({
        type: 'success',
        message: `Evaluation completed! Tested ${prompts.length} prompts across ${selectedModels.length} models.`,
      });
    } catch (error) {
      addNotification({
        type: 'error',
        message: 'Failed to run evaluation. Please try again.',
      });
    } finally {
      setIsRunning(false);
      setLoading({ isLoading: false });
    }
  };

  const generateMockResponse = (prompt: string, modelId: string): string => {
    const responses = [
      "This is a comprehensive response that addresses the key points in your prompt. The analysis shows several important considerations that should be taken into account.",
      "Based on the information provided, here's a detailed explanation that covers the main aspects of your question with relevant examples and practical insights.",
      "The answer to your prompt involves multiple factors. Let me break this down systematically to provide you with a clear and actionable response.",
      "Here's a thoughtful response that takes into consideration the context and requirements outlined in your prompt, with specific recommendations.",
      "This response provides a balanced perspective on the topic, incorporating both theoretical understanding and practical applications relevant to your needs.",
    ];
    
    const baseResponse = responses[Math.floor(Math.random() * responses.length)];
    
    // Add model-specific variations
    if (modelId.includes('gpt-4')) {
      return `${baseResponse}\n\nAdditionally, from an advanced AI perspective, this approach offers enhanced accuracy and nuanced understanding of complex scenarios.`;
    } else if (modelId.includes('claude')) {
      return `${baseResponse}\n\nI should note that this analysis considers ethical implications and strives for helpful, harmless, and honest communication.`;
    } else if (modelId.includes('fine-tuned')) {
      return `${baseResponse}\n\n[Fine-tuned response] This specialized model provides domain-specific insights tailored to your particular use case.`;
    }
    
    return baseResponse;
  };

  const stopEvaluation = () => {
    setIsRunning(false);
    setLoading({ isLoading: false });
    addNotification({
      type: 'info',
      message: 'Evaluation stopped by user',
    });
  };

  const clearResults = () => {
    setResults([]);
    addNotification({
      type: 'info',
      message: 'Results cleared',
    });
  };

  return (
    <div>
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Prompt Testing</h1>
            <p className="mt-2 text-gray-600">
              Test and compare prompts across different models
            </p>
          </div>
          
          <div className="flex space-x-3">
            {results.length > 0 && (
              <Button variant="secondary" onClick={clearResults}>
                Clear Results
              </Button>
            )}
            
            {isRunning ? (
              <Button variant="danger" onClick={stopEvaluation}>
                <StopIcon className="h-4 w-4 mr-2" />
                Stop Evaluation
              </Button>
            ) : (
              <Button 
                onClick={runEvaluation}
                disabled={prompts.length === 0 || selectedModels.length === 0}
              >
                <PlayIcon className="h-4 w-4 mr-2" />
                Run Evaluation
              </Button>
            )}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
        <PromptInput
          prompts={prompts}
          onPromptsChange={setPrompts}
          maxPrompts={3}
        />
        
        <ModelSelector
          models={models}
          selectedModels={selectedModels}
          onSelectionChange={setSelectedModels}
          loading={loading.isLoading}
        />
      </div>

      <ComparisonView
        results={results}
        loading={isRunning}
        onResultUpdate={(updatedResult) => {
          setResults(prev => prev.map(r => 
            r.id === updatedResult.id ? updatedResult : r
          ));
          addNotification({
            type: 'success',
            message: `Rating updated for ${updatedResult.model_id}`,
          });
        }}
      />
    </div>
  );
};

export default PromptTesting;