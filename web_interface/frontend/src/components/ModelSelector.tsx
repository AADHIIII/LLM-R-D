import React from 'react';
import { CheckIcon } from '@heroicons/react/24/outline';
import { Model } from '../types';
import Card from './Card';

interface ModelSelectorProps {
  models: Model[];
  selectedModels: string[];
  onSelectionChange: (selectedModels: string[]) => void;
  loading?: boolean;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({
  models,
  selectedModels,
  onSelectionChange,
  loading = false,
}) => {
  const toggleModel = (modelId: string) => {
    if (selectedModels.includes(modelId)) {
      onSelectionChange(selectedModels.filter(id => id !== modelId));
    } else {
      onSelectionChange([...selectedModels, modelId]);
    }
  };

  const selectAll = () => {
    onSelectionChange(models.map(model => model.id));
  };

  const clearAll = () => {
    onSelectionChange([]);
  };

  const getModelTypeColor = (type: string) => {
    switch (type) {
      case 'fine-tuned':
        return 'bg-green-100 text-green-800';
      case 'commercial':
        return 'bg-blue-100 text-blue-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getModelTypeIcon = (type: string) => {
    switch (type) {
      case 'fine-tuned':
        return '🔧';
      case 'commercial':
        return '🏢';
      default:
        return '🤖';
    }
  };

  if (loading) {
    return (
      <Card title="Model Selection" subtitle="Loading available models...">
        <div className="animate-pulse space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="flex items-center space-x-3">
              <div className="w-4 h-4 bg-gray-300 rounded"></div>
              <div className="flex-1 space-y-2">
                <div className="h-4 bg-gray-300 rounded w-3/4"></div>
                <div className="h-3 bg-gray-300 rounded w-1/2"></div>
              </div>
            </div>
          ))}
        </div>
      </Card>
    );
  }

  return (
    <Card title="Model Selection" subtitle="Choose models to test your prompts against">
      <div className="space-y-4">
        {/* Selection controls */}
        <div className="flex items-center justify-between">
          <div className="text-sm text-gray-600">
            {selectedModels.length} of {models.length} models selected
          </div>
          <div className="flex space-x-2">
            <button
              onClick={selectAll}
              className="text-sm text-primary-600 hover:text-primary-700"
              type="button"
            >
              Select All
            </button>
            <span className="text-gray-300">|</span>
            <button
              onClick={clearAll}
              className="text-sm text-gray-600 hover:text-gray-700"
              type="button"
            >
              Clear All
            </button>
          </div>
        </div>

        {/* Model list */}
        {models.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <div className="text-4xl mb-2">🤖</div>
            <p>No models available</p>
            <p className="text-sm mt-1">
              Create a fine-tuned model or check your API configuration
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            {models.map((model) => {
              const isSelected = selectedModels.includes(model.id);
              
              return (
                <div
                  key={model.id}
                  className={`relative flex items-center p-3 rounded-lg border cursor-pointer transition-colors ${
                    isSelected
                      ? 'border-primary-300 bg-primary-50'
                      : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                  }`}
                  onClick={() => toggleModel(model.id)}
                >
                  <div className="flex items-center">
                    <div
                      className={`w-4 h-4 rounded border-2 flex items-center justify-center ${
                        isSelected
                          ? 'border-primary-500 bg-primary-500'
                          : 'border-gray-300'
                      }`}
                    >
                      {isSelected && (
                        <CheckIcon className="w-3 h-3 text-white" />
                      )}
                    </div>
                  </div>
                  
                  <div className="ml-3 flex-1">
                    <div className="flex items-center space-x-2">
                      <span className="text-lg">
                        {getModelTypeIcon(model.type)}
                      </span>
                      <h3 className="text-sm font-medium text-gray-900">
                        {model.name}
                      </h3>
                      <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${getModelTypeColor(model.type)}`}>
                        {model.type}
                      </span>
                    </div>
                    
                    <div className="mt-1 flex items-center space-x-4 text-xs text-gray-500">
                      {model.base_model && (
                        <span>Base: {model.base_model}</span>
                      )}
                      <span>
                        Created: {new Date(model.created_at).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Model type legend */}
        <div className="bg-gray-50 rounded-lg p-3">
          <h4 className="text-sm font-medium text-gray-900 mb-2">Model Types</h4>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="flex items-center space-x-2">
              <span>🔧</span>
              <span className="text-gray-600">Fine-tuned models</span>
            </div>
            <div className="flex items-center space-x-2">
              <span>🏢</span>
              <span className="text-gray-600">Commercial APIs</span>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
};

export default ModelSelector;