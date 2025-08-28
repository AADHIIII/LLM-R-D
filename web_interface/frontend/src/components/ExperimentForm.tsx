import React, { useState } from 'react';
import { ExperimentFormData, TrainingConfig } from '../types';
import Button from './Button';
import Input from './Input';
import Card from './Card';

interface ExperimentFormProps {
  onSubmit: (data: ExperimentFormData) => void;
  loading?: boolean;
}

const ExperimentForm: React.FC<ExperimentFormProps> = ({ onSubmit, loading = false }) => {
  const [formData, setFormData] = useState<ExperimentFormData>({
    name: '',
    description: '',
    dataset_id: '',
    training_config: {
      base_model: 'gpt2',
      epochs: 3,
      batch_size: 4,
      learning_rate: 5e-5,
      warmup_steps: 100,
      use_lora: true,
      lora_rank: 16,
    },
  });

  const [errors, setErrors] = useState<Partial<ExperimentFormData>>({});

  const validateForm = (): boolean => {
    const newErrors: Partial<ExperimentFormData> = {};

    if (!formData.name.trim()) {
      newErrors.name = 'Experiment name is required';
    }

    if (!formData.description.trim()) {
      newErrors.description = 'Description is required';
    }

    if (!formData.dataset_id) {
      newErrors.dataset_id = 'Dataset selection is required';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (validateForm()) {
      onSubmit(formData);
    }
  };

  const updateTrainingConfig = (key: keyof TrainingConfig, value: any) => {
    setFormData(prev => ({
      ...prev,
      training_config: {
        ...prev.training_config,
        [key]: value,
      },
    }));
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <Card title="Experiment Details">
        <div className="space-y-4">
          <Input
            label="Experiment Name"
            value={formData.name}
            onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
            error={errors.name}
            placeholder="Enter experiment name"
            required
          />

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Description
            </label>
            <textarea
              value={formData.description}
              onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
              className={`block w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-1 sm:text-sm ${
                errors.description
                  ? 'border-red-300 focus:ring-red-500 focus:border-red-500'
                  : 'border-gray-300 focus:ring-primary-500 focus:border-primary-500'
              }`}
              rows={3}
              placeholder="Describe your experiment"
              required
            />
            {errors.description && (
              <p className="mt-1 text-sm text-red-600">{errors.description}</p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Dataset
            </label>
            <select
              value={formData.dataset_id}
              onChange={(e) => setFormData(prev => ({ ...prev, dataset_id: e.target.value }))}
              className={`block w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none focus:ring-1 sm:text-sm ${
                errors.dataset_id
                  ? 'border-red-300 focus:ring-red-500 focus:border-red-500'
                  : 'border-gray-300 focus:ring-primary-500 focus:border-primary-500'
              }`}
              required
            >
              <option value="">Select a dataset</option>
              <option value="sample-1">Sample Dataset 1</option>
              <option value="sample-2">Sample Dataset 2</option>
            </select>
            {errors.dataset_id && (
              <p className="mt-1 text-sm text-red-600">{errors.dataset_id}</p>
            )}
          </div>
        </div>
      </Card>

      <Card title="Training Configuration">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Base Model
            </label>
            <select
              value={formData.training_config.base_model}
              onChange={(e) => updateTrainingConfig('base_model', e.target.value)}
              className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-1 focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
            >
              <option value="gpt2">GPT-2</option>
              <option value="gpt2-medium">GPT-2 Medium</option>
              <option value="gpt2-large">GPT-2 Large</option>
              <option value="distilgpt2">DistilGPT-2</option>
            </select>
          </div>

          <Input
            label="Epochs"
            type="number"
            value={formData.training_config.epochs}
            onChange={(e) => updateTrainingConfig('epochs', parseInt(e.target.value))}
            min="1"
            max="10"
          />

          <Input
            label="Batch Size"
            type="number"
            value={formData.training_config.batch_size}
            onChange={(e) => updateTrainingConfig('batch_size', parseInt(e.target.value))}
            min="1"
            max="32"
          />

          <Input
            label="Learning Rate"
            type="number"
            step="0.00001"
            value={formData.training_config.learning_rate}
            onChange={(e) => updateTrainingConfig('learning_rate', parseFloat(e.target.value))}
            min="0.00001"
            max="0.001"
          />

          <Input
            label="Warmup Steps"
            type="number"
            value={formData.training_config.warmup_steps}
            onChange={(e) => updateTrainingConfig('warmup_steps', parseInt(e.target.value))}
            min="0"
            max="1000"
          />

          <Input
            label="LoRA Rank"
            type="number"
            value={formData.training_config.lora_rank}
            onChange={(e) => updateTrainingConfig('lora_rank', parseInt(e.target.value))}
            min="4"
            max="64"
            disabled={!formData.training_config.use_lora}
          />

          <div className="flex items-center">
            <input
              type="checkbox"
              id="use_lora"
              checked={formData.training_config.use_lora}
              onChange={(e) => updateTrainingConfig('use_lora', e.target.checked)}
              className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
            />
            <label htmlFor="use_lora" className="ml-2 block text-sm text-gray-900">
              Use LoRA (Low-Rank Adaptation)
            </label>
          </div>
        </div>
      </Card>

      <div className="flex justify-end space-x-3">
        <Button type="button" variant="secondary">
          Cancel
        </Button>
        <Button type="submit" loading={loading}>
          Create Experiment
        </Button>
      </div>
    </form>
  );
};

export default ExperimentForm;