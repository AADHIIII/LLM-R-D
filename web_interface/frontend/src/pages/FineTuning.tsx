import React, { useState, useEffect } from 'react';
import { PlusIcon } from '@heroicons/react/24/outline';
import { Experiment, TrainingJob, ExperimentFormData } from '../types';
import { useNotifications, useLoading } from '../context/AppContext';
import Button from '../components/Button';
import ExperimentForm from '../components/ExperimentForm';
import DatasetUpload from '../components/DatasetUpload';
import ExperimentDashboard from '../components/ExperimentDashboard';

const FineTuning: React.FC = () => {
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [showDatasetUpload, setShowDatasetUpload] = useState(false);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [trainingJobs, setTrainingJobs] = useState<TrainingJob[]>([]);
  
  const { addNotification } = useNotifications();
  const { loading, setLoading } = useLoading();

  // Mock data for demonstration
  useEffect(() => {
    // Simulate loading experiments
    const mockExperiments: Experiment[] = [
      {
        id: '1',
        name: 'Customer Support Fine-tuning',
        description: 'Fine-tune GPT-2 on customer support conversations',
        dataset_id: 'dataset-1',
        status: 'completed',
        progress: 100,
        created_at: '2024-01-15T10:00:00Z',
        updated_at: '2024-01-15T12:30:00Z',
      },
      {
        id: '2',
        name: 'Medical Q&A Training',
        description: 'Training on medical question-answer pairs',
        dataset_id: 'dataset-2',
        status: 'running',
        progress: 65,
        created_at: '2024-01-16T09:00:00Z',
        updated_at: '2024-01-16T11:15:00Z',
      },
      {
        id: '3',
        name: 'Code Generation Experiment',
        description: 'Fine-tuning for Python code generation',
        dataset_id: 'dataset-3',
        status: 'pending',
        progress: 0,
        created_at: '2024-01-17T08:00:00Z',
        updated_at: '2024-01-17T08:00:00Z',
      },
    ];

    const mockTrainingJobs: TrainingJob[] = [
      {
        id: 'job-2',
        experiment_id: '2',
        status: 'running',
        progress: 65,
        current_epoch: 2,
        loss: 0.3245,
        created_at: '2024-01-16T09:00:00Z',
        updated_at: '2024-01-16T11:15:00Z',
        config: {
          base_model: 'gpt2',
          epochs: 3,
          batch_size: 4,
          learning_rate: 5e-5,
          warmup_steps: 100,
          use_lora: true,
          lora_rank: 16,
        },
      },
    ];

    setExperiments(mockExperiments);
    setTrainingJobs(mockTrainingJobs);
  }, []);

  const handleCreateExperiment = async (data: ExperimentFormData) => {
    setLoading({ isLoading: true, message: 'Creating experiment...' });
    
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const newExperiment: Experiment = {
        id: Date.now().toString(),
        name: data.name,
        description: data.description,
        dataset_id: data.dataset_id,
        status: 'pending',
        progress: 0,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };
      
      setExperiments(prev => [newExperiment, ...prev]);
      setShowCreateForm(false);
      
      addNotification({
        type: 'success',
        message: 'Experiment created successfully!',
      });
    } catch (error) {
      addNotification({
        type: 'error',
        message: 'Failed to create experiment. Please try again.',
      });
    } finally {
      setLoading({ isLoading: false });
    }
  };

  const handleDatasetUpload = async (file: File) => {
    setLoading({ isLoading: true, message: 'Uploading dataset...' });
    
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      addNotification({
        type: 'success',
        message: `Dataset "${file.name}" uploaded successfully!`,
      });
      
      setShowDatasetUpload(false);
    } catch (error) {
      addNotification({
        type: 'error',
        message: 'Failed to upload dataset. Please try again.',
      });
    } finally {
      setLoading({ isLoading: false });
    }
  };

  const handleStartExperiment = async (id: string) => {
    setLoading({ isLoading: true, message: 'Starting experiment...' });
    
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      setExperiments(prev =>
        prev.map(exp =>
          exp.id === id ? { ...exp, status: 'running', updated_at: new Date().toISOString() } : exp
        )
      );
      
      addNotification({
        type: 'success',
        message: 'Experiment started successfully!',
      });
    } catch (error) {
      addNotification({
        type: 'error',
        message: 'Failed to start experiment. Please try again.',
      });
    } finally {
      setLoading({ isLoading: false });
    }
  };

  const handleStopExperiment = async (id: string) => {
    setLoading({ isLoading: true, message: 'Stopping experiment...' });
    
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setExperiments(prev =>
        prev.map(exp =>
          exp.id === id ? { ...exp, status: 'pending', updated_at: new Date().toISOString() } : exp
        )
      );
      
      addNotification({
        type: 'success',
        message: 'Experiment stopped successfully!',
      });
    } catch (error) {
      addNotification({
        type: 'error',
        message: 'Failed to stop experiment. Please try again.',
      });
    } finally {
      setLoading({ isLoading: false });
    }
  };

  const handleViewResults = (id: string) => {
    // Navigate to results page (would use router in real implementation)
    addNotification({
      type: 'info',
      message: 'Navigating to results page...',
    });
  };

  return (
    <div>
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Fine-Tuning</h1>
            <p className="mt-2 text-gray-600">
              Create and manage fine-tuning experiments
            </p>
          </div>
          
          <div className="flex space-x-3">
            <Button
              variant="secondary"
              onClick={() => setShowDatasetUpload(true)}
            >
              Upload Dataset
            </Button>
            <Button onClick={() => setShowCreateForm(true)}>
              <PlusIcon className="h-4 w-4 mr-2" />
              New Experiment
            </Button>
          </div>
        </div>
      </div>

      {showDatasetUpload && (
        <div className="mb-8">
          <DatasetUpload
            onUpload={handleDatasetUpload}
            loading={loading.isLoading}
          />
          <div className="mt-4 flex justify-end">
            <Button
              variant="secondary"
              onClick={() => setShowDatasetUpload(false)}
            >
              Cancel
            </Button>
          </div>
        </div>
      )}

      {showCreateForm && (
        <div className="mb-8">
          <ExperimentForm
            onSubmit={handleCreateExperiment}
            loading={loading.isLoading}
          />
          <div className="mt-4 flex justify-end">
            <Button
              variant="secondary"
              onClick={() => setShowCreateForm(false)}
            >
              Cancel
            </Button>
          </div>
        </div>
      )}

      <ExperimentDashboard
        experiments={experiments}
        trainingJobs={trainingJobs}
        onStartExperiment={handleStartExperiment}
        onStopExperiment={handleStopExperiment}
        onViewResults={handleViewResults}
        loading={loading.isLoading}
      />
    </div>
  );
};

export default FineTuning;