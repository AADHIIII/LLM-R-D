import React from 'react';
import { 
  PlayIcon, 
  PauseIcon, 
  StopIcon, 
  ChartBarIcon,
  ClockIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline';
import { Experiment, TrainingJob } from '../types';
import Button from './Button';
import Card from './Card';

interface ExperimentDashboardProps {
  experiments: Experiment[];
  trainingJobs: TrainingJob[];
  onStartExperiment: (id: string) => void;
  onStopExperiment: (id: string) => void;
  onViewResults: (id: string) => void;
  loading?: boolean;
}

const ExperimentDashboard: React.FC<ExperimentDashboardProps> = ({
  experiments,
  trainingJobs,
  onStartExperiment,
  onStopExperiment,
  onViewResults,
  loading = false,
}) => {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <PlayIcon className="h-5 w-5 text-green-500" />;
      case 'completed':
        return <CheckCircleIcon className="h-5 w-5 text-blue-500" />;
      case 'failed':
        return <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />;
      default:
        return <ClockIcon className="h-5 w-5 text-gray-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'bg-green-100 text-green-800';
      case 'completed':
        return 'bg-blue-100 text-blue-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getTrainingJob = (experimentId: string) => {
    return trainingJobs.find(job => job.experiment_id === experimentId);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">Experiments</h2>
        <div className="flex items-center space-x-2">
          <span className="text-sm text-gray-500">
            {experiments.length} total experiments
          </span>
        </div>
      </div>

      {experiments.length === 0 ? (
        <Card>
          <div className="text-center py-12">
            <ChartBarIcon className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">No experiments</h3>
            <p className="mt-1 text-sm text-gray-500">
              Get started by creating your first experiment.
            </p>
          </div>
        </Card>
      ) : (
        <div className="grid grid-cols-1 gap-6">
          {experiments.map((experiment) => {
            const trainingJob = getTrainingJob(experiment.id);
            
            return (
              <Card key={experiment.id}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    {getStatusIcon(experiment.status)}
                    <div>
                      <h3 className="text-lg font-medium text-gray-900">
                        {experiment.name}
                      </h3>
                      <p className="text-sm text-gray-500">{experiment.description}</p>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-3">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(experiment.status)}`}>
                      {experiment.status}
                    </span>
                    
                    {experiment.status === 'pending' && (
                      <Button
                        size="sm"
                        onClick={() => onStartExperiment(experiment.id)}
                        loading={loading}
                      >
                        <PlayIcon className="h-4 w-4 mr-1" />
                        Start
                      </Button>
                    )}
                    
                    {experiment.status === 'running' && (
                      <Button
                        size="sm"
                        variant="danger"
                        onClick={() => onStopExperiment(experiment.id)}
                        loading={loading}
                      >
                        <StopIcon className="h-4 w-4 mr-1" />
                        Stop
                      </Button>
                    )}
                    
                    {experiment.status === 'completed' && (
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={() => onViewResults(experiment.id)}
                      >
                        <ChartBarIcon className="h-4 w-4 mr-1" />
                        View Results
                      </Button>
                    )}
                  </div>
                </div>

                <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-gray-500">Created:</span>
                    <span className="ml-2 text-gray-900">
                      {formatDate(experiment.created_at)}
                    </span>
                  </div>
                  
                  <div>
                    <span className="text-gray-500">Updated:</span>
                    <span className="ml-2 text-gray-900">
                      {formatDate(experiment.updated_at)}
                    </span>
                  </div>
                  
                  <div>
                    <span className="text-gray-500">ID:</span>
                    <span className="ml-2 text-gray-900 font-mono text-xs">
                      {experiment.id.slice(0, 8)}...
                    </span>
                  </div>
                </div>

                {trainingJob && (
                  <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                    <h4 className="text-sm font-medium text-gray-900 mb-3">Training Progress</h4>
                    
                    <div className="space-y-3">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-gray-600">Progress</span>
                        <span className="text-gray-900">{trainingJob.progress}%</span>
                      </div>
                      
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-primary-500 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${trainingJob.progress}%` }}
                        />
                      </div>
                      
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <span className="text-gray-500">Epoch:</span>
                          <span className="ml-2 text-gray-900">
                            {trainingJob.current_epoch}/{trainingJob.config.epochs}
                          </span>
                        </div>
                        
                        <div>
                          <span className="text-gray-500">Loss:</span>
                          <span className="ml-2 text-gray-900">
                            {trainingJob.loss.toFixed(4)}
                          </span>
                        </div>
                        
                        <div>
                          <span className="text-gray-500">Batch Size:</span>
                          <span className="ml-2 text-gray-900">
                            {trainingJob.config.batch_size}
                          </span>
                        </div>
                        
                        <div>
                          <span className="text-gray-500">Learning Rate:</span>
                          <span className="ml-2 text-gray-900">
                            {trainingJob.config.learning_rate}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default ExperimentDashboard;