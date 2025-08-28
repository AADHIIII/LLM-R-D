import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
} from 'recharts';

interface TrainingMetrics {
  epoch: number;
  training_loss: number;
  validation_loss: number;
  learning_rate: number;
  timestamp: string;
}

interface EvaluationMetrics {
  model_name: string;
  bleu_score: number;
  rouge_score: number;
  perplexity: number;
  semantic_similarity: number;
  llm_judge_score: number;
  cost_usd: number;
  latency_ms: number;
}

interface PerformanceMetricsChartProps {
  trainingMetrics?: TrainingMetrics[];
  evaluationMetrics?: EvaluationMetrics[];
  type: 'training' | 'evaluation' | 'comparison';
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

const PerformanceMetricsChart: React.FC<PerformanceMetricsChartProps> = ({
  trainingMetrics = [],
  evaluationMetrics = [],
  type,
}) => {
  const renderTrainingChart = () => (
    <div className="space-y-6">
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Training Loss Over Time</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={trainingMetrics}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="epoch" 
              label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
            />
            <YAxis 
              label={{ value: 'Loss', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip 
              formatter={(value: number, name: string) => [
                value.toFixed(4), 
                name === 'training_loss' ? 'Training Loss' : 'Validation Loss'
              ]}
            />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="training_loss" 
              stroke="#8884d8" 
              strokeWidth={2}
              name="Training Loss"
            />
            <Line 
              type="monotone" 
              dataKey="validation_loss" 
              stroke="#82ca9d" 
              strokeWidth={2}
              name="Validation Loss"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Learning Rate Schedule</h3>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={trainingMetrics}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="epoch" 
              label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }}
            />
            <YAxis 
              label={{ value: 'Learning Rate', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip 
              formatter={(value: number) => [value.toExponential(2), 'Learning Rate']}
            />
            <Line 
              type="monotone" 
              dataKey="learning_rate" 
              stroke="#ff7300" 
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );

  const renderEvaluationChart = () => (
    <div className="space-y-6">
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Model Performance Comparison</h3>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={evaluationMetrics} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="model_name" 
              angle={-45}
              textAnchor="end"
              height={100}
            />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="bleu_score" fill="#8884d8" name="BLEU Score" />
            <Bar dataKey="rouge_score" fill="#82ca9d" name="ROUGE Score" />
            <Bar dataKey="semantic_similarity" fill="#ffc658" name="Semantic Similarity" />
            <Bar dataKey="llm_judge_score" fill="#ff7300" name="LLM Judge Score" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Cost Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={evaluationMetrics}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ model_name, cost_usd }) => `${model_name}: $${cost_usd.toFixed(3)}`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="cost_usd"
              >
                {evaluationMetrics.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip formatter={(value: number) => [`$${value.toFixed(3)}`, 'Cost']} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Response Latency</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={evaluationMetrics}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="model_name" 
                angle={-45}
                textAnchor="end"
                height={100}
              />
              <YAxis label={{ value: 'Latency (ms)', angle: -90, position: 'insideLeft' }} />
              <Tooltip formatter={(value: number) => [`${value}ms`, 'Latency']} />
              <Bar dataKey="latency_ms" fill="#82ca9d" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );

  const renderComparisonChart = () => (
    <div className="bg-white p-6 rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-4">Performance vs Cost Analysis</h3>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={evaluationMetrics}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="cost_usd" 
            type="number"
            domain={['dataMin', 'dataMax']}
            label={{ value: 'Cost (USD)', position: 'insideBottom', offset: -5 }}
          />
          <YAxis 
            dataKey="llm_judge_score"
            type="number"
            domain={['dataMin', 'dataMax']}
            label={{ value: 'LLM Judge Score', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip 
            formatter={(value: number, name: string) => [
              name === 'cost_usd' ? `$${value.toFixed(3)}` : value.toFixed(2),
              name === 'cost_usd' ? 'Cost' : 'Score'
            ]}
            labelFormatter={(label) => `Model: ${label}`}
          />
          <Line 
            type="monotone" 
            dataKey="llm_judge_score" 
            stroke="#8884d8" 
            strokeWidth={2}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );

  return (
    <div className="space-y-6">
      {type === 'training' && renderTrainingChart()}
      {type === 'evaluation' && renderEvaluationChart()}
      {type === 'comparison' && renderComparisonChart()}
    </div>
  );
};

export default PerformanceMetricsChart;