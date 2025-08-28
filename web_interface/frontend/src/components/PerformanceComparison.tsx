import React, { useState } from 'react';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  BarChart,
  Bar,
  Cell,
} from 'recharts';

interface ExperimentResult {
  id: string;
  name: string;
  model_name: string;
  metrics: {
    bleu_score: number;
    rouge_score: number;
    perplexity: number;
    semantic_similarity: number;
    llm_judge_score: number;
    human_rating: number;
  };
  performance: {
    avg_latency_ms: number;
    total_cost_usd: number;
    tokens_per_second: number;
    success_rate: number;
  };
  created_at: string;
}

interface PerformanceComparisonProps {
  experiments: ExperimentResult[];
  selectedExperiments: string[];
  onExperimentSelect: (experimentIds: string[]) => void;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82ca9d'];

const PerformanceComparison: React.FC<PerformanceComparisonProps> = ({
  experiments,
  selectedExperiments,
  onExperimentSelect,
}) => {
  const [comparisonView, setComparisonView] = useState<'radar' | 'scatter' | 'bar'>('radar');
  const [metricFocus, setMetricFocus] = useState<'quality' | 'performance' | 'cost'>('quality');

  const selectedExperimentData = experiments.filter(exp => 
    selectedExperiments.includes(exp.id)
  );

  // Prepare radar chart data
  const radarData = [
    {
      metric: 'BLEU',
      ...selectedExperimentData.reduce((acc, exp, index) => ({
        ...acc,
        [exp.name]: exp.metrics.bleu_score * 100
      }), {})
    },
    {
      metric: 'ROUGE',
      ...selectedExperimentData.reduce((acc, exp, index) => ({
        ...acc,
        [exp.name]: exp.metrics.rouge_score * 100
      }), {})
    },
    {
      metric: 'Semantic Sim.',
      ...selectedExperimentData.reduce((acc, exp, index) => ({
        ...acc,
        [exp.name]: exp.metrics.semantic_similarity * 100
      }), {})
    },
    {
      metric: 'LLM Judge',
      ...selectedExperimentData.reduce((acc, exp, index) => ({
        ...acc,
        [exp.name]: exp.metrics.llm_judge_score * 20 // Scale to 0-100
      }), {})
    },
    {
      metric: 'Human Rating',
      ...selectedExperimentData.reduce((acc, exp, index) => ({
        ...acc,
        [exp.name]: exp.metrics.human_rating * 20 // Scale to 0-100
      }), {})
    },
  ];

  // Prepare scatter plot data (Cost vs Quality)
  const scatterData = selectedExperimentData.map(exp => ({
    x: exp.performance.total_cost_usd,
    y: exp.metrics.llm_judge_score,
    name: exp.name,
    model: exp.model_name,
    latency: exp.performance.avg_latency_ms,
  }));

  // Prepare bar chart data based on focus
  const getBarData = () => {
    switch (metricFocus) {
      case 'quality':
        return selectedExperimentData.map(exp => ({
          name: exp.name,
          bleu: exp.metrics.bleu_score,
          rouge: exp.metrics.rouge_score,
          semantic: exp.metrics.semantic_similarity,
          llm_judge: exp.metrics.llm_judge_score,
          human: exp.metrics.human_rating,
        }));
      case 'performance':
        return selectedExperimentData.map(exp => ({
          name: exp.name,
          latency: exp.performance.avg_latency_ms,
          tokens_per_sec: exp.performance.tokens_per_second,
          success_rate: exp.performance.success_rate * 100,
        }));
      case 'cost':
        return selectedExperimentData.map(exp => ({
          name: exp.name,
          total_cost: exp.performance.total_cost_usd,
          cost_per_token: exp.performance.total_cost_usd / (exp.performance.tokens_per_second * 100), // Estimated
        }));
      default:
        return [];
    }
  };

  const renderExperimentSelector = () => (
    <div className="bg-white p-4 rounded-lg shadow mb-6">
      <h3 className="text-lg font-semibold mb-3">Select Experiments to Compare</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {experiments.map((exp) => (
          <label key={exp.id} className="flex items-center space-x-2 cursor-pointer">
            <input
              type="checkbox"
              checked={selectedExperiments.includes(exp.id)}
              onChange={(e) => {
                if (e.target.checked) {
                  onExperimentSelect([...selectedExperiments, exp.id]);
                } else {
                  onExperimentSelect(selectedExperiments.filter(id => id !== exp.id));
                }
              }}
              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <span className="text-sm text-gray-700">
              {exp.name} ({exp.model_name})
            </span>
          </label>
        ))}
      </div>
    </div>
  );

  const renderViewSelector = () => (
    <div className="flex space-x-4 mb-6">
      <div className="flex space-x-2">
        <span className="text-sm font-medium text-gray-700">View:</span>
        {(['radar', 'scatter', 'bar'] as const).map((view) => (
          <button
            key={view}
            onClick={() => setComparisonView(view)}
            className={`px-3 py-1 rounded-md text-sm font-medium ${
              comparisonView === view
                ? 'bg-blue-100 text-blue-700'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            {view.charAt(0).toUpperCase() + view.slice(1)}
          </button>
        ))}
      </div>
      
      {comparisonView === 'bar' && (
        <div className="flex space-x-2">
          <span className="text-sm font-medium text-gray-700">Focus:</span>
          {(['quality', 'performance', 'cost'] as const).map((focus) => (
            <button
              key={focus}
              onClick={() => setMetricFocus(focus)}
              className={`px-3 py-1 rounded-md text-sm font-medium ${
                metricFocus === focus
                  ? 'bg-green-100 text-green-700'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              {focus.charAt(0).toUpperCase() + focus.slice(1)}
            </button>
          ))}
        </div>
      )}
    </div>
  );

  const renderRadarChart = () => (
    <div className="bg-white p-6 rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-4">Quality Metrics Comparison</h3>
      <ResponsiveContainer width="100%" height={400}>
        <RadarChart data={radarData}>
          <PolarGrid />
          <PolarAngleAxis dataKey="metric" />
          <PolarRadiusAxis angle={90} domain={[0, 100]} />
          {selectedExperimentData.map((exp, index) => (
            <Radar
              key={exp.id}
              name={exp.name}
              dataKey={exp.name}
              stroke={COLORS[index % COLORS.length]}
              fill={COLORS[index % COLORS.length]}
              fillOpacity={0.1}
              strokeWidth={2}
            />
          ))}
          <Legend />
          <Tooltip />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );

  const renderScatterChart = () => (
    <div className="bg-white p-6 rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-4">Cost vs Quality Analysis</h3>
      <ResponsiveContainer width="100%" height={400}>
        <ScatterChart>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            type="number" 
            dataKey="x" 
            name="Cost"
            label={{ value: 'Total Cost (USD)', position: 'insideBottom', offset: -5 }}
          />
          <YAxis 
            type="number" 
            dataKey="y" 
            name="Quality"
            label={{ value: 'LLM Judge Score', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip 
            cursor={{ strokeDasharray: '3 3' }}
            formatter={(value: number, name: string) => [
              name === 'Cost' ? `$${value.toFixed(3)}` : value.toFixed(2),
              name === 'Cost' ? 'Total Cost' : 'LLM Judge Score'
            ]}
            labelFormatter={(label, payload) => {
              if (payload && payload[0]) {
                const data = payload[0].payload;
                return `${data.name} (${data.model})`;
              }
              return '';
            }}
          />
          <Scatter 
            data={scatterData} 
            fill="#8884d8"
          >
            {scatterData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );

  const renderBarChart = () => {
    const data = getBarData();
    const keys = data.length > 0 ? Object.keys(data[0]).filter(key => key !== 'name') : [];
    
    return (
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">
          {metricFocus.charAt(0).toUpperCase() + metricFocus.slice(1)} Metrics Comparison
        </h3>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            {keys.map((key, index) => (
              <Bar 
                key={key}
                dataKey={key} 
                fill={COLORS[index % COLORS.length]}
                name={key.replace('_', ' ').toUpperCase()}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const renderStatisticalSummary = () => (
    <div className="bg-white p-6 rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-4">Statistical Summary</h3>
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Experiment
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Model
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Avg Quality Score
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Avg Latency (ms)
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Total Cost ($)
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Cost Efficiency
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {selectedExperimentData.map((exp, index) => {
              const avgQuality = (
                exp.metrics.bleu_score + 
                exp.metrics.rouge_score + 
                exp.metrics.semantic_similarity + 
                exp.metrics.llm_judge_score + 
                exp.metrics.human_rating
              ) / 5;
              const costEfficiency = avgQuality / exp.performance.total_cost_usd;
              
              return (
                <tr key={exp.id}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    {exp.name}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {exp.model_name}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {avgQuality.toFixed(3)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {exp.performance.avg_latency_ms.toFixed(0)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    ${exp.performance.total_cost_usd.toFixed(3)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {costEfficiency.toFixed(2)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );

  if (selectedExperiments.length === 0) {
    return (
      <div className="space-y-6">
        {renderExperimentSelector()}
        <div className="bg-gray-50 p-8 rounded-lg text-center">
          <p className="text-gray-500">Select experiments to compare their performance</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {renderExperimentSelector()}
      {renderViewSelector()}
      
      {comparisonView === 'radar' && renderRadarChart()}
      {comparisonView === 'scatter' && renderScatterChart()}
      {comparisonView === 'bar' && renderBarChart()}
      
      {renderStatisticalSummary()}
    </div>
  );
};

export default PerformanceComparison;