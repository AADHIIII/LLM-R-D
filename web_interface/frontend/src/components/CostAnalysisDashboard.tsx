import React, { useState, useEffect } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  Area,
  AreaChart,
} from 'recharts';

interface CostData {
  date: string;
  openai_cost: number;
  anthropic_cost: number;
  fine_tuned_cost: number;
  total_cost: number;
  token_count: number;
  request_count: number;
}

interface ModelCostBreakdown {
  model_name: string;
  total_cost: number;
  request_count: number;
  avg_cost_per_request: number;
  token_count: number;
  cost_per_token: number;
}

interface CostAnalysisDashboardProps {
  costData: CostData[];
  modelBreakdown: ModelCostBreakdown[];
  totalBudget?: number;
  currentSpend?: number;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

const CostAnalysisDashboard: React.FC<CostAnalysisDashboardProps> = ({
  costData,
  modelBreakdown,
  totalBudget = 1000,
  currentSpend = 0,
}) => {
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d'>('30d');
  const [alertThreshold, setAlertThreshold] = useState(80); // 80% of budget

  const budgetUtilization = (currentSpend / totalBudget) * 100;
  const isOverBudget = budgetUtilization > 100;
  const isNearBudget = budgetUtilization > alertThreshold;

  const filteredCostData = costData.slice(-parseInt(timeRange.replace('d', '')));

  const totalCostByProvider = [
    {
      name: 'OpenAI',
      value: costData.reduce((sum, item) => sum + item.openai_cost, 0),
      color: '#0088FE',
    },
    {
      name: 'Anthropic',
      value: costData.reduce((sum, item) => sum + item.anthropic_cost, 0),
      color: '#00C49F',
    },
    {
      name: 'Fine-tuned',
      value: costData.reduce((sum, item) => sum + item.fine_tuned_cost, 0),
      color: '#FFBB28',
    },
  ];

  const renderBudgetAlert = () => {
    if (isOverBudget) {
      return (
        <div className="bg-red-50 border border-red-200 rounded-md p-4 mb-6">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">Budget Exceeded</h3>
              <p className="text-sm text-red-700 mt-1">
                You have exceeded your budget by ${(currentSpend - totalBudget).toFixed(2)}
              </p>
            </div>
          </div>
        </div>
      );
    } else if (isNearBudget) {
      return (
        <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4 mb-6">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-yellow-800">Budget Warning</h3>
              <p className="text-sm text-yellow-700 mt-1">
                You have used {budgetUtilization.toFixed(1)}% of your budget
              </p>
            </div>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-6">
      {renderBudgetAlert()}

      {/* Budget Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="w-8 h-8 bg-blue-500 rounded-md flex items-center justify-center">
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1" />
                </svg>
              </div>
            </div>
            <div className="ml-5 w-0 flex-1">
              <dl>
                <dt className="text-sm font-medium text-gray-500 truncate">Total Budget</dt>
                <dd className="text-lg font-medium text-gray-900">${totalBudget.toFixed(2)}</dd>
              </dl>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className={`w-8 h-8 ${isOverBudget ? 'bg-red-500' : 'bg-green-500'} rounded-md flex items-center justify-center`}>
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                </svg>
              </div>
            </div>
            <div className="ml-5 w-0 flex-1">
              <dl>
                <dt className="text-sm font-medium text-gray-500 truncate">Current Spend</dt>
                <dd className={`text-lg font-medium ${isOverBudget ? 'text-red-600' : 'text-gray-900'}`}>
                  ${currentSpend.toFixed(2)}
                </dd>
              </dl>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="w-8 h-8 bg-yellow-500 rounded-md flex items-center justify-center">
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
            </div>
            <div className="ml-5 w-0 flex-1">
              <dl>
                <dt className="text-sm font-medium text-gray-500 truncate">Remaining</dt>
                <dd className="text-lg font-medium text-gray-900">
                  ${Math.max(0, totalBudget - currentSpend).toFixed(2)}
                </dd>
              </dl>
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="w-8 h-8 bg-purple-500 rounded-md flex items-center justify-center">
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                </svg>
              </div>
            </div>
            <div className="ml-5 w-0 flex-1">
              <dl>
                <dt className="text-sm font-medium text-gray-500 truncate">Utilization</dt>
                <dd className={`text-lg font-medium ${isOverBudget ? 'text-red-600' : 'text-gray-900'}`}>
                  {budgetUtilization.toFixed(1)}%
                </dd>
              </dl>
            </div>
          </div>
        </div>
      </div>

      {/* Time Range Selector */}
      <div className="bg-white p-6 rounded-lg shadow">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold">Cost Trends</h3>
          <div className="flex space-x-2">
            {(['7d', '30d', '90d'] as const).map((range) => (
              <button
                key={range}
                onClick={() => setTimeRange(range)}
                className={`px-3 py-1 rounded-md text-sm font-medium ${
                  timeRange === range
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                {range}
              </button>
            ))}
          </div>
        </div>

        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={filteredCostData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis label={{ value: 'Cost ($)', angle: -90, position: 'insideLeft' }} />
            <Tooltip formatter={(value: number) => [`$${value.toFixed(3)}`, 'Cost']} />
            <Legend />
            <Area
              type="monotone"
              dataKey="openai_cost"
              stackId="1"
              stroke="#0088FE"
              fill="#0088FE"
              name="OpenAI"
            />
            <Area
              type="monotone"
              dataKey="anthropic_cost"
              stackId="1"
              stroke="#00C49F"
              fill="#00C49F"
              name="Anthropic"
            />
            <Area
              type="monotone"
              dataKey="fine_tuned_cost"
              stackId="1"
              stroke="#FFBB28"
              fill="#FFBB28"
              name="Fine-tuned"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Cost Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Cost by Provider</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={totalCostByProvider}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, value }) => `${name}: $${value?.toFixed(2) || '0.00'}`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {totalCostByProvider.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip formatter={(value: number) => [`$${value.toFixed(3)}`, 'Cost']} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Model Cost Efficiency</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={modelBreakdown}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="model_name" 
                angle={-45}
                textAnchor="end"
                height={100}
              />
              <YAxis label={{ value: 'Cost per Token', angle: -90, position: 'insideLeft' }} />
              <Tooltip 
                formatter={(value: number) => [`$${value.toFixed(6)}`, 'Cost per Token']}
              />
              <Bar dataKey="cost_per_token" fill="#8884d8" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Detailed Model Breakdown Table */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Detailed Model Breakdown</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Model
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Total Cost
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Requests
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Avg Cost/Request
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Tokens
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Cost/Token
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {modelBreakdown.map((model, index) => (
                <tr key={index}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    {model.model_name}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    ${model.total_cost.toFixed(3)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {model.request_count.toLocaleString()}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    ${model.avg_cost_per_request.toFixed(4)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {model.token_count.toLocaleString()}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    ${model.cost_per_token.toFixed(6)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default CostAnalysisDashboard;