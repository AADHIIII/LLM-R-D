import React, { useState, useEffect } from 'react';
import Card from './Card';

interface SystemHealth {
  status: string;
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
}

interface APIPerformance {
  total_requests: number;
  avg_response_time: number;
  error_rate: number;
  unique_endpoints: number;
}

interface AlertSummary {
  total_active: number;
  by_severity: Record<string, number>;
  critical_count: number;
  high_count: number;
}

interface DashboardOverview {
  timestamp: string;
  system_health: SystemHealth;
  api_performance: APIPerformance;
  alerts: AlertSummary;
  trends: Record<string, any>;
}

interface Alert {
  id: string;
  name: string;
  description: string;
  severity: string;
  status: string;
  created_at: string;
  updated_at: string;
}

interface MetricPoint {
  timestamp: string;
  cpu_percent: number;
  memory_percent: number;
  disk_usage_percent: number;
}

const MonitoringDashboard: React.FC = () => {
  const [overview, setOverview] = useState<DashboardOverview | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [systemMetrics, setSystemMetrics] = useState<MetricPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshInterval, setRefreshInterval] = useState<NodeJS.Timeout | null>(null);

  const fetchDashboardData = async () => {
    try {
      const [overviewRes, alertsRes, metricsRes] = await Promise.all([
        fetch('/api/v1/monitoring/dashboard/overview'),
        fetch('/api/v1/monitoring/alerts'),
        fetch('/api/v1/monitoring/metrics/system?minutes=60')
      ]);

      if (!overviewRes.ok || !alertsRes.ok || !metricsRes.ok) {
        throw new Error('Failed to fetch monitoring data');
      }

      const overviewData = await overviewRes.json();
      const alertsData = await alertsRes.json();
      const metricsData = await metricsRes.json();

      setOverview(overviewData.data);
      setAlerts(alertsData.data);
      setSystemMetrics(metricsData.data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const acknowledgeAlert = async (alertId: string) => {
    try {
      const response = await fetch(`/api/v1/monitoring/alerts/${alertId}/acknowledge`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to acknowledge alert');
      }

      // Refresh alerts after acknowledging
      fetchDashboardData();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to acknowledge alert');
    }
  };

  useEffect(() => {
    fetchDashboardData();

    // Set up auto-refresh every 30 seconds
    const interval = setInterval(fetchDashboardData, 30000);
    setRefreshInterval(interval);

    return () => {
      if (refreshInterval) {
        clearInterval(refreshInterval);
      }
    };
  }, []);

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'healthy':
        return 'text-green-600';
      case 'warning':
        return 'text-yellow-600';
      case 'critical':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'low':
        return 'bg-blue-100 text-blue-800';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800';
      case 'high':
        return 'bg-orange-100 text-orange-800';
      case 'critical':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-md p-4">
        <div className="flex">
          <div className="ml-3">
            <h3 className="text-sm font-medium text-red-800">Error</h3>
            <div className="mt-2 text-sm text-red-700">
              <p>{error}</p>
            </div>
            <div className="mt-4">
              <button
                onClick={fetchDashboardData}
                className="bg-red-100 px-3 py-2 rounded-md text-sm font-medium text-red-800 hover:bg-red-200"
              >
                Retry
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-gray-900">Monitoring Dashboard</h1>
        <div className="flex items-center space-x-4">
          <span className="text-sm text-gray-500">
            Last updated: {overview ? formatTimestamp(overview.timestamp) : 'Never'}
          </span>
          <button
            onClick={fetchDashboardData}
            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* System Health Overview */}
      {overview && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <Card>
            <div className="p-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                    overview.system_health.status === 'healthy' ? 'bg-green-100' : 'bg-yellow-100'
                  }`}>
                    <span className={`text-sm font-medium ${getStatusColor(overview.system_health.status)}`}>
                      {overview.system_health.status === 'healthy' ? '✓' : '⚠'}
                    </span>
                  </div>
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">System Status</dt>
                    <dd className={`text-lg font-medium ${getStatusColor(overview.system_health.status)}`}>
                      {overview.system_health.status.charAt(0).toUpperCase() + overview.system_health.status.slice(1)}
                    </dd>
                  </dl>
                </div>
              </div>
            </div>
          </Card>

          <Card>
            <div className="p-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                    <span className="text-sm font-medium text-blue-600">CPU</span>
                  </div>
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">CPU Usage</dt>
                    <dd className="text-lg font-medium text-gray-900">
                      {overview.system_health.cpu_usage.toFixed(1)}%
                    </dd>
                  </dl>
                </div>
              </div>
            </div>
          </Card>

          <Card>
            <div className="p-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <div className="w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center">
                    <span className="text-sm font-medium text-purple-600">MEM</span>
                  </div>
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">Memory Usage</dt>
                    <dd className="text-lg font-medium text-gray-900">
                      {overview.system_health.memory_usage.toFixed(1)}%
                    </dd>
                  </dl>
                </div>
              </div>
            </div>
          </Card>

          <Card>
            <div className="p-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <div className="w-8 h-8 bg-indigo-100 rounded-full flex items-center justify-center">
                    <span className="text-sm font-medium text-indigo-600">API</span>
                  </div>
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">Error Rate</dt>
                    <dd className="text-lg font-medium text-gray-900">
                      {overview.api_performance.error_rate.toFixed(1)}%
                    </dd>
                  </dl>
                </div>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* API Performance */}
      {overview && (
        <Card>
          <div className="p-6">
            <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">API Performance</h3>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div>
                <dt className="text-sm font-medium text-gray-500">Total Requests</dt>
                <dd className="mt-1 text-2xl font-semibold text-gray-900">
                  {overview.api_performance.total_requests.toLocaleString()}
                </dd>
              </div>
              <div>
                <dt className="text-sm font-medium text-gray-500">Avg Response Time</dt>
                <dd className="mt-1 text-2xl font-semibold text-gray-900">
                  {overview.api_performance.avg_response_time.toFixed(0)}ms
                </dd>
              </div>
              <div>
                <dt className="text-sm font-medium text-gray-500">Error Rate</dt>
                <dd className="mt-1 text-2xl font-semibold text-gray-900">
                  {overview.api_performance.error_rate.toFixed(1)}%
                </dd>
              </div>
              <div>
                <dt className="text-sm font-medium text-gray-500">Unique Endpoints</dt>
                <dd className="mt-1 text-2xl font-semibold text-gray-900">
                  {overview.api_performance.unique_endpoints}
                </dd>
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Active Alerts */}
      <Card>
        <div className="p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg leading-6 font-medium text-gray-900">Active Alerts</h3>
            {overview && (
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                {overview.alerts.total_active} active
              </span>
            )}
          </div>
          
          {alerts.length === 0 ? (
            <div className="text-center py-8">
              <div className="text-green-500 text-4xl mb-2">✓</div>
              <p className="text-gray-500">No active alerts</p>
            </div>
          ) : (
            <div className="space-y-3">
              {alerts.map((alert) => (
                <div key={alert.id} className="border border-gray-200 rounded-lg p-4">
                  <div className="flex justify-between items-start">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getSeverityColor(alert.severity)}`}>
                          {alert.severity.toUpperCase()}
                        </span>
                        <h4 className="text-sm font-medium text-gray-900">{alert.name}</h4>
                      </div>
                      <p className="mt-1 text-sm text-gray-600">{alert.description}</p>
                      <p className="mt-1 text-xs text-gray-500">
                        Created: {formatTimestamp(alert.created_at)}
                      </p>
                    </div>
                    <div className="flex space-x-2">
                      {alert.status === 'active' && (
                        <button
                          onClick={() => acknowledgeAlert(alert.id)}
                          className="inline-flex items-center px-3 py-1 border border-transparent text-xs font-medium rounded text-blue-700 bg-blue-100 hover:bg-blue-200"
                        >
                          Acknowledge
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </Card>

      {/* System Metrics Chart */}
      <Card>
        <div className="p-6">
          <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">System Metrics (Last Hour)</h3>
          {systemMetrics.length > 0 ? (
            <div className="h-64 flex items-end space-x-1">
              {systemMetrics.slice(-20).map((metric, index) => (
                <div key={index} className="flex-1 flex flex-col items-center space-y-1">
                  <div className="w-full bg-gray-200 rounded-full h-32 flex flex-col justify-end">
                    <div
                      className="bg-blue-500 rounded-full"
                      style={{ height: `${metric.cpu_percent}%` }}
                      title={`CPU: ${metric.cpu_percent.toFixed(1)}%`}
                    ></div>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-32 flex flex-col justify-end">
                    <div
                      className="bg-purple-500 rounded-full"
                      style={{ height: `${metric.memory_percent}%` }}
                      title={`Memory: ${metric.memory_percent.toFixed(1)}%`}
                    ></div>
                  </div>
                  <span className="text-xs text-gray-500 transform rotate-45 origin-left">
                    {new Date(metric.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              No metrics data available
            </div>
          )}
          <div className="mt-4 flex justify-center space-x-6">
            <div className="flex items-center">
              <div className="w-3 h-3 bg-blue-500 rounded-full mr-2"></div>
              <span className="text-sm text-gray-600">CPU Usage</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-3 bg-purple-500 rounded-full mr-2"></div>
              <span className="text-sm text-gray-600">Memory Usage</span>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default MonitoringDashboard;