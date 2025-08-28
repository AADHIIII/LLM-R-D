import axios from 'axios';

// API configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api/v1';

// Create axios instance
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('authToken');
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized access
      localStorage.removeItem('authToken');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// API service types
export interface GenerationRequest {
  prompt: string;
  model_type: string;
  max_tokens?: number;
  temperature?: number;
}

export interface GenerationResponse {
  response: string;
  model: string;
  metadata: {
    tokens_used: number;
    latency_ms: number;
    cost_usd?: number;
  };
}

export interface Model {
  id: string;
  name: string;
  type: 'fine-tuned' | 'commercial';
  base_model?: string;
  created_at: string;
}

export interface Experiment {
  id: string;
  name: string;
  description: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  created_at: string;
  updated_at: string;
}

// API service methods
export const apiService = {
  // Text generation
  generateText: async (request: GenerationRequest): Promise<GenerationResponse> => {
    const response = await apiClient.post('/generate', request);
    return response.data as GenerationResponse;
  },

  // Models
  getModels: async (): Promise<Model[]> => {
    const response = await apiClient.get('/models');
    return response.data as Model[];
  },

  // Experiments
  getExperiments: async (): Promise<Experiment[]> => {
    const response = await apiClient.get('/experiments');
    return response.data as Experiment[];
  },

  createExperiment: async (experiment: Partial<Experiment>): Promise<Experiment> => {
    const response = await apiClient.post('/experiments', experiment);
    return response.data as Experiment;
  },

  getExperiment: async (id: string): Promise<Experiment> => {
    const response = await apiClient.get(`/experiments/${id}`);
    return response.data as Experiment;
  },

  // Health check
  healthCheck: async (): Promise<{ status: string; timestamp: string }> => {
    const response = await apiClient.get('/health');
    return response.data as { status: string; timestamp: string };
  },
};

export default apiService;