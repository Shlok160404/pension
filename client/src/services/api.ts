// API service for Pension AI backend
const API_BASE_URL = 'http://4.240.80.252';

// Types for API responses
export interface LoginResponse {
  access_token: string;
  token_type: string;
  user_id: number;
  role: string;
  full_name: string;
}

export interface SignupRequest {
  full_name: string;
  email: string;
  password: string;
  role: string;
}

export interface UserResponse {
  id: number;
  full_name: string;
  email: string;
  role: string;
}

export interface PromptRequest {
  query: string;
}

export interface PromptResponse {
  summary: string;
  chart_data?: any;
  plotly_figures?: any;
  chart_images?: any;
  metadata?: any;
  data_source?: string;
  search_type?: string;
  pdf_status?: string;
}

// PDF Upload Types
export interface PDFUploadResponse {
  status: string;
  filename: string;
  message: string;
}

// Advisor Dashboard Types
export interface AdvisorClient {
  user_id: number;
  full_name: string;
  age: number;
  annual_income: number;
  current_savings: number;
  risk_tolerance: string;
  anomaly_score: number;
}

export interface AdvisorDashboardData {
  advisor_id: number;
  total_clients: number;
  grouped_data: {
    by_risk_tolerance: Record<string, { count: number; clients: AdvisorClient[] }>;
    by_age_group: Record<string, { count: number; clients: AdvisorClient[] }>;
    by_income_level: Record<string, { count: number; clients: AdvisorClient[] }>;
  };
  summary: {
    total_clients: number;
    avg_age: number;
    avg_income: number;
    avg_savings: number;
    risk_distribution: Record<string, number>;
    fraud_risk_summary: { high: number; medium: number; low: number };
  };
}

export interface ClientDetails {
  client: {
    id: number;
    full_name: string;
    email: string;
  };
  pension_data: {
    demographics: any;
    financial: any;
    pension: any;
    investment: any;
    risk_indicators: any;
  };
}

// JWT Token Management
class TokenManager {
  private static TOKEN_KEY = 'pension_ai_token';
  private static USER_KEY = 'pension_ai_user';

  static setToken(token: string): void {
    localStorage.setItem(this.TOKEN_KEY, token);
  }

  static getToken(): string | null {
    return localStorage.getItem(this.TOKEN_KEY);
  }

  static removeToken(): void {
    localStorage.removeItem(this.TOKEN_KEY);
  }

  static setUser(user: Omit<LoginResponse, 'access_token' | 'token_type'>): void {
    localStorage.setItem(this.USER_KEY, JSON.stringify(user));
  }

  static getUser(): Omit<LoginResponse, 'access_token' | 'token_type'> | null {
    const user = localStorage.getItem(this.USER_KEY);
    return user ? JSON.parse(user) : null;
  }

  static removeUser(): void {
    localStorage.removeItem(this.USER_KEY);
  }

  static isAuthenticated(): boolean {
    return !!this.getToken();
  }

  static logout(): void {
    this.removeToken();
    this.removeUser();
  }
}

// API Client with authentication
class ApiClient {
  private baseURL: string;

  constructor(baseURL: string) {
    this.baseURL = baseURL;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
    // Add auth header if token exists
    const token = TokenManager.getToken();
    if (token) {
      options.headers = {
        ...options.headers,
        'Authorization': `Bearer ${token}`,
      };
    }

    // Add content type for JSON requests
    if (options.body && !(options.body instanceof FormData)) {
      options.headers = {
        ...options.headers,
        'Content-Type': 'application/json',
      };
    }

    try {
      const response = await fetch(url, options);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  // Authentication endpoints
  async login(email: string, password: string): Promise<LoginResponse> {
    const formData = new FormData();
    formData.append('username', email); // OAuth2 expects 'username'
    formData.append('password', password);

    const response = await this.request<LoginResponse>('/login', {
      method: 'POST',
      body: formData,
    });

    // Store token and user info
    TokenManager.setToken(response.access_token);
    TokenManager.setUser({
      user_id: response.user_id,
      role: response.role,
      full_name: response.full_name,
    });

    return response;
  }

  async signup(userData: SignupRequest): Promise<UserResponse> {
    return this.request<UserResponse>('/signup', {
      method: 'POST',
      body: JSON.stringify(userData),
    });
  }

  // AI Query endpoint
  async processPrompt(query: string): Promise<PromptResponse> {
    return this.request<PromptResponse>('/prompt', {
      method: 'POST',
      body: JSON.stringify({ query }),
    });
  }

  // User profile
  async getCurrentUser(): Promise<UserResponse> {
    return this.request<UserResponse>('/users/me');
  }

  // Dashboard data
  async getDashboardAnalytics(category?: string): Promise<any> {
    const endpoint = category ? `/dashboard/analytics?category=${category}` : '/dashboard/analytics';
    return this.request(endpoint);
  }

  // Advisor-specific endpoints
  async getAdvisorDashboard(): Promise<AdvisorDashboardData> {
    return this.request<AdvisorDashboardData>('/advisor/dashboard');
  }

  async getAdvisorClientDetails(clientId: number): Promise<ClientDetails> {
    return this.request<ClientDetails>(`/advisor/client/${clientId}/details`);
  }

  // Regulator-specific endpoints
  async getRegulatorDashboard(): Promise<any> {
    return this.request('/regulator/dashboard');
  }

  // Get all users for regulator oversight
  async getAllUsers(): Promise<any> {
    return this.request('/users/all');
  }

  // Resident-specific endpoints
  async getResidentDashboard(): Promise<any> {
    return this.request('/resident/dashboard');
  }

  // PDF Upload endpoint
  async uploadPDF(file: File): Promise<PDFUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseURL}/pension/me/upload_document`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${TokenManager.getToken()}`,
      },
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `PDF upload failed! status: ${response.status}`);
    }

    return await response.json();
  }

  // Health check
  async healthCheck(): Promise<any> {
    return this.request('/health');
  }
}

// Export singleton instance
export const apiClient = new ApiClient(API_BASE_URL);

// Export token manager for use in components
export const tokenManager = TokenManager;


