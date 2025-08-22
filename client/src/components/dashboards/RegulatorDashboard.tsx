import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { apiClient, tokenManager } from '../../services/api';
import { 
  Shield, 
  Users, 
  TrendingUp, 
  AlertTriangle, 
  BarChart3, 
  PieChart, 
  Filter,
  RefreshCw,
  Send,

} from 'lucide-react';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  chartData?: any;
}

interface RegulatorData {
  regulator_id: number;
  total_users: number;
  grouped_data: {
    by_risk_tolerance: Record<string, { count: number; users: RegulatorUser[] }>;
    by_age_group: Record<string, { count: number; users: RegulatorUser[] }>;
    by_income_level: Record<string, { count: number; users: RegulatorUser[] }>;
  };
  summary: {
    total_users: number;
    avg_age: number;
    avg_income: number;
    avg_savings: number;
    risk_distribution: Record<string, number>;
    fraud_risk_summary: { high: number; medium: number; low: number };
  };
}

interface RegulatorUser {
  user_id: number;
  full_name: string;
  age: number;
  annual_income: number;
  current_savings: number;
  risk_tolerance: string;
  anomaly_score: number;
  role: string;
}

const RegulatorDashboard: React.FC = () => {
  const [regulatorData, setRegulatorData] = useState<RegulatorData | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [loadingDashboard, setLoadingDashboard] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Simple filters
  const [selectedRiskLevel, setSelectedRiskLevel] = useState<string>('all');
  const [selectedAgeGroup, setSelectedAgeGroup] = useState<string>('all');
  const [selectedIncomeLevel, setSelectedIncomeLevel] = useState<string>('all');
  const [currentUser, setCurrentUser] = useState<any>(null);

  const quickQueries = [
    "Show high-risk members",
    "Fraud detection summary",
    "Geographic risk analysis",
    "Portfolio performance trends",
    "Suspicious activity report",
    "Compliance overview"
  ];

  // Load regulator dashboard data on component mount
  useEffect(() => {
    loadRegulatorDashboard();
  }, []);

  // Get current user on component mount
  useEffect(() => {
    const user = tokenManager.getUser();
    setCurrentUser(user);
  }, []);

  // Initialize welcome message
  useEffect(() => {
    if (regulatorData) {
      setMessages([
        {
          id: Date.now().toString(),
          type: 'assistant',
          content: `Welcome to the Regulatory Oversight Dashboard! I can help you analyze ${regulatorData.total_users} members across the pension system. What would you like to investigate?`,
          timestamp: new Date()
        }
      ]);
    }
  }, [regulatorData]);

  const handleLogout = () => {
    tokenManager.logout();
    // Redirect to landing page or refresh the page
    window.location.href = '/';
  };

  const loadRegulatorDashboard = async () => {
    try {
      setLoadingDashboard(true);
      setError(null);
      const data = await apiClient.getRegulatorDashboard();
      setRegulatorData(data);
    } catch (err) {
      setError('Failed to load regulatory dashboard data. Please try again.');
      console.error('Error loading regulator dashboard:', err);
    } finally {
      setLoadingDashboard(false);
    }
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await apiClient.processPrompt(inputValue);
      
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: response.summary,
        timestamp: new Date(),
        chartData: response.chart_data,
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuickQuery = (query: string) => {
    setInputValue(query);
  };

  if (loadingDashboard) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <h2 className="text-xl font-semibold text-gray-800">Loading Regulatory Dashboard</h2>
          <p className="text-gray-600">Please wait...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-50">
        <div className="text-center">
          <AlertTriangle className="h-16 w-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-800 mb-2">Error Loading Dashboard</h2>
          <p className="text-gray-600 mb-4">{error}</p>
          <button
            onClick={loadRegulatorDashboard}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <RefreshCw className="h-5 w-5 inline mr-2" />
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!regulatorData) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-50">
        <div className="text-center">
          <Shield className="h-16 w-16 text-gray-400 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-800 mb-2">No Regulatory Data Available</h2>
          <p className="text-gray-600">Unable to load regulatory oversight data.</p>
        </div>
      </div>
    );
  }

  // Prepare data for interactive charts
  const riskData = Object.entries(regulatorData.summary.risk_distribution).map(([risk, count]) => ({
    risk,
    count,
    color: risk.toLowerCase().includes('high') ? '#ef4444' : 
           risk.toLowerCase().includes('medium') ? '#f59e0b' : '#10b981'
  }));

  const ageGroups = Object.entries(regulatorData.grouped_data.by_age_group).map(([age, data]) => ({
    age,
    count: data.count,
    avgIncome: data.users.reduce((sum, user) => sum + (user.annual_income || 0), 0) / data.count,
    avgSavings: data.users.reduce((sum, user) => sum + (user.current_savings || 0), 0) / data.count
  }));

  const incomeGroups = Object.entries(regulatorData.grouped_data.by_income_level).map(([income, data]) => ({
    income,
    count: data.count,
    avgAge: data.users.reduce((sum, user) => sum + (user.age || 0), 0) / data.count,
    avgSavings: data.users.reduce((sum, user) => sum + (user.current_savings || 0), 0) / data.count
  }));

  const fraudRiskData = [
    { risk: 'High Risk', count: regulatorData.summary.fraud_risk_summary.high, color: '#ef4444' },
    { risk: 'Medium Risk', count: regulatorData.summary.fraud_risk_summary.medium, color: '#f59e0b' },
    { risk: 'Low Risk', count: regulatorData.summary.fraud_risk_summary.low, color: '#10b981' }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Compact Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="h-10 w-10 bg-blue-600 rounded-lg flex items-center justify-center">
                <Shield className="h-5 w-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">Regulatory Oversight Dashboard</h1>
                <p className="text-sm text-gray-600">Comprehensive pension system monitoring and fraud detection</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              {/* Status Indicator */}
              <div className="flex items-center space-x-2 px-3 py-2 bg-green-50 rounded-full border border-green-200">
                <div className="h-2 w-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-sm text-green-700 font-medium">Connected</span>
              </div>
              
              {/* Dashboard Stats */}
              <div className="text-center">
                <p className="text-xs text-gray-500">Total Members</p>
                <p className="text-xl font-bold text-blue-600">{regulatorData.total_users.toLocaleString()}</p>
              </div>
              <div className="text-center">
                <p className="text-xs text-gray-500">High Risk</p>
                <p className="text-xl font-bold text-red-600">{regulatorData.summary.fraud_risk_summary.high}</p>
              </div>
              <div className="text-center">
                <p className="text-xs text-gray-500">Avg Portfolio</p>
                <p className="text-lg font-bold text-green-600">£{(regulatorData.summary.avg_savings / 1000).toFixed(0)}k</p>
              </div>

              {/* User Details and Logout */}
              <div className="flex items-center space-x-3">
                <div className="text-right">
                  <p className="text-sm font-medium text-gray-900">
                    Welcome, {currentUser?.full_name || currentUser?.username || 'Regulator'}
                  </p>
                  <p className="text-xs text-gray-500">
                    ID: {currentUser?.id || 'N/A'}
                  </p>
                </div>
                <button
                  onClick={handleLogout}
                  className="flex items-center space-x-2 px-3 py-2 bg-red-50 text-red-600 rounded-lg hover:bg-red-100 transition-colors duration-200"
                >
                  <Shield className="h-4 w-4" />
                  <span className="text-sm font-medium">Logout</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Compact Filters */}
      <div className="bg-white shadow-sm border-b border-gray-200 px-4 py-3">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <Filter className="h-4 w-4 text-gray-500" />
            <span className="text-sm font-medium text-gray-700">Filters:</span>
          </div>
          
          <select
            value={selectedRiskLevel}
            onChange={(e) => setSelectedRiskLevel(e.target.value)}
            className="px-2 py-1 border border-gray-300 rounded text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="all">All Risk Levels</option>
            <option value="high">High Risk</option>
            <option value="medium">Medium Risk</option>
            <option value="low">Low Risk</option>
          </select>

          <select
            value={selectedAgeGroup}
            onChange={(e) => setSelectedAgeGroup(e.target.value)}
            className="px-2 py-1 border border-gray-300 rounded text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="all">All Age Groups</option>
            <option value="18-30">18-30</option>
            <option value="31-45">31-45</option>
            <option value="46-60">46-60</option>
            <option value="60+">60+</option>
          </select>

          <select
            value={selectedIncomeLevel}
            onChange={(e) => setSelectedIncomeLevel(e.target.value)}
            className="px-2 py-1 border border-gray-300 rounded text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="all">All Income Levels</option>
            <option value="low">Low (&lt;£30k)</option>
            <option value="medium">Medium (£30k-£70k)</option>
            <option value="high">High (&gt;£70k)</option>
          </select>
        </div>
      </div>

      {/* Compact Key Metrics */}
      <div className="px-4 py-4">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
          {/* Average Age Card */}
          <div className="bg-white rounded-xl p-4 shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between mb-3">
              <div className="h-12 w-12 bg-blue-50 rounded-xl flex items-center justify-center">
                <Users className="h-6 w-6 text-blue-600" />
              </div>
              <div className="text-right">
                <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">Average Age</p>
                <p className="text-2xl font-bold text-gray-900">{regulatorData.summary.avg_age}</p>
              </div>
            </div>
            <div className="flex items-center text-sm text-gray-600">
              <span className="text-blue-600 font-medium">Years</span>
            </div>
          </div>

          {/* Average Income Card */}
          <div className="bg-white rounded-xl p-4 shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between mb-3">
              <div className="h-12 w-12 bg-green-50 rounded-xl flex items-center justify-center">
                <TrendingUp className="h-6 w-6 text-green-600" />
              </div>
              <div className="text-right">
                <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">Average Income</p>
                <p className="text-2xl font-bold text-gray-900">£{(regulatorData.summary.avg_income / 1000).toFixed(0)}k</p>
              </div>
            </div>
            <div className="flex items-center text-sm text-gray-600">
              <span className="text-green-600 font-medium">Annual</span>
            </div>
          </div>
              
          {/* Suspicious Transactions Card */}
          <div className="bg-white rounded-xl p-4 shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between mb-3">
              <div className="h-12 w-12 bg-red-50 rounded-xl flex items-center justify-center">
                <AlertTriangle className="h-6 w-6 text-red-600" />
              </div>
              <div className="text-right">
                <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">Suspicious</p>
                <p className="text-2xl font-bold text-red-600">{regulatorData.summary.fraud_risk_summary.medium + regulatorData.summary.fraud_risk_summary.high}</p>
              </div>
            </div>
            <div className="flex items-center text-sm text-gray-600">
              <span className="text-red-600 font-medium">Transactions</span>
            </div>
          </div>

          {/* Average Savings Card */}
          <div className="bg-white rounded-xl p-4 shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
            <div className="flex items-center justify-between mb-3">
              <div className="h-12 w-12 bg-purple-50 rounded-xl flex items-center justify-center">
                <BarChart3 className="h-6 w-6 text-purple-600" />
              </div>
              <div className="text-right">
                <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">Average Savings</p>
                <p className="text-2xl font-bold text-gray-900">£{(regulatorData.summary.avg_savings / 1000).toFixed(0)}k</p>
              </div>
            </div>
            <div className="flex items-center text-sm text-gray-600">
              <span className="text-purple-600 font-medium">Portfolio</span>
            </div>
          </div>
        </div>
            
        {/* All 4 Charts in One View - 2x2 Grid - Fixed TypeScript errors */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
          {/* Risk Distribution Chart */}
          <div className="bg-white rounded-lg p-3 shadow-sm border border-gray-200">
            <div className="flex items-center space-x-2 mb-2">
              <BarChart3 className="h-4 w-4 text-blue-600" />
              <h3 className="text-sm font-semibold text-gray-900">Risk Distribution</h3>
            </div>
            <div className="h-48">
              <Plot
                data={[
                  {
                    x: riskData.map(d => d.risk),
                    y: riskData.map(d => d.count),
                    type: 'bar',
                    marker: {
                      color: riskData.map(d => d.color),
                      line: { width: 2, color: 'white' }
                    },
                    hovertemplate: '<b>%{x}</b><br>Members: %{y}<extra></extra>',
                    hoverlabel: { bgcolor: 'white', font: { color: 'black' } }
                  }
                ]}
                layout={{
                  xaxis: { title: { text: '' }, showgrid: false },
                  yaxis: { title: { text: '' }, showgrid: true, gridcolor: '#e5e7eb' },
                  plot_bgcolor: 'rgba(0,0,0,0)',
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  margin: { t: 20, b: 30, l: 40, r: 20 },
                  showlegend: false,
                  hovermode: 'closest'
              }}
              config={{ displayModeBar: false, displaylogo: false }}
              style={{ width: '100%', height: '100%' }}
              />
            </div>
          </div>

          {/* Fraud Risk Summary Chart */}
          <div className="bg-white rounded-lg p-3 shadow-sm border border-gray-200">
            <div className="flex items-center space-x-2 mb-2">
              <PieChart className="h-4 w-4 text-green-600" />
              <h3 className="text-sm font-semibold text-gray-900">Fraud Risk</h3>
            </div>
            <div className="relative h-48 w-full">
              <div className="absolute inset-0 flex items-center justify-center">
                <Plot
                  data={[
                    {
                      labels: fraudRiskData.map(d => d.risk),
                      values: fraudRiskData.map(d => d.count),
                      type: 'pie',
                      marker: {
                        colors: fraudRiskData.map(d => d.color),
                        line: { width: 1, color: 'white' }
                      },
                      hovertemplate: '<b>%{label}</b><br>Members: %{value}<br>Percentage: %{percent}<extra></extra>',
                      hoverlabel: { bgcolor: 'white', font: { color: 'black' } },
                      textinfo: 'label+percent',
                      textposition: 'inside',
                      textfont: { size: 10 }
                    }
                  ]}
                  layout={{
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    margin: { t: 0, b: 0, l: 0, r: 0 },
                    showlegend: false,
                    hovermode: 'closest',
                    autosize: false,
                    width: 180,
                    height: 180
                  }}
                  config={{ displayModeBar: false, displaylogo: false }}
                  style={{ width: '180px', height: '180px' }}
                  useResizeHandler={false}
                />
              </div>
            </div>
          </div>

          {/* Age Group Analysis */}
          <div className="bg-white rounded-lg p-3 shadow-sm border border-gray-200">
            <div className="flex items-center space-x-2 mb-2">
              <Users className="h-4 w-4 text-purple-600" />
              <h3 className="text-sm font-semibold text-gray-900">Age Analysis</h3>
            </div>
            <div className="h-48">
              <Plot
                data={[
                  {
                    x: ageGroups.map(d => d.age),
                    y: ageGroups.map(d => d.avgIncome / 1000),
                    type: 'bar',
                    name: 'Avg Income (£k)',
                    marker: { color: '#3b82f6' },
                    yaxis: 'y'
                  },
                  {
                    x: ageGroups.map(d => d.age),
                    y: ageGroups.map(d => d.avgSavings / 1000),
                    type: 'bar',
                    name: 'Avg Savings (£k)',
                    marker: { color: '#10b981' },
                    yaxis: 'y2'
                  }
                ]}
                layout={{
                  xaxis: { title: { text: '' }, showgrid: false },
                  yaxis: { title: { text: '' }, side: 'left', showgrid: true, gridcolor: '#e5e7eb' },
                  yaxis2: { title: { text: '' }, side: 'right', overlaying: 'y', showgrid: false },
                  plot_bgcolor: 'rgba(0,0,0,0)',
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  margin: { t: 20, b: 30, l: 40, r: 40 },
                  hovermode: 'closest',
                  showlegend: false
              }}
              config={{ displayModeBar: false, displaylogo: false }}
              style={{ width: '100%', height: '100%' }}
              />
            </div>
          </div>

          {/* Income Level Analysis */}
          <div className="bg-white rounded-lg p-3 shadow-sm border border-gray-200">
            <div className="flex items-center space-x-2 mb-2">
              <TrendingUp className="h-4 w-4 text-orange-600" />
              <h3 className="text-sm font-semibold text-gray-900">Income Analysis</h3>
            </div>
            <div className="h-48">
              <Plot
                data={[
                  {
                    x: incomeGroups.map(d => d.income),
                    y: incomeGroups.map(d => d.avgAge),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Average Age',
                    marker: { size: 10, color: '#f59e0b' },
                    line: { width: 3 }
                  },
                  {
                    x: incomeGroups.map(d => d.income),
                    y: incomeGroups.map(d => d.avgSavings / 1000),
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Average Savings (£k)',
                    marker: { size: 10, color: '#8b5cf6' },
                    line: { width: 3 },
                    yaxis: 'y2'
                  }
                ]}
                layout={{
                  xaxis: { title: { text: '' }, showgrid: false },
                  yaxis: { title: { text: '' }, showgrid: true, gridcolor: '#e5e7eb' },
                  yaxis2: { title: { text: '' }, side: 'right', overlaying: 'y', showgrid: false },
                  plot_bgcolor: 'rgba(0,0,0,0)',
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  margin: { t: 20, b: 30, l: 40, r: 40 },
                  hovermode: 'closest',
                  showlegend: false
              }}
              config={{ displayModeBar: false, displaylogo: false }}
              style={{ width: '100%', height: '100%' }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Compact AI Assistant Chat Box */}
      <div className="px-4 pb-4">
        <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-4">
          <div className="flex items-center space-x-3 mb-4">
            <div className="h-8 w-8 bg-blue-600 rounded-lg flex items-center justify-center">
              <Shield className="h-4 w-4 text-white" />
            </div>
            <div>
              <h3 className="text-base font-semibold text-gray-900">AI Regulatory Assistant</h3>
              <p className="text-sm text-gray-600">Ask questions about system health and compliance</p>
            </div>
          </div>

          {/* Compact Quick Action Buttons */}
          <div className="grid grid-cols-3 md:grid-cols-6 gap-2 mb-4">
            {quickQueries.map((query, index) => (
              <button
                key={index}
                onClick={() => handleQuickQuery(query)}
                className="p-2 text-xs font-medium text-white bg-blue-600 rounded hover:bg-blue-700 transition-colors hover:shadow-md"
              >
                {query}
              </button>
            ))}
          </div>

          {/* Compact Chat Interface */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Chat Input */}
            <div className="lg:col-span-2">
              <div className="flex space-x-2">
                <input
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                  placeholder="Ask about compliance, risks, or trends..."
                  className="flex-1 px-3 py-2 border border-gray-300 rounded text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
                <button
                  onClick={handleSendMessage}
                  disabled={isLoading || !inputValue.trim()}
                  className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isLoading ? 'Processing...' : (
                    <>
                      <Send className="h-3 w-3 inline mr-1" />
                      Send
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Compact Quick Stats */}
            <div className="lg:col-span-1">
              <div className="bg-gray-50 rounded p-3">
                <h4 className="font-medium text-gray-900 mb-2 text-sm">Quick Stats</h4>
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Total Members:</span>
                    <span className="font-medium">{regulatorData.total_users}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">High Risk:</span>
                    <span className="font-medium text-red-600">{regulatorData.summary.fraud_risk_summary.high}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Avg Age:</span>
                    <span className="font-medium">{regulatorData.summary.avg_age}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Avg Income:</span>
                    <span className="font-medium">£{(regulatorData.summary.avg_income / 1000).toFixed(0)}k</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Compact Messages Display */}
          {messages.length > 1 && (
            <div className="mt-4 max-h-48 overflow-y-auto space-y-2">
              {messages.slice(1).map((message) => (
                <div
                  key={message.id}
                  className={`p-2 rounded text-xs ${
                    message.type === 'user'
                      ? 'bg-blue-100 text-blue-800 ml-6'
                      : 'bg-gray-100 text-gray-800 mr-6'
                  }`}
                >
                  <p className="font-medium text-xs mb-1">
                    {message.type === 'user' ? 'You' : 'AI Assistant'}
                  </p>
                  <p>{message.content}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RegulatorDashboard;





