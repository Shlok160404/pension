import React, { useState, useEffect } from 'react';
import { 
  CpuChipIcon, 
  PaperAirplaneIcon, 
  ChevronLeftIcon, 
  ChevronRightIcon,
  UserIcon,
  ChartBarIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon,
  CurrencyPoundIcon,
  UsersIcon,
  ArrowRightOnRectangleIcon
} from '@heroicons/react/24/outline';
import Plot from 'react-plotly.js';
import { apiClient, tokenManager } from '../../services/api';
import type { AdvisorDashboardData, AdvisorClient, ClientDetails } from '../../services/api';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  clientId?: number;
  chartData?: any;
}

const AdvisorDashboard: React.FC = () => {
  const [dashboardData, setDashboardData] = useState<AdvisorDashboardData | null>(null);
  const [selectedClient, setSelectedClient] = useState<AdvisorClient | null>(null);
  const [_clientDetails, setClientDetails] = useState<ClientDetails | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [loadingDashboard, setLoadingDashboard] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentClientIndex, setCurrentClientIndex] = useState(0);
  const [currentUser, setCurrentUser] = useState<any>(null);



  // Load dashboard data on component mount
  useEffect(() => {
    loadDashboardData();
  }, []);

  // Get current user on component mount
  useEffect(() => {
    const user = tokenManager.getUser();
    setCurrentUser(user);
  }, []);

  // Load client details when a client is selected
  useEffect(() => {
    if (selectedClient) {
      loadClientDetails(selectedClient.user_id);
      // Update welcome message for selected client
    setMessages([
      {
        id: Date.now().toString(),
        type: 'assistant',
          content: `Now viewing ${selectedClient.full_name}'s dashboard. How can I help you analyze their pension data?`,
        timestamp: new Date(),
          clientId: selectedClient.user_id
        }
      ]);
    }
  }, [selectedClient]);

  const loadDashboardData = async () => {
    try {
      setLoadingDashboard(true);
      setError(null);
      const data = await apiClient.getAdvisorDashboard();
      setDashboardData(data);
      
      // Set first client as selected if available
      if (data.total_clients > 0) {
        const firstRiskGroup = Object.values(data.grouped_data.by_risk_tolerance)[0];
        if (firstRiskGroup && firstRiskGroup.clients.length > 0) {
          setSelectedClient(firstRiskGroup.clients[0]);
        }
      }
    } catch (err) {
      setError('Failed to load dashboard data. Please try again.');
      console.error('Error loading dashboard:', err);
    } finally {
      setLoadingDashboard(false);
    }
  };

  const loadClientDetails = async (clientId: number) => {
    try {
      const details = await apiClient.getAdvisorClientDetails(clientId);
      setClientDetails(details);
    } catch (err) {
      console.error('Error loading client details:', err);
    }
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading || !selectedClient) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue,
      timestamp: new Date(),
      clientId: selectedClient.user_id
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Call the AI API with the query
      const response = await apiClient.processPrompt(inputValue);
      
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: response.summary,
        timestamp: new Date(),
        clientId: selectedClient.user_id,
        chartData: response.chart_data,
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        timestamp: new Date(),
        clientId: selectedClient.user_id
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };





  const getClientList = (): AdvisorClient[] => {
    if (!dashboardData) return [];
    
    const allClients: AdvisorClient[] = [];
    Object.values(dashboardData.grouped_data.by_risk_tolerance).forEach(group => {
      allClients.push(...group.clients);
    });
    
    return allClients;
  };

  const nextClient = () => {
    const allClients = getClientList();
    if (allClients.length > 0) {
      setCurrentClientIndex((prev) => (prev + 1) % allClients.length);
      setSelectedClient(allClients[(currentClientIndex + 1) % allClients.length]);
    }
  };

  const prevClient = () => {
    const allClients = getClientList();
    if (allClients.length > 0) {
      setCurrentClientIndex((prev) => (prev - 1 + allClients.length) % allClients.length);
      setSelectedClient(allClients[(currentClientIndex - 1 + allClients.length) % allClients.length]);
    }
  };

  const getRiskColor = (risk: string): string => {
    switch (risk.toLowerCase()) {
      case 'high': return 'text-red-600 bg-red-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      case 'low': return 'text-green-600 bg-green-100';
      case 'conservative': return 'text-blue-600 bg-blue-100';
      case 'moderate': return 'text-orange-600 bg-orange-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getFraudRiskColor = (score: number): string => {
    if (score > 0.8) return 'text-red-600 bg-red-100';
    if (score > 0.5) return 'text-yellow-600 bg-yellow-100';
    return 'text-green-600 bg-green-100';
  };

  // Enhanced message formatting function for better UX - only for AI responses
  const formatMessageContent = (content: string, messageType: 'user' | 'assistant') => {
    if (!content) return null;

    // For user messages, return simple text without any formatting or highlighting
    if (messageType === 'user') {
      return <p className="leading-relaxed text-white">{content}</p>;
    }

    // For AI assistant messages, apply the enhanced formatting
    const paragraphs = content.split('\n').filter(p => p.trim());
    
    return (
      <div className="space-y-3">
        {paragraphs.map((paragraph, index) => {
          const hasNumbers = /\d+%|\d+\.\d+|\d+/.test(paragraph);
          
          return (
            <div key={index} className={`${hasNumbers ? 'bg-blue-50 p-3 rounded-lg border-l-4 border-blue-200' : ''}`}>
              <p className="leading-relaxed">
                {paragraph.split(' ').map((word, wordIndex) => {
                  if (/\d+%|\d+\.\d+|\d+/.test(word) || 
                      ['risk', 'fraud', 'client', 'portfolio', 'allocation', 'growth', 'return', 'score', 'value', 'total', 'income', 'savings', 'equity', 'bonds', 'cash', 'volatility', 'sharpe', 'drawdown', 'benchmark', 'performance'].some(term => word.toLowerCase().includes(term))) {
                    return (
                      <span key={wordIndex} className="inline-block bg-blue-100 text-blue-800 px-2 py-1 rounded-md font-semibold mx-1 border border-blue-200 shadow-sm hover:bg-blue-200 hover:shadow-md transition-all duration-200 cursor-default">
                        {word}
                      </span>
                    );
                  }
                  return <span key={wordIndex}>{word} </span>;
                })}
              </p>
            </div>
          );
        })}
      </div>
    );
  };

  const handleLogout = () => {
    tokenManager.logout();
    // Redirect to landing page or refresh the page
    window.location.href = '/';
  };

  if (loadingDashboard) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading advisor dashboard...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <ExclamationTriangleIcon className="h-16 w-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-800 mb-2">Error Loading Dashboard</h2>
          <p className="text-gray-600 mb-4">{error}</p>
          <button
            onClick={loadDashboardData}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!dashboardData || dashboardData.total_clients === 0) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <UsersIcon className="h-16 w-16 text-gray-400 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-800 mb-2">No Clients Assigned</h2>
          <p className="text-gray-600">You don't have any clients assigned yet.</p>
        </div>
      </div>
    );
  }

  const allClients = getClientList();

     return (
     <div className="flex flex-col min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="h-10 w-10 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
              <CpuChipIcon className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">AegisAI Professional Advisor</h1>
              <p className="text-sm text-gray-500">Portfolio Analytics & Risk Management Platform</p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            {/* Status Indicator */}
            <div className="flex items-center space-x-2 px-3 py-2 bg-green-50 rounded-full border border-green-200">
              <div className="h-2 w-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-green-700 font-medium">Connected</span>
            </div>
            
            {/* User Info */}
            <div className="flex items-center space-x-2 text-sm text-gray-600">
              <UserIcon className="h-5 w-5" />
              <span>Advisor Dashboard</span>
            </div>

            {/* Dashboard Stats */}
            <div className="text-right">
              <p className="text-sm font-medium text-gray-900">{dashboardData.total_clients} Active Clients</p>
              <p className="text-xs text-gray-500">Avg Age: {dashboardData.summary.avg_age}</p>
            </div>

            {/* User Details and Logout */}
            <div className="flex items-center space-x-3">
              <div className="text-right">
                <p className="text-sm font-medium text-gray-900">
                  Welcome, {currentUser?.full_name || currentUser?.username || 'Advisor'}
                </p>
                <p className="text-xs text-gray-500">
                  ID: {currentUser?.id || 'N/A'}
                </p>
              </div>
              <button
                onClick={handleLogout}
                className="flex items-center space-x-2 px-3 py-2 bg-red-50 text-red-600 rounded-lg hover:bg-red-100 transition-colors duration-200"
              >
                <ArrowRightOnRectangleIcon className="h-4 w-4" />
                <span className="text-sm font-medium">Logout</span>
              </button>
            </div>
          </div>
        </div>
      </div>

             {/* Main Content */}
       <div className="flex-1 flex overflow-hidden">
         {/* Client Dashboard Area */}
         <div className="flex-1 flex flex-col overflow-y-auto">
                      {/* Unified Client Dashboard with Charts */}
            <div className="bg-white border-b border-gray-200 p-4">
             <div className="flex items-center justify-between mb-3">
               <h2 className="text-lg font-semibold text-gray-900">Client Dashboard</h2>
               <div className="flex items-center space-x-2 text-sm text-gray-500">
                <span>{currentClientIndex + 1} of {allClients.length}</span>
               </div>
             </div>
             
             <div className="relative">
               {/* Navigation Arrows */}
               <button
                 onClick={prevClient}
                 className="absolute left-0 top-1/2 transform -translate-y-1/2 z-10 bg-white border border-gray-200 rounded-full p-2 shadow-lg hover:shadow-xl transition-shadow duration-200"
               >
                 <ChevronLeftIcon className="h-5 w-5 text-gray-600" />
               </button>
               
               <button
                 onClick={nextClient}
                 className="absolute right-0 top-1/2 transform -translate-y-1/2 z-10 bg-white border border-gray-200 rounded-full p-2 shadow-lg hover:shadow-xl transition-shadow duration-200"
               >
                 <ChevronRightIcon className="h-5 w-5 text-gray-600" />
               </button>

               {/* Unified Dashboard Content */}
              <div className="mx-12">
                 <div className="bg-gradient-to-br from-blue-50 to-purple-50 rounded-2xl p-4 border border-blue-200">
                   {/* Client Info Row */}
                   <div className="flex items-center space-x-3 mb-4">
                     <div className="h-12 w-12 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
                       <UserIcon className="h-6 w-6 text-white" />
                     </div>
                     <div>
                      <h3 className="text-xl font-bold text-gray-900">{selectedClient?.full_name}</h3>
                      <p className="text-sm text-gray-600">Age {selectedClient?.age} • {selectedClient?.risk_tolerance} Risk Profile</p>
                     </div>
                   </div>

                   {/* Key Metrics Row */}
                   <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                     <div className="bg-white rounded-lg p-3 border border-gray-200">
                       <div className="flex items-center space-x-2 mb-1">
                         <CurrencyPoundIcon className="h-4 w-4 text-green-600" />
                        <span className="text-xs font-medium text-gray-700">Current Savings</span>
                       </div>
                      <p className="text-lg font-bold text-gray-900">£{selectedClient?.current_savings.toLocaleString()}</p>
                     </div>
                     
                     <div className="bg-white rounded-lg p-3 border border-gray-200">
                       <div className="flex items-center space-x-2 mb-1">
                         <ChartBarIcon className="h-4 w-4 text-blue-600" />
                        <span className="text-xs font-medium text-gray-700">Annual Income</span>
                       </div>
                      <p className="text-lg font-bold text-gray-900">£{selectedClient?.annual_income.toLocaleString()}</p>
                     </div>
                     
                     <div className="bg-white rounded-lg p-3 border border-gray-200">
                       <div className="flex items-center space-x-2 mb-1">
                         <CheckCircleIcon className="h-4 w-4 text-purple-600" />
                        <span className="text-xs font-medium text-gray-700">Risk Score</span>
                       </div>
                                              <p className="text-lg font-bold text-gray-900">{((selectedClient?.anomaly_score || 0) * 100).toFixed(0)}%</p>
                     </div>
                     
                     <div className="bg-white rounded-lg p-3 border border-gray-200">
                       <div className="flex items-center space-x-2 mb-1">
                         <ClockIcon className="h-4 w-4 text-gray-600" />
                        <span className="text-xs font-medium text-gray-700">Risk Level</span>
                       </div>
                      <p className="text-sm font-semibold text-gray-900">
                        {(selectedClient?.anomaly_score || 0) > 0.8 ? 'High' : (selectedClient?.anomaly_score || 0) > 0.5 ? 'Medium' : 'Low'}
                      </p>
                     </div>
                   </div>

                   {/* Risk Metrics Row */}
                   <div className="grid grid-cols-2 gap-3 mb-4">
                                                                 <div className="bg-white rounded-lg p-4 border border-gray-200">
                        <h4 className="text-xs font-medium text-gray-900 mb-3">Risk Profile Analysis</h4>
                        
                        {/* Risk Score Visualization */}
                        <div className="mb-3">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-xs font-medium text-gray-700">Risk Score</span>
                            <span className={`inline-flex px-2 py-1 rounded-full text-xs font-medium ${getFraudRiskColor(selectedClient?.anomaly_score || 0)}`}>
                              {(selectedClient?.anomaly_score || 0).toFixed(2)}
                            </span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div 
                              className={`h-2 rounded-full transition-all duration-300 ${
                                (selectedClient?.anomaly_score || 0) > 0.8 ? 'bg-red-500' : 
                                (selectedClient?.anomaly_score || 0) > 0.5 ? 'bg-yellow-500' : 'bg-green-500'
                              }`}
                              style={{ width: `${((selectedClient?.anomaly_score || 0) * 100)}%` }}
                            ></div>
                          </div>
                        </div>

                        {/* Risk Tolerance Badge */}
                        <div className="mb-3">
                          <span className={`inline-flex px-3 py-2 rounded-lg text-sm font-semibold ${getRiskColor(selectedClient?.risk_tolerance || '')}`}>
                            {selectedClient?.risk_tolerance} Risk Profile
                          </span>
                        </div>

                        {/* Risk Assessment */}
                        <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
                          <div className="flex items-center space-x-2 mb-2">
                            {selectedClient?.anomaly_score && selectedClient.anomaly_score < 0.5 ? 
                              <span className="text-green-600">✅</span> :
                             selectedClient?.anomaly_score && selectedClient.anomaly_score < 0.8 ? 
                              <span className="text-yellow-600">⚠️</span> :
                              <span className="text-red-600">🚨</span>
                            }
                            <span className="text-xs font-semibold text-gray-800">
                              {selectedClient?.anomaly_score && selectedClient.anomaly_score < 0.5 ? 
                                'Excellent Risk Profile' :
                               selectedClient?.anomaly_score && selectedClient.anomaly_score < 0.8 ? 
                                'Moderate Risk Profile' :
                                'High Risk Profile'}
                            </span>
                          </div>
                          <p className="text-xs text-gray-700 leading-relaxed">
                            {selectedClient?.anomaly_score && selectedClient.anomaly_score < 0.5 ? 
                              'Portfolio shows excellent risk-adjusted returns with controlled volatility. Current risk metrics are well within acceptable parameters.' :
                             selectedClient?.anomaly_score && selectedClient.anomaly_score < 0.8 ? 
                              'Portfolio demonstrates moderate risk exposure requiring regular monitoring. Consider risk mitigation strategies.' :
                              'Portfolio exhibits elevated risk levels requiring immediate attention. Implement defensive positioning and risk controls.'}
                          </p>
                        </div>
                      </div>
                     
                                                                 <div className="bg-white rounded-lg p-4 border border-gray-200">
                       <h4 className="text-xs font-medium text-gray-900 mb-3">Annual Income vs Savings Analysis</h4>
                                               <div className="flex items-center justify-center h-80">
                                                     <Plot
                             data={[
                               {
                                 type: 'pie',
                                 labels: ['Annual Income', 'Current Savings'],
                                 values: [
                                   selectedClient?.annual_income || 0,
                                   selectedClient?.current_savings || 0
                                 ],
                                 marker: { 
                                   colors: ['#3B82F6', '#10B981'],
                                   line: { color: '#ffffff', width: 2 }
                                 },
                                 textinfo: 'label+percent',
                                 textposition: 'outside',
                                 hole: 0.3,
                                 textfont: { size: 10, family: 'Arial', color: '#374151' },
                                 texttemplate: '<b>%{label}</b><br>£%{value:,.0f}<br>(%{percent:.1%})',
                                 hoverinfo: 'label+percent',
                                 hovertemplate: '<b>%{label}</b><br>£%{value:,.0f}<br>(%{percent:.1%})<extra></extra>'
                               }
                             ]}
                             layout={{
                               width: 300,
                               height: 350,
                               showlegend: false,
                               margin: { t: 40, b: 40, l: 40, r: 40 },
                               plot_bgcolor: 'rgba(0,0,0,0)',
                               paper_bgcolor: 'rgba(0,0,0,0)',
                               title: {
                                 text: 'Income vs Savings',
                                 font: { size: 14, color: '#374151' },
                                 x: 0.5,
                                 xanchor: 'center'
                               },
                               annotations: [
                                 {
                                   text: `Total: £${((selectedClient?.annual_income || 0) + (selectedClient?.current_savings || 0)).toLocaleString()}`,
                                   showarrow: false,
                                   x: 0.5,
                                   y: 0.5,
                                   font: { size: 12, color: '#374151' },
                                   bgcolor: 'rgba(255,255,255,0.8)',
                                   bordercolor: '#e5e7eb',
                                   borderwidth: 1
                                 }
                               ]
                             }}
                             config={{ displayModeBar: false }}
                           />
                        </div>
                     </div>
                   </div>

                   {/* Enhanced Charts Row */}
                   <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Risk Distribution Chart */}
                     <div className="bg-white rounded-lg p-6 border border-gray-200 shadow-sm">
                      <h4 className="text-sm font-semibold text-gray-900 mb-4">Risk Distribution</h4>
                       <Plot
                         data={[
                           {
                            type: 'pie',
                            labels: Object.keys(dashboardData.summary.risk_distribution),
                            values: Object.values(dashboardData.summary.risk_distribution),
                            marker: {
                              colors: ['#10B981', '#F59E0B', '#EF4444', '#3B82F6', '#8B5CF6'],
                              line: { color: '#ffffff', width: 2 }
                            },
                            textinfo: 'label+percent',
                            textposition: 'outside',
                            hole: 0.3,
                            textfont: { size: 11 }
                           }
                         ]}
                         layout={{
                           width: 280,
                           height: 220,
                           showlegend: false,
                           margin: { t: 30, b: 30, l: 30, r: 30 },
                           plot_bgcolor: 'rgba(0,0,0,0)',
                           paper_bgcolor: 'rgba(0,0,0,0)',
                           title: {
                             text: 'Risk Distribution',
                             font: { size: 12, color: '#374151' }
                           }
                         }}
                         config={{ displayModeBar: false }}
                       />
                     </div>

                    {/* Fraud Risk Summary Chart */}
                     <div className="bg-white rounded-lg p-6 border border-gray-200 shadow-sm">
                      <h4 className="text-sm font-semibold text-gray-900 mb-4">Fraud Risk Summary</h4>
                       <Plot
                         data={[
                           {
                            labels: ['Low Risk', 'Medium Risk', 'High Risk'],
                            values: [
                              dashboardData.summary.fraud_risk_summary.low,
                              dashboardData.summary.fraud_risk_summary.medium,
                              dashboardData.summary.fraud_risk_summary.high
                            ],
                            type: 'pie',
                             marker: {
                              colors: ['#10B981', '#F59E0B', '#EF4444'],
                              line: { color: '#ffffff', width: 2 }
                            },
                            textinfo: 'label+percent',
                            textposition: 'outside',
                            hole: 0.4,
                            textfont: { size: 11 }
                           }
                         ]}
                         layout={{
                           width: 280,
                           height: 220,
                           showlegend: false,
                          margin: { t: 30, b: 30, l: 30, r: 30 },
                           plot_bgcolor: 'rgba(0,0,0,0)',
                           paper_bgcolor: 'rgba(0,0,0,0)',
                         }}
                         config={{ displayModeBar: false }}
                       />
                     </div>

                                         {/* Portfolio Performance Trend */}
                     <div className="bg-white rounded-lg p-6 border border-gray-200 shadow-sm">
                      <h4 className="text-sm font-semibold text-gray-900 mb-4">Portfolio Trend</h4>
                       <Plot
                         data={[
                           {
                            x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                            y: [
                              (selectedClient?.current_savings || 100000) * 0.95,
                              (selectedClient?.current_savings || 100000) * 0.97,
                              (selectedClient?.current_savings || 100000) * 0.99,
                              (selectedClient?.current_savings || 100000) * 1.02,
                              (selectedClient?.current_savings || 100000) * 1.05,
                              selectedClient?.current_savings || 100000
                            ],
                            type: 'scatter',
                            mode: 'lines+markers',
                            line: { color: '#3B82F6', width: 3 },
                            marker: { size: 6, color: '#3B82F6' }
                           }
                         ]}
                         layout={{
                           width: 280,
                           height: 220,
                           xaxis: { 
                            title: { text: 'Month', font: { size: 12 } },
                             zeroline: false,
                             gridcolor: '#f3f4f6',
                             tickangle: 0
                           },
                           yaxis: { 
                            title: { text: 'Value (£)', font: { size: 12 } },
                             zeroline: false,
                             gridcolor: '#f3f4f6',
                             tickformat: ',.0f'
                           },
                           showlegend: false,
                           margin: { t: 30, b: 60, l: 70, r: 30 },
                           plot_bgcolor: 'rgba(0,0,0,0)',
                           paper_bgcolor: 'rgba(0,0,0,0)',
                         }}
                         config={{ displayModeBar: false }}
                       />
                     </div>
                   </div>

             
                    {/* Professional Insights Section */}
                    <div className="mt-6 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-4 border border-blue-200">
                     <div className="flex items-center space-x-3 mb-3">
                       <div className="p-2 bg-blue-500 rounded-lg">
                         <ChartBarIcon className="h-5 w-5 text-white" />
                       </div>
                       <h4 className="text-sm font-semibold text-blue-900">AegisAI Professional Insights</h4>
                     </div>
                     
                     <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                       <div className="bg-white rounded-lg p-3 border border-blue-200">
                         <div className="flex items-center space-x-2 mb-2">
                           <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                           <span className="text-xs font-semibold text-green-700">Portfolio Health</span>
                         </div>
                         <p className="text-xs text-gray-700">
                           {selectedClient?.anomaly_score && selectedClient.anomaly_score < 0.5 ? 
                             'Excellent portfolio health with strong risk-adjusted returns' :
                            selectedClient?.anomaly_score && selectedClient.anomaly_score < 0.8 ? 
                             'Good portfolio health with moderate risk exposure' :
                             'Portfolio requires attention due to elevated risk levels'}
                         </p>
                       </div>
                       
                       <div className="bg-white rounded-lg p-3 border border-blue-200">
                         <div className="flex items-center space-x-2 mb-2">
                           <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                           <span className="text-xs font-semibold text-blue-700">Risk Assessment</span>
                         </div>
                         <p className="text-xs text-gray-700">
                           Risk score of {(selectedClient?.anomaly_score || 0).toFixed(2)} indicates 
                           {(selectedClient?.anomaly_score || 0) > 0.8 ? ' high risk requiring immediate review' :
                            (selectedClient?.anomaly_score || 0) > 0.5 ? ' moderate risk with monitoring needed' :
                             ' low risk with stable performance'}
                         </p>
                       </div>
                       
                       <div className="bg-white rounded-lg p-3 border border-blue-200">
                         <div className="flex items-center space-x-2 mb-2">
                           <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                           <span className="text-xs font-semibold text-purple-700">Recommendations</span>
                         </div>
                         <p className="text-xs text-gray-700">
                           {selectedClient?.age && selectedClient.age < 50 ? 
                             'Consider growth-focused strategies for long-term horizon' :
                            selectedClient?.age && selectedClient.age < 65 ? 
                             'Balanced approach recommended for mid-career planning' :
                             'Conservative strategies advised for retirement preparation'}
                         </p>
                       </div>
                     </div>
                   </div>
                 </div>
               </div>
             </div>
           </div>

                     {/* Chat Interface */}
           <div className="flex flex-col">
            <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-2xl rounded-2xl px-4 py-3 ${
                      message.type === 'user'
                        ? 'bg-blue-500 text-white'
                        : 'bg-white text-gray-900 shadow-sm border border-gray-200'
                    }`}
                  >
                    <div className="flex items-start space-x-3">
                      {message.type === 'assistant' && (
                        <div className="h-6 w-6 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center flex-shrink-0">
                          <CpuChipIcon className="h-3 w-3 text-white" />
                        </div>
                      )}
                      <div className="flex-1 min-w-0">
                        <div className="text-sm leading-relaxed">
                          {formatMessageContent(message.content, message.type)}
                        </div>
                        
                        <div className="text-xs text-gray-400 mt-2">
                          {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
              
              {/* Loading Indicator */}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-white rounded-2xl px-4 py-3 shadow-sm border border-gray-200">
                    <div className="flex items-center space-x-3">
                      <div className="h-6 w-6 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
                        <CpuChipIcon className="h-3 w-3 text-white" />
                      </div>
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Input Area */}
            <div className="border-t border-gray-200 bg-white px-4 py-3">
              <div className="flex space-x-3">
                <div className="flex-1">
                  <input
                    type="text"
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                    placeholder={`Ask about ${selectedClient?.full_name}'s pension data...`}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors duration-200"
                    disabled={isLoading || !selectedClient}
                  />
                </div>
                <button
                  onClick={handleSendMessage}
                  disabled={!inputValue.trim() || isLoading || !selectedClient}
                  className="px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-lg hover:from-blue-600 hover:to-purple-600 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center space-x-2"
                >
                  <PaperAirplaneIcon className="h-4 w-4" />
                  <span>Send</span>
                </button>
              </div>
            </div>
          </div>
        </div>

        
      </div>
    </div>
  );
};

export default AdvisorDashboard;