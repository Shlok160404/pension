import { useState, useEffect, useRef } from 'react';
import { 
  ArrowRightIcon, 
  CpuChipIcon, 
  ShieldCheckIcon, 
  ArrowTrendingUpIcon, 
  BoltIcon, 
  UserGroupIcon, 
  CheckCircleIcon, 
  ChartBarIcon,
  ArrowRightOnRectangleIcon
} from '@heroicons/react/24/outline';
import { tokenManager } from '../services/api';

interface AegisAILandingProps {
  onNavigate: (page: string) => void;
}

const AegisAILanding: React.FC<AegisAILandingProps> = ({ onNavigate }) => {
  const [isVisible, setIsVisible] = useState<Record<string, boolean>>({});
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [showContent, setShowContent] = useState(false);
  const sectionsRef = useRef<(HTMLElement | null)[]>([]);

  // Loading animation effect
  useEffect(() => {
    const timer = setInterval(() => {
      setLoadingProgress(prev => {
        if (prev >= 100) {
          clearInterval(timer);
          setTimeout(() => setShowContent(true), 500);
          return 100;
        }
        return prev + 2;
      });
    }, 50);

    return () => clearInterval(timer);
  }, []);

  // Intersection Observer for scroll animations
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && entry.target instanceof HTMLElement) {
            const section = entry.target.dataset.section;
            if (section) {
              setIsVisible(prev => ({
                ...prev,
                [section]: true
              }));
            }
          }
        });
      },
      { threshold: 0.1 }
    );

    sectionsRef.current.forEach((section) => {
      if (section) observer.observe(section);
    });

    return () => observer.disconnect();
  }, [showContent]);

  // SVG Line Drawing Component
  const AnimatedLine = ({ path, delay = 0, duration = 2 }: { path: string; delay?: number; duration?: number }) => (
    <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 400 300">
      <path
        d={path}
        fill="none"
        stroke="url(#gradient)"
        strokeWidth="2"
        strokeLinecap="round"
        className="animate-pulse"
        style={{
          strokeDasharray: 1000,
          strokeDashoffset: showContent ? 0 : 1000,
          transition: `stroke-dashoffset ${duration}s ease-in-out ${delay}s`
        }}
      />
      <defs>
        <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#3B82F6" />
          <stop offset="50%" stopColor="#8B5CF6" />
          <stop offset="100%" stopColor="#06B6D4" />
        </linearGradient>
      </defs>
    </svg>
  );

  // Loading Screen - temporarily bypassed
  if (!showContent) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 flex items-center justify-center">
        <div className="text-center">
          <div className="relative w-32 h-32 mx-auto mb-8">
            <div className="absolute inset-0 rounded-full border-4 border-gray-200"></div>
            <div 
              className="absolute inset-0 rounded-full border-4 border-blue-500 border-t-transparent animate-spin"
              style={{ animationDuration: '1s' }}
            ></div>
            <div className="absolute inset-0 flex items-center justify-center">
              <CpuChipIcon className="w-12 h-12 text-blue-600" />
            </div>
          </div>
          <h2 className="text-2xl font-bold text-gray-800 mb-4">Aegis AI</h2>
          <div className="w-64 h-2 bg-gray-200 rounded-full mx-auto overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full transition-all duration-100"
              style={{ width: `${loadingProgress}%` }}
            ></div>
          </div>
          <p className="text-gray-600 mt-2">{loadingProgress}%</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Navigation */}
      <nav className="relative z-10 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg flex items-center justify-center">
              <CpuChipIcon className="w-6 h-6 text-white" />
            </div>
            <span className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Aegis AI
            </span>
          </div>
          <div className="flex items-center space-x-4">
            {tokenManager.isAuthenticated() ? (
              <button 
                onClick={() => {
                  tokenManager.logout();
                  window.location.reload();
                }}
                className="flex items-center space-x-2 px-3 py-2 bg-red-50 text-red-600 rounded-lg hover:bg-red-100 transition-colors duration-200"
              >
                <ArrowRightOnRectangleIcon className="h-4 w-4" />
                <span className="text-sm font-medium">Logout</span>
              </button>
            ) : (
              <button 
                onClick={() => onNavigate('login')}
                className="bg-gradient-to-r from-blue-500 to-purple-500 text-white px-6 py-2 rounded-full hover:shadow-lg transform hover:scale-105 transition-all duration-300 font-medium"
              >
                Login
              </button>
            )}
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative px-6 py-20">
        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div className="space-y-8">
              <div className={`transform transition-all duration-1000 ${showContent ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'}`}>
                <h1 className="text-5xl lg:text-6xl font-bold text-gray-900 leading-tight">
                  Aegis AI: Where Financial Complexity
                  <br />
                  <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">Meets AI Simplicity</span>
                </h1>
              </div>
              
              <div className={`transform transition-all duration-1000 delay-300 ${showContent ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'}`}>
                <p className="text-xl text-gray-600 leading-relaxed">
                  Where Financial Complexity Meets AI Simplicity
                </p>
              </div>

              <div className={`flex flex-wrap gap-4 transform transition-all duration-1000 delay-500 ${showContent ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'}`}>
                <button 
                  onClick={() => onNavigate('signup')}
                  className="bg-gradient-to-r from-blue-500 to-purple-500 text-white px-8 py-4 rounded-full hover:shadow-xl transform hover:scale-105 transition-all duration-300 flex items-center space-x-2"
                >
                  <span className="font-semibold">Start Analysis</span>
                  <ArrowRightIcon className="w-5 h-5" />
                </button>
                <button 
                  onClick={() => onNavigate('login')}
                  className="border-2 border-gray-300 text-gray-700 px-8 py-4 rounded-full hover:border-blue-500 hover:text-blue-500 transition-all duration-300"
                >
                  Live Demo
                </button>
              </div>
            </div>

            <div className="relative">
              {/* Animated Background Lines */}
              <AnimatedLine 
                path="M50,50 Q200,20 350,80 T300,200 Q150,250 50,200 Z" 
                delay={0.5}
              />
              <AnimatedLine 
                path="M100,100 L300,50 L350,150 L150,200 Z" 
                delay={1}
              />
              
              {/* Floating AI Mascot - Always Animated */}
              <div className="absolute top-0 right-0 w-80 h-80 pointer-events-none">
                {/* Main AI Brain Mascot */}
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 animate-bounce">
                  <div className="relative w-32 h-32">
                    {/* Brain Core */}
                    <div className="absolute inset-0 bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 rounded-full animate-pulse shadow-2xl">
                      <div className="absolute inset-2 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                        <CpuChipIcon className="w-16 h-16 text-white animate-spin" style={{ animationDuration: '8s' }} />
                      </div>
                    </div>
                    
                    {/* Glowing Ring */}
                    <div className="absolute -inset-4 bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 rounded-full opacity-30 animate-ping" style={{ animationDuration: '3s' }}></div>
                  </div>
                </div>

                {/* Orbiting Elements */}
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-64 h-64 animate-spin" style={{ animationDuration: '20s' }}>
                  {/* Data Nodes */}
                  <div className="absolute top-0 left-1/2 w-4 h-4 bg-blue-400 rounded-full transform -translate-x-1/2 animate-pulse"></div>
                  <div className="absolute bottom-0 left-1/2 w-4 h-4 bg-purple-400 rounded-full transform -translate-x-1/2 animate-pulse" style={{ animationDelay: '1s' }}></div>
                  <div className="absolute left-0 top-1/2 w-4 h-4 bg-pink-400 rounded-full transform -translate-y-1/2 animate-pulse" style={{ animationDelay: '2s' }}></div>
                  <div className="absolute right-0 top-1/2 w-4 h-4 bg-cyan-400 rounded-full transform -translate-y-1/2 animate-pulse" style={{ animationDelay: '3s' }}></div>
                </div>

                {/* Secondary Orbit - Counter Rotation */}
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-48 h-48 animate-spin" style={{ animationDuration: '15s', animationDirection: 'reverse' }}>
                  <div className="absolute top-4 right-4 w-3 h-3 bg-yellow-400 rounded-full animate-bounce"></div>
                  <div className="absolute bottom-4 left-4 w-3 h-3 bg-green-400 rounded-full animate-bounce" style={{ animationDelay: '0.5s' }}></div>
                  <div className="absolute top-4 left-4 w-3 h-3 bg-red-400 rounded-full animate-bounce" style={{ animationDelay: '1s' }}></div>
                  <div className="absolute bottom-4 right-4 w-3 h-3 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '1.5s' }}></div>
                </div>

                {/* Floating Data Streams */}
                <div className="absolute top-8 left-8 w-2 h-16 bg-gradient-to-b from-blue-400 to-transparent rounded-full animate-pulse opacity-60" style={{ animationDelay: '0s' }}></div>
                <div className="absolute top-16 right-8 w-2 h-12 bg-gradient-to-b from-purple-400 to-transparent rounded-full animate-pulse opacity-60" style={{ animationDelay: '1s' }}></div>
                <div className="absolute bottom-16 left-16 w-2 h-20 bg-gradient-to-b from-pink-400 to-transparent rounded-full animate-pulse opacity-60" style={{ animationDelay: '2s' }}></div>
                <div className="absolute bottom-8 right-16 w-2 h-14 bg-gradient-to-b from-cyan-400 to-transparent rounded-full animate-pulse opacity-60" style={{ animationDelay: '3s' }}></div>

                {/* Neural Network Lines */}
                <svg className="absolute inset-0 w-full h-full opacity-30" viewBox="0 0 320 320">
                  <path
                    d="M160,160 L80,80 M160,160 L240,80 M160,160 L80,240 M160,160 L240,240"
                    stroke="url(#neuralGradient)"
                    strokeWidth="2"
                    fill="none"
                    className="animate-pulse"
                  />
                  <defs>
                    <linearGradient id="neuralGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#3B82F6" />
                      <stop offset="50%" stopColor="#8B5CF6" />
                      <stop offset="100%" stopColor="#EC4899" />
                    </linearGradient>
                  </defs>
                </svg>
              </div>
              

            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section 
        ref={(el) => { sectionsRef.current[0] = el; }}
        data-section="features"
        className="px-6 py-20"
      >
        <div className="max-w-7xl mx-auto">
          <div className={`text-center mb-16 transform transition-all duration-1000 ${isVisible.features ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'}`}>
            <h2 className="text-4xl font-bold text-gray-900 mb-6">
              The <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">Aegis Engine</span>
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Powered by a sophisticated multi-agent system—a team of specialized AIs working in concert, orchestrated by LangGraph.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                icon: BoltIcon,
                title: 'AI Supervisor',
                description: 'Central supervisor agent analyzes strategic intent and delegates to specialists',
                color: 'from-yellow-400 to-orange-500',
                delay: 0
              },
              {
                icon: ChartBarIcon,
                title: 'Specialist Agents',
                description: 'Fraud Detection, Risk Analysis, and Projection Specialists working in harmony',
                color: 'from-blue-400 to-purple-500',
                delay: 200
              },
              {
                icon: CheckCircleIcon,
                title: 'Hybrid Intelligence',
                description: 'ML-powered predictions + LLM reasoning for explainable, actionable insights',
                color: 'from-green-400 to-blue-500',
                delay: 400
              }
            ].map((feature) => {
              const IconComponent = feature.icon;
              return (
                <div 
                  key={feature.title}
                  className={`transform transition-all duration-1000 ${isVisible.features ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'}`}
                  style={{ transitionDelay: `${feature.delay}ms` }}
                >
                  <div className="bg-white rounded-2xl p-8 shadow-lg hover:shadow-xl transition-shadow duration-300 group">
                    <div className={`w-12 h-12 bg-gradient-to-r ${feature.color} rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300`}>
                      <IconComponent className="w-6 h-6 text-white" />
                    </div>
                    <h3 className="text-xl font-bold text-gray-900 mb-4">{feature.title}</h3>
                    <p className="text-gray-600 leading-relaxed">{feature.description}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Multi-Agent System Section */}
      <section 
        ref={(el) => { sectionsRef.current[2] = el; }}
        data-section="multiagent"
        className="px-6 py-20 bg-gradient-to-r from-purple-50 to-pink-50"
      >
        <div className="max-w-7xl mx-auto">
          <div className={`text-center mb-16 transform transition-all duration-1000 ${isVisible.multiagent ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'}`}>
            <h2 className="text-4xl font-bold text-gray-900 mb-6">
              <span className="bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">Multi-Agent</span> Intelligence
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              This isn't just another AI chatbot. Aegis Ai is powered by a sophisticated multi-agent system—a team of specialized AIs working in concert.
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div className="space-y-6">
              <div className="bg-white rounded-2xl p-8 shadow-lg">
                <h3 className="text-2xl font-bold text-gray-900 mb-4">How It Works</h3>
                <div className="space-y-4">
                  <div className="flex items-start space-x-3">
                    <div className="w-6 h-6 bg-blue-500 rounded-full flex-shrink-0 mt-1"></div>
                    <div>
                      <h4 className="font-semibold text-gray-900">User Query</h4>
                      <p className="text-gray-600">A user asks a question through their tailored dashboard</p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-6 h-6 bg-purple-500 rounded-full flex-shrink-0 mt-1"></div>
                    <div>
                      <h4 className="font-semibold text-gray-900">AI Supervisor</h4>
                      <p className="text-gray-600">Central supervisor agent analyzes the query's strategic intent</p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-6 h-6 bg-pink-500 rounded-full flex-shrink-0 mt-1"></div>
                    <div>
                      <h4 className="font-semibold text-gray-900">Intelligent Delegation</h4>
                      <p className="text-gray-600">Routes to the right specialist: Fraud Detection, Risk Analysis, or Projection Specialist</p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-6 h-6 bg-green-500 rounded-full flex-shrink-0 mt-1"></div>
                    <div>
                      <h4 className="font-semibold text-gray-900">Trusted Results</h4>
                      <p className="text-gray-600">Final structured answer backed by both ML accuracy and LLM-powered reasoning</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-2xl p-8 shadow-lg">
              <h3 className="text-2xl font-bold text-gray-900 mb-4">Hybrid Intelligence Approach</h3>
              <div className="space-y-4">
                <div className="border-l-4 border-blue-500 pl-4">
                  <h4 className="font-semibold text-gray-900">ML First</h4>
                  <p className="text-gray-600">Custom-trained XGBoost models deliver rapid, statistically powerful predictions</p>
                </div>
                <div className="border-l-4 border-purple-500 pl-4">
                  <h4 className="font-semibold text-gray-900">LLM Reasoning</h4>
                  <p className="text-gray-600">Google Gemini provides human-readable explanations and actionable recommendations</p>
                </div>
                <div className="border-l-4 border-green-500 pl-4">
                  <h4 className="font-semibold text-gray-900">Best of Both Worlds</h4>
                  <p className="text-gray-600">Predictive power + explainable reasoning for complete confidence</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Workflow Visualization */}
      <section 
        ref={(el) => { sectionsRef.current[1] = el; }}
        data-section="workflow"
        className="px-6 py-20 bg-gradient-to-r from-blue-50 to-purple-50"
      >
        <div className="max-w-7xl mx-auto">
          <div className={`text-center mb-16 transform transition-all duration-1000 ${isVisible.workflow ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'}`}>
            <h2 className="text-4xl font-bold text-gray-900 mb-6">
              Intelligent <span className="bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">Workflow</span>
            </h2>
          </div>

          <div className="relative">
            {/* Workflow Steps */}
            <div className="grid md:grid-cols-4 gap-8 items-center">
              {[
                { icon: UserGroupIcon, title: 'User Query', desc: 'Ask any pension question' },
                { icon: CpuChipIcon, title: 'AI Supervisor', desc: 'Routes to specialist' },
                { icon: ArrowTrendingUpIcon, title: 'Analysis', desc: 'Deep insights generated' },
                { icon: ShieldCheckIcon, title: 'Results', desc: 'Secure, accurate answers' }
              ].map((step, index) => {
                const IconComponent = step.icon;
                return (
                  <div 
                    key={step.title}
                    className={`text-center transform transition-all duration-1000 ${isVisible.workflow ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'}`}
                    style={{ transitionDelay: `${index * 200}ms` }}
                  >
                    <div className="relative">
                      <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center mx-auto mb-4 animate-pulse">
                        <IconComponent className="w-8 h-8 text-white" />
                      </div>
                      {index < 3 && (
                        <div className="hidden md:block absolute top-8 left-full w-full h-0.5 bg-gradient-to-r from-blue-300 to-purple-300 transform -translate-x-4"></div>
                      )}
                    </div>
                    <h3 className="font-bold text-gray-900 mb-2">{step.title}</h3>
                    <p className="text-sm text-gray-600">{step.desc}</p>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="px-6 py-20">
        <div className="max-w-4xl mx-auto text-center">
          <div className="bg-gradient-to-r from-blue-500 to-purple-500 rounded-3xl p-12 text-white relative overflow-hidden">
            <div className="relative z-10">
              <h2 className="text-4xl font-bold mb-6">Ready to Transform Your Pension Planning?</h2>
              <p className="text-xl mb-8 opacity-90">
                Experience the power of Aegis AI's multi-agent intelligence system that delivers clarity from complexity.
              </p>
              <button className="bg-white text-gray-900 px-8 py-4 rounded-full font-bold hover:shadow-xl transform hover:scale-105 transition-all duration-300">
                Start Your Analysis Now
              </button>
            </div>
            
            {/* Background Animation */}
            <div className="absolute inset-0 opacity-10">
              <AnimatedLine 
                path="M0,50 Q100,0 200,50 T400,50 Q300,100 200,100 T0,100 Z" 
                delay={0}
                duration={3}
              />
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="px-6 py-8 bg-gray-50">
        <div className="max-w-7xl mx-auto text-center">
          <div className="flex items-center justify-center space-x-2 mb-4">
            <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg flex items-center justify-center">
              <CpuChipIcon className="w-4 h-4 text-white" />
            </div>
            <span className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Aegis AI
            </span>
          </div>
          <p className="text-gray-600">
            © 2025 Aegis AI. Multi-agent intelligence for smarter pension planning.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default AegisAILanding;