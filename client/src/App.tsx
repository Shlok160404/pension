import { useState, useEffect } from 'react';
import { tokenManager } from './services/api';
import PensionAILanding from './pages/LandingPage';
import Login from './components/auth/Login';
import Signup from './components/auth/Signup';
import ResidentDashboard from './components/dashboards/ResidentDashboard';
import AdvisorDashboard from './components/dashboards/AdvisorDashboard';
import RegulatorDashboard from './components/dashboards/RegulatorDashboard';

const pages = [
  { id: 'login', component: Login },
  { id: 'signup', component: Signup },
  { id: 'resident', component: ResidentDashboard },
  { id: 'advisor', component: AdvisorDashboard },
  { id: 'regulator', component: RegulatorDashboard },
];

function App() {
  const [currentPage, setCurrentPage] = useState('landing');
  const [_isAuthenticated, setIsAuthenticated] = useState(false);
  const [_currentUser, setCurrentUser] = useState<any>(null);

  useEffect(() => {
    const checkAuth = () => {
      const authenticated = tokenManager.isAuthenticated();
      const user = tokenManager.getUser();
      
      setIsAuthenticated(authenticated);
      setCurrentUser(user);
      
      // Only redirect to dashboard if user explicitly navigates there
      // Don't auto-redirect on app load - always start with landing page
      // if (authenticated && user) {
      //   setCurrentPage(user.role === 'resident' ? 'resident' : 
      //                user.role === 'advisor' ? 'advisor' : 
      //                user.role === 'regulator' ? 'regulator' : 'resident');
      // }
    };

    checkAuth();
  }, []);

  // Handle successful login
  const handleLoginSuccess = (userData: any) => {
    setIsAuthenticated(true);
    setCurrentUser(userData);
    
    // Redirect to appropriate dashboard based on role
    const targetPage = userData.role === 'resident' ? 'resident' : 
                      userData.role === 'advisor' ? 'advisor' : 
                      userData.role === 'regulator' ? 'regulator' : 'resident';
    setCurrentPage(targetPage);
  };



  const CurrentComponent = pages.find(page => page.id === currentPage)?.component;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Main Content */}
      <main>
        {currentPage === 'login' ? (
          <Login onLoginSuccess={handleLoginSuccess} />
        ) : currentPage === 'signup' ? (
          <Signup />
        ) : currentPage === 'landing' ? (
          <PensionAILanding onNavigate={setCurrentPage} />
        ) : CurrentComponent ? (
          <CurrentComponent />
        ) : (
          <PensionAILanding onNavigate={setCurrentPage} />
        )}
      </main>
    </div>
  );
}

export default App;
