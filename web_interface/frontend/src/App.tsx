import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AppProvider } from './context/AppContext';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import FineTuning from './pages/FineTuning';
import PromptTesting from './pages/PromptTesting';
import Results from './pages/Results';
import Analytics from './pages/Analytics';

function App() {
  return (
    <AppProvider>
      <Router>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="fine-tuning" element={<FineTuning />} />
            <Route path="prompt-testing" element={<PromptTesting />} />
            <Route path="results" element={<Results />} />
            <Route path="analytics" element={<Analytics />} />
          </Route>
        </Routes>
      </Router>
    </AppProvider>
  );
}

export default App;
