import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { motion } from 'framer-motion';
import HomePage from './pages/HomePage';
import DesignStudioPage from './pages/DesignStudioPage';
import GalleryPage from './pages/GalleryPage';
import Navigation from './components/Navigation';
import CircuitBackground from './components/CircuitBackground';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-circuit-darker relative overflow-hidden">
        {/* Animated circuit background */}
        <CircuitBackground />
        
        {/* Main content */}
        <div className="relative z-10">
          <Navigation />
          
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/studio" element={<DesignStudioPage />} />
            <Route path="/gallery" element={<GalleryPage />} />
          </Routes>
        </div>

        {/* Ambient glow effects */}
        <div className="fixed top-0 left-0 w-full h-full pointer-events-none overflow-hidden">
          <motion.div
            className="absolute top-1/4 left-1/4 w-96 h-96 bg-circuit-electric/20 rounded-full blur-3xl"
            animate={{
              scale: [1, 1.2, 1],
              opacity: [0.3, 0.5, 0.3],
            }}
            transition={{
              duration: 8,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />
          <motion.div
            className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-circuit-cyan/20 rounded-full blur-3xl"
            animate={{
              scale: [1.2, 1, 1.2],
              opacity: [0.5, 0.3, 0.5],
            }}
            transition={{
              duration: 10,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />
        </div>
      </div>
    </Router>
  );
}

export default App;
