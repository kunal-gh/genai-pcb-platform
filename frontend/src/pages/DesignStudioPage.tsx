import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Sparkles, Send, Loader2, Download, Eye, 
  Settings, Layers, CheckCircle2, AlertCircle 
} from 'lucide-react';
import PCBCanvas from '../components/PCBCanvas';
import ToolPanel from '../components/ToolPanel';

interface DesignState {
  status: 'idle' | 'generating' | 'success' | 'error';
  progress: number;
  message: string;
}

const DesignStudioPage: React.FC = () => {
  const [prompt, setPrompt] = useState('');
  const [designState, setDesignState] = useState<DesignState>({
    status: 'idle',
    progress: 0,
    message: '',
  });
  const [showTools, setShowTools] = useState(true);
  const [activeTab, setActiveTab] = useState<'design' | 'preview' | 'export'>('design');

  const handleGenerate = async () => {
    if (!prompt.trim()) return;

    setDesignState({ status: 'generating', progress: 0, message: 'Initializing AI...' });

    // Simulate design generation process
    const steps = [
      { progress: 20, message: 'Parsing circuit description...' },
      { progress: 40, message: 'Placing components...' },
      { progress: 60, message: 'Routing traces with RL agent...' },
      { progress: 80, message: 'Running DRC validation...' },
      { progress: 100, message: 'Design complete!' },
    ];

    for (const step of steps) {
      await new Promise(resolve => setTimeout(resolve, 1500));
      setDesignState({ status: 'generating', ...step });
    }

    setDesignState({ status: 'success', progress: 100, message: 'PCB design ready!' });
  };

  const examplePrompts = [
    'LED blinker circuit with 555 timer',
    'Arduino Uno compatible board',
    'USB-C power delivery circuit',
    'Audio amplifier with TDA2030',
  ];

  return (
    <div className="min-h-screen pt-20 px-4 pb-8">
      <div className="max-w-[1800px] mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-4xl font-bold mb-2">
                <span className="gradient-text">Design Studio</span>
              </h1>
              <p className="text-white/60">
                Describe your circuit and watch AI bring it to life
              </p>
            </div>
            
            <motion.button
              className="btn-secondary flex items-center space-x-2"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setShowTools(!showTools)}
            >
              <Settings className="w-4 h-4" />
              <span>{showTools ? 'Hide' : 'Show'} Tools</span>
            </motion.button>
          </div>
        </motion.div>

        <div className="grid lg:grid-cols-[1fr_400px] gap-6">
          {/* Main Design Area */}
          <div className="space-y-6">
            {/* Input Section */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="glass-card p-6"
            >
              <div className="flex items-center space-x-2 mb-4">
                <Sparkles className="w-5 h-5 text-circuit-cyan" />
                <h2 className="text-xl font-semibold">Describe Your Circuit</h2>
              </div>

              <div className="relative">
                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Example: Create a simple LED blinker circuit using a 555 timer IC, with adjustable frequency using a potentiometer..."
                  className="input-glass min-h-[120px] resize-none"
                  disabled={designState.status === 'generating'}
                />
                
                <motion.button
                  className="absolute bottom-4 right-4 btn-primary flex items-center space-x-2"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={handleGenerate}
                  disabled={!prompt.trim() || designState.status === 'generating'}
                >
                  {designState.status === 'generating' ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Generating...</span>
                    </>
                  ) : (
                    <>
                      <Send className="w-4 h-4" />
                      <span>Generate PCB</span>
                    </>
                  )}
                </motion.button>
              </div>

              {/* Example Prompts */}
              <div className="mt-4 flex flex-wrap gap-2">
                {examplePrompts.map((example) => (
                  <motion.button
                    key={example}
                    className="text-sm px-3 py-1.5 glass-card hover:bg-glass-medium transition-all rounded-lg text-white/70 hover:text-white"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => setPrompt(example)}
                  >
                    {example}
                  </motion.button>
                ))}
              </div>
            </motion.div>

            {/* Progress Section */}
            {designState.status !== 'idle' && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="glass-card p-6"
              >
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      {designState.status === 'generating' && (
                        <Loader2 className="w-5 h-5 text-circuit-cyan animate-spin" />
                      )}
                      {designState.status === 'success' && (
                        <CheckCircle2 className="w-5 h-5 text-circuit-green" />
                      )}
                      {designState.status === 'error' && (
                        <AlertCircle className="w-5 h-5 text-red-500" />
                      )}
                      <span className="font-medium">{designState.message}</span>
                    </div>
                    <span className="text-sm text-white/60">{designState.progress}%</span>
                  </div>

                  <div className="relative h-2 bg-glass-dark rounded-full overflow-hidden">
                    <motion.div
                      className="absolute inset-y-0 left-0 bg-gradient-to-r from-circuit-electric to-circuit-cyan"
                      initial={{ width: 0 }}
                      animate={{ width: `${designState.progress}%` }}
                      transition={{ duration: 0.5 }}
                    />
                  </div>
                </motion.div>
              )}

            {/* Tabs */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="glass-card p-2 flex space-x-2"
            >
              {[
                { id: 'design', label: 'Design View', icon: Layers },
                { id: 'preview', label: '3D Preview', icon: Eye },
                { id: 'export', label: 'Export', icon: Download },
              ].map((tab) => {
                const Icon = tab.icon;
                return (
                  <motion.button
                    key={tab.id}
                    className={`flex-1 flex items-center justify-center space-x-2 px-4 py-3 rounded-lg transition-all ${
                      activeTab === tab.id
                        ? 'bg-gradient-to-r from-circuit-electric to-circuit-cyan text-white'
                        : 'text-white/60 hover:text-white hover:bg-glass-light'
                    }`}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => setActiveTab(tab.id as any)}
                  >
                    <Icon className="w-4 h-4" />
                    <span className="font-medium">{tab.label}</span>
                  </motion.button>
                );
              })}
            </motion.div>

            {/* Canvas Area */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="glass-card p-6 min-h-[600px]"
            >
              <PCBCanvas activeTab={activeTab} designState={designState} />
            </motion.div>
          </div>

          {/* Tool Panel */}
          {showTools && (
            <motion.div
              initial={{ opacity: 0, x: 100 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 100 }}
              transition={{ type: "spring", damping: 20 }}
            >
              <ToolPanel />
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DesignStudioPage;
