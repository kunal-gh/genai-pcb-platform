import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  Zap, Cpu, Layers, Sparkles, ArrowRight, 
  Brain, Gauge, Shield, Code2 
} from 'lucide-react';

const HomePage: React.FC = () => {
  const features = [
    {
      icon: Brain,
      title: 'AI-Powered Design',
      description: 'Transform natural language into production-ready PCB designs using advanced machine learning',
      color: 'from-circuit-electric to-circuit-cyan',
    },
    {
      icon: Cpu,
      title: 'Smart Routing',
      description: 'Hybrid RL + A* algorithm for optimal trace routing with 40% faster performance',
      color: 'from-circuit-cyan to-circuit-accent',
    },
    {
      icon: Gauge,
      title: 'Real-time Validation',
      description: 'Automated DRC, ERC, and DFM checks with 95%+ pass rate',
      color: 'from-circuit-accent to-circuit-electric',
    },
    {
      icon: Shield,
      title: 'Production Ready',
      description: 'Export Gerber files, drill files, and complete manufacturing documentation',
      color: 'from-circuit-green to-circuit-cyan',
    },
  ];

  const stats = [
    { value: '40%', label: 'Faster Routing' },
    { value: '95%', label: 'DFM Pass Rate' },
    { value: '33%', label: 'Fewer Vias' },
    { value: '<100ms', label: 'Inference Time' },
  ];

  return (
    <div className="min-h-screen pt-20">
      {/* Hero Section */}
      <section className="relative px-4 py-20 sm:py-32">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center"
          >
            {/* Badge */}
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.2 }}
              className="inline-flex items-center space-x-2 glass-card px-4 py-2 rounded-full mb-8"
            >
              <Sparkles className="w-4 h-4 text-circuit-cyan" />
              <span className="text-sm text-circuit-cyan font-medium">
                AI-Powered PCB Design Platform
              </span>
            </motion.div>

            {/* Main Heading */}
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="text-5xl sm:text-7xl font-bold mb-6 leading-tight"
            >
              <span className="gradient-text">Stuff Made</span>
              <br />
              <span className="text-white">Incredibly</span>{' '}
              <span className="gradient-text">Easy</span>
            </motion.h1>

            {/* Subtitle */}
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="text-xl sm:text-2xl text-white/60 mb-12 max-w-3xl mx-auto"
            >
              Transform your circuit ideas into production-ready PCB designs
              using state-of-the-art machine learning and intelligent automation
            </motion.p>

            {/* CTA Buttons */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="flex flex-col sm:flex-row items-center justify-center gap-4"
            >
              <Link to="/studio">
                <motion.button
                  className="btn-primary flex items-center space-x-2 text-lg px-8 py-4"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Layers className="w-5 h-5" />
                  <span>Start Designing</span>
                  <ArrowRight className="w-5 h-5" />
                </motion.button>
              </Link>
              
              <Link to="/gallery">
                <motion.button
                  className="btn-secondary flex items-center space-x-2 text-lg px-8 py-4"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Code2 className="w-5 h-5" />
                  <span>View Examples</span>
                </motion.button>
              </Link>
            </motion.div>
          </motion.div>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="grid grid-cols-2 md:grid-cols-4 gap-6 mt-20"
          >
            {stats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.7 + index * 0.1 }}
                className="glass-card p-6 text-center"
                whileHover={{ scale: 1.05, y: -5 }}
              >
                <div className="text-3xl sm:text-4xl font-bold gradient-text mb-2">
                  {stat.value}
                </div>
                <div className="text-sm text-white/60">{stat.label}</div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="relative px-4 py-20">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl sm:text-5xl font-bold mb-4">
              <span className="gradient-text">Powerful Features</span>
            </h2>
            <p className="text-xl text-white/60 max-w-2xl mx-auto">
              Everything you need to design professional PCBs with AI assistance
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-6">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.1 }}
                  className="card-3d p-8 group cursor-pointer"
                  whileHover={{ scale: 1.02 }}
                >
                  <div className={`w-14 h-14 rounded-xl bg-gradient-to-br ${feature.color} flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300`}>
                    <Icon className="w-7 h-7 text-white" />
                  </div>
                  
                  <h3 className="text-2xl font-bold mb-3 text-white group-hover:gradient-text transition-all duration-300">
                    {feature.title}
                  </h3>
                  
                  <p className="text-white/60 leading-relaxed">
                    {feature.description}
                  </p>

                  <motion.div
                    className="mt-6 flex items-center text-circuit-cyan group-hover:text-circuit-electric transition-colors"
                    whileHover={{ x: 5 }}
                  >
                    <span className="text-sm font-medium">Learn more</span>
                    <ArrowRight className="w-4 h-4 ml-2" />
                  </motion.div>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative px-4 py-20">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            className="glass-card p-12 text-center relative overflow-hidden"
          >
            {/* Animated background glow */}
            <motion.div
              className="absolute inset-0 bg-gradient-to-r from-circuit-electric/20 via-circuit-cyan/20 to-circuit-accent/20"
              animate={{
                backgroundPosition: ['0% 50%', '100% 50%', '0% 50%'],
              }}
              transition={{
                duration: 5,
                repeat: Infinity,
                ease: "linear",
              }}
              style={{ backgroundSize: '200% 200%' }}
            />

            <div className="relative z-10">
              <Zap className="w-16 h-16 mx-auto mb-6 text-circuit-cyan" />
              <h2 className="text-3xl sm:text-4xl font-bold mb-4">
                Ready to <span className="gradient-text">revolutionize</span> your PCB design?
              </h2>
              <p className="text-lg text-white/60 mb-8 max-w-2xl mx-auto">
                Join the future of PCB design with AI-powered automation and intelligent routing
              </p>
              <Link to="/studio">
                <motion.button
                  className="btn-primary text-lg px-10 py-4"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  Get Started Now
                </motion.button>
              </Link>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default HomePage;
