import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Sliders, Layers, Zap, Shield, Settings, 
  ChevronDown, ChevronUp 
} from 'lucide-react';

interface Section {
  id: string;
  title: string;
  icon: any;
  content: React.ReactNode;
}

const ToolPanel: React.FC = () => {
  const [expandedSections, setExpandedSections] = useState<string[]>(['board', 'routing']);

  const toggleSection = (id: string) => {
    setExpandedSections(prev =>
      prev.includes(id) ? prev.filter(s => s !== id) : [...prev, id]
    );
  };

  const sections: Section[] = [
    {
      id: 'board',
      title: 'Board Settings',
      icon: Layers,
      content: (
        <div className="space-y-4">
          <div>
            <label className="text-sm text-white/60 mb-2 block">Board Size</label>
            <div className="grid grid-cols-2 gap-2">
              <input
                type="number"
                placeholder="Width (mm)"
                className="input-glass text-sm py-2"
                defaultValue="100"
              />
              <input
                type="number"
                placeholder="Height (mm)"
                className="input-glass text-sm py-2"
                defaultValue="80"
              />
            </div>
          </div>

          <div>
            <label className="text-sm text-white/60 mb-2 block">Layers</label>
            <select className="input-glass text-sm py-2">
              <option>2 Layers</option>
              <option>4 Layers</option>
              <option>6 Layers</option>
              <option>8 Layers</option>
            </select>
          </div>

          <div>
            <label className="text-sm text-white/60 mb-2 block">Board Thickness</label>
            <select className="input-glass text-sm py-2">
              <option>1.6 mm (Standard)</option>
              <option>1.0 mm</option>
              <option>2.0 mm</option>
            </select>
          </div>
        </div>
      ),
    },
    {
      id: 'routing',
      title: 'Routing Options',
      icon: Zap,
      content: (
        <div className="space-y-4">
          <div>
            <label className="text-sm text-white/60 mb-2 block">Algorithm</label>
            <select className="input-glass text-sm py-2">
              <option>Hybrid (RL + A*)</option>
              <option>RL Only</option>
              <option>A* Only</option>
            </select>
          </div>

          <div>
            <label className="text-sm text-white/60 mb-2 flex items-center justify-between">
              <span>Trace Width</span>
              <span className="text-circuit-cyan">0.2 mm</span>
            </label>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              defaultValue="0.2"
              className="w-full accent-circuit-electric"
            />
          </div>

          <div>
            <label className="text-sm text-white/60 mb-2 flex items-center justify-between">
              <span>Clearance</span>
              <span className="text-circuit-cyan">0.15 mm</span>
            </label>
            <input
              type="range"
              min="0.1"
              max="0.5"
              step="0.05"
              defaultValue="0.15"
              className="w-full accent-circuit-electric"
            />
          </div>

          <div className="flex items-center justify-between glass-card p-3 rounded-lg">
            <span className="text-sm">Optimize for speed</span>
            <label className="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" className="sr-only peer" defaultChecked />
              <div className="w-11 h-6 bg-glass-dark rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-gradient-to-r peer-checked:from-circuit-electric peer-checked:to-circuit-cyan"></div>
            </label>
          </div>
        </div>
      ),
    },
    {
      id: 'validation',
      title: 'Validation',
      icon: Shield,
      content: (
        <div className="space-y-3">
          {[
            { name: 'Design Rule Check (DRC)', enabled: true },
            { name: 'Electrical Rule Check (ERC)', enabled: true },
            { name: 'Design for Manufacturing (DFM)', enabled: true },
            { name: 'Signal Integrity Analysis', enabled: false },
          ].map((rule) => (
            <div key={rule.name} className="flex items-center justify-between glass-card p-3 rounded-lg">
              <span className="text-sm">{rule.name}</span>
              <label className="relative inline-flex items-center cursor-pointer">
                <input type="checkbox" className="sr-only peer" defaultChecked={rule.enabled} />
                <div className="w-11 h-6 bg-glass-dark rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-gradient-to-r peer-checked:from-circuit-electric peer-checked:to-circuit-cyan"></div>
              </label>
            </div>
          ))}
        </div>
      ),
    },
    {
      id: 'advanced',
      title: 'Advanced',
      icon: Settings,
      content: (
        <div className="space-y-4">
          <div>
            <label className="text-sm text-white/60 mb-2 block">Via Size</label>
            <input
              type="number"
              placeholder="0.6 mm"
              className="input-glass text-sm py-2"
              defaultValue="0.6"
            />
          </div>

          <div>
            <label className="text-sm text-white/60 mb-2 block">Copper Weight</label>
            <select className="input-glass text-sm py-2">
              <option>1 oz (35 µm)</option>
              <option>2 oz (70 µm)</option>
              <option>3 oz (105 µm)</option>
            </select>
          </div>

          <div>
            <label className="text-sm text-white/60 mb-2 block">Surface Finish</label>
            <select className="input-glass text-sm py-2">
              <option>HASL</option>
              <option>ENIG</option>
              <option>OSP</option>
              <option>Immersion Silver</option>
            </select>
          </div>
        </div>
      ),
    },
  ];

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-4"
    >
      <div className="glass-card p-4">
        <div className="flex items-center space-x-2 mb-4">
          <Sliders className="w-5 h-5 text-circuit-cyan" />
          <h2 className="text-lg font-semibold">Design Tools</h2>
        </div>

        <div className="space-y-2">
          {sections.map((section) => {
            const Icon = section.icon;
            const isExpanded = expandedSections.includes(section.id);

            return (
              <div key={section.id} className="glass-card rounded-lg overflow-hidden">
                <motion.button
                  className="w-full px-4 py-3 flex items-center justify-between hover:bg-glass-light transition-colors"
                  onClick={() => toggleSection(section.id)}
                  whileHover={{ scale: 1.01 }}
                  whileTap={{ scale: 0.99 }}
                >
                  <div className="flex items-center space-x-3">
                    <Icon className="w-4 h-4 text-circuit-cyan" />
                    <span className="font-medium text-sm">{section.title}</span>
                  </div>
                  {isExpanded ? (
                    <ChevronUp className="w-4 h-4 text-white/60" />
                  ) : (
                    <ChevronDown className="w-4 h-4 text-white/60" />
                  )}
                </motion.button>

                <motion.div
                  initial={false}
                  animate={{
                    height: isExpanded ? 'auto' : 0,
                    opacity: isExpanded ? 1 : 0,
                  }}
                  transition={{ duration: 0.3 }}
                  className="overflow-hidden"
                >
                  <div className="px-4 pb-4 pt-2">
                    {section.content}
                  </div>
                </motion.div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="glass-card p-4">
        <h3 className="text-sm font-semibold mb-3 text-white/60">Quick Actions</h3>
        <div className="space-y-2">
          <motion.button
            className="w-full btn-secondary text-sm py-2"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            Reset to Defaults
          </motion.button>
          <motion.button
            className="w-full btn-ghost text-sm py-2"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            Save Preset
          </motion.button>
        </div>
      </div>
    </motion.div>
  );
};

export default ToolPanel;
