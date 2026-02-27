import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Search, Filter, Download, Eye, Heart, Clock } from 'lucide-react';

interface Design {
  id: string;
  title: string;
  description: string;
  image: string;
  layers: number;
  components: number;
  likes: number;
  date: string;
  tags: string[];
}

const GalleryPage: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedFilter, setSelectedFilter] = useState('all');

  const designs: Design[] = [
    {
      id: '1',
      title: 'Arduino Uno Compatible',
      description: 'Full Arduino Uno R3 compatible board with USB-C',
      image: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      layers: 2,
      components: 45,
      likes: 234,
      date: '2024-02-15',
      tags: ['Arduino', 'USB-C', 'Development'],
    },
    {
      id: '2',
      title: 'LED Matrix Driver',
      description: '8x8 LED matrix controller with MAX7219',
      image: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
      layers: 2,
      components: 12,
      likes: 189,
      date: '2024-02-14',
      tags: ['LED', 'Display', 'Driver'],
    },
    {
      id: '3',
      title: 'USB-C Power Delivery',
      description: 'PD 3.0 compatible power supply with 100W output',
      image: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
      layers: 4,
      components: 67,
      likes: 312,
      date: '2024-02-13',
      tags: ['Power', 'USB-C', 'PD'],
    },
    {
      id: '4',
      title: 'Audio Amplifier',
      description: 'Class D amplifier with TDA2030, 2x15W stereo',
      image: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)',
      layers: 2,
      components: 28,
      likes: 156,
      date: '2024-02-12',
      tags: ['Audio', 'Amplifier', 'Stereo'],
    },
    {
      id: '5',
      title: 'ESP32 IoT Module',
      description: 'WiFi + Bluetooth development board',
      image: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
      layers: 2,
      components: 34,
      likes: 278,
      date: '2024-02-11',
      tags: ['IoT', 'WiFi', 'Bluetooth'],
    },
    {
      id: '6',
      title: 'Motor Controller',
      description: 'H-bridge motor driver with L298N, 2A per channel',
      image: 'linear-gradient(135deg, #30cfd0 0%, #330867 100%)',
      layers: 2,
      components: 19,
      likes: 201,
      date: '2024-02-10',
      tags: ['Motor', 'Driver', 'Robotics'],
    },
  ];

  const filters = ['all', 'popular', 'recent', 'power', 'audio', 'iot'];

  return (
    <div className="min-h-screen pt-20 px-4 pb-12">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12"
        >
          <h1 className="text-4xl sm:text-5xl font-bold mb-4">
            <span className="gradient-text">Design Gallery</span>
          </h1>
          <p className="text-xl text-white/60">
            Explore community-created PCB designs and get inspired
          </p>
        </motion.div>

        {/* Search and Filters */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="mb-8 space-y-4"
        >
          {/* Search Bar */}
          <div className="relative">
            <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-white/40" />
            <input
              type="text"
              placeholder="Search designs..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="input-glass pl-12 pr-4 py-4 text-lg"
            />
          </div>

          {/* Filter Tabs */}
          <div className="flex items-center space-x-2 overflow-x-auto pb-2">
            <Filter className="w-5 h-5 text-white/60 flex-shrink-0" />
            {filters.map((filter) => (
              <motion.button
                key={filter}
                className={`px-4 py-2 rounded-lg font-medium capitalize whitespace-nowrap transition-all ${
                  selectedFilter === filter
                    ? 'bg-gradient-to-r from-circuit-electric to-circuit-cyan text-white'
                    : 'glass-card text-white/60 hover:text-white hover:bg-glass-medium'
                }`}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setSelectedFilter(filter)}
              >
                {filter}
              </motion.button>
            ))}
          </div>
        </motion.div>

        {/* Gallery Grid */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {designs.map((design, index) => (
            <motion.div
              key={design.id}
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 * index }}
              className="card-3d group cursor-pointer"
              whileHover={{ y: -8 }}
            >
              {/* Image */}
              <div 
                className="h-48 rounded-t-2xl relative overflow-hidden"
                style={{ background: design.image }}
              >
                <div className="absolute inset-0 bg-gradient-to-t from-circuit-darker/80 to-transparent" />
                
                {/* Hover Actions */}
                <motion.div
                  initial={{ opacity: 0 }}
                  whileHover={{ opacity: 1 }}
                  className="absolute inset-0 flex items-center justify-center space-x-3"
                >
                  <motion.button
                    className="glass-card p-3 rounded-full"
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                  >
                    <Eye className="w-5 h-5" />
                  </motion.button>
                  <motion.button
                    className="glass-card p-3 rounded-full"
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                  >
                    <Download className="w-5 h-5" />
                  </motion.button>
                </motion.div>

                {/* Stats Badge */}
                <div className="absolute top-4 right-4 glass-card px-3 py-1.5 rounded-full flex items-center space-x-2">
                  <Heart className="w-4 h-4 text-red-400" />
                  <span className="text-sm font-medium">{design.likes}</span>
                </div>
              </div>

              {/* Content */}
              <div className="p-6">
                <h3 className="text-xl font-bold mb-2 group-hover:gradient-text transition-all">
                  {design.title}
                </h3>
                <p className="text-white/60 text-sm mb-4 line-clamp-2">
                  {design.description}
                </p>

                {/* Tags */}
                <div className="flex flex-wrap gap-2 mb-4">
                  {design.tags.slice(0, 3).map((tag) => (
                    <span
                      key={tag}
                      className="text-xs px-2 py-1 glass-card rounded-full text-circuit-cyan"
                    >
                      {tag}
                    </span>
                  ))}
                </div>

                {/* Meta Info */}
                <div className="flex items-center justify-between text-sm text-white/60">
                  <div className="flex items-center space-x-4">
                    <span>{design.layers} layers</span>
                    <span>•</span>
                    <span>{design.components} parts</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <Clock className="w-4 h-4" />
                    <span>{new Date(design.date).toLocaleDateString()}</span>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Load More */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="mt-12 text-center"
        >
          <motion.button
            className="btn-secondary px-8 py-3"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            Load More Designs
          </motion.button>
        </motion.div>
      </div>
    </div>
  );
};

export default GalleryPage;
