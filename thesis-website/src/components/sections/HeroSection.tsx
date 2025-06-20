'use client';

import { motion } from 'framer-motion';
import { ChevronDownIcon } from '@heroicons/react/24/outline';

export default function HeroSection() {
  const scrollToBackground = () => {
    document.getElementById('background')?.scrollIntoView({ 
      behavior: 'smooth' 
    });
  };

  return (
    <section className="min-h-screen flex items-center justify-center relative bg-gradient-to-br from-gray-50 via-white to-gray-100">
      {/* Subtle Academic Pattern */}
      <div className="absolute inset-0 opacity-3">
        <div className="absolute inset-0" style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg width='40' height='40' viewBox='0 0 40 40' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23374151' fill-opacity='0.1'%3E%3Cpath d='M20 20c0-5.5-4.5-10-10-10s-10 4.5-10 10 4.5 10 10 10 10-4.5 10-10zm10 0c0-5.5-4.5-10-10-10s-10 4.5-10 10 4.5 10 10 10 10-4.5 10-10z'/%3E%3C/g%3E%3C/svg%3E")`,
        }} />
      </div>

      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 text-center relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          {/* Academic Badge */}
          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="inline-flex items-center px-4 py-2 bg-gray-100 text-gray-700 rounded-full text-sm font-medium mb-8"
          >
            Mathematical Research • Neural Network Optimization
          </motion.div>

          <h1 className="text-4xl md:text-6xl lg:text-7xl font-light text-gray-900 mb-6 tracking-tight">
            Empirical Comparison of{' '}
            <span className="font-semibold text-gray-800">
              NLP Solvers
            </span>
          </h1>
          
          <h2 className="text-xl md:text-2xl lg:text-3xl text-gray-600 mb-8 font-light">
            for Small Neural Network Training
          </h2>
          
          <p className="text-lg md:text-xl text-gray-700 max-w-4xl mx-auto mb-12 leading-relaxed font-light">
            An interactive mathematical exploration of optimization algorithms used in neural network training.
            Designed for <strong className="font-semibold text-gray-900">mathematics professors</strong> with rigorous 
            mathematical foundations, accessible explanations, and engaging 3D visualizations.
          </p>

          {/* Key Features Grid */}
          <div className="grid md:grid-cols-3 gap-8 mb-12 max-w-4xl mx-auto">
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.4 }}
              className="text-center group"
            >
              <div className="w-16 h-16 bg-gray-100 border border-gray-300 rounded-lg flex items-center justify-center mx-auto mb-4 group-hover:bg-gray-200 transition-colors duration-300">
                <span className="text-2xl text-gray-600">∇</span>
              </div>
              <h3 className="font-semibold text-gray-900 mb-2 text-lg">Mathematical Rigor</h3>
              <p className="text-sm text-gray-600 leading-relaxed">
                Complete derivations, equations, and calculus foundations with LaTeX rendering
              </p>
            </motion.div>
            
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.5 }}
              className="text-center group"
            >
              <div className="w-16 h-16 bg-gray-100 border border-gray-300 rounded-lg flex items-center justify-center mx-auto mb-4 group-hover:bg-gray-200 transition-colors duration-300">
                <span className="text-2xl text-gray-600">3D</span>
              </div>
              <h3 className="font-semibold text-gray-900 mb-2 text-lg">Interactive Visualization</h3>
              <p className="text-sm text-gray-600 leading-relaxed">
                Real-time visualizations of optimizer trajectories on loss surfaces
              </p>
            </motion.div>
            
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.6 }}
              className="text-center group"
            >
              <div className="w-16 h-16 bg-gray-100 border border-gray-300 rounded-lg flex items-center justify-center mx-auto mb-4 group-hover:bg-gray-200 transition-colors duration-300">
                <span className="text-2xl text-gray-600">⚖</span>
              </div>
              <h3 className="font-semibold text-gray-900 mb-2 text-lg">Empirical Analysis</h3>
              <p className="text-sm text-gray-600 leading-relaxed">
                Quantitative comparison of convergence, efficiency, and trade-offs
              </p>
            </motion.div>
          </div>

          <div className="flex flex-col sm:flex-row gap-6 justify-center items-center">
            <motion.button
              onClick={scrollToBackground}
              className="bg-gray-900 hover:bg-gray-800 text-white px-8 py-4 rounded-lg font-semibold text-lg shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-105"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Begin Mathematical Journey
            </motion.button>
            
            <div className="text-sm text-gray-500 font-medium">
              Scroll to explore • Interactive content ahead
            </div>
          </div>
        </motion.div>
      </div>

      {/* Elegant Scroll indicator */}
      <motion.div
        className="absolute bottom-8 left-1/2 transform -translate-x-1/2 cursor-pointer"
        animate={{ y: [0, 8, 0] }}
        transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
        onClick={scrollToBackground}
      >
        <div className="flex flex-col items-center space-y-2">
          <ChevronDownIcon className="w-6 h-6 text-gray-400 hover:text-gray-600 transition-colors" />
          <div className="w-0.5 h-8 bg-gradient-to-b from-gray-300 to-transparent"></div>
        </div>
      </motion.div>
    </section>
  );
} 