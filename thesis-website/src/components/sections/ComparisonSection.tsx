'use client';

import { motion } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import { useState } from 'react';

export default function ComparisonSection() {
  const { ref, inView } = useInView({
    threshold: 0.1,
    triggerOnce: true
  });

  const [selectedMetric, setSelectedMetric] = useState('accuracy');

  const optimizers = [
    {
      name: 'Gradient Descent',
      accuracy: 16.8,
      loss: 2.26,
      time: 17.3,
      memory: 416,
      category: 'First-Order',
      description: 'Simple baseline method with fixed learning rate'
    },
    {
      name: 'Adam',
      accuracy: 88.5,
      loss: 0.39,
      time: 17.4,
      memory: 416,
      category: 'Adaptive First-Order',
      description: 'Momentum with per-parameter learning rates'
    },
    {
      name: 'L-BFGS',
      accuracy: 97.5,
      loss: 0.28,
      time: 10.0,
      memory: 427,
      category: 'Quasi-Newton',
      description: 'Limited-memory approximation of Newton method'
    },
    {
      name: 'Trust-Region Newton',
      accuracy: 97.8,
      loss: 0.22,
      time: 25.4,
      memory: 469,
      category: 'Newton Method',
      description: 'Full second-order with trust region control'
    },
    {
      name: 'Interior-Point',
      accuracy: 95.1,
      loss: 0.18,
      time: 299.5,
      memory: 438,
      category: 'Constrained Newton',
      description: 'Newton method with constraint handling'
    }
  ];

  const getMetricValue = (optimizer: any, metric: string) => {
    switch (metric) {
      case 'accuracy': return optimizer.accuracy;
      case 'loss': return optimizer.loss;
      case 'time': return optimizer.time;
      case 'memory': return optimizer.memory;
      default: return 0;
    }
  };

  const getMetricUnit = (metric: string) => {
    switch (metric) {
      case 'accuracy': return '%';
      case 'loss': return '';
      case 'time': return 's';
      case 'memory': return 'MB';
      default: return '';
    }
  };

  const getBestValue = (metric: string) => {
    const values = optimizers.map(opt => getMetricValue(opt, metric));
    return metric === 'accuracy' ? Math.max(...values) :
           metric === 'loss' ? Math.min(...values.filter(v => v > 0)) :
           Math.min(...values);
  };

  const getRelativePerformance = (optimizer: any, metric: string) => {
    const value = getMetricValue(optimizer, metric);
    const bestValue = getBestValue(metric);
    
    if (metric === 'accuracy') {
      return (value / bestValue) * 100;
    } else {
      return (bestValue / value) * 100;
    }
  };

  return (
    <section ref={ref} id="comparison" className="py-20 bg-gradient-to-br from-white to-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
        >
          <div className="inline-flex items-center px-4 py-2 bg-gray-100 text-gray-700 rounded-full text-sm font-medium mb-6">
            Performance Analysis • Comprehensive Comparison
          </div>
          <h2 className="text-4xl md:text-5xl font-light text-gray-900 mb-6">
            Optimization Comparison
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
            A systematic evaluation of algorithm performance across accuracy, efficiency, and computational cost.
          </p>
        </motion.div>

        {/* Performance Table */}
        <motion.div
          className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 mb-16"
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          <h3 className="text-2xl font-semibold text-gray-900 mb-8 text-center">
            Experimental Results Summary
          </h3>
          
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-4 px-4 font-semibold text-gray-900">Optimizer</th>
                  <th className="text-left py-4 px-4 font-semibold text-gray-900">Category</th>
                  <th className="text-center py-4 px-4 font-semibold text-gray-900">Accuracy (%)</th>
                  <th className="text-center py-4 px-4 font-semibold text-gray-900">Final Loss</th>
                  <th className="text-center py-4 px-4 font-semibold text-gray-900">Time (s)</th>
                  <th className="text-center py-4 px-4 font-semibold text-gray-900">Memory (MB)</th>
                </tr>
              </thead>
              <tbody>
                {optimizers.map((optimizer, index) => (
                  <tr key={optimizer.name} className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="py-4 px-4">
                      <div className="font-medium text-gray-900">{optimizer.name}</div>
                      <div className="text-sm text-gray-500">{optimizer.description}</div>
                    </td>
                    <td className="py-4 px-4 text-sm text-gray-600">
                      {optimizer.category}
                    </td>
                    <td className="py-4 px-4 text-center">
                      <div className="font-semibold text-gray-900">{optimizer.accuracy}%</div>
                      {optimizer.accuracy === getBestValue('accuracy') && (
                        <div className="text-xs text-gray-500 mt-1">Best</div>
                      )}
                    </td>
                    <td className="py-4 px-4 text-center">
                      <div className="font-semibold text-gray-900">{optimizer.loss}</div>
                      {optimizer.loss === getBestValue('loss') && (
                        <div className="text-xs text-gray-500 mt-1">Best</div>
                      )}
                    </td>
                    <td className="py-4 px-4 text-center">
                      <div className="font-semibold text-gray-900">{optimizer.time}s</div>
                      {optimizer.time === getBestValue('time') && (
                        <div className="text-xs text-gray-500 mt-1">Best</div>
                      )}
                    </td>
                    <td className="py-4 px-4 text-center">
                      <div className="font-semibold text-gray-900">{optimizer.memory}MB</div>
                      {optimizer.memory === getBestValue('memory') && (
                        <div className="text-xs text-gray-500 mt-1">Best</div>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>

        {/* Interactive Performance Visualization */}
        <motion.div
          className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 mb-16"
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, delay: 0.4 }}
        >
          <h3 className="text-2xl font-semibold text-gray-900 mb-8 text-center">
            Performance Visualization
          </h3>
          
          {/* Metric Selector */}
          <div className="flex flex-wrap justify-center gap-4 mb-8">
            {['accuracy', 'loss', 'time', 'memory'].map((metric) => (
              <button
                key={metric}
                onClick={() => setSelectedMetric(metric)}
                className={`px-6 py-3 rounded-lg font-medium text-sm transition-colors ${
                  selectedMetric === metric
                    ? 'bg-gray-900 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {metric.charAt(0).toUpperCase() + metric.slice(1)}
              </button>
            ))}
          </div>

          {/* Bar Chart */}
          <div className="space-y-6">
            {optimizers.map((optimizer, index) => {
              const performance = getRelativePerformance(optimizer, selectedMetric);
              const value = getMetricValue(optimizer, selectedMetric);
              const unit = getMetricUnit(selectedMetric);
              
              return (
                <motion.div
                  key={optimizer.name}
                  className="flex items-center space-x-4"
                  initial={{ opacity: 0, x: -20 }}
                  animate={inView ? { opacity: 1, x: 0 } : {}}
                  transition={{ duration: 0.6, delay: 0.6 + index * 0.1 }}
                >
                  <div className="w-32 text-sm font-medium text-gray-900 text-right">
                    {optimizer.name}
                  </div>
                  <div className="flex-1 bg-gray-100 rounded-full h-8 relative overflow-hidden">
                    <motion.div
                      className="h-full bg-gray-600 rounded-full flex items-center justify-end pr-3"
                      initial={{ width: 0 }}
                      animate={inView ? { width: `${performance}%` } : {}}
                      transition={{ duration: 1, delay: 0.8 + index * 0.1 }}
                    >
                      <span className="text-white text-xs font-medium">
                        {value}{unit}
                      </span>
                    </motion.div>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </motion.div>

        {/* Key Insights */}
        <motion.div
          className="grid md:grid-cols-2 gap-12"
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, delay: 0.6 }}
        >
          {/* Mathematical Insights */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
            <h4 className="text-xl font-semibold text-gray-900 mb-6">Mathematical Insights</h4>
            <div className="space-y-4 text-gray-700">
              <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
                <h5 className="font-medium text-gray-900 mb-2">First-Order vs Second-Order</h5>
                <p className="text-sm leading-relaxed">
                  Second-order methods (L-BFGS, Trust-Region) achieved 97%+ accuracy compared to 
                  first-order methods' 16-88%, demonstrating the value of curvature information.
                </p>
              </div>
              <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
                <h5 className="font-medium text-gray-900 mb-2">Adaptive Mechanisms</h5>
                <p className="text-sm leading-relaxed">
                  Adam's 5.26× improvement over Gradient Descent shows how adaptive learning 
                  rates and momentum dramatically enhance first-order optimization.
                </p>
              </div>
              <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
                <h5 className="font-medium text-gray-900 mb-2">Convergence Rates</h5>
                <p className="text-sm leading-relaxed">
                  Trust-Region Newton achieved the lowest loss (0.22) through its quadratic 
                  convergence near the optimum, while L-BFGS provided the best time efficiency.
                </p>
              </div>
            </div>
          </div>

          {/* Practical Recommendations */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
            <h4 className="text-xl font-semibold text-gray-900 mb-6">Practical Recommendations</h4>
            <div className="space-y-4 text-gray-700">
              <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
                <h5 className="font-medium text-gray-900 mb-2">For High Accuracy</h5>
                <p className="text-sm leading-relaxed">
                  Use Trust-Region Newton when maximum precision is required and computational 
                  cost is acceptable (scientific computing, critical applications).
                </p>
              </div>
              <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
                <h5 className="font-medium text-gray-900 mb-2">For Speed & Efficiency</h5>
                <p className="text-sm leading-relaxed">
                  L-BFGS offers the best balance of high accuracy (97.5%) and fast convergence 
                  (10.0s) for small to medium-scale problems.
                </p>
              </div>
              <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
                <h5 className="font-medium text-gray-900 mb-2">For Teaching & Understanding</h5>
                <p className="text-sm leading-relaxed">
                  Gradient Descent provides clear conceptual foundation, while Adam demonstrates 
                  modern adaptive techniques used in deep learning.
                </p>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Final Summary */}
        <motion.div
          className="mt-16 bg-gray-50 border border-gray-200 rounded-lg p-8 text-center"
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, delay: 0.8 }}
        >
          <h4 className="text-xl font-semibold text-gray-900 mb-4">
            Conclusion
          </h4>
          <p className="text-gray-700 leading-relaxed max-w-4xl mx-auto">
            This empirical comparison demonstrates the fundamental trade-offs in optimization algorithm design. 
            While first-order methods offer simplicity and scalability, second-order methods provide superior 
            convergence quality for problems where computational resources permit. The choice of optimizer 
            should align with specific requirements for accuracy, speed, and computational constraints.
          </p>
        </motion.div>
      </div>
    </section>
  );
} 