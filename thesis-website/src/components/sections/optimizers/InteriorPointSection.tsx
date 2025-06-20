'use client';

import { motion } from 'framer-motion';
import { useInView } from 'react-intersection-observer';

export default function InteriorPointSection() {
  const { ref, inView } = useInView({
    triggerOnce: true,
    threshold: 0.1,
  });

  return (
    <section ref={ref} className="py-20 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
        >
          <div className="text-center mb-12">
            <div className="inline-flex items-center space-x-3 mb-6">
              <div className="w-12 h-12 bg-orange-500 rounded-full flex items-center justify-center">
                <span className="text-2xl">🐌</span>
              </div>
              <h2 className="text-3xl md:text-4xl font-bold text-gray-900">
                Interior-Point (Trust-Constr)
              </h2>
            </div>
            <p className="text-lg text-gray-600 max-w-3xl mx-auto">
              A Newton method designed for constrained optimization, applied here without constraints.
              Shows the overhead of generality when constraints aren't needed.
            </p>
          </div>

          <div className="bg-gradient-to-br from-orange-50 to-red-50 p-8 rounded-xl">
            <div className="grid md:grid-cols-3 gap-8">
              <div className="text-center">
                <div className="text-4xl font-bold text-orange-600 mb-2">95.1%</div>
                <div className="text-sm text-gray-600">Final Accuracy</div>
                <div className="text-xs text-gray-500">Good but not best</div>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold text-red-600 mb-2">299.5s</div>
                <div className="text-sm text-gray-600">Training Time</div>
                <div className="text-xs text-gray-500">Extremely slow!</div>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold text-orange-600 mb-2">Glacial</div>
                <div className="text-sm text-gray-600">Progress Rate</div>
                <div className="text-xs text-gray-500">Many tiny steps</div>
              </div>
            </div>
            
            <div className="mt-6 p-4 bg-white rounded-lg">
              <p className="text-center text-gray-700 text-sm">
                <strong>Lesson:</strong> Interior-point methods are powerful for constrained problems, 
                but add unnecessary overhead for unconstrained optimization. This demonstrates the 
                importance of choosing the right tool for the specific problem.
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
} 