'use client';

import { motion } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import { useEffect } from 'react';
import { InlineMath } from 'react-katex';
import InteractiveOptimizerDemo from '../../3d/InteractiveOptimizerDemo';
import 'katex/dist/katex.min.css';
import katex from 'katex';

export default function GradientDescentSection() {
  const { ref, inView } = useInView({
    threshold: 0.1,
    triggerOnce: true
  });

  useEffect(() => {
    if (inView) {
      // Trigger animations when section comes into view
    }
  }, [inView]);

  const renderMath = (mathString: string) => {
    try {
      return katex.renderToString(mathString, {
        displayMode: false,
        throwOnError: false,
      });
    } catch (error) {
      return mathString;
    }
  };

  return (
    <section ref={ref} id="gradient-descent" className="py-20 bg-gradient-to-br from-gray-50 to-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
        >
          <div className="inline-flex items-center px-4 py-2 bg-gray-100 text-gray-700 rounded-full text-sm font-medium mb-6">
            Baseline Algorithm • First-Order Method
          </div>
          <h2 className="text-4xl md:text-5xl font-light text-gray-900 mb-6">
            Gradient Descent
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
            The foundational first-order optimization method that iteratively moves parameters in the direction of steepest descent.
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-12 mb-16">
            <motion.div
                className="bg-white rounded-lg shadow-sm border border-gray-200 p-8"
                initial={{ opacity: 0, x: -50 }}
                animate={inView ? { opacity: 1, x: 0 } : {}}
                transition={{ duration: 0.8, delay: 0.4 }}
            >
                <h3 className="text-2xl font-semibold text-gray-900 mb-6">
                Practical Properties
                </h3>
                
                <div className="space-y-6">
                <div>
                    <h4 className="text-lg font-medium text-gray-900 mb-3">Core Update Rule</h4>
                    <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                    <div 
                        className="text-xl text-center mb-3"
                        dangerouslySetInnerHTML={{ 
                        __html: renderMath('w_{t+1} = w_t - \\eta \\nabla L(w_t)') 
                        }}
                    />
                    <div className="text-sm text-gray-600 space-y-1">
                        <p><strong>Where:</strong></p>
                        <p>• <span dangerouslySetInnerHTML={{ __html: renderMath('w_t') }} /> is the parameter vector</p>
                        <p>• <span dangerouslySetInnerHTML={{ __html: renderMath('\\nabla L(w_t)') }} /> is the gradient</p>
                        <p>• <span dangerouslySetInnerHTML={{ __html: renderMath('\\eta > 0') }} /> is the learning rate</p>
                    </div>
                    </div>
                </div>

                <div>
                    <h4 className="text-lg font-medium text-gray-900 mb-3">Key Limitations</h4>
                    <ul className="space-y-2 text-gray-700">
                    <li className="flex items-start">
                        <span className="w-2 h-2 bg-gray-400 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                        <span><strong>Learning rate sensitivity:</strong> Requires careful tuning</span>
                    </li>
                    <li className="flex items-start">
                        <span className="w-2 h-2 bg-gray-400 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                        <span><strong>Ill-conditioning:</strong> Zig-zag behavior in narrow valleys</span>
                    </li>
                    <li className="flex items-start">
                        <span className="w-2 h-2 bg-gray-400 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                        <span><strong>Plateau problems:</strong> Slow progress in flat regions</span>
                    </li>
                    <li className="flex items-start">
                        <span className="w-2 h-2 bg-gray-400 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                        <span><strong>No curvature info:</strong> Ignores second-order structure</span>
                    </li>
                    </ul>
                </div>
                </div>
            </motion.div>

          {/* 3D Algorithm Demonstration */}
          <motion.div
            className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden"
            initial={{ opacity: 0, x: 50 }}
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.6 }}
          >
            <div className="p-6 bg-gray-50 border-b border-gray-200">
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                Algorithm Demonstration
              </h3>
              <p className="text-sm text-gray-600 mb-3">
                Watch Gradient Descent follow the path of steepest descent on our complex loss surface.
              </p>
            </div>
            <div className="h-[500px] relative">
              <InteractiveOptimizerDemo optimizer="gradient-descent" />
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
} 