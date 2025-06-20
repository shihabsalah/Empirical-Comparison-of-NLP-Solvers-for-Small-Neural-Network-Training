'use client';

import { motion } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import { useEffect } from 'react';
import { InlineMath } from 'react-katex';
import InteractiveOptimizerDemo from '../../3d/InteractiveOptimizerDemo';
import 'katex/dist/katex.min.css';
import katex from 'katex';

export default function TrustRegionSection() {
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
    <section ref={ref} id="trust-region" className="py-20 bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
        >
          <div className="inline-flex items-center px-4 py-2 bg-white text-gray-700 rounded-full text-sm font-medium mb-6 shadow-sm">
            Newton Method • Constrained Optimization
          </div>
          <h2 className="text-4xl md:text-5xl font-light text-gray-900 mb-6">
            Trust-Region Newton
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
            Robust Newton method with adaptive trust region radius for controlled convergence.
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-12 mb-16">
          {/* Mathematical Foundation */}
          <motion.div
            className="bg-white rounded-lg shadow-sm border border-gray-200 p-8"
            initial={{ opacity: 0, x: -50 }}
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            <h3 className="text-2xl font-semibold text-gray-900 mb-6">
              Trust-Region Methodology
            </h3>
            
            <div className="space-y-6">
              <div>
                <h4 className="text-lg font-medium text-gray-900 mb-3">Quadratic Model</h4>
                <p className="text-gray-700 leading-relaxed mb-3">
                  At each iteration, we build a quadratic model of the loss function that we "trust" to be accurate in a local neighborhood.
                </p>
              </div>

              <div>
                <h4 className="text-lg font-medium text-gray-900 mb-3">Trust-Region Subproblem</h4>
                <p className="text-gray-700 leading-relaxed mb-3">
                  We then solve for the step that minimizes this model within the "trust region" (a sphere of a certain radius).
                </p>
              </div>

              <div>
                <h4 className="text-lg font-medium text-gray-900 mb-3">Radius Update Strategy</h4>
                <p className="text-gray-700 leading-relaxed mb-3">
                  The radius of the trust region is expanded or shrunk based on how well our quadratic model predicted the actual change in the loss function.
                </p>
              </div>
            </div>
          </motion.div>

          {/* 3D Visualization */}
          <motion.div
            className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden"
            initial={{ opacity: 0, x: 50 }}
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            <div className="p-6 bg-gray-50 border-b border-gray-200">
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                Controlled Newton Steps
              </h3>
              <p className="text-sm text-gray-600">
                Watch the Trust-Region method make large, confident steps, guided by the dynamically adjusting trust radius.
              </p>
            </div>
            <div className="h-[500px]">
              <InteractiveOptimizerDemo optimizer="trust-region" />
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
} 