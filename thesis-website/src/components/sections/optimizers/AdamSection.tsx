'use client';

import { motion } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import { useEffect } from 'react';
import { InlineMath, BlockMath } from 'react-katex';
import InteractiveOptimizerDemo from '../../3d/InteractiveOptimizerDemo';
import 'katex/dist/katex.min.css';
import katex from 'katex';

export default function AdamSection() {
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
    <section ref={ref} id="adam" className="py-20 bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
        >
          <div className="inline-flex items-center px-4 py-2 bg-white text-gray-700 rounded-full text-sm font-medium mb-6 shadow-sm">
            Adaptive Method • First-Order with Momentum
          </div>
          <h2 className="text-4xl md:text-5xl font-light text-gray-900 mb-6">
            Adam Optimizer
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
            Adaptive Moment Estimation combining momentum with per-parameter learning rate adaptation.
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
              Mathematical Formulation
            </h3>
            
            <div className="space-y-6">
              <div>
                <h4 className="text-lg font-medium text-gray-900 mb-3">Moment Updates</h4>
                <div className="bg-gray-50 p-4 rounded-lg border border-gray-200 space-y-3">
                  <div 
                    className="text-center"
                    dangerouslySetInnerHTML={{ 
                      __html: renderMath('m_t = \\beta_1 m_{t-1} + (1-\\beta_1) \\nabla L(w_{t-1})') 
                    }}
                  />
                  <div 
                    className="text-center"
                    dangerouslySetInnerHTML={{ 
                      __html: renderMath('v_t = \\beta_2 v_{t-1} + (1-\\beta_2) (\\nabla L(w_{t-1}))^2') 
                    }}
                  />
                </div>
                <p className="text-sm text-gray-600 mt-2">
                  First and second moment estimates with exponential decay rates.
                </p>
              </div>

              <div>
                <h4 className="text-lg font-medium text-gray-900 mb-3">Bias Correction</h4>
                <div className="bg-gray-50 p-4 rounded-lg border border-gray-200 space-y-3">
                  <div 
                    className="text-center"
                    dangerouslySetInnerHTML={{ 
                      __html: renderMath('\\hat{m}_t = \\frac{m_t}{1-\\beta_1^t}') 
                    }}
                  />
                  <div 
                    className="text-center"
                    dangerouslySetInnerHTML={{ 
                      __html: renderMath('\\hat{v}_t = \\frac{v_t}{1-\\beta_2^t}') 
                    }}
                  />
                </div>
              </div>

              <div>
                <h4 className="text-lg font-medium text-gray-900 mb-3">Parameter Update</h4>
                <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                  <div 
                    className="text-lg text-center"
                    dangerouslySetInnerHTML={{ 
                      __html: renderMath('w_t = w_{t-1} - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t') 
                    }}
                  />
                </div>
                <p className="text-sm text-gray-600 mt-2">
                  Typical values: β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸
                </p>
              </div>

              <div>
                <h4 className="text-lg font-medium text-gray-900 mb-3">Key Advantages</h4>
                <ul className="space-y-2 text-gray-700">
                  <li className="flex items-start">
                    <span className="w-2 h-2 bg-gray-400 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                    <span>Adaptive per-parameter learning rates</span>
                  </li>
                  <li className="flex items-start">
                    <span className="w-2 h-2 bg-gray-400 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                    <span>Momentum acceleration with exponential averaging</span>
                  </li>
                  <li className="flex items-start">
                    <span className="w-2 h-2 bg-gray-400 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                    <span>Robust to hyperparameter choices</span>
                  </li>
                </ul>
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
                Adaptive Path Visualization
              </h3>
              <p className="text-sm text-gray-600">
                Observe Adam's curved trajectory with momentum-driven overshooting and correction
              </p>
            </div>
'use client';

import { motion } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import { useEffect } from 'react';
import { InlineMath, BlockMath } from 'react-katex';
import OptimizerPathAnimation from '../../3d/OptimizerPathAnimation';
import 'katex/dist/katex.min.css';
import katex from 'katex';

export default function AdamSection() {
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
    <section ref={ref} id="adam" className="py-20 bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
        >
          <div className="inline-flex items-center px-4 py-2 bg-white text-gray-700 rounded-full text-sm font-medium mb-6 shadow-sm">
            Adaptive Method • First-Order with Momentum
          </div>
          <h2 className="text-4xl md:text-5xl font-light text-gray-900 mb-6">
            Adam Optimizer
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
            Adaptive Moment Estimation combining momentum with per-parameter learning rate adaptation.
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
              Mathematical Formulation
            </h3>
            
            <div className="space-y-6">
              <div>
                <h4 className="text-lg font-medium text-gray-900 mb-3">Moment Updates</h4>
                <div className="bg-gray-50 p-4 rounded-lg border border-gray-200 space-y-3">
                  <div 
                    className="text-center"
                    dangerouslySetInnerHTML={{ 
                      __html: renderMath('m_t = \\beta_1 m_{t-1} + (1-\\beta_1) \\nabla L(w_{t-1})') 
                    }}
                  />
                  <div 
                    className="text-center"
                    dangerouslySetInnerHTML={{ 
                      __html: renderMath('v_t = \\beta_2 v_{t-1} + (1-\\beta_2) (\\nabla L(w_{t-1}))^2') 
                    }}
                  />
                </div>
                <p className="text-sm text-gray-600 mt-2">
                  First and second moment estimates with exponential decay rates.
                </p>
              </div>

              <div>
                <h4 className="text-lg font-medium text-gray-900 mb-3">Bias Correction</h4>
                <div className="bg-gray-50 p-4 rounded-lg border border-gray-200 space-y-3">
                  <div 
                    className="text-center"
                    dangerouslySetInnerHTML={{ 
                      __html: renderMath('\\hat{m}_t = \\frac{m_t}{1-\\beta_1^t}') 
                    }}
                  />
                  <div 
                    className="text-center"
                    dangerouslySetInnerHTML={{ 
                      __html: renderMath('\\hat{v}_t = \\frac{v_t}{1-\\beta_2^t}') 
                    }}
                  />
                </div>
              </div>

              <div>
                <h4 className="text-lg font-medium text-gray-900 mb-3">Parameter Update</h4>
                <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                  <div 
                    className="text-lg text-center"
                    dangerouslySetInnerHTML={{ 
                      __html: renderMath('w_t = w_{t-1} - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t') 
                    }}
                  />
                </div>
                <p className="text-sm text-gray-600 mt-2">
                  Typical values: β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸
                </p>
              </div>

              <div>
                <h4 className="text-lg font-medium text-gray-900 mb-3">Key Advantages</h4>
                <ul className="space-y-2 text-gray-700">
                  <li className="flex items-start">
                    <span className="w-2 h-2 bg-gray-400 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                    <span>Adaptive per-parameter learning rates</span>
                  </li>
                  <li className="flex items-start">
                    <span className="w-2 h-2 bg-gray-400 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                    <span>Momentum acceleration with exponential averaging</span>
                  </li>
                  <li className="flex items-start">
                    <span className="w-2 h-2 bg-gray-400 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                    <span>Robust to hyperparameter choices</span>
                  </li>
                </ul>
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
                Adaptive Path Visualization
              </h3>
              <p className="text-sm text-gray-600">
                Observe Adam's curved trajectory with momentum-driven overshooting and correction
              </p>
            </div>
            <div className="h-96">
              <OptimizerPathAnimation optimizer="adam" />
            </div>
          </motion.div>
        </div>

        {/* Performance Results */}
        <motion.div
          className="bg-white rounded-lg shadow-sm border border-gray-200 p-8"
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, delay: 0.6 }}
        >
          <h3 className="text-2xl font-semibold text-gray-900 mb-8 text-center">
            Experimental Results
          </h3>
          
          <div className="grid md:grid-cols-4 gap-6 mb-8">
            <div className="text-center p-6 bg-gray-50 rounded-lg border border-gray-200">
              <div className="text-3xl font-light text-gray-900 mb-2">88.5%</div>
              <div className="text-sm text-gray-600 font-medium">Final Accuracy</div>
              <div className="text-xs text-gray-500 mt-1">5.26× better than GD</div>
            </div>
            <div className="text-center p-6 bg-gray-50 rounded-lg border border-gray-200">
              <div className="text-3xl font-light text-gray-900 mb-2">0.39</div>
              <div className="text-sm text-gray-600 font-medium">Final Loss</div>
              <div className="text-xs text-gray-500 mt-1">0.17× of GD loss</div>
            </div>
            <div className="text-center p-6 bg-gray-50 rounded-lg border border-gray-200">
              <div className="text-3xl font-light text-gray-900 mb-2">17.4s</div>
              <div className="text-sm text-gray-600 font-medium">Training Time</div>
              <div className="text-xs text-gray-500 mt-1">Similar to GD</div>
            </div>
            <div className="text-center p-6 bg-gray-50 rounded-lg border border-gray-200">
              <div className="text-3xl font-light text-gray-900 mb-2">416MB</div>
              <div className="text-sm text-gray-600 font-medium">Memory Usage</div>
              <div className="text-xs text-gray-500 mt-1">Same as GD</div>
            </div>
          </div>

          <div className="bg-gray-50 border border-gray-200 rounded-lg p-6">
            <h4 className="text-lg font-medium text-gray-900 mb-4">Analysis</h4>
            <p className="text-gray-700 leading-relaxed mb-4">
              Adam demonstrates the power of adaptive optimization, achieving 88.5% accuracy compared to Gradient Descent's 
              16.8% — a remarkable <strong>5.26× improvement</strong> with nearly identical computational cost. 
              The algorithm's momentum and per-parameter learning rate adaptation enabled rapid initial convergence 
              and effective navigation of the loss landscape.
            </p>
            <p className="text-gray-700 leading-relaxed">
              While not reaching the ultimate precision of second-order methods (L-BFGS and Trust-Region achieved 97%+), 
              Adam provides an excellent balance of performance, efficiency, and robustness that explains its widespread 
              adoption in modern deep learning applications.
            </p>
          </div>
        </motion.div>
      </div>
    </section>
  );
} 