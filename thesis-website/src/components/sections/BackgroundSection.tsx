'use client';

import { motion } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import { BlockMath, InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';

export default function BackgroundSection() {
  const { ref, inView } = useInView({
    triggerOnce: true,
    threshold: 0.1,
  });

  return (
    <section ref={ref} className="py-20 bg-white">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
        >
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-6">
              Neural Network Optimization Problem
            </h2>
            <p className="text-lg text-gray-600 max-w-3xl mx-auto">
              Understanding the mathematical foundation of neural network training through the lens of optimization theory
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-12 items-start">
            {/* Problem Definition */}
            <motion.div
              initial={{ opacity: 0, x: -50 }}
              animate={inView ? { opacity: 1, x: 0 } : {}}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="bg-blue-50 p-8 rounded-xl"
            >
              <h3 className="text-xl font-semibold text-gray-900 mb-4">
                Problem Definition
              </h3>
              <p className="text-gray-700 mb-6">
                Training a neural network is equivalent to solving a high-dimensional{' '}
                <strong>nonlinear optimization</strong> problem. If <InlineMath math="w" /> represents 
                all the parameters (weights) of the network, and <InlineMath math="L(w)" /> is the 
                loss function measuring training error, then training seeks to find:
              </p>
              
              <div className="bg-white p-4 rounded-lg shadow-sm mb-6">
                <BlockMath math="w^* = \arg\min_w L(w)" />
              </div>

              <p className="text-gray-700 mb-4">
                For a neural network with parameters <InlineMath math="w \in \mathbb{R}^d" /> and 
                training set <InlineMath math="\{(x_i, y_i)\}_{i=1}^N" />, the training loss can be written as:
              </p>

              <div className="bg-white p-4 rounded-lg shadow-sm">
                <BlockMath math="L(w) = \frac{1}{N}\sum_{i=1}^N \ell(f(w; x_i), y_i)" />
              </div>

              <p className="text-gray-700 mt-4 text-sm">
                where <InlineMath math="f(w; x)" /> is the network's output for input <InlineMath math="x" />, 
                and <InlineMath math="\ell(\hat{y}, y)" /> is the loss for prediction <InlineMath math="\hat{y}" />.
              </p>
            </motion.div>

            {/* Non-convex Nature */}
            <motion.div
              initial={{ opacity: 0, x: 50 }}
              animate={inView ? { opacity: 1, x: 0 } : {}}
              transition={{ duration: 0.8, delay: 0.4 }}
              className="bg-purple-50 p-8 rounded-xl"
            >
              <h3 className="text-xl font-semibold text-gray-900 mb-4">
                Non-convex Challenges
              </h3>
              <p className="text-gray-700 mb-6">
                The loss function <InlineMath math="L(w)" /> is typically <strong>highly non-convex</strong>, 
                meaning it has many local minima and saddle points rather than a single bowl-shaped minimum.
              </p>

              <div className="space-y-4">
                <div className="flex items-start space-x-3">
                  <div className="w-3 h-3 bg-red-400 rounded-full mt-2 flex-shrink-0"></div>
                  <div>
                    <h4 className="font-medium text-gray-900">Local Minima</h4>
                    <p className="text-sm text-gray-600">
                      Points where small movements in any direction increase the loss
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start space-x-3">
                  <div className="w-3 h-3 bg-yellow-400 rounded-full mt-2 flex-shrink-0"></div>
                  <div>
                    <h4 className="font-medium text-gray-900">Saddle Points</h4>
                    <p className="text-sm text-gray-600">
                      Points with mixed curvature: some directions go up, others down
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start space-x-3">
                  <div className="w-3 h-3 bg-green-400 rounded-full mt-2 flex-shrink-0"></div>
                  <div>
                    <h4 className="font-medium text-gray-900">Global Minimum</h4>
                    <p className="text-sm text-gray-600">
                      The absolute lowest point we're trying to find
                    </p>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>

          {/* Algorithm Categories */}
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="mt-16"
          >
            <h3 className="text-2xl font-semibold text-gray-900 mb-8 text-center">
              Two Broad Classes of Optimization Methods
            </h3>

            <div className="grid md:grid-cols-2 gap-8">
              <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-8 rounded-xl">
                <h4 className="text-xl font-semibold text-blue-900 mb-4">
                  First-Order Methods
                </h4>
                <p className="text-blue-800 mb-4">
                  Use only the gradient (first derivative) of <InlineMath math="L" /> to guide the search.
                </p>
                <ul className="space-y-2 text-sm text-blue-700">
                  <li className="flex items-center">
                    <span className="w-2 h-2 bg-blue-400 rounded-full mr-2"></span>
                    <strong>Gradient Descent (GD):</strong> Basic iterative method
                  </li>
                  <li className="flex items-center">
                    <span className="w-2 h-2 bg-blue-400 rounded-full mr-2"></span>
                    <strong>Adam:</strong> Adaptive method with momentum
                  </li>
                </ul>
                <div className="mt-4 bg-white p-3 rounded-lg">
                  <BlockMath math="w_{t+1} = w_t - \eta \nabla L(w_t)" />
                </div>
              </div>

              <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-8 rounded-xl">
                <h4 className="text-xl font-semibold text-purple-900 mb-4">
                  Second-Order Methods
                </h4>
                <p className="text-purple-800 mb-4">
                  Use both gradient and curvature (Hessian) information for more informed steps.
                </p>
                <ul className="space-y-2 text-sm text-purple-700">
                  <li className="flex items-center">
                    <span className="w-2 h-2 bg-purple-400 rounded-full mr-2"></span>
                    <strong>L-BFGS:</strong> Quasi-Newton method
                  </li>
                  <li className="flex items-center">
                    <span className="w-2 h-2 bg-purple-400 rounded-full mr-2"></span>
                    <strong>Trust-Region Newton:</strong> Constrained Newton steps
                  </li>
                  <li className="flex items-center">
                    <span className="w-2 h-2 bg-purple-400 rounded-full mr-2"></span>
                    <strong>Interior-Point:</strong> Constraint-handling Newton variant
                  </li>
                </ul>
                <div className="mt-4 bg-white p-3 rounded-lg">
                  <BlockMath math="w_{t+1} = w_t - H_t^{-1} \nabla L(w_t)" />
                </div>
              </div>
            </div>
          </motion.div>

          {/* Small Network Context */}
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.8 }}
            className="mt-16 bg-gray-50 p-8 rounded-xl"
          >
            <h3 className="text-xl font-semibold text-gray-900 mb-4">
              Our Experimental Context
            </h3>
            <div className="grid md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-600 mb-2">~10⁵</div>
                <div className="text-sm text-gray-600">Parameters</div>
                <div className="text-xs text-gray-500 mt-1">
                  Small enough for sophisticated solvers
                </div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-purple-600 mb-2">MNIST</div>
                <div className="text-sm text-gray-600">Dataset</div>
                <div className="text-xs text-gray-500 mt-1">
                  Handwritten digit classification
                </div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-green-600 mb-2">5</div>
                <div className="text-sm text-gray-600">Optimizers</div>
                <div className="text-xs text-gray-500 mt-1">
                  Empirical comparison study
                </div>
              </div>
            </div>
            <p className="text-gray-700 mt-6 text-center">
              This setup allows us to experiment with computationally intensive second-order methods
              that would be infeasible for larger networks.
            </p>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
} 