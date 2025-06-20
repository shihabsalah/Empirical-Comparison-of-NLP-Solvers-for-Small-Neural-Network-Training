'use client';

import { useEffect } from 'react';
import HeroSection from '@/components/sections/HeroSection';
import BackgroundSection from '@/components/sections/BackgroundSection';
import GradientDescentSection from '@/components/sections/optimizers/GradientDescentSection';
import AdamSection from '@/components/sections/optimizers/AdamSection';
import LBFGSSection from '@/components/sections/optimizers/LBFGSSection';
import TrustRegionSection from '@/components/sections/optimizers/TrustRegionSection';
import InteriorPointSection from '@/components/sections/optimizers/InteriorPointSection';
import ComparisonSection from '@/components/sections/ComparisonSection';

export default function Home() {
  useEffect(() => {
    // Import KaTeX CSS
    const link = document.createElement('link');
    link.href = 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css';
    link.rel = 'stylesheet';
    document.head.appendChild(link);
  }, []);

  return (
    <main className="min-h-screen bg-white">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-white/90 backdrop-blur-md border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gray-800 rounded flex items-center justify-center">
                <span className="text-white text-sm font-semibold">O</span>
              </div>
              <span className="font-semibold text-gray-900">Optimizer Analysis</span>
            </div>
            
            <div className="hidden md:flex items-center space-x-8">
              <a href="#background" className="text-gray-600 hover:text-gray-900 text-sm font-medium transition-colors">
                Background
              </a>
              <a href="#gradient-descent" className="text-gray-600 hover:text-gray-900 text-sm font-medium transition-colors">
                Gradient Descent
              </a>
              <a href="#adam" className="text-gray-600 hover:text-gray-900 text-sm font-medium transition-colors">
                Adam
              </a>
              <a href="#lbfgs" className="text-gray-600 hover:text-gray-900 text-sm font-medium transition-colors">
                L-BFGS
              </a>
              <a href="#trust-region" className="text-gray-600 hover:text-gray-900 text-sm font-medium transition-colors">
                Trust-Region
              </a>
              <a href="#comparison" className="text-gray-600 hover:text-gray-900 text-sm font-medium transition-colors">
                Comparison
              </a>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="pt-16">
        <HeroSection />
        <BackgroundSection />
        <GradientDescentSection />
        <AdamSection />
        <LBFGSSection />
        <TrustRegionSection />
        <InteriorPointSection />
        <ComparisonSection />
      </div>

      {/* Footer */}
      <footer className="bg-gray-50 border-t border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="grid md:grid-cols-3 gap-8">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">About This Research</h3>
              <p className="text-gray-600 text-sm leading-relaxed">
                An empirical comparison of nonlinear programming solvers for neural network training, 
                designed to provide mathematics professors with rigorous analysis and accessible explanations.
              </p>
            </div>
            
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Methodology</h3>
              <ul className="text-gray-600 text-sm space-y-2">
                <li>• MNIST dataset with 100k parameter network</li>
                <li>• Identical initialization and hardware conditions</li>
                <li>• 20 epochs training with precision tolerance</li>
                <li>• Multiple metrics: accuracy, loss, time, memory</li>
              </ul>
            </div>
            
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Technical Stack</h3>
              <ul className="text-gray-600 text-sm space-y-2">
                <li>• PyTorch for neural network implementation</li>
                <li>• SciPy for advanced optimization algorithms</li>
                <li>• Three.js for 3D visualizations</li>
                <li>• React and Next.js for interactive interface</li>
              </ul>
            </div>
          </div>
          
          <div className="mt-8 pt-8 border-t border-gray-200 text-center">
            <p className="text-gray-500 text-sm">
              © 2024 Neural Network Optimization Analysis. Created for educational purposes.
            </p>
          </div>
        </div>
      </footer>
    </main>
  );
}
