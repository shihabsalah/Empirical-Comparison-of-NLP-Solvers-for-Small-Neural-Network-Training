/**
 * This file contains utility functions for generating the data needed for
 * the 3D optimizer path visualizations. It defines a complex loss surface
 * and simulates the trajectories of various optimizers on that surface.
 */

import * as THREE from 'three';

// --- Complex Loss Surface Definition ---

// A collection of Gaussian functions to build the landscape.
// [amplitude, centerX, centerZ, widthX, widthZ]
const surfaceFeatures: [number, number, number, number, number][] = [
  // Minima (Valleys)
  [-4.5, 3.5, -3.5, 2.0, 2.0],  // Global Minimum
  [-3.0, -3.5, 3.5, 2.0, 2.0],  // Local Minimum 1
  [-2.5, -3.5, -3.5, 2.5, 2.5], // Local Minimum 2

  // Maxima (Hills)
  [3.5, 3.5, 3.5, 2.0, 2.0],    // Maximum 1
  [3.0, 0, 0, 2.0, 2.0],        // Maximum 2 (near saddle)
  [2.5, -3.5, 1.5, 1.5, 1.5],   // Maximum 3
];

// Locations of key points for markers [x, z, name]
export const keyFeaturePoints: [number, number, string][] = [
    [3.5, -3.5, 'Global Minimum'],
    [-3.5, 3.5, 'Local Minimum'],
    [-3.5, -3.5, 'Local Minimum'],
    [0.5, -0.5, 'Saddle Point'],
];

/**
 * Defines a complex, non-convex loss surface with multiple features.
 * This function creates a landscape with 3 minima, 3 maxima, and a saddle point.
 *
 * @param parameter1 The first dimension in the parameter space (x-coordinate).
 * @param parameter2 The second dimension in the parameter space (z-coordinate in 3D).
 * @returns The computed loss value for the given parameters.
 */
export function complexLossSurface(parameter1: number, parameter2: number): number {
  let lossValue = 0;

  // Add Gaussian features
  for (const [amp, cx, cz, wx, wz] of surfaceFeatures) {
    lossValue += amp * Math.exp(
      -((parameter1 - cx) ** 2 / wx + (parameter2 - cz) ** 2 / wz)
    );
  }

  // Add a saddle point
  const saddleX = parameter1 - 0.5;
  const saddleZ = parameter2 + 0.5;
  lossValue += 2.5 * (saddleX**2 - saddleZ**2) * Math.exp(-(saddleX**2 + saddleZ**2) / 5);

  return lossValue;
}


// --- Optimizer Simulation Logic ---

type OptimizerPath = Array<[number, number, number]>;

/**
 * Numerically calculates the gradient of the loss surface at a given point.
 * @param p1 The first parameter (x-coordinate).
 * @param p2 The second parameter (z-coordinate).
 * @returns A THREE.Vector2 representing the gradient [dLoss/dp1, dLoss/dp2].
 */
function getGradient(p1: number, p2: number): THREE.Vector2 {
  const delta = 0.01;
  const loss = complexLossSurface(p1, p2);
  const dLoss_dp1 = (complexLossSurface(p1 + delta, p2) - loss) / delta;
  const dLoss_dp2 = (complexLossSurface(p1, p2 + delta) - loss) / delta;
  return new THREE.Vector2(dLoss_dp1, dLoss_dp2);
}

function generateGradientDescentPath(start: [number, number]): OptimizerPath {
  const path: OptimizerPath = [];
  let currentP = new THREE.Vector2(start[0], start[1]);
  const learningRate = 0.1;

  for (let i = 0; i < 30; i++) {
    path.push([currentP.x, currentP.y, complexLossSurface(currentP.x, currentP.y)]);
    const gradient = getGradient(currentP.x, currentP.y);
    currentP.sub(gradient.multiplyScalar(learningRate));
  }
  return path;
}

function generateAdamPath(start: [number, number]): OptimizerPath {
  const path: OptimizerPath = [];
  let currentP = new THREE.Vector2(start[0], start[1]);
  
  // Adam parameters
  const learningRate = 0.2;
  const beta1 = 0.9;
  const beta2 = 0.999;
  const epsilon = 1e-8;
  let m = new THREE.Vector2(0, 0); // 1st moment vector
  let v = new THREE.Vector2(0, 0); // 2nd moment vector
  
  for (let i = 1; i <= 25; i++) {
    path.push([currentP.x, currentP.y, complexLossSurface(currentP.x, currentP.y)]);
    
    const gradient = getGradient(currentP.x, currentP.y);
    
    m.lerp(gradient, 1 - beta1);
    v.lerp(gradient.clone().multiply(gradient), 1 - beta2);
    
    const m_hat = m.clone().divideScalar(1 - beta1 ** i);
    const v_hat = v.clone().divideScalar(1 - beta2 ** i);
    
    const update = m_hat.divide(v_hat.sqrt().addScalar(epsilon));
    currentP.sub(update.multiplyScalar(learningRate));
  }
  return path;
}

function generateLbfgsPath(start: [number, number]): OptimizerPath {
    const path: OptimizerPath = [];
    let currentP = new THREE.Vector2(start[0], start[1]);
    
    // Simplified L-BFGS: take larger, more direct steps
    const numSteps = 10;
    for (let i = 0; i < numSteps; i++) {
        path.push([currentP.x, currentP.y, complexLossSurface(currentP.x, currentP.y)]);
        const gradient = getGradient(currentP.x, currentP.y);

        // A bigger step size simulates the aggressive nature of L-BFGS
        currentP.sub(gradient.multiplyScalar(0.3));
    }
    return path;
}

function generateTrustRegionPath(start: [number, number]): OptimizerPath {
    const path: OptimizerPath = [];
    let currentP = new THREE.Vector2(start[0], start[1]);

    // Simulate a few large, confident jumps
    const trustRadius = 1.5;
    for (let i = 0; i < 5; i++) {
        path.push([currentP.x, currentP.y, complexLossSurface(currentP.x, currentP.y)]);
        const gradient = getGradient(currentP.x, currentP.y).normalize();
        
        // Move a large, fixed distance in the gradient direction
        currentP.sub(gradient.multiplyScalar(trustRadius));
    }
    return path;
}


/**
 * Generates a simulated trajectory for a given optimization algorithm on the complex surface.
 *
 * @param optimizer The name of the optimizer.
 * @returns An array of points representing the optimizer's path.
 */
export function generateOptimizerPath(optimizer: 'gradient-descent' | 'adam' | 'lbfgs' | 'trust-region'): OptimizerPath {
  const startPosition: [number, number] = [3, 4];

  switch (optimizer) {
    case 'gradient-descent':
      return generateGradientDescentPath(startPosition);
    case 'adam':
      return generateAdamPath(startPosition);
    case 'lbfgs':
      return generateLbfgsPath(startPosition);
    case 'trust-region':
      return generateTrustRegionPath(startPosition);
  }
}
