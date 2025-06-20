'use client';

import { useMemo, useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Text, Line } from '@react-three/drei';
import * as THREE from 'three';
import { ParametricGeometry } from 'three-stdlib';
import { complexLossSurface, generateOptimizerPath, keyFeaturePoints } from '@/lib/optimizer-paths';

// Reusable Loss Surface Mesh Component
interface LossSurfaceMeshProps {
  showWireframe?: boolean;
}

function LossSurfaceMesh({ showWireframe = false }: LossSurfaceMeshProps) {
  const geometry = useMemo(() => new ParametricGeometry(complexLossSurface, 100, 100), []);
  
  const colors = useMemo(() => {
    const vertices = geometry.attributes.position.array as Float32Array;
    const vertexColors = new Float32Array(vertices.length);
    const minLoss = -5, maxLoss = 4;
    
    for (let i = 0; i < vertices.length; i += 3) {
      const lossValue = vertices[i + 1]; // y is loss
      const normalizedLoss = (lossValue - minLoss) / (maxLoss - minLoss);
      const color = new THREE.Color();
      color.setHSL(0.7 - normalizedLoss * 0.7, 0.9, 0.6);
      vertexColors[i] = color.r;
      vertexColors[i + 1] = color.g;
      vertexColors[i + 2] = color.b;
    }
    return vertexColors;
  }, [geometry]);

  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  return (
    <mesh geometry={geometry} rotation={[-Math.PI / 2, 0, 0]}>
      <meshStandardMaterial
        vertexColors
        side={THREE.DoubleSide}
        wireframe={showWireframe}
        transparent
        opacity={showWireframe ? 0.7 : 1.0}
      />
    </mesh>
  );
}

// Main Interactive Demo Component
type OptimizerType = 'gradient-descent' | 'adam' | 'lbfgs' | 'trust-region';

interface InteractiveOptimizerDemoProps {
  optimizer: OptimizerType;
  className?: string;
}

const OPTIMIZER_CONFIG = {
  'gradient-descent': { color: '#eab308', name: 'Gradient Descent' },
  'adam': { color: '#3b82f6', name: 'Adam' },
  'lbfgs': { color: '#10b981', name: 'L-BFGS' },
  'trust-region': { color: '#8b5cf6', name: 'Trust-Region Newton' },
};

/**
 * A complete 3D visualization scene that shows an optimizer's path on a complex loss surface.
 * This component combines the loss surface mesh, an animated optimizer path, and UI controls.
 */
export default function InteractiveOptimizerDemo({ optimizer, className = "" }: InteractiveOptimizerDemoProps) {
  const [isPlaying, setIsPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [showWireframe, setShowWireframe] = useState(false);
  const [showMarkers, setShowMarkers] = useState(true);

  const path = useMemo(() => generateOptimizerPath(optimizer), [optimizer]);
  const config = OPTIMIZER_CONFIG[optimizer];

  return (
    <div className={`relative w-full h-[500px] bg-white rounded-xl overflow-hidden border border-gray-200 shadow-lg ${className}`}>
      {/* UI Controls */}
      <div className="absolute top-4 left-4 z-10 flex flex-col space-y-2">
        <div className="flex items-center space-x-3 bg-black/30 backdrop-blur-sm p-2 rounded-lg">
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="bg-white/20 hover:bg-white/30 text-white px-3 py-1.5 rounded-md text-xs font-medium"
            >
              {isPlaying ? '⏸️ Pause' : '▶️ Play'}
            </button>
            <div className="flex items-center space-x-1">
              <label htmlFor="speed-control" className="text-white text-xs">Speed:</label>
              <input id="speed-control" type="range" min="0.5" max="5" step="0.5" value={speed} onChange={(e) => setSpeed(Number(e.target.value))} className="w-12 h-1"/>
            </div>
        </div>
        <div className="flex items-center space-x-3 bg-black/30 backdrop-blur-sm p-2 rounded-lg">
            <label className="flex items-center space-x-1.5 cursor-pointer">
              <input type="checkbox" checked={showWireframe} onChange={(e) => setShowWireframe(e.target.checked)} className="h-3 w-3 rounded border-gray-300 text-blue-500 focus:ring-blue-500"/>
              <span className="text-white text-xs">Wireframe</span>
            </label>
            <label className="flex items-center space-x-1.5 cursor-pointer">
              <input type="checkbox" checked={showMarkers} onChange={(e) => setShowMarkers(e.target.checked)} className="h-3 w-3 rounded border-gray-300 text-blue-500 focus:ring-blue-500"/>
              <span className="text-white text-xs">Markers</span>
            </label>
        </div>
      </div>
      
      {/* Optimizer Label */}
      <div className="absolute top-4 right-4 z-10 px-3 py-1.5 rounded-lg text-white font-semibold text-xs" style={{ backgroundColor: config.color }}>
          {config.name}
      </div>

      <Canvas camera={{ position: [0, 8, 9], fov: 60 }} style={{ background: 'linear-gradient(to bottom, #d1d5db, #ffffff)' }}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[10, 10, 5]} intensity={1.0} />
        <pointLight position={[-5, -10, -5]} intensity={0.3} color="#ffffff" />
        
        <LossSurfaceMesh showWireframe={showWireframe} />
        
        <AnimatedPath path={path} color={config.color} isPlaying={isPlaying} speed={speed} />

        {showMarkers && keyFeaturePoints.map(([x, z, name], index) => (
            <group key={index}>
                <mesh position={[x, complexLossSurface(x, z) + 0.1, z]}>
                    <sphereGeometry args={[0.12, 16, 16]} />
                    <meshStandardMaterial color="#374151" roughness={0.5} />
                </mesh>
                <Text position={[x, complexLossSurface(x, z) + 0.5, z]} fontSize={0.2} color="#374151" anchorX="center">
                    {name}
                </Text>
            </group>
        ))}

        <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} minDistance={4} maxDistance={20} />
      </Canvas>
    </div>
  );
}

// Minimal AnimatedPath component, kept in the same file for simplicity
type OptimizerPathPoint = [number, number, number];

interface AnimatedPathProps {
  path: OptimizerPathPoint[];
  color: string;
  isPlaying: boolean;
  speed: number;
}

function AnimatedPath({ path, color, isPlaying, speed }: AnimatedPathProps) {
  const [currentStep, setCurrentStep] = useState(0);
  useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(() => {
      setCurrentStep(prev => (prev >= path.length - 1 ? 0 : prev + 1));
    }, 1000 / (speed * 5)); // speed is a multiplier
    return () => clearInterval(interval);
  }, [isPlaying, speed, path.length]);

  const animatedPath = path.slice(0, currentStep + 1);
  const linePoints = animatedPath.map(([p1, p2, loss]) => new THREE.Vector3(p1, loss, p2));
  const lastPoint = animatedPath[animatedPath.length - 1];

  return (
    <group>
      {linePoints.length > 1 && <Line points={linePoints} color={color} lineWidth={4} />}
      {lastPoint && (
        <mesh position={[lastPoint[0], lastPoint[2], lastPoint[1]]}>
          <sphereGeometry args={[0.15, 32, 32]} />
          <meshStandardMaterial color={color} metalness={0.5} roughness={0.3} />
        </mesh>
      )}
    </group>
  );
} 