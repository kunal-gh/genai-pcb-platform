import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Points, PointMaterial } from '@react-three/drei';
import * as THREE from 'three';

function AnimatedPoints() {
  const ref = useRef<THREE.Points>(null);
  
  // Generate random points in 3D space
  const particlesCount = 2000;
  const positions = new Float32Array(particlesCount * 3);
  
  for (let i = 0; i < particlesCount; i++) {
    positions[i * 3] = (Math.random() - 0.5) * 10;
    positions[i * 3 + 1] = (Math.random() - 0.5) * 10;
    positions[i * 3 + 2] = (Math.random() - 0.5) * 10;
  }

  useFrame((state) => {
    if (ref.current) {
      ref.current.rotation.x = state.clock.getElapsedTime() * 0.05;
      ref.current.rotation.y = state.clock.getElapsedTime() * 0.075;
    }
  });

  return (
    <Points ref={ref} positions={positions} stride={3} frustumCulled={false}>
      <PointMaterial
        transparent
        color="#3B82F6"
        size={0.02}
        sizeAttenuation={true}
        depthWrite={false}
        opacity={0.6}
      />
    </Points>
  );
}

const CircuitBackground: React.FC = () => {
  return (
    <div className="fixed inset-0 z-0">
      {/* Animated grid background */}
      <div className="absolute inset-0 grid-background opacity-30" />
      
      {/* Circuit pattern overlay */}
      <div className="absolute inset-0 bg-circuit-pattern opacity-20" />
      
      {/* 3D particle field */}
      <div className="absolute inset-0 opacity-40">
        <Canvas camera={{ position: [0, 0, 5], fov: 75 }}>
          <ambientLight intensity={0.5} />
          <AnimatedPoints />
        </Canvas>
      </div>

      {/* Gradient overlays */}
      <div className="absolute inset-0 bg-gradient-to-b from-circuit-darker via-transparent to-circuit-darker" />
      <div className="absolute inset-0 bg-gradient-to-r from-circuit-darker/50 via-transparent to-circuit-darker/50" />
      
      {/* Animated circuit lines */}
      <svg className="absolute inset-0 w-full h-full opacity-20" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#3B82F6" stopOpacity="0" />
            <stop offset="50%" stopColor="#3B82F6" stopOpacity="1" />
            <stop offset="100%" stopColor="#3B82F6" stopOpacity="0" />
          </linearGradient>
        </defs>
        
        {/* Horizontal lines */}
        <line x1="0" y1="20%" x2="100%" y2="20%" stroke="url(#lineGradient)" strokeWidth="1">
          <animate attributeName="x1" from="-100%" to="100%" dur="8s" repeatCount="indefinite" />
          <animate attributeName="x2" from="0%" to="200%" dur="8s" repeatCount="indefinite" />
        </line>
        <line x1="0" y1="50%" x2="100%" y2="50%" stroke="url(#lineGradient)" strokeWidth="1">
          <animate attributeName="x1" from="-100%" to="100%" dur="10s" repeatCount="indefinite" />
          <animate attributeName="x2" from="0%" to="200%" dur="10s" repeatCount="indefinite" />
        </line>
        <line x1="0" y1="80%" x2="100%" y2="80%" stroke="url(#lineGradient)" strokeWidth="1">
          <animate attributeName="x1" from="-100%" to="100%" dur="12s" repeatCount="indefinite" />
          <animate attributeName="x2" from="0%" to="200%" dur="12s" repeatCount="indefinite" />
        </line>
        
        {/* Vertical lines */}
        <line x1="20%" y1="0" x2="20%" y2="100%" stroke="url(#lineGradient)" strokeWidth="1">
          <animate attributeName="y1" from="-100%" to="100%" dur="9s" repeatCount="indefinite" />
          <animate attributeName="y2" from="0%" to="200%" dur="9s" repeatCount="indefinite" />
        </line>
        <line x1="50%" y1="0" x2="50%" y2="100%" stroke="url(#lineGradient)" strokeWidth="1">
          <animate attributeName="y1" from="-100%" to="100%" dur="11s" repeatCount="indefinite" />
          <animate attributeName="y2" from="0%" to="200%" dur="11s" repeatCount="indefinite" />
        </line>
        <line x1="80%" y1="0" x2="80%" y2="100%" stroke="url(#lineGradient)" strokeWidth="1">
          <animate attributeName="y1" from="-100%" to="100%" dur="13s" repeatCount="indefinite" />
          <animate attributeName="y2" from="0%" to="200%" dur="13s" repeatCount="indefinite" />
        </line>
      </svg>
    </div>
  );
};

export default CircuitBackground;
