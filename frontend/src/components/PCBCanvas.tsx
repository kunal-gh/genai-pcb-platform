import React, { useRef, Suspense } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Environment } from '@react-three/drei';
import { motion } from 'framer-motion';
import * as THREE from 'three';
import { Layers, Eye, Download } from 'lucide-react';

interface PCBCanvasProps {
  activeTab: 'design' | 'preview' | 'export';
  designState: {
    status: string;
    progress: number;
    message: string;
  };
}

// 3D PCB Board Component
function PCBBoard() {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.3) * 0.1;
    }
  });

  return (
    <group>
      {/* PCB Base */}
      <mesh ref={meshRef} position={[0, 0, 0]}>
        <boxGeometry args={[4, 0.1, 3]} />
        <meshStandardMaterial 
          color="#10B981" 
          roughness={0.3}
          metalness={0.6}
        />
      </mesh>

      {/* Copper Traces */}
      {[...Array(8)].map((_, i) => (
        <mesh key={i} position={[-1.5 + i * 0.4, 0.06, 0]}>
          <boxGeometry args={[0.05, 0.01, 2]} />
          <meshStandardMaterial 
            color="#D97706" 
            roughness={0.2}
            metalness={0.8}
            emissive="#D97706"
            emissiveIntensity={0.2}
          />
        </mesh>
      ))}

      {/* Components */}
      <mesh position={[0, 0.3, 0]}>
        <boxGeometry args={[0.8, 0.5, 0.8]} />
        <meshStandardMaterial color="#1E3A8A" roughness={0.4} metalness={0.3} />
      </mesh>

      <mesh position={[-1, 0.15, 0.8]}>
        <cylinderGeometry args={[0.15, 0.15, 0.3, 16]} />
        <meshStandardMaterial color="#3B82F6" roughness={0.3} metalness={0.5} />
      </mesh>

      <mesh position={[1, 0.15, -0.8]}>
        <cylinderGeometry args={[0.15, 0.15, 0.3, 16]} />
        <meshStandardMaterial color="#3B82F6" roughness={0.3} metalness={0.5} />
      </mesh>

      {/* Vias */}
      {[...Array(12)].map((_, i) => {
        const x = (Math.random() - 0.5) * 3;
        const z = (Math.random() - 0.5) * 2;
        return (
          <mesh key={`via-${i}`} position={[x, 0.06, z]}>
            <cylinderGeometry args={[0.03, 0.03, 0.02, 8]} />
            <meshStandardMaterial 
              color="#F59E0B" 
              roughness={0.1}
              metalness={0.9}
            />
          </mesh>
        );
      })}
    </group>
  );
}

const PCBCanvas: React.FC<PCBCanvasProps> = ({ activeTab, designState }) => {
  if (activeTab === 'design') {
    return (
      <div className="w-full h-full flex items-center justify-center">
        <div className="text-center space-y-4">
          <div className="w-20 h-20 mx-auto glass-card rounded-2xl flex items-center justify-center">
            <Layers className="w-10 h-10 text-circuit-cyan" />
          </div>
          <h3 className="text-2xl font-semibold">2D Design View</h3>
          <p className="text-white/60 max-w-md">
            {designState.status === 'success'
              ? 'Your PCB design is ready! Switch to 3D Preview to see the board.'
              : 'Generate a design to see the 2D schematic view here'}
          </p>
        </div>
      </div>
    );
  }

  if (activeTab === 'preview') {
    return (
      <div className="w-full h-full relative">
        {designState.status === 'success' ? (
          <Suspense fallback={
            <div className="w-full h-full flex items-center justify-center">
              <div className="spinner w-12 h-12" />
            </div>
          }>
            <Canvas shadows>
              <PerspectiveCamera makeDefault position={[5, 3, 5]} />
              {/* @ts-ignore */}
              <OrbitControls 
                enablePan
                enableZoom
                enableRotate
                minDistance={3}
                maxDistance={15}
              />
              
              <ambientLight intensity={0.5} />
              <directionalLight 
                position={[10, 10, 5]} 
                intensity={1}
                castShadow
                shadow-mapSize-width={2048}
                shadow-mapSize-height={2048}
              />
              <pointLight position={[-10, -10, -5]} intensity={0.5} color="#3B82F6" />
              
              <PCBBoard />
              
              <Environment preset="city" />
              
              {/* Grid */}
              <gridHelper args={[10, 20, '#3B82F6', '#1E3A8A']} position={[0, -0.5, 0]} />
            </Canvas>

            {/* Controls hint */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="absolute bottom-4 left-4 glass-card px-4 py-2 text-sm text-white/60"
            >
              <p>🖱️ Drag to rotate • Scroll to zoom • Right-click to pan</p>
            </motion.div>
          </Suspense>
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <div className="text-center space-y-4">
              <div className="w-20 h-20 mx-auto glass-card rounded-2xl flex items-center justify-center">
                <Eye className="w-10 h-10 text-circuit-cyan" />
              </div>
              <h3 className="text-2xl font-semibold">3D Preview</h3>
              <p className="text-white/60 max-w-md">
                Generate a design to see the interactive 3D preview
              </p>
            </div>
          </div>
        )}
      </div>
    );
  }

  // Export tab
  return (
    <div className="w-full h-full flex items-center justify-center">
      <div className="text-center space-y-6 max-w-md">
        <div className="w-20 h-20 mx-auto glass-card rounded-2xl flex items-center justify-center">
          <Download className="w-10 h-10 text-circuit-cyan" />
        </div>
        <h3 className="text-2xl font-semibold">Export Files</h3>
        <p className="text-white/60">
          {designState.status === 'success'
            ? 'Download your manufacturing files'
            : 'Generate a design to export manufacturing files'}
        </p>

        {designState.status === 'success' && (
          <div className="space-y-3">
            {[
              { name: 'Gerber Files', format: 'ZIP', size: '2.4 MB' },
              { name: 'Drill Files', format: 'DRL', size: '156 KB' },
              { name: 'Bill of Materials', format: 'CSV', size: '12 KB' },
              { name: 'Assembly Drawing', format: 'PDF', size: '890 KB' },
            ].map((file) => (
              <motion.button
                key={file.name}
                className="w-full glass-card-hover p-4 flex items-center justify-between"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="flex items-center space-x-3">
                  <Download className="w-5 h-5 text-circuit-cyan" />
                  <div className="text-left">
                    <div className="font-medium">{file.name}</div>
                    <div className="text-sm text-white/60">{file.format} • {file.size}</div>
                  </div>
                </div>
                <div className="text-circuit-cyan text-sm font-medium">Download</div>
              </motion.button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default PCBCanvas;
