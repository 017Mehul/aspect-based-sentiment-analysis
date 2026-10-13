"use client";

import { Canvas, useFrame } from "@react-three/fiber";
import { Float, MeshDistortMaterial, Sparkles } from "@react-three/drei";
import { useRef } from "react";
import type { Mesh } from "three";

function FloatingShape({
  position,
  scale,
  color,
  speed,
}: {
  position: [number, number, number];
  scale: number;
  color: string;
  speed: number;
}) {
  const mesh = useRef<Mesh>(null);

  useFrame(({ clock }) => {
    if (!mesh.current) return;
    mesh.current.rotation.x = clock.elapsedTime * 0.12 * speed;
    mesh.current.rotation.y = clock.elapsedTime * 0.19 * speed;
  });

  return (
    <Float speed={1.15 * speed} rotationIntensity={0.55} floatIntensity={0.7}>
      <mesh ref={mesh} position={position} scale={scale}>
        <icosahedronGeometry args={[1, 2]} />
        <MeshDistortMaterial
          color={color}
          roughness={0.2}
          metalness={0.12}
          distort={0.28}
          speed={1.7}
        />
      </mesh>
    </Float>
  );
}

function SceneContents() {
  return (
    <>
      <ambientLight intensity={1.15} />
      <directionalLight position={[4, 5, 4]} intensity={2.1} color="#ffffff" />
      <pointLight position={[-4, -2, 3]} intensity={18} color="#c8f34a" />
      <pointLight position={[3, 2, -2]} intensity={12} color="#91b83c" />
      <Float speed={0.8} rotationIntensity={0.12} floatIntensity={0.25}>
        <mesh position={[0, 0, -0.7]} rotation={[0.4, 0.2, 0.3]}>
          <torusGeometry args={[1.48, 0.025, 16, 120]} />
          <meshStandardMaterial color="#a9d52f" metalness={0.45} roughness={0.28} />
        </mesh>
        <mesh position={[0, 0, -0.75]} rotation={[-0.55, 0.4, -0.4]}>
          <torusGeometry args={[1.72, 0.012, 12, 120]} />
          <meshStandardMaterial color="#d7f48a" metalness={0.35} roughness={0.3} />
        </mesh>
      </Float>
      <FloatingShape position={[-0.25, 0.05, 0.25]} scale={1.04} color="#d8f36a" speed={0.85} />
      <FloatingShape position={[1.4, 0.82, -0.1]} scale={0.34} color="#8dbd35" speed={1.25} />
      <FloatingShape position={[-1.45, -0.78, 0.1]} scale={0.25} color="#f0d94b" speed={1.4} />
      <FloatingShape position={[1.38, -1.02, -0.3]} scale={0.16} color="#b2d94a" speed={1.8} />
      <Sparkles count={42} scale={4.4} size={2.1} speed={0.22} color="#91b83c" opacity={0.65} />
    </>
  );
}

export default function HeroScene() {
  return (
    <div className="hero-scene" aria-label="Animated three-dimensional lime-green abstract forms">
      <Canvas camera={{ position: [0, 0, 5.2], fov: 42 }} dpr={[1, 1.6]}>
        <SceneContents />
      </Canvas>
      <div className="scene-orbit-label scene-label-one"><span /> aspect signals</div>
      <div className="scene-orbit-label scene-label-two"><span /> context aware</div>
    </div>
  );
}
