import { SPH } from '../sph/constants';

export type Algorithm = 'sph' | 'flip' | 'euler';

export interface SimConfig {
  algorithm: Algorithm;
  particleCount: number;
  stiffnessMultiplier: number;
  viscosity: number;
  gravity: number;
  boundaryDamping: number;
  xsphEpsilon: number;
  surfaceTension: number;
  splatRadius: number;
  threshold: number;
  renderScale: number;
  paused: boolean;
  substepLimit: number;
  lightEnabled: boolean;
  fxaaEnabled: boolean;
  debugMode: boolean;
}

export const DEFAULT_PARTICLE_COUNT = 100000;

export function createDefaultConfig(): SimConfig {
  return {
    algorithm: 'sph',
    particleCount: DEFAULT_PARTICLE_COUNT,
    stiffnessMultiplier: 1.0,
    viscosity: SPH.viscosity,
    gravity: SPH.gravity,
    boundaryDamping: SPH.boundaryDamping,
    xsphEpsilon: SPH.xsphEpsilon,
    surfaceTension: SPH.surfaceTension,
    splatRadius: 0.05,
    threshold: 0.3,
    renderScale: 0.5,
    paused: false,
    substepLimit: 3,
    lightEnabled: true,
    fxaaEnabled: true,
    debugMode: false,
  };
}
