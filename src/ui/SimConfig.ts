import { SPH } from '../sph/constants';

export interface SimConfig {
  particleCount: number;
  stiffness: number;
  viscosity: number;
  gravity: number;
  boundaryDamping: number;
  maxVelocity: number;
  xsphEpsilon: number;
  surfaceTension: number;
  threshold: number;
  renderScale: number;
  paused: boolean;
  substepLimit: number;
}

export const DEFAULT_PARTICLE_COUNT = 100000;

export function createDefaultConfig(): SimConfig {
  return {
    particleCount: DEFAULT_PARTICLE_COUNT,
    stiffness: SPH.stiffness,
    viscosity: SPH.viscosity,
    gravity: SPH.gravity,
    boundaryDamping: SPH.boundaryDamping,
    maxVelocity: SPH.maxVelocity,
    xsphEpsilon: SPH.xsphEpsilon,
    surfaceTension: SPH.surfaceTension,
    threshold: 0.75,
    renderScale: 0.5,
    paused: false,
    substepLimit: 3,
  };
}
