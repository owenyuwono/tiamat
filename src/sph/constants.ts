export const SPH = {
  smoothingRadius: 0.1,
  restDensity: 1000,
  stiffness: 150,
  taitGamma: 7,
  viscosity: 2.5,
  gravity: -9.81,
  boundaryDamping: -0.5,
  maxVelocity: 5.0,
  collisionRadius: 0.035,
  collisionStiffness: 8000,
  xsphEpsilon: 0.15,
  surfaceTension: 0.2,
} as const;

export const GRID_CELL_SIZE = SPH.smoothingRadius;
