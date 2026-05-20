# fluid3

Real-time SPH (Smoothed Particle Hydrodynamics) fluid simulation with raymarched water rendering.

## Stack

- Three.js (v0.184) + WebGL2
- TypeScript, Vite
- No custom build plugins — GLSL shaders imported via `?raw`

## Architecture

### Simulation (`src/sph/`)
- **CPU-based SPH** with spatial hash grid for neighbor lookup
- `simulation.ts` — SoA particle data (Float32Array), Tait equation pressure, XSPH velocity smoothing, surface tension, mirror boundary particles
- `constants.ts` — all physics tuning parameters
- Particles expose position/velocity arrays (direct references, no copy) for rendering

### Rendering (`src/rendering/`)
- **3D density texture** (64^3, RG format): R = density, G = velocity-weighted density
- `DensityField.ts` — CPU splatting of particles into the 3D texture with kernel `(1-r^2/R^2)^3`; tracks dirty region for partial clears
- `WaterRenderer.ts` — orchestrates Data3DTexture (RGFormat, FloatType, LinearFilter), ShaderMaterial with BackSide rendering on a box mesh
- **Raymarching shader** (`shaders/water.frag.glsl`): fixed-step march with binary refinement, Fresnel + Blinn-Phong water shading, velocity-based foam (G/R channel), gradient-magnitude thin-region foam, depth estimation via secondary march

## Key design decisions

- `BackSide` box rendering for raymarching — handles camera inside/outside volume
- Ray direction computed from `gl_FragCoord` + inverse view-projection matrix (not interpolated vertex position) to eliminate triangle-seam artifacts
- DensityField shares its Float32Array backing with the 3D texture to avoid per-frame copy
- Mirror particles at 6 walls restore density at boundaries (prevents gaps)
- Negative pressure clamped to 0 (Tait equation) — prevents tensile instability at free surface

## Tuning knobs

- `constants.ts` — viscosity, XSPH epsilon, surface tension, stiffness, Tait gamma
- `main.ts` — particle count, splat radius, density threshold
- `water.frag.glsl` — foam velocity/gradient thresholds, step size, depth colors
