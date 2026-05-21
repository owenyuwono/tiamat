# fluid3

Real-time SPH (Smoothed Particle Hydrodynamics) fluid simulation with raymarched water rendering.

## Stack

- Three.js (v0.184) + WebGL2 for rendering
- WebGPU compute shaders for simulation (CPU fallback in `src/sph/`)
- TypeScript, Vite
- GLSL shaders imported via `?raw`, WGSL shaders via `?raw`

## Architecture

### GPU Simulation (`src/gpu/`)
- **WebGPU compute SPH** — 70k particles at 60fps
- `GPUCompute.ts` — device init, buffer allocation, 7 compute pipelines, double-buffered density field readback
- Compute pass order per frame: clearGrid → insertParticles → [substep: computeDensity → computeForces → integrate] → clearDensityField → splatDensity → copy to staging
- Spatial hash grid with fixed-size cells (`MAX_PER_CELL=32`), `atomicAdd` for lock-free insertion
- Density field splatting uses fixed-point `u32` atomics (scale 10000) since WebGPU lacks atomic floats
- Table size is dynamic: `nextPowerOfTwo(particleCount * 3)`
- Params struct (128 bytes, 32 fields) shared across all shaders — WGSL struct layout must match TypeScript array indices exactly

### CPU Simulation fallback (`src/sph/`)
- `simulation.ts` — SoA particle data (Float32Array), prefix-sum spatial hash, same physics as GPU
- `constants.ts` — all physics tuning parameters (shared by both GPU and CPU paths)
- Used when WebGPU is unavailable (`GPUCompute.create()` returns null)

### Rendering (`src/rendering/`)
- **3D density texture** (80^3, RG format): R = density, G = XSPH-weighted density (collision impact indicator)
- `DensityField.ts` — CPU splatting (used by CPU fallback path only)
- `WaterRenderer.ts` — Data3DTexture (RGFormat, FloatType, LinearFilter), ShaderMaterial with BackSide rendering, `transparent: true`, `depthWrite: false`
- **Raymarching shader** (`shaders/water.frag.glsl`): fixed-step march (step 0.025, 400 iterations for 4x4x4 container) with binary refinement, Fresnel + Blinn-Phong shading, depth-based alpha transparency, XSPH-based collision foam restricted to upward-facing surfaces

## Key design decisions

- `BackSide` box rendering for raymarching — handles camera inside/outside volume
- Ray direction computed from `gl_FragCoord` + inverse view-projection matrix (not interpolated vertex position) to eliminate triangle-seam artifacts
- GPU path: density field read back via double-buffered staging buffers (MAP_READ), 1-frame latency
- DensityField (CPU path) shares its Float32Array backing with the 3D texture to avoid per-frame copy
- Mirror particles at 6 walls restore density at boundaries (prevents gaps)
- Negative pressure clamped to 0 (Tait equation) — prevents tensile instability at free surface
- Density guard in integrate.wgsl: `max(density, 10.0)` prevents division-by-zero freezing particles
- Foam uses XSPH magnitude (not velocity) — detects particle collisions, not free-fall. Restricted to `N.y > 0` surfaces
- Real transparency: alpha driven by `depthFactor` (secondary ray march thickness), Fresnel mixes background as reflection
- Spatial hash cell indices offset by `halfContainerX`/`halfContainerZ` so all indices are non-negative — the prime-number hash function produces asymmetric bucket distributions for negative i32 inputs, causing one-sided turbulence from uneven `MAX_PER_CELL` overflow. Must keep offset consistent across insertParticles, computeDensity, computeForces (GPU), and simulation.ts (CPU)
- Water color uses blue absorption filter `vec3(0.8, 0.88, 1.0)` — mimics real water absorbing red/green light. Fresnel reflection tint is suppressed over foam regions to keep foam white

## Tuning knobs

- `constants.ts` — stiffness (150), viscosity (2.5), XSPH epsilon, surface tension, Tait gamma
- `main.ts` — particle count (70k), container size (4x4x4), splat radius (0.1), density threshold (0.75), fixed dt (0.005), max substeps (3)
- `water.frag.glsl` — foam XSPH thresholds (0.2–1.5), step size (0.025), depth absorption rate (3.0), shallow/deep colors, blue absorption filter, Fresnel reflection strength (0.4)

## Known issues

- **Initial drop hollow**: A visible hollow/concavity appears on the particle block during the initial drop. Root cause is undiagnosed. Attempted fixes that did NOT work: stiffness ramp (0→150 over 1s cubic ease-in), density threshold ramp (3.0→0.75), particle position jitter, swapping Y/Z init order to hide unfilled grid slots. The issue persists across all these changes, suggesting it is NOT caused by: pressure explosion from lattice packing, threshold sensitivity, lattice regularity, or unfilled grid slots at high-Z. Needs proper visual debugging (e.g. rendering normals/depth/density separately) to isolate which rendering component creates the artifact.

## Scaling notes

- Container must be large enough for the particle block: `blockWidth = (ceil(cbrt(N)) - 1) * 0.05`
- At 70k: block is 2.05m in 4m container (51%). Larger counts need bigger containers or the physics becomes unstable (pressure waves with nowhere to dissipate)
- Raymarching iterations must cover the container diagonal: `iterations * stepSize > sqrt(3) * containerSize`
- Stiffness of 50 (original) causes supersonic compression at high particle counts — increased to 150 for stability
