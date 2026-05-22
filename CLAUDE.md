# fluid3

Real-time fluid simulation with GPU solvers and raymarched water rendering. SPH is production-ready; FLIP and Euler are disabled in UI (coming soon).

## Stack

- Three.js (v0.184) for camera/controls math only (WebGPU path) or full rendering (CPU fallback)
- WebGPU compute shaders for simulation + WebGPU render pipeline for raymarching
- CPU fallback: Three.js WebGL2 rendering + CPU SPH (`src/sph/`)
- TypeScript, Vite
- GLSL shaders imported via `?raw`, WGSL shaders via `?raw`

## Architecture

### GPU Pipeline (`src/gpu/`) — zero CPU readback

Single WebGPU device handles both compute and render. One command encoder per frame.

**Three solvers, switchable at runtime:**
- **`GPUCompute.ts`** (SPH) — Lagrangian particle solver. Spatial hash grid (`MAX_PER_CELL=16`), pairwise kernel density/force computation, Tait pressure, XSPH smoothing, mirror boundary forces
- **`FLIPCompute.ts`** (FLIP) — Hybrid PIC/FLIP solver. Particles → staggered MAC grid (P2G), pressure projection via Jacobi iteration, grid → particles (G2P). Uses 6 staggered face velocity buffers for proper incompressibility
- **`EulerCompute.ts`** (Euler) — Pure grid Navier-Stokes solver. Semi-Lagrangian advection (RK2) on staggered MAC grid, 80-iteration Jacobi pressure projection, velocity extrapolation into air cells (Bridson §6.3), upwind density advection with 4 substeps. No particles during simulation — fixed O(n³) cost independent of fluid volume. Uses `copyBufferToBuffer` for velocity advection ping-pong, bind group ping-pong for density and pressure Jacobi

All three implement the same interface: `encodeStep()`, `uploadInitialPositions()`, `resetVelocities()`, `getDensityFieldBuffer()`, `getParamsBuffer()`, `updateSimConfig()`

**`WebGPURenderer.ts`** — render pipeline orchestrator
- Creates `rg32float` 3D texture (100^3, STORAGE_BINDING | TEXTURE_BINDING) shared between compute and render
- `bufferToTexture.wgsl` compute pass converts u32 density buffer → 3D texture with 3x3x3 box filter (workgroup 4,4,4)
- `waterRaymarch.wgsl` renders fullscreen triangle with WGSL raymarching (400 iterations, step 0.025)
- `floor.wgsl` renders textured sand floor slab matching container footprint
- Wireframe pipeline exists but is not drawn (removed from render pass)
- Water bind group includes sand texture + repeat sampler (bindings 3-4) for refraction
- `rebindComputeBuffers()` swaps density/params buffers when switching solvers without recreating pipelines/textures
- `loadFloorTexture()` recreates both floor AND water bind groups when sand texture changes

**SPH frame pipeline:**
```
1. clearGrid → insertParticles → [substeps × (density → forces → integrate)]
2. clearDensityField → splatDensity (atomic u32 buffer)
3. bufferToTexture compute (u32 → rg32float 3D texture)
4. render pass: clear bg → draw floor → draw water (fullscreen tri, alpha blend)
```

**FLIP frame pipeline (disabled — see known issues):**
```
1. encoder.clearBuffer × 15 (zero all grid buffers via DMA, no shader needed)
2. flipP2G → flipNormalizeA (U+V faces) → flipNormalizeB (W faces + fluid marker)
3. [substeps × (flipDivergence → 40× jacobi → flipProject → flipProjectStaggered)]
4. flipG2P → clearDensityField → flipSplatDensity
5. bufferToTexture → render pass (same as SPH)
```

**Euler frame pipeline:**
```
1. eulerClearGrid (zero pressure/divergence)
2. eulerAdvectVelocity (semi-Lagrangian RK2 → temp buffers)
3. copyBufferToBuffer (temp → main velocity)
4. eulerApplyForces (gravity on v-faces)
5. eulerDivergence → [80 × eulerJacobi] → eulerProject
6. eulerExtrapolateInit (classify face validity from density markers)
7. eulerExtrapolateSweep × 3 (extend velocity into air, ping-pong valid buffers)
8. eulerAdvectDensity × 4 (upwind finite-difference, ping-pong density)
9. eulerWriteDensity (f32 marker → atomic u32 densityField)
10. bufferToTexture → render pass (same as SPH/FLIP)
```

**Params struct** (128 bytes, 32 fields) shared across all shaders — WGSL struct layout must match TypeScript array indices exactly. All three solvers write the same layout so the renderer doesn't care which solver is active.

### UI (`src/ui/`)
- `AlgorithmPicker.ts` — standalone SPH/FLIP/Euler toggle with technical descriptions
- `ControlPanel.ts` — particle count slider, physics sliders, render sliders, pause/reset, light toggle
- `SimConfig.ts` — all runtime-tunable parameters, `Algorithm` type (`'sph' | 'flip' | 'euler'`)
- `StatsPanel.ts` — FPS, GPU timing (when `timestamp-query` available), substep count

### CPU Simulation fallback (`src/sph/`)
- `simulation.ts` — SoA particle data (Float32Array), prefix-sum spatial hash, same physics as GPU SPH
- `constants.ts` — all physics tuning parameters (shared by both GPU and CPU paths)
- Used when WebGPU is unavailable (`GPUCompute.create()` returns null)
- Falls back to Three.js WebGLRenderer with `WaterRenderer.ts` (Data3DTexture + GLSL raymarching)

### Rendering fallback (`src/rendering/`)
- `WaterRenderer.ts` — Three.js ShaderMaterial with BackSide box rendering, `transparent: true`, `depthWrite: false`
- `DensityField.ts` — CPU splatting into Float32Array backing a Data3DTexture
- `water.frag.glsl` — GLSL raymarching shader (reference implementation, WGSL port in `waterRaymarch.wgsl`)

## Key design decisions

- **Fixed container size (4×4×4)**: container does not scale with particle count. More particles = denser fluid, not a bigger box. Avoids raymarcher coverage issues and GPU memory explosion at high counts
- **Zero CPU readback (GPU path)**: density field stays on GPU as a 3D texture shared between compute (atomic u32 buffer → bufferToTexture copy) and render (hardware-filtered sampling). Eliminates the old 4MB staging buffer round-trip
- **`float32-filterable` device feature**: required for `rg32float` textures with linear sampling. Requested conditionally in `GPUCompute.create()` — without it, bind group creation fails silently and the entire command encoder becomes a no-op
- **WGSL Y-axis flip**: `@builtin(position).y` is 0 at top (opposite of GLSL `gl_FragCoord.y`). NDC Y must be negated in `waterRaymarch.wgsl`
- **GPU throttling**: `await device.queue.onSubmittedWorkDone()` prevents GPU queue buildup. Without it, `queue.submit()` is non-blocking — JS queues frames faster than GPU processes them, causing increasing input lag and thermal throttling on laptops
- Fullscreen triangle for raymarching — simpler than BackSide box, AABB test discards non-intersecting pixels
- Ray direction computed from `@builtin(position)` + inverse view-projection matrix
- Mirror particles at 6 walls restore density at boundaries (prevents gaps)
- Negative pressure clamped to 0 (Tait equation) — prevents tensile instability at free surface
- Density guard in integrate.wgsl: `max(density, 10.0)` prevents division-by-zero freezing particles
- **PBR water shader** (`waterRaymarch.wgsl`): 6-tap central difference normals, GGX/Cook-Torrance specular, volumetric shadow rays (16 steps toward light), density-field ambient occlusion (4 samples along normal), per-channel Beer-Lambert absorption (tropical turquoise palette), ACES filmic tone mapping
- **Refraction**: water shader samples sand texture via refracted ray (Snell's law, IOR 1.33). Refracted ray intersects floor plane (y=0), samples sand texture at that position. Shallow water blends toward floor color, deep water stays blue
- **Reflection**: sky gradient (warm horizon → sky blue zenith) based on `reflect(rd, N)` direction, blended by Fresnel
- **Foam system**: three sources combined — (1) XSPH impact magnitude (turbulence detection, amplified 5x during splatting), (2) curvature foam (gradLen/density for stretched surfaces), (3) spray (thin fluid edges just above threshold with steep gradient). Foam persists via temporal decay: `clearDensityField.wgsl` decays impact channel by 0.97/frame instead of zeroing, giving ~0.77s half-life
- Real transparency: alpha driven by `depthFactor` (secondary ray march thickness), Fresnel mixes sky gradient as reflection
- Spatial hash cell indices offset by `halfContainerX`/`halfContainerZ` so all indices are non-negative — the prime-number hash function produces asymmetric bucket distributions for negative i32 inputs, causing one-sided turbulence from uneven `MAX_PER_CELL` overflow. Must keep offset consistent across insertParticles, computeDensity, computeForces (GPU), and simulation.ts (CPU)
- **Tropical water color**: Beer-Lambert absorption `vec3(3.0, 1.0, 0.4)` — red absorbs fastest, blue least. Scatter color `(0.2, 0.7, 0.65)` turquoise, deep color `(0.01, 0.15, 0.4)` navy. Background is sky blue `0x87CEEB`
- **SPH cell entries buffer scales with particle count**: `tableSize = nextPowerOfTwo(N * 3)`, buffer = `tableSize * MAX_PER_CELL * 4`. At 100k particles ≈ 67MB. Reducing MAX_PER_CELL below 32 or tableSize multiplier below 3 causes neighbor-miss artifacts (pressure spikes, particles exploding upward on impact) — do not reduce for memory savings without validating settling behavior
- **splatRadiusCells floor of 2**: prevents single-cell splatting when splatRadius ≈ fieldCellSize, which produces blocky/voxel artifacts in the density field

## Tuning knobs

- `constants.ts` — stiffness (150), viscosity (2.5), XSPH epsilon (0.15), surface tension (0.2), boundary damping (-0.5), Tait gamma (7)
- `SimConfig.ts` defaults — particle count (100k), splat radius (0.10), density threshold (0.65), render scale (0.5), fixed dt (0.004), max substeps (6), lightEnabled toggle
- `main.ts` — container size (4×4×4), field resolution (100)
- `waterRaymarch.wgsl` — foam thresholds (`smoothstep(0.002, 0.1)` for impact, `smoothstep(0.3, 0.8)` for curvature), step size (0.025), iterations (400), absorption `vec3(3.0, 1.0, 0.4)`, GGX roughness (0.06 water, 0.5 foam), shadow extinction (3.0), Fresnel reflection strength (0.6), alpha floor (0.3)
- `splatDensity.wgsl` — impact amplification (5x) to survive box filter
- `clearDensityField.wgsl` — foam decay rate (0.97/frame)
- `computeForces.wgsl` — mirror boundary forces (pressure-based, replaces old wall springs)
- `integrate.wgsl` — velocity damping (`1.0 - 1.5 * dt`), XSPH applied to position (not velocity)

## SPH parameter coupling — critical constraints

**Do not change SPH physics parameters independently.** Wall springs, XSPH, dt, boundary damping, stiffness, and MAX_PER_CELL are tightly coupled. Changing one without revalidating the others causes instability. The current parameter set (commit `7c79d3f` baseline + gentle velocity damping) was validated as a working unit.

Specific constraints discovered through debugging:
- **Mirror boundary forces replaced wall springs**: `computeForces.wgsl` now uses pressure-based mirror forces at all 6 walls (same approach as `computeDensity.wgsl` mirror particles). The old wall springs are removed. Mirror forces produce smoother boundary behavior than spring forces
- **XSPH on position**: XSPH correction is now applied to position update (`pos += (vel + eps * xsph) * dt`) rather than velocity. This is the Monaghan formulation. Current ε=0.15 (from SimConfig xsphEpsilon)
- **dt must be consistent**: `main.ts` fixedDt and `GPUCompute.ts` fixedDt must match (both 0.008). A mismatch halves the effective damping and changes substep count
- **MAX_PER_CELL=32 and tableSize=N*3 are minimum**: reducing these causes neighbor-miss artifacts — density is underestimated during compression, then pressure explodes when neighbors are rediscovered. At 100k particles this uses ~67MB which is acceptable
- **Velocity damping**: `vel *= 1.0 - 1.5 * dt` — higher than original (was 0.5) to help settling with the new mirror forces

## Known issues

- **Initial drop hollow**: A visible hollow/concavity appears on the particle block during the initial drop. Root cause is undiagnosed. Attempted fixes that did NOT work: stiffness ramp (0→150 over 1s cubic ease-in), density threshold ramp (3.0→0.75), particle position jitter, swapping Y/Z init order to hide unfilled grid slots. The issue persists across all these changes, suggesting it is NOT caused by: pressure explosion from lattice packing, threshold sensitivity, lattice regularity, or unfilled grid slots at high-Z. Needs proper visual debugging (e.g. rendering normals/depth/density separately) to isolate which rendering component creates the artifact.
- **FLIP solver disabled**: All WebGPU pipeline binding-limit issues are fixed (every shader ≤ 8 storage bindings). The remaining bug is **gravity double-counting in the FLIP delta**: `vOldVel` must store the pre-gravity velocity so the G2P delta (`vVel_projected - vOldVel`) correctly carries gravity + pressure correction to particles. Without this, gravity cancels in the delta (present in both old and new grid velocities), floor particles get pushed upward by the unbalanced pressure correction, and the simulation produces a spongy convection pattern. A partial fix exists in `flipNormalizeA.wgsl` (gravity applied after storing `vOldVel`) but the solver still needs: (a) gravity restricted to fluid faces only (not air), (b) Jacobi iteration count increase or better pressure solver for 100³ grids, (c) end-to-end validation. The Euler solver has not been validated either.
- **FLIP/Euler binding limit fixes already done**: flipClearGrid replaced with `encoder.clearBuffer()`, flipNormalize split into A (8 storage) + B (7 storage), xsph buffer removed from flipG2P (9→8), flipProjectStaggered V-face decomposition fixed, `requiredLimits` in device creation uses selective key-value syntax to avoid `GPUSupportedLimits` prototype getter trap (gpuweb #4277)

## Scaling notes

- Container is fixed at 4×4×4. At 100k particles the block fills ~51% of the container. Higher counts pack denser
- Raymarching iterations (400) × step size (0.025) = 10 units, covers the 4×4×4 diagonal (√48 ≈ 6.93) with margin
- SPH cell entries buffer: `nextPowerOfTwo(N * 3) * 32 * 4` bytes. At 100k ≈ 67MB, at 1M ≈ 512MB — test on target GPU before going higher
- Stiffness of 50 (original) causes supersonic compression at high particle counts — increased to 150 for stability
- Raymarching is the main GPU bottleneck on laptops — fullscreen shader runs 400 steps × multiple texture samples per hit pixel. Reducing render scale or step count are the main perf levers
