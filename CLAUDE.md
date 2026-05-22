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
- **`GPUCompute.ts`** (SPH) — Lagrangian particle solver. Spatial hash grid (`MAX_PER_CELL=16`), pairwise kernel density/force computation, Tait pressure, XSPH smoothing
- **`FLIPCompute.ts`** (FLIP) — Hybrid PIC/FLIP solver. Particles → staggered MAC grid (P2G), pressure projection via Jacobi iteration, grid → particles (G2P). Uses 6 staggered face velocity buffers for proper incompressibility
- **`EulerCompute.ts`** (Euler) — Pure grid Navier-Stokes solver. Semi-Lagrangian advection (RK2) on staggered MAC grid, 80-iteration Jacobi pressure projection, velocity extrapolation into air cells (Bridson §6.3), upwind density advection with 4 substeps. No particles during simulation — fixed O(n³) cost independent of fluid volume. Uses `copyBufferToBuffer` for velocity advection ping-pong, bind group ping-pong for density and pressure Jacobi

All three implement the same interface: `encodeStep()`, `uploadInitialPositions()`, `resetVelocities()`, `getDensityFieldBuffer()`, `getParamsBuffer()`, `updateSimConfig()`

**`WebGPURenderer.ts`** — render pipeline orchestrator
- Creates `rg32float` 3D texture (100^3, STORAGE_BINDING | TEXTURE_BINDING) shared between compute and render
- `bufferToTexture.wgsl` compute pass converts u32 density buffer → 3D texture with 3x3x3 box filter (workgroup 4,4,4)
- `waterRaymarch.wgsl` renders fullscreen triangle with WGSL raymarching (400 iterations, step 0.025)
- `wireframe.wgsl` renders container box as line-list (24 verts, 12 edges)
- `rebindComputeBuffers()` swaps density/params buffers when switching solvers without recreating pipelines/textures

**SPH frame pipeline:**
```
1. clearGrid → insertParticles → [substeps × (density → forces → integrate)]
2. clearDensityField → splatDensity (atomic u32 buffer)
3. bufferToTexture compute (u32 → rg32float 3D texture)
4. render pass: clear bg → draw wireframe → draw water (fullscreen tri, alpha blend)
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
- `ControlPanel.ts` — particle count slider, physics sliders, render sliders, pause/reset
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
- Foam uses XSPH magnitude (not velocity) — detects particle collisions, not free-fall. Restricted to `N.y > 0` surfaces
- Real transparency: alpha driven by `depthFactor` (secondary ray march thickness), Fresnel mixes background as reflection
- Spatial hash cell indices offset by `halfContainerX`/`halfContainerZ` so all indices are non-negative — the prime-number hash function produces asymmetric bucket distributions for negative i32 inputs, causing one-sided turbulence from uneven `MAX_PER_CELL` overflow. Must keep offset consistent across insertParticles, computeDensity, computeForces (GPU), and simulation.ts (CPU)
- Water color uses blue absorption filter `vec3(0.8, 0.88, 1.0)` — mimics real water absorbing red/green light. Fresnel reflection tint is suppressed over foam regions to keep foam white
- **SPH cell entries buffer scales with particle count**: `tableSize = nextPowerOfTwo(N * 2)`, buffer = `tableSize * MAX_PER_CELL * 4`. At 360k+ this was 268MB with old settings (3x multiplier, MAX_PER_CELL=32) — reduced to 2x and 16 to keep under ~67MB
- **splatRadiusCells floor of 2**: prevents single-cell splatting when splatRadius ≈ fieldCellSize, which produces blocky/voxel artifacts in the density field

## Tuning knobs

- `constants.ts` — stiffness (150), viscosity (2.5), XSPH epsilon (0.15), surface tension (0.2), boundary damping (-0.5), Tait gamma (7)
- `SimConfig.ts` defaults — particle count (100k), splat radius (0.15), density threshold (0.55), render scale (0.5), fixed dt (0.004), max substeps (6)
- `main.ts` — container size (4×4×4), field resolution (100)
- `waterRaymarch.wgsl` / `water.frag.glsl` — foam XSPH thresholds (0.2–1.5), step size (0.025), iterations (400), depth absorption rate (3.0), shallow/deep colors, blue absorption filter, Fresnel reflection strength (0.4)
- `computeForces.wgsl` — wall collision radius and stiffness (softened: `collisionRadius * 1.0`, `collisionStiffness * 0.5`)
- `integrate.wgsl` — velocity damping (`1.0 - 0.5 * dt`)

## Known issues

- **Initial drop hollow**: A visible hollow/concavity appears on the particle block during the initial drop. Root cause is undiagnosed. Attempted fixes that did NOT work: stiffness ramp (0→150 over 1s cubic ease-in), density threshold ramp (3.0→0.75), particle position jitter, swapping Y/Z init order to hide unfilled grid slots. The issue persists across all these changes, suggesting it is NOT caused by: pressure explosion from lattice packing, threshold sensitivity, lattice regularity, or unfilled grid slots at high-Z. Needs proper visual debugging (e.g. rendering normals/depth/density separately) to isolate which rendering component creates the artifact.
- **FLIP solver disabled**: All WebGPU pipeline binding-limit issues are fixed (every shader ≤ 8 storage bindings). The remaining bug is **gravity double-counting in the FLIP delta**: `vOldVel` must store the pre-gravity velocity so the G2P delta (`vVel_projected - vOldVel`) correctly carries gravity + pressure correction to particles. Without this, gravity cancels in the delta (present in both old and new grid velocities), floor particles get pushed upward by the unbalanced pressure correction, and the simulation produces a spongy convection pattern. A partial fix exists in `flipNormalizeA.wgsl` (gravity applied after storing `vOldVel`) but the solver still needs: (a) gravity restricted to fluid faces only (not air), (b) Jacobi iteration count increase or better pressure solver for 100³ grids, (c) end-to-end validation. The Euler solver has not been validated either.
- **FLIP/Euler binding limit fixes already done**: flipClearGrid replaced with `encoder.clearBuffer()`, flipNormalize split into A (8 storage) + B (7 storage), xsph buffer removed from flipG2P (9→8), flipProjectStaggered V-face decomposition fixed, `requiredLimits` in device creation uses selective key-value syntax to avoid `GPUSupportedLimits` prototype getter trap (gpuweb #4277)

## Scaling notes

- Container is fixed at 4×4×4. At 100k particles the block fills ~51% of the container. Higher counts pack denser
- Raymarching iterations (400) × step size (0.025) = 10 units, covers the 4×4×4 diagonal (√48 ≈ 6.93) with margin
- SPH cell entries buffer: `nextPowerOfTwo(N * 2) * 16 * 4` bytes. At 1M particles ≈ 128MB — test on target GPU before going higher
- Stiffness of 50 (original) causes supersonic compression at high particle counts — increased to 150 for stability
- Raymarching is the main GPU bottleneck on laptops — fullscreen shader runs 400 steps × multiple texture samples per hit pixel. Reducing render scale or step count are the main perf levers
