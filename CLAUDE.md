# fluid3

Real-time SPH (Smoothed Particle Hydrodynamics) fluid simulation with raymarched water rendering.

## Stack

- Three.js (v0.184) for camera/controls math only (WebGPU path) or full rendering (CPU fallback)
- WebGPU compute shaders for simulation + WebGPU render pipeline for raymarching
- CPU fallback: Three.js WebGL2 rendering + CPU SPH (`src/sph/`)
- TypeScript, Vite
- GLSL shaders imported via `?raw`, WGSL shaders via `?raw`

## Architecture

### GPU Pipeline (`src/gpu/`) — zero CPU readback

Single WebGPU device handles both compute and render. One command encoder per frame.

**`GPUCompute.ts`** — device init (requests `float32-filterable` feature), buffer allocation, 7 compute pipelines
- `encodeStep(encoder, substeps)` — encodes all compute passes onto an external encoder (no submit, no staging copy)
- `step()` + `readDensityField()` — legacy path with staging buffer readback, kept for backward compat
- Compute pass order: clearGrid → insertParticles → [substep: computeDensity → computeForces → integrate] → clearDensityField → splatDensity
- Spatial hash grid with fixed-size cells (`MAX_PER_CELL=32`), `atomicAdd` for lock-free insertion
- Density field splatting uses fixed-point `u32` atomics (scale 10000) since WebGPU lacks atomic floats
- Params struct (128 bytes, 32 fields) shared across all shaders — WGSL struct layout must match TypeScript array indices exactly

**`WebGPURenderer.ts`** — render pipeline orchestrator
- Creates `rg32float` 3D texture (80^3, STORAGE_BINDING | TEXTURE_BINDING) shared between compute and render
- `bufferToTexture.wgsl` compute pass converts u32 density buffer → 3D texture (workgroup 4,4,4)
- `waterRaymarch.wgsl` renders fullscreen triangle with WGSL raymarching (ported from GLSL)
- `wireframe.wgsl` renders container box as line-list (24 verts, 12 edges)
- Linear sampler with clamp-to-edge for hardware trilinear filtering
- Uniforms: render params (192 bytes, 16-byte aligned) + wireframe params (80 bytes)
- Caches Matrix4/Vector3 to avoid per-frame allocations

**Frame pipeline:**
```
1. clearGrid → insertParticles → [substeps × (density → forces → integrate)]
2. clearDensityField → splatDensity (atomic u32 buffer)
3. bufferToTexture compute (u32 → rg32float 3D texture)
4. render pass: clear bg → draw wireframe → draw water (fullscreen tri, alpha blend)
```

### CPU Simulation fallback (`src/sph/`)
- `simulation.ts` — SoA particle data (Float32Array), prefix-sum spatial hash, same physics as GPU
- `constants.ts` — all physics tuning parameters (shared by both GPU and CPU paths)
- Used when WebGPU is unavailable (`GPUCompute.create()` returns null)
- Falls back to Three.js WebGLRenderer with `WaterRenderer.ts` (Data3DTexture + GLSL raymarching)

### Rendering fallback (`src/rendering/`)
- `WaterRenderer.ts` — Three.js ShaderMaterial with BackSide box rendering, `transparent: true`, `depthWrite: false`
- `DensityField.ts` — CPU splatting into Float32Array backing a Data3DTexture
- `water.frag.glsl` — GLSL raymarching shader (reference implementation, WGSL port in `waterRaymarch.wgsl`)

## Key design decisions

- **Zero CPU readback (GPU path)**: density field stays on GPU as a 3D texture shared between compute (atomic u32 buffer → bufferToTexture copy) and render (hardware-filtered sampling). Eliminates the old 4MB staging buffer round-trip.
- **`float32-filterable` device feature**: required for `rg32float` textures with linear sampling. Requested conditionally in `GPUCompute.create()` — without it, bind group creation fails silently and the entire command encoder becomes a no-op.
- **WGSL Y-axis flip**: `@builtin(position).y` is 0 at top (opposite of GLSL `gl_FragCoord.y`). NDC Y must be negated in `waterRaymarch.wgsl`.
- **GPU throttling**: `await device.queue.onSubmittedWorkDone()` prevents GPU queue buildup. Without it, `queue.submit()` is non-blocking — JS queues frames faster than GPU processes them, causing increasing input lag and thermal throttling on laptops.
- Fullscreen triangle for raymarching — simpler than BackSide box, AABB test discards non-intersecting pixels
- Ray direction computed from `@builtin(position)` + inverse view-projection matrix
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
- `waterRaymarch.wgsl` / `water.frag.glsl` — foam XSPH thresholds (0.2–1.5), step size (0.025), depth absorption rate (3.0), shallow/deep colors, blue absorption filter, Fresnel reflection strength (0.4)

## Known issues

- **Initial drop hollow**: A visible hollow/concavity appears on the particle block during the initial drop. Root cause is undiagnosed. Attempted fixes that did NOT work: stiffness ramp (0→150 over 1s cubic ease-in), density threshold ramp (3.0→0.75), particle position jitter, swapping Y/Z init order to hide unfilled grid slots. The issue persists across all these changes, suggesting it is NOT caused by: pressure explosion from lattice packing, threshold sensitivity, lattice regularity, or unfilled grid slots at high-Z. Needs proper visual debugging (e.g. rendering normals/depth/density separately) to isolate which rendering component creates the artifact.

## Scaling notes

- Container must be large enough for the particle block: `blockWidth = (ceil(cbrt(N)) - 1) * 0.05`
- At 70k: block is 2.05m in 4m container (51%). Larger counts need bigger containers or the physics becomes unstable (pressure waves with nowhere to dissipate)
- Raymarching iterations must cover the container diagonal: `iterations * stepSize > sqrt(3) * containerSize`
- Stiffness of 50 (original) causes supersonic compression at high particle counts — increased to 150 for stability
- Raymarching is the main GPU bottleneck on laptops — fullscreen shader runs 400 steps × multiple texture samples per hit pixel. Reducing pixel ratio or step count are the main perf levers.
