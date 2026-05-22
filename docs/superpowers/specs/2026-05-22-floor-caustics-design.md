# Floor Caustics

## Problem
The floor has flat Lambertian lighting. Real water projects dancing light patterns (caustics) onto surfaces beneath it.

## Approach
Physics-driven caustics from the density field Laplacian, plus animated Worley noise shimmer.

### Compute pass: `caustics.wgsl` (256x256, `r32float`)
For each texel:
1. Map to world XZ on the floor plane
2. March Y top→down through the 3D density texture — find surface height (first voxel where density > threshold)
3. Sample 4 XZ neighbors to get their surface heights
4. Laplacian = `h_left + h_right + h_front + h_back - 4*h_center`
5. Caustic intensity = `clamp(1.0 + strength * laplacian, 0.0, 3.0)`
6. Write to 2D caustics texture

### Floor shader changes
- Add caustics texture binding + time uniform
- Sample caustics texture at `worldPos.xz` mapped to floor UV
- Sample existing Worley noise texture with time-scrolling UV for shimmer
- Blend: `causticFinal = physicsIntensity * (0.7 + 0.3 * shimmer)`
- Apply: `color *= causticFinal` (only on top face, not sides)

### Pipeline integration
- Caustics compute runs after `bufferToTexture`, before render pass (same encoder)
- Floor bind group gains 2 new bindings: caustics texture + foam noise texture (already exists)
- Floor uniform gains `time: f32`

### Resources
- 256x256 `r32float` texture (~256KB)
- Reuse existing `foamNoiseTexture` (Worley, 256x256) for shimmer
- Reuse existing `linearSampler`

## Files
- NEW: `src/gpu/shaders/caustics.wgsl`
- MODIFY: `src/gpu/shaders/floor.wgsl` — add caustics + noise sampling
- MODIFY: `src/gpu/WebGPURenderer.ts` — caustics texture, compute pipeline, floor bind group expansion, time uniform
