struct Params {
  particleCount: u32,
  tableSize: u32,
  maxPerCell: u32,
  fieldResolution: u32,
  H: f32,
  H2: f32,
  mass: f32,
  restDensity: f32,
  stiffness: f32,
  viscosity: f32,
  gravity: f32,
  boundaryDamping: f32,
  maxVelocity: f32,
  collisionRadius: f32,
  collisionStiffness: f32,
  xsphEpsilon: f32,
  surfaceTension: f32,
  dt: f32,
  poly6Coeff: f32,
  spikyCoeff: f32,
  viscLapCoeff: f32,
  invCellSize: f32,
  halfContainerX: f32,
  halfContainerZ: f32,
  containerMaxY: f32,
  fieldDomainMinX: f32,
  fieldDomainMinY: f32,
  fieldDomainMinZ: f32,
  fieldCellSize: f32,
  fieldInvCellSize: f32,
  splatRadius2: f32,
  splatRadiusCells: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> uVel: array<f32>;
@group(0) @binding(2) var<storage, read_write> vVel: array<f32>;
@group(0) @binding(3) var<storage, read_write> wVel: array<f32>;
@group(0) @binding(4) var<storage, read_write> uOldVel: array<f32>;
@group(0) @binding(5) var<storage, read_write> vOldVel: array<f32>;
@group(0) @binding(6) var<storage, read_write> wOldVel: array<f32>;
@group(0) @binding(7) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(8) var<storage, read_write> divergence: array<f32>;
@group(0) @binding(9) var<storage, read_write> accumU: array<atomic<i32>>;
@group(0) @binding(10) var<storage, read_write> accumV: array<atomic<i32>>;
@group(0) @binding(11) var<storage, read_write> accumW: array<atomic<i32>>;
@group(0) @binding(12) var<storage, read_write> weightU: array<atomic<u32>>;
@group(0) @binding(13) var<storage, read_write> weightV: array<atomic<u32>>;
@group(0) @binding(14) var<storage, read_write> weightW: array<atomic<u32>>;
@group(0) @binding(15) var<storage, read_write> gridVel: array<vec4<f32>>;

// Shared staggered MAC indexing helpers
fn uFaceIndex(i: u32, j: u32, k: u32, R: u32) -> u32 {
  return i + (R + 1u) * (j + R * k);
}
fn vFaceIndex(i: u32, j: u32, k: u32, R: u32) -> u32 {
  return j + (R + 1u) * (i + R * k);
}
fn wFaceIndex(i: u32, j: u32, k: u32, R: u32) -> u32 {
  return k + (R + 1u) * (j + R * i);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let R = params.fieldResolution;

  let uCount = (R + 1u) * R * R;
  let vCount = R * (R + 1u) * R;
  let wCount = R * R * (R + 1u);
  let cellCount = R * R * R;

  if (idx < uCount) {
    uVel[idx] = 0.0;
    uOldVel[idx] = 0.0;
  }
  if (idx < vCount) {
    vVel[idx] = 0.0;
    vOldVel[idx] = 0.0;
  }
  if (idx < wCount) {
    wVel[idx] = 0.0;
    wOldVel[idx] = 0.0;
  }
  if (idx < cellCount) {
    pressure[idx] = 0.0;
    divergence[idx] = 0.0;
    gridVel[idx] = vec4f(0.0);
  }

  // Clear P2G accumulators (same size as face buffers)
  if (idx < uCount) {
    atomicStore(&accumU[idx], 0);
    atomicStore(&weightU[idx], 0u);
  }
  if (idx < vCount) {
    atomicStore(&accumV[idx], 0);
    atomicStore(&weightV[idx], 0u);
  }
  if (idx < wCount) {
    atomicStore(&accumW[idx], 0);
    atomicStore(&weightW[idx], 0u);
  }
}
