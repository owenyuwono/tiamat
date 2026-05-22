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
@group(0) @binding(1) var<storage, read_write> accumVelX: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> accumVelY: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> accumVelZ: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> accumWeight: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> pressure: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let res = params.fieldResolution;
  let total = res * res * res;
  if (idx >= total) { return; }

  atomicStore(&accumVelX[idx], 0);
  atomicStore(&accumVelY[idx], 0);
  atomicStore(&accumVelZ[idx], 0);
  atomicStore(&accumWeight[idx], 0u);
  pressure[idx] = 0.0;
}
