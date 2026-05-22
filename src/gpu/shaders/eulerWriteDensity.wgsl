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
@group(0) @binding(1) var<storage, read> densityMarker: array<f32>;
@group(0) @binding(2) var<storage, read_write> densityField: array<atomic<u32>>;

const FIXED_POINT_SCALE: f32 = 10000.0;
const DENSITY_SCALE: f32 = 5.0;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let res = params.fieldResolution;
  let total = res * res * res;
  if (gid.x >= total) { return; }

  let d = densityMarker[gid.x];
  let outIdx = gid.x * 2u;

  let scaledDensity = u32(clamp(d, 0.0, 1.0) * DENSITY_SCALE * FIXED_POINT_SCALE);
  atomicStore(&densityField[outIdx], scaledDensity);
  atomicStore(&densityField[outIdx + 1u], 0u);
}
