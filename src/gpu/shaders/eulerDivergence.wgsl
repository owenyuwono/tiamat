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
@group(0) @binding(1) var<storage, read> uVel: array<f32>;
@group(0) @binding(2) var<storage, read> vVel: array<f32>;
@group(0) @binding(3) var<storage, read> wVel: array<f32>;
@group(0) @binding(4) var<storage, read_write> divergence: array<f32>;

fn idxU(ix: i32, iy: i32, iz: i32, res: i32) -> u32 {
  return u32(((iz * res + iy) * (res + 1)) + ix);
}
fn idxV(ix: i32, iy: i32, iz: i32, res: i32) -> u32 {
  return u32(((iz * (res + 1) + iy) * res) + ix);
}
fn idxW(ix: i32, iy: i32, iz: i32, res: i32) -> u32 {
  return u32(((iz * res + iy) * res) + ix);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let res = i32(params.fieldResolution);
  let total = u32(res * res * res);
  if (gid.x >= total) { return; }

  let ires = i32(gid.x);
  let iz = ires / (res * res);
  let iy = (ires / res) % res;
  let ix = ires % res;

  let h = params.fieldCellSize;

  let du = uVel[idxU(ix + 1, iy, iz, res)] - uVel[idxU(ix, iy, iz, res)];
  let dv = vVel[idxV(ix, iy + 1, iz, res)] - vVel[idxV(ix, iy, iz, res)];
  let dw = wVel[idxW(ix, iy, iz + 1, res)] - wVel[idxW(ix, iy, iz, res)];

  divergence[gid.x] = (du + dv + dw) / h;
}
