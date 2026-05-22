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
@group(0) @binding(4) var<storage, read> densityIn: array<f32>;
@group(0) @binding(5) var<storage, read_write> densityOut: array<f32>;

const DENSITY_SUBSTEPS: f32 = 4.0;

fn idxU(ix: i32, iy: i32, iz: i32, res: i32) -> u32 {
  return u32(((iz * res + iy) * (res + 1)) + ix);
}
fn idxV(ix: i32, iy: i32, iz: i32, res: i32) -> u32 {
  return u32(((iz * (res + 1) + iy) * res) + ix);
}
fn idxW(ix: i32, iy: i32, iz: i32, res: i32) -> u32 {
  return u32(((iz * res + iy) * res) + ix);
}
fn cellIdx(ix: i32, iy: i32, iz: i32, res: i32) -> u32 {
  return u32(iz * res * res + iy * res + ix);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let R = i32(params.fieldResolution);
  let total = u32(R * R * R);
  if (gid.x >= total) { return; }

  let iz = i32(gid.x) / (R * R);
  let iy = (i32(gid.x) / R) % R;
  let ix = i32(gid.x) % R;

  let subDt = params.dt / DENSITY_SUBSTEPS;
  let invH = params.fieldInvCellSize;

  let d = densityIn[gid.x];

  let uL = uVel[idxU(ix, iy, iz, R)];
  let uR = uVel[idxU(ix + 1, iy, iz, R)];
  let vB = vVel[idxV(ix, iy, iz, R)];
  let vT = vVel[idxV(ix, iy + 1, iz, R)];
  let wBk = wVel[idxW(ix, iy, iz, R)];
  let wFr = wVel[idxW(ix, iy, iz + 1, R)];

  let dXm = select(d, densityIn[cellIdx(ix - 1, iy, iz, R)], ix > 0);
  let dXp = select(d, densityIn[cellIdx(ix + 1, iy, iz, R)], ix < R - 1);
  let dYm = select(d, densityIn[cellIdx(ix, iy - 1, iz, R)], iy > 0);
  let dYp = select(d, densityIn[cellIdx(ix, iy + 1, iz, R)], iy < R - 1);
  let dZm = select(d, densityIn[cellIdx(ix, iy, iz - 1, R)], iz > 0);
  let dZp = select(d, densityIn[cellIdx(ix, iy, iz + 1, R)], iz < R - 1);

  let fxR = max(uR, 0.0) * d + min(uR, 0.0) * dXp;
  let fxL = max(uL, 0.0) * dXm + min(uL, 0.0) * d;
  let fyT = max(vT, 0.0) * d + min(vT, 0.0) * dYp;
  let fyB = max(vB, 0.0) * dYm + min(vB, 0.0) * d;
  let fzF = max(wFr, 0.0) * d + min(wFr, 0.0) * dZp;
  let fzBk = max(wBk, 0.0) * dZm + min(wBk, 0.0) * d;

  let netFlux = (fxR - fxL) + (fyT - fyB) + (fzF - fzBk);
  densityOut[gid.x] = clamp(d - subDt * invH * netFlux, 0.0, 1.0);
}
