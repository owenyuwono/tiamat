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
@group(0) @binding(1) var<storage, read_write> gridVel: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> pressure: array<f32>;

fn idx3(ix: i32, iy: i32, iz: i32, res: i32) -> u32 {
  return u32(iz * res * res + iy * res + ix);
}

fn isFluid(ix: i32, iy: i32, iz: i32, res: i32) -> bool {
  if (ix < 0 || ix >= res || iy < 0 || iy >= res || iz < 0 || iz >= res) {
    return false;
  }
  return gridVel[idx3(ix, iy, iz, res)].w > 0.01;
}

fn pressureAt(ix: i32, iy: i32, iz: i32, res: i32) -> f32 {
  let cx = clamp(ix, 0, res - 1);
  let cy = clamp(iy, 0, res - 1);
  let cz = clamp(iz, 0, res - 1);

  if (!isFluid(cx, cy, cz, res)) {
    return 0.0;
  }
  return pressure[idx3(cx, cy, cz, res)];
}

fn enforceWallBC(ix: i32, iy: i32, iz: i32, res: i32, vel: vec3f) -> vec3f {
  var v = vel;
  if (ix == 0) { v.x = max(v.x, 0.0); }
  if (ix == res - 1) { v.x = min(v.x, 0.0); }
  if (iy == 0) { v.y = max(v.y, 0.0); }
  if (iy == res - 1) { v.y = min(v.y, 0.0); }
  if (iz == 0) { v.z = max(v.z, 0.0); }
  if (iz == res - 1) { v.z = min(v.z, 0.0); }
  return v;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let flatIdx = gid.x;
  let res = i32(params.fieldResolution);
  let total = u32(res * res * res);
  if (flatIdx >= total) { return; }

  let cell = gridVel[flatIdx];
  if (cell.w < 0.01) { return; }

  let ires = i32(flatIdx);
  let iz = ires / (res * res);
  let iy = (ires / res) % res;
  let ix = ires % res;

  let invDx2 = 1.0 / (2.0 * params.fieldCellSize);
  let scale = params.dt / params.restDensity;

  let gradP = vec3f(
    (pressureAt(ix + 1, iy, iz, res) - pressureAt(ix - 1, iy, iz, res)) * invDx2,
    (pressureAt(ix, iy + 1, iz, res) - pressureAt(ix, iy - 1, iz, res)) * invDx2,
    (pressureAt(ix, iy, iz + 1, res) - pressureAt(ix, iy, iz - 1, res)) * invDx2,
  );

  var newVel = cell.xyz - scale * gradP;
  newVel = enforceWallBC(ix, iy, iz, res, newVel);
  gridVel[flatIdx] = vec4f(newVel, cell.w);
}
