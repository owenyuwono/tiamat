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
@group(0) @binding(1) var<storage, read> gridVel: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> divergence: array<f32>;

fn idx3(ix: i32, iy: i32, iz: i32, res: i32) -> u32 {
  return u32(iz * res * res + iy * res + ix);
}

// Solid walls: mirror ghost velocity. Air outside domain is not zero-velocity free surface.
fn velAt(ix: i32, iy: i32, iz: i32, res: i32) -> vec3f {
  var cx = ix;
  var cy = iy;
  var cz = iz;
  var mirrorX = false;
  var mirrorY = false;
  var mirrorZ = false;

  if (cx < 0) { cx = 0; mirrorX = true; }
  else if (cx >= res) { cx = res - 1; mirrorX = true; }
  if (cy < 0) { cy = 0; mirrorY = true; }
  else if (cy >= res) { cy = res - 1; mirrorY = true; }
  if (cz < 0) { cz = 0; mirrorZ = true; }
  else if (cz >= res) { cz = res - 1; mirrorZ = true; }

  var v = gridVel[idx3(cx, cy, cz, res)].xyz;
  if (mirrorX) { v.x = -v.x; }
  if (mirrorY) { v.y = -v.y; }
  if (mirrorZ) { v.z = -v.z; }
  return v;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let flatIdx = gid.x;
  let res = i32(params.fieldResolution);
  let total = u32(res * res * res);
  if (flatIdx >= total) { return; }

  let ires = i32(flatIdx);
  let iz = ires / (res * res);
  let iy = (ires / res) % res;
  let ix = ires % res;

  let w = gridVel[flatIdx].w;
  if (w < 0.01) {
    divergence[flatIdx] = 0.0;
    return;
  }

  let invDx2 = 1.0 / (2.0 * params.fieldCellSize);

  let dvx = velAt(ix + 1, iy, iz, res).x - velAt(ix - 1, iy, iz, res).x;
  let dvy = velAt(ix, iy + 1, iz, res).y - velAt(ix, iy - 1, iz, res).y;
  let dvz = velAt(ix, iy, iz + 1, res).z - velAt(ix, iy, iz - 1, res).z;

  let div = (dvx + dvy + dvz) * invDx2;
  divergence[flatIdx] = div * params.restDensity / params.dt;
}
