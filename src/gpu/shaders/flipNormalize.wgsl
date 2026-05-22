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
@group(0) @binding(5) var<storage, read_write> gridVel: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> gridOldVel: array<vec4<f32>>;

const INV_SCALE: f32 = 1.0 / 10000.0;
const MIN_WEIGHT: f32 = 0.01;

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
  let idx = gid.x;
  let res = params.fieldResolution;
  let resI = i32(res);
  if (idx >= res * res * res) { return; }

  let ix = i32(idx % res);
  let iy = i32((idx / res) % res);
  let iz = i32(idx / (res * res));

  let w = f32(atomicLoad(&accumWeight[idx])) * INV_SCALE;

  if (w > MIN_WEIGHT) {
    let invW = 1.0 / w;
    var vel = vec3f(
      f32(atomicLoad(&accumVelX[idx])) * INV_SCALE * invW,
      f32(atomicLoad(&accumVelY[idx])) * INV_SCALE * invW,
      f32(atomicLoad(&accumVelZ[idx])) * INV_SCALE * invW,
    );

    vel.y += params.gravity * params.dt;

    vel = enforceWallBC(ix, iy, iz, resI, vel);

    gridVel[idx] = vec4f(vel, w);
    gridOldVel[idx] = vec4f(vel, w);
  } else {
    gridVel[idx] = vec4f(0.0, 0.0, 0.0, 0.0);
    gridOldVel[idx] = vec4f(0.0, 0.0, 0.0, 0.0);
  }
}
