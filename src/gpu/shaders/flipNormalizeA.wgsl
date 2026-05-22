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
@group(0) @binding(1) var<storage, read_write> accumU: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> weightU: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> accumV: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> weightV: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> uVel: array<f32>;
@group(0) @binding(6) var<storage, read_write> vVel: array<f32>;
@group(0) @binding(7) var<storage, read_write> uOldVel: array<f32>;
@group(0) @binding(8) var<storage, read_write> vOldVel: array<f32>;

const INV_SCALE: f32 = 1.0 / 10000.0;
const MIN_WEIGHT: f32 = 0.01;

fn enforceUWallBC(i: i32, res: i32, vel: f32) -> f32 {
  if (i == 0) { return max(vel, 0.0); }
  if (i == res) { return min(vel, 0.0); }
  return vel;
}
fn enforceVWallBC(j: i32, res: i32, vel: f32) -> f32 {
  if (j == 0) { return max(vel, 0.0); }
  if (j == res) { return min(vel, 0.0); }
  return vel;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let R = i32(params.fieldResolution);
  let uCount = (R + 1) * R * R;
  let vCount = R * (R + 1) * R;

  if (idx < u32(uCount)) {
    let w = f32(atomicLoad(&weightU[idx])) * INV_SCALE;
    var vel = 0.0;
    if (w > MIN_WEIGHT) {
      vel = f32(atomicLoad(&accumU[idx])) * INV_SCALE / w;
    }
    vel = enforceUWallBC(i32(idx % u32(R + 1)), R, vel);
    uVel[idx] = vel;
    uOldVel[idx] = vel;
  }

  if (idx < u32(vCount)) {
    let w = f32(atomicLoad(&weightV[idx])) * INV_SCALE;
    var vel = 0.0;
    if (w > MIN_WEIGHT) {
      vel = f32(atomicLoad(&accumV[idx])) * INV_SCALE / w;
    }
    let iy = i32((idx / u32(R)) % u32(R + 1));
    vel = enforceVWallBC(iy, R, vel);
    vOldVel[idx] = vel;
    if (w > MIN_WEIGHT) {
      vel += params.gravity * params.dt;
      vel = enforceVWallBC(iy, R, vel);
    }
    vVel[idx] = vel;
  }
}
