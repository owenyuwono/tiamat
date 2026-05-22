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
@group(0) @binding(2) var<storage, read_write> accumV: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> accumW: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> weightU: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> weightV: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> weightW: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> uVel: array<f32>;
@group(0) @binding(8) var<storage, read_write> vVel: array<f32>;
@group(0) @binding(9) var<storage, read_write> wVel: array<f32>;
@group(0) @binding(10) var<storage, read_write> uOldVel: array<f32>;
@group(0) @binding(11) var<storage, read_write> vOldVel: array<f32>;
@group(0) @binding(12) var<storage, read_write> wOldVel: array<f32>;
@group(0) @binding(13) var<storage, read_write> gridVel: array<vec4<f32>>;

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
fn enforceWWallBC(k: i32, res: i32, vel: f32) -> f32 {
  if (k == 0) { return max(vel, 0.0); }
  if (k == res) { return min(vel, 0.0); }
  return vel;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let R = i32(params.fieldResolution);

  let uCount = (R + 1) * R * R;
  let vCount = R * (R + 1) * R;
  let wCount = R * R * (R + 1);

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
    vel += params.gravity * params.dt;
    vel = enforceVWallBC(i32(idx % u32(R + 1)), R, vel);
    vVel[idx] = vel;
    vOldVel[idx] = vel;
  }

  if (idx < u32(wCount)) {
    let w = f32(atomicLoad(&weightW[idx])) * INV_SCALE;
    var vel = 0.0;
    if (w > MIN_WEIGHT) {
      vel = f32(atomicLoad(&accumW[idx])) * INV_SCALE / w;
    }
    vel = enforceWWallBC(i32(idx % u32(R + 1)), R, vel);
    wVel[idx] = vel;
    wOldVel[idx] = vel;
  }

  let cellCount = u32(R * R * R);
  if (idx < cellCount) {
    let iz = i32(idx / u32(R * R));
    let iy = i32((idx / u32(R)) % u32(R));
    let ix = i32(idx % u32(R));

    let wuL = f32(atomicLoad(&weightU[u32(ix) + u32(R + 1) * (u32(iy) + u32(R) * u32(iz))])) * INV_SCALE;
    let wuR = f32(atomicLoad(&weightU[u32(ix + 1) + u32(R + 1) * (u32(iy) + u32(R) * u32(iz))])) * INV_SCALE;
    let wvD = f32(atomicLoad(&weightV[u32(ix) + u32(R) * (u32(iy) + u32(R + 1) * u32(iz))])) * INV_SCALE;
    let wvU = f32(atomicLoad(&weightV[u32(ix) + u32(R) * (u32(iy + 1) + u32(R + 1) * u32(iz))])) * INV_SCALE;
    let wwB = f32(atomicLoad(&weightW[u32(ix) + u32(R) * (u32(iy) + u32(R) * u32(iz))])) * INV_SCALE;
    let wwF = f32(atomicLoad(&weightW[u32(ix) + u32(R) * (u32(iy) + u32(R) * u32(iz + 1))])) * INV_SCALE;

    let totalWeight = wuL + wuR + wvD + wvU + wwB + wwF;
    gridVel[idx] = vec4f(0.0, 0.0, 0.0, select(0.0, 1.0, totalWeight > MIN_WEIGHT));
  }
}
