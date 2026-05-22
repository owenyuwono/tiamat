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
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> accumVelX: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> accumVelY: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> accumVelZ: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> accumWeight: array<atomic<u32>>;

const SCALE: f32 = 10000.0;

fn cellIdx(ix: i32, iy: i32, iz: i32, res: i32) -> u32 {
  return u32(iz * res * res + iy * res + ix);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.particleCount) { return; }

  let pos = positions[i].xyz;
  let vel = velocities[i].xyz;
  let res = i32(params.fieldResolution);
  let invDx = params.fieldInvCellSize;

  let gx = (pos.x - params.fieldDomainMinX) * invDx - 0.5;
  let gy = (pos.y - params.fieldDomainMinY) * invDx - 0.5;
  let gz = (pos.z - params.fieldDomainMinZ) * invDx - 0.5;

  let i0 = i32(floor(gx));
  let j0 = i32(floor(gy));
  let k0 = i32(floor(gz));

  let fx = gx - f32(i0);
  let fy = gy - f32(j0);
  let fz = gz - f32(k0);

  for (var dz = 0; dz <= 1; dz++) {
    let wz = select(1.0 - fz, fz, dz == 1);
    let kk = k0 + dz;
    if (kk < 0 || kk >= res) { continue; }
    for (var dy = 0; dy <= 1; dy++) {
      let wy = select(1.0 - fy, fy, dy == 1);
      let jj = j0 + dy;
      if (jj < 0 || jj >= res) { continue; }
      for (var dx = 0; dx <= 1; dx++) {
        let wx = select(1.0 - fx, fx, dx == 1);
        let ii = i0 + dx;
        if (ii < 0 || ii >= res) { continue; }

        let w = wx * wy * wz;
        let idx = cellIdx(ii, jj, kk, res);

        atomicAdd(&accumVelX[idx], i32(vel.x * w * SCALE));
        atomicAdd(&accumVelY[idx], i32(vel.y * w * SCALE));
        atomicAdd(&accumVelZ[idx], i32(vel.z * w * SCALE));
        atomicAdd(&accumWeight[idx], u32(w * SCALE));
      }
    }
  }
}
