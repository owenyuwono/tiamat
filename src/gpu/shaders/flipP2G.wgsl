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
@group(0) @binding(3) var<storage, read_write> accumU: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> accumV: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> accumW: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> weightU: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> weightV: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> weightW: array<atomic<u32>>;

const SCALE: f32 = 10000.0;

fn idxU(i: i32, j: i32, k: i32, res: i32) -> u32 {
  let sx = res + 1;
  let sy = res;
  return u32(i) + u32(sx) * (u32(j) + u32(sy) * u32(k));
}

fn idxV(i: i32, j: i32, k: i32, res: i32) -> u32 {
  let sx = res;
  let sy = res + 1;
  return u32(i) + u32(sx) * (u32(j) + u32(sy) * u32(k));
}

fn idxW(i: i32, j: i32, k: i32, res: i32) -> u32 {
  let sx = res;
  let sy = res;
  return u32(i) + u32(sx) * (u32(j) + u32(sy) * u32(k));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pid = gid.x;
  if (pid >= params.particleCount) { return; }

  let pos = positions[pid].xyz;
  let vel = velocities[pid].xyz;
  let res = i32(params.fieldResolution);
  let invDx = params.fieldInvCellSize;
  let minX = params.fieldDomainMinX;
  let minY = params.fieldDomainMinY;
  let minZ = params.fieldDomainMinZ;

  // U (x-faces)
  {
    let gx = (pos.x - minX) * invDx;
    let gy = (pos.y - minY) * invDx - 0.5;
    let gz = (pos.z - minZ) * invDx - 0.5;

    let i0 = i32(floor(gx));
    let j0 = i32(floor(gy));
    let k0 = i32(floor(gz));
    let fx = gx - f32(i0);
    let fy = gy - f32(j0);
    let fz = gz - f32(k0);

    for (var di = 0; di < 2; di++) {
      let ii = i0 + di;
      if (ii < 0 || ii > res) { continue; }
      let wx = select(1.0 - fx, fx, di == 1);
      for (var dj = 0; dj < 2; dj++) {
        let jj = j0 + dj;
        if (jj < 0 || jj >= res) { continue; }
        let wy = select(1.0 - fy, fy, dj == 1);
        for (var dk = 0; dk < 2; dk++) {
          let kk = k0 + dk;
          if (kk < 0 || kk >= res) { continue; }
          let wz = select(1.0 - fz, fz, dk == 1);
          let w = wx * wy * wz;
          let idx = idxU(ii, jj, kk, res);
          atomicAdd(&accumU[idx], i32(vel.x * w * SCALE));
          atomicAdd(&weightU[idx], u32(w * SCALE));
        }
      }
    }
  }

  // V (y-faces)
  {
    let gx = (pos.x - minX) * invDx - 0.5;
    let gy = (pos.y - minY) * invDx;
    let gz = (pos.z - minZ) * invDx - 0.5;

    let i0 = i32(floor(gx));
    let j0 = i32(floor(gy));
    let k0 = i32(floor(gz));
    let fx = gx - f32(i0);
    let fy = gy - f32(j0);
    let fz = gz - f32(k0);

    for (var di = 0; di < 2; di++) {
      let ii = i0 + di;
      if (ii < 0 || ii >= res) { continue; }
      let wx = select(1.0 - fx, fx, di == 1);
      for (var dj = 0; dj < 2; dj++) {
        let jj = j0 + dj;
        if (jj < 0 || jj > res) { continue; }
        let wy = select(1.0 - fy, fy, dj == 1);
        for (var dk = 0; dk < 2; dk++) {
          let kk = k0 + dk;
          if (kk < 0 || kk >= res) { continue; }
          let wz = select(1.0 - fz, fz, dk == 1);
          let w = wx * wy * wz;
          let idx = idxV(ii, jj, kk, res);
          atomicAdd(&accumV[idx], i32(vel.y * w * SCALE));
          atomicAdd(&weightV[idx], u32(w * SCALE));
        }
      }
    }
  }

  // W (z-faces)
  {
    let gx = (pos.x - minX) * invDx - 0.5;
    let gy = (pos.y - minY) * invDx - 0.5;
    let gz = (pos.z - minZ) * invDx;

    let i0 = i32(floor(gx));
    let j0 = i32(floor(gy));
    let k0 = i32(floor(gz));
    let fx = gx - f32(i0);
    let fy = gy - f32(j0);
    let fz = gz - f32(k0);

    for (var di = 0; di < 2; di++) {
      let ii = i0 + di;
      if (ii < 0 || ii >= res) { continue; }
      let wx = select(1.0 - fx, fx, di == 1);
      for (var dj = 0; dj < 2; dj++) {
        let jj = j0 + dj;
        if (jj < 0 || jj >= res) { continue; }
        let wy = select(1.0 - fy, fy, dj == 1);
        for (var dk = 0; dk < 2; dk++) {
          let kk = k0 + dk;
          if (kk < 0 || kk > res) { continue; }
          let wz = select(1.0 - fz, fz, dk == 1);
          let w = wx * wy * wz;
          let idx = idxW(ii, jj, kk, res);
          atomicAdd(&accumW[idx], i32(vel.z * w * SCALE));
          atomicAdd(&weightW[idx], u32(w * SCALE));
        }
      }
    }
  }
}
