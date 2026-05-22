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
@group(0) @binding(4) var<storage, read_write> uVelOut: array<f32>;
@group(0) @binding(5) var<storage, read_write> vVelOut: array<f32>;
@group(0) @binding(6) var<storage, read_write> wVelOut: array<f32>;

fn idxU(i: i32, j: i32, k: i32, res: i32) -> u32 {
  return u32(i) + u32(res + 1) * (u32(j) + u32(res) * u32(k));
}
fn idxV(i: i32, j: i32, k: i32, res: i32) -> u32 {
  return u32(i) + u32(res) * (u32(j) + u32(res + 1) * u32(k));
}
fn idxW(i: i32, j: i32, k: i32, res: i32) -> u32 {
  return u32(i) + u32(res) * (u32(j) + u32(res) * u32(k));
}

fn sampleU(gx: f32, gy: f32, gz: f32, res: i32) -> f32 {
  let i0 = i32(floor(gx));
  let j0 = i32(floor(gy - 0.5));
  let k0 = i32(floor(gz - 0.5));
  let fx = gx - f32(i0);
  let fy = gy - 0.5 - f32(j0);
  let fz = gz - 0.5 - f32(k0);

  var sum = 0.0;
  var wsum = 0.0;
  for (var di = 0; di < 2; di++) {
    let ii = clamp(i0 + di, 0, res);
    let wx = select(1.0 - fx, fx, di == 1);
    for (var dj = 0; dj < 2; dj++) {
      let jj = clamp(j0 + dj, 0, res - 1);
      let wy = select(1.0 - fy, fy, dj == 1);
      for (var dk = 0; dk < 2; dk++) {
        let kk = clamp(k0 + dk, 0, res - 1);
        let wz = select(1.0 - fz, fz, dk == 1);
        let w = wx * wy * wz;
        sum += w * uVel[idxU(ii, jj, kk, res)];
        wsum += w;
      }
    }
  }
  return select(0.0, sum / wsum, wsum > 0.0);
}

fn sampleV(gx: f32, gy: f32, gz: f32, res: i32) -> f32 {
  let i0 = i32(floor(gx - 0.5));
  let j0 = i32(floor(gy));
  let k0 = i32(floor(gz - 0.5));
  let fx = gx - 0.5 - f32(i0);
  let fy = gy - f32(j0);
  let fz = gz - 0.5 - f32(k0);

  var sum = 0.0;
  var wsum = 0.0;
  for (var di = 0; di < 2; di++) {
    let ii = clamp(i0 + di, 0, res - 1);
    let wx = select(1.0 - fx, fx, di == 1);
    for (var dj = 0; dj < 2; dj++) {
      let jj = clamp(j0 + dj, 0, res);
      let wy = select(1.0 - fy, fy, dj == 1);
      for (var dk = 0; dk < 2; dk++) {
        let kk = clamp(k0 + dk, 0, res - 1);
        let wz = select(1.0 - fz, fz, dk == 1);
        let w = wx * wy * wz;
        sum += w * vVel[idxV(ii, jj, kk, res)];
        wsum += w;
      }
    }
  }
  return select(0.0, sum / wsum, wsum > 0.0);
}

fn sampleW(gx: f32, gy: f32, gz: f32, res: i32) -> f32 {
  let i0 = i32(floor(gx - 0.5));
  let j0 = i32(floor(gy - 0.5));
  let k0 = i32(floor(gz));
  let fx = gx - 0.5 - f32(i0);
  let fy = gy - 0.5 - f32(j0);
  let fz = gz - f32(k0);

  var sum = 0.0;
  var wsum = 0.0;
  for (var di = 0; di < 2; di++) {
    let ii = clamp(i0 + di, 0, res - 1);
    let wx = select(1.0 - fx, fx, di == 1);
    for (var dj = 0; dj < 2; dj++) {
      let jj = clamp(j0 + dj, 0, res - 1);
      let wy = select(1.0 - fy, fy, dj == 1);
      for (var dk = 0; dk < 2; dk++) {
        let kk = clamp(k0 + dk, 0, res);
        let wz = select(1.0 - fz, fz, dk == 1);
        let w = wx * wy * wz;
        sum += w * wVel[idxW(ii, jj, kk, res)];
        wsum += w;
      }
    }
  }
  return select(0.0, sum / wsum, wsum > 0.0);
}

fn sampleVelocity(gx: f32, gy: f32, gz: f32, res: i32) -> vec3f {
  return vec3f(sampleU(gx, gy, gz, res), sampleV(gx, gy, gz, res), sampleW(gx, gy, gz, res));
}

fn clampGrid(g: vec3f, res: i32) -> vec3f {
  let r = f32(res);
  return clamp(g, vec3f(0.0), vec3f(r));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let R = i32(params.fieldResolution);
  let Rf = f32(R);
  let dt = params.dt;
  let invDx = params.fieldInvCellSize;
  let h = params.fieldCellSize;

  let uCount = (R + 1) * R * R;
  let vCount = R * (R + 1) * R;
  let wCount = R * R * (R + 1);
  let maxCount = max(uCount, max(vCount, wCount));
  if (idx >= u32(maxCount)) { return; }

  // Advect U faces: face center at grid coords (i, j+0.5, k+0.5)
  if (idx < u32(uCount)) {
    let iz = i32(idx) / ((R + 1) * R);
    let iy = (i32(idx) / (R + 1)) % R;
    let ix = i32(idx) % (R + 1);

    let g = vec3f(f32(ix), f32(iy) + 0.5, f32(iz) + 0.5);

    // RK2 backtrace
    let vel0 = sampleVelocity(g.x, g.y, g.z, R);
    let gMid = clampGrid(g - 0.5 * dt * invDx * vel0, R);
    let velMid = sampleVelocity(gMid.x, gMid.y, gMid.z, R);
    let gDep = clampGrid(g - dt * invDx * velMid, R);

    uVelOut[idx] = sampleU(gDep.x, gDep.y, gDep.z, R);
  }

  // Advect V faces: face center at grid coords (i+0.5, j, k+0.5)
  if (idx < u32(vCount)) {
    let iz = i32(idx) / ((R + 1) * R);
    let iy = (i32(idx) / R) % (R + 1);
    let ix = i32(idx) % R;

    let g = vec3f(f32(ix) + 0.5, f32(iy), f32(iz) + 0.5);

    let vel0 = sampleVelocity(g.x, g.y, g.z, R);
    let gMid = clampGrid(g - 0.5 * dt * invDx * vel0, R);
    let velMid = sampleVelocity(gMid.x, gMid.y, gMid.z, R);
    let gDep = clampGrid(g - dt * invDx * velMid, R);

    vVelOut[idx] = sampleV(gDep.x, gDep.y, gDep.z, R);
  }

  // Advect W faces: face center at grid coords (i+0.5, j+0.5, k)
  if (idx < u32(wCount)) {
    let iz = i32(idx) / (R * R);
    let iy = (i32(idx) / R) % R;
    let ix = i32(idx) % R;

    let g = vec3f(f32(ix) + 0.5, f32(iy) + 0.5, f32(iz));

    let vel0 = sampleVelocity(g.x, g.y, g.z, R);
    let gMid = clampGrid(g - 0.5 * dt * invDx * vel0, R);
    let velMid = sampleVelocity(gMid.x, gMid.y, gMid.z, R);
    let gDep = clampGrid(g - dt * invDx * velMid, R);

    wVelOut[idx] = sampleW(gDep.x, gDep.y, gDep.z, R);
  }
}
