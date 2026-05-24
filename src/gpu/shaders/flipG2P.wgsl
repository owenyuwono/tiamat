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
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> uVel: array<f32>;
@group(0) @binding(4) var<storage, read> vVel: array<f32>;
@group(0) @binding(5) var<storage, read> wVel: array<f32>;
@group(0) @binding(6) var<storage, read> uOldVel: array<f32>;
@group(0) @binding(7) var<storage, read> vOldVel: array<f32>;
@group(0) @binding(8) var<storage, read> wOldVel: array<f32>;

fn idxU(i: i32, j: i32, k: i32, res: i32) -> u32 {
  return u32(i) + u32(res + 1) * (u32(j) + u32(res) * u32(k));
}
fn idxV(i: i32, j: i32, k: i32, res: i32) -> u32 {
  return u32(i) + u32(res) * (u32(j) + u32(res + 1) * u32(k));
}
fn idxW(i: i32, j: i32, k: i32, res: i32) -> u32 {
  return u32(i) + u32(res) * (u32(j) + u32(res) * u32(k));
}

// Proper trilinear sampling for staggered MAC faces.
// u faces are located at x = i*h, y = (j+0.5)*h, z = (k+0.5)*h
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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.particleCount) { return; }

  var pos = positions[i].xyz;
  let res = i32(params.fieldResolution);
  let invDx = params.fieldInvCellSize;

  let gx = (pos.x - params.fieldDomainMinX) * invDx;
  let gy = (pos.y - params.fieldDomainMinY) * invDx;
  let gz = (pos.z - params.fieldDomainMinZ) * invDx;

  let uNew = sampleU(gx, gy, gz, res);
  let vNew = sampleV(gx, gy, gz, res);
  let wNew = sampleW(gx, gy, gz, res);

  // Sample old face velocities for FLIP delta
  // (we reuse the same sampling functions on the Old buffers)
  let uOld = uOldVel[idxU(clamp(i32(floor(gx)), 0, res), clamp(i32(floor(gy-0.5)), 0, res-1), clamp(i32(floor(gz-0.5)), 0, res-1), res)];
  let vOld = vOldVel[idxV(clamp(i32(floor(gx-0.5)), 0, res-1), clamp(i32(floor(gy)), 0, res), clamp(i32(floor(gz-0.5)), 0, res-1), res)];
  let wOld = wOldVel[idxW(clamp(i32(floor(gx-0.5)), 0, res-1), clamp(i32(floor(gy-0.5)), 0, res-1), clamp(i32(floor(gz)), 0, res), res)];

  let alpha: f32 = 0.85;
  let du = uNew - uOld;
  let dv = vNew - vOld;
  let dw = wNew - wOld;
  var vel = velocities[i].xyz;
  vel = (1.0 - alpha) * vec3f(uNew, vNew, wNew) + alpha * (vel + vec3f(du, dv, dw));

  let speed2 = dot(vel, vel);
  if (speed2 > 10000.0) {
    vel *= 100.0 / sqrt(speed2);
  }

  pos += vel * params.dt;

  // Particle wall collisions
  if (pos.x < -params.halfContainerX) { pos.x = -params.halfContainerX; vel.x *= -params.boundaryDamping; }
  if (pos.x > params.halfContainerX)  { pos.x = params.halfContainerX;  vel.x *= -params.boundaryDamping; }
  if (pos.y < 0.0) { pos.y = 0.0; vel.y *= -params.boundaryDamping; }
  if (pos.y > params.containerMaxY) { pos.y = params.containerMaxY; vel.y *= -params.boundaryDamping; }
  if (pos.z < -params.halfContainerZ) { pos.z = -params.halfContainerZ; vel.z *= -params.boundaryDamping; }
  if (pos.z > params.halfContainerZ)  { pos.z = params.halfContainerZ;  vel.z *= -params.boundaryDamping; }

  positions[i] = vec4f(pos, positions[i].w);
  velocities[i] = vec4f(vel, velocities[i].w);
}
