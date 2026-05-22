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
@group(0) @binding(3) var<storage, read_write> xsph: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> gridVel: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> gridOldVel: array<vec4<f32>>;

fn cellIdx(ix: i32, iy: i32, iz: i32, res: i32) -> u32 {
  return u32(iz * res * res + iy * res + ix);
}

fn sampleGrid(grid: ptr<storage, array<vec4<f32>>, read>, gx: f32, gy: f32, gz: f32, res: i32) -> vec3f {
  let i0 = i32(floor(gx));
  let j0 = i32(floor(gy));
  let k0 = i32(floor(gz));
  let fx = gx - f32(i0);
  let fy = gy - f32(j0);
  let fz = gz - f32(k0);

  var result = vec3f(0.0);
  var weightSum = 0.0;
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
        let cell = (*grid)[cellIdx(ii, jj, kk, res)];
        if (cell.w > 0.01) {
          result += w * cell.xyz;
          weightSum += w;
        }
      }
    }
  }
  if (weightSum > 0.0) {
    return result / weightSum;
  }
  return vec3f(0.0);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.particleCount) { return; }

  var pos = positions[i].xyz;
  let res = i32(params.fieldResolution);
  let invDx = params.fieldInvCellSize;

  let gx = (pos.x - params.fieldDomainMinX) * invDx - 0.5;
  let gy = (pos.y - params.fieldDomainMinY) * invDx - 0.5;
  let gz = (pos.z - params.fieldDomainMinZ) * invDx - 0.5;

  // FLIP: particle keeps its velocity + grid delta (v_new - v_old), blended toward PIC for stability.
  // alpha ~0.85 gives lively water-like motion while damping grid-scale noise.
  let vNew = sampleGrid(&gridVel, gx, gy, gz, res);
  let vOld = sampleGrid(&gridOldVel, gx, gy, gz, res);
  let delta = vNew - vOld;
  let alpha: f32 = 0.85;
  var vel = (1.0 - alpha) * vNew + alpha * (velocities[i].xyz + delta);

  let speed2 = dot(vel, vel);
  if (speed2 > params.maxVelocity * params.maxVelocity) {
    vel *= params.maxVelocity / sqrt(speed2);
  }

  pos += vel * params.dt;

  if (pos.x < -params.halfContainerX) {
    pos.x = -params.halfContainerX;
    vel.x = max(vel.x, 0.0);
  }
  if (pos.x > params.halfContainerX) {
    pos.x = params.halfContainerX;
    vel.x = min(vel.x, 0.0);
  }
  if (pos.y < 0.0) {
    pos.y = 0.0;
    vel.y = max(vel.y, 0.0);
  }
  if (pos.y > params.containerMaxY) {
    pos.y = params.containerMaxY;
    vel.y = min(vel.y, 0.0);
  }
  if (pos.z < -params.halfContainerZ) {
    pos.z = -params.halfContainerZ;
    vel.z = max(vel.z, 0.0);
  }
  if (pos.z > params.halfContainerZ) {
    pos.z = params.halfContainerZ;
    vel.z = min(vel.z, 0.0);
  }

  positions[i] = vec4f(pos, positions[i].w);
  velocities[i] = vec4f(vel, velocities[i].w);
  xsph[i] = vec4f(vel, 0.0);
}
