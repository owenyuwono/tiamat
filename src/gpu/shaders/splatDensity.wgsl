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
@group(0) @binding(2) var<storage, read> xsph: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> velocities: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> densityField: array<atomic<u32>>;

const FIXED_POINT_SCALE: u32 = 10000u;

fn splatAt(splatPos: vec3<f32>, foamSignal: f32) {
  let res = i32(params.fieldResolution);
  let resM1 = res - 1;
  let r = i32(params.splatRadiusCells);

  let cx = i32(floor((splatPos.x - params.fieldDomainMinX) * params.fieldInvCellSize));
  let cy = i32(floor((splatPos.y - params.fieldDomainMinY) * params.fieldInvCellSize));
  let cz = i32(floor((splatPos.z - params.fieldDomainMinZ) * params.fieldInvCellSize));

  let x0 = max(0, cx - r);
  let x1 = min(resM1, cx + r);
  let y0 = max(0, cy - r);
  let y1 = min(resM1, cy + r);
  let z0 = max(0, cz - r);
  let z1 = min(resM1, cz + r);

  for (var iz = z0; iz <= z1; iz++) {
    let vz = (f32(iz) + 0.5) * params.fieldCellSize + params.fieldDomainMinZ - splatPos.z;
    let vz2 = vz * vz;
    if (vz2 >= params.splatRadius2) {
      continue;
    }
    for (var iy = y0; iy <= y1; iy++) {
      let vy = (f32(iy) + 0.5) * params.fieldCellSize + params.fieldDomainMinY - splatPos.y;
      let vyz2 = vy * vy + vz2;
      if (vyz2 >= params.splatRadius2) {
        continue;
      }
      for (var ix = x0; ix <= x1; ix++) {
        let vx = (f32(ix) + 0.5) * params.fieldCellSize + params.fieldDomainMinX - splatPos.x;
        let dist2 = vx * vx + vyz2;
        if (dist2 < params.splatRadius2) {
          let t = 1.0 - dist2 / params.splatRadius2;
          let w = t * t * t;
          let idx = u32((iz * res * res + iy * res + ix) * 2);
          atomicAdd(&densityField[idx], u32(w * f32(FIXED_POINT_SCALE)));
          atomicAdd(&densityField[idx + 1u], u32(w * foamSignal * f32(FIXED_POINT_SCALE)));
        }
      }
    }
  }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.particleCount) {
    return;
  }

  let pos = positions[i].xyz;
  let impact = length(xsph[i].xyz);
  let trappedAir = xsph[i].w;
  let kineticSurface = velocities[i].w;
  let foamSignal = impact * 5.0 + trappedAir * 3.0 + kineticSurface * 0.5;

  splatAt(pos, foamSignal);

  // Mirror splat across container walls to fill density at boundaries
  let splatR = sqrt(params.splatRadius2);

  if (pos.x > params.halfContainerX - splatR) {
    splatAt(vec3(2.0 * params.halfContainerX - pos.x, pos.y, pos.z), foamSignal);
  }
  if (pos.x < -params.halfContainerX + splatR) {
    splatAt(vec3(-2.0 * params.halfContainerX - pos.x, pos.y, pos.z), foamSignal);
  }
  if (pos.z > params.halfContainerZ - splatR) {
    splatAt(vec3(pos.x, pos.y, 2.0 * params.halfContainerZ - pos.z), foamSignal);
  }
  if (pos.z < -params.halfContainerZ + splatR) {
    splatAt(vec3(pos.x, pos.y, -2.0 * params.halfContainerZ - pos.z), foamSignal);
  }
  if (pos.y < splatR) {
    splatAt(vec3(pos.x, -pos.y, pos.z), foamSignal);
  }
  if (pos.y > params.containerMaxY - splatR) {
    splatAt(vec3(pos.x, 2.0 * params.containerMaxY - pos.y, pos.z), foamSignal);
  }
}
