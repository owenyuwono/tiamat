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
@group(0) @binding(3) var<storage, read_write> densityField: array<atomic<u32>>;

const FIXED_POINT_SCALE: u32 = 10000u;

fn splatJitter(particleIdx: u32) -> vec3f {
  let n = f32(particleIdx);
  let jx = fract(sin(n * 127.1) * 43758.5453) * 2.0 - 1.0;
  let jy = fract(sin(n * 269.5) * 43758.5453) * 2.0 - 1.0;
  let jz = fract(sin(n * 419.2) * 43758.5453) * 2.0 - 1.0;
  return vec3f(jx, jy, jz) * params.fieldCellSize * 0.45;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.particleCount) {
    return;
  }

  let pos = positions[i].xyz + splatJitter(i);
  let impact = length(xsph[i].xyz);

  let res = i32(params.fieldResolution);
  let resM1 = res - 1;
  let r = i32(params.splatRadiusCells);

  let cx = i32(floor((pos.x - params.fieldDomainMinX) * params.fieldInvCellSize));
  let cy = i32(floor((pos.y - params.fieldDomainMinY) * params.fieldInvCellSize));
  let cz = i32(floor((pos.z - params.fieldDomainMinZ) * params.fieldInvCellSize));

  let x0 = max(0, cx - r);
  let x1 = min(resM1, cx + r);
  let y0 = max(0, cy - r);
  let y1 = min(resM1, cy + r);
  let z0 = max(0, cz - r);
  let z1 = min(resM1, cz + r);

  for (var iz = z0; iz <= z1; iz++) {
    let vz = (f32(iz) + 0.5) * params.fieldCellSize + params.fieldDomainMinZ - pos.z;
    let vz2 = vz * vz;
    if (vz2 >= params.splatRadius2) {
      continue;
    }
    for (var iy = y0; iy <= y1; iy++) {
      let vy = (f32(iy) + 0.5) * params.fieldCellSize + params.fieldDomainMinY - pos.y;
      let vyz2 = vy * vy + vz2;
      if (vyz2 >= params.splatRadius2) {
        continue;
      }
      for (var ix = x0; ix <= x1; ix++) {
        let vx = (f32(ix) + 0.5) * params.fieldCellSize + params.fieldDomainMinX - pos.x;
        let dist2 = vx * vx + vyz2;
        if (dist2 < params.splatRadius2) {
          let t = 1.0 - dist2 / params.splatRadius2;
          let w = t * t * t;
          let idx = u32((iz * res * res + iy * res + ix) * 2);
          atomicAdd(&densityField[idx], u32(w * f32(FIXED_POINT_SCALE)));
          atomicAdd(&densityField[idx + 1u], u32(w * impact * f32(FIXED_POINT_SCALE)));
        }
      }
    }
  }
}
