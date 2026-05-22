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
@group(0) @binding(1) var<storage, read> densityMarker: array<f32>;
@group(0) @binding(2) var<storage, read_write> validFlags: array<u32>;

fn cellIdx(ix: i32, iy: i32, iz: i32, res: i32) -> u32 {
  return u32(iz * res * res + iy * res + ix);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let R = i32(params.fieldResolution);

  let uCount = u32((R + 1) * R * R);
  let vCount = u32(R * (R + 1) * R);
  let wCount = u32(R * R * (R + 1));
  let maxCount = max(uCount, max(vCount, wCount));
  if (idx >= maxCount) { return; }

  let vOffset = uCount;
  let wOffset = uCount + vCount;

  if (idx < uCount) {
    let iz = i32(idx / u32((R + 1) * R));
    let iy = i32((idx / u32(R + 1)) % u32(R));
    let ix = i32(idx % u32(R + 1));

    if (ix == 0 || ix == R) {
      validFlags[idx] = 1u;
    } else {
      let fluid = densityMarker[cellIdx(ix - 1, iy, iz, R)] > 0.5
               || densityMarker[cellIdx(ix, iy, iz, R)] > 0.5;
      validFlags[idx] = select(0u, 1u, fluid);
    }
  }

  if (idx < vCount) {
    let iz = i32(idx / u32((R + 1) * R));
    let iy = i32((idx / u32(R)) % u32(R + 1));
    let ix = i32(idx % u32(R));

    if (iy == 0 || iy == R) {
      validFlags[vOffset + idx] = 1u;
    } else {
      let fluid = densityMarker[cellIdx(ix, iy - 1, iz, R)] > 0.5
               || densityMarker[cellIdx(ix, iy, iz, R)] > 0.5;
      validFlags[vOffset + idx] = select(0u, 1u, fluid);
    }
  }

  if (idx < wCount) {
    let iz = i32(idx / u32(R * R));
    let iy = i32((idx / u32(R)) % u32(R));
    let ix = i32(idx % u32(R));

    if (iz == 0 || iz == R) {
      validFlags[wOffset + idx] = 1u;
    } else {
      let fluid = densityMarker[cellIdx(ix, iy, iz - 1, R)] > 0.5
               || densityMarker[cellIdx(ix, iy, iz, R)] > 0.5;
      validFlags[wOffset + idx] = select(0u, 1u, fluid);
    }
  }
}
