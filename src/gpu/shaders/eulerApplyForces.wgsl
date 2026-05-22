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
@group(0) @binding(1) var<storage, read_write> vVel: array<f32>;
@group(0) @binding(2) var<storage, read> densityMarker: array<f32>;

fn idx3(ix: i32, iy: i32, iz: i32, res: i32) -> u32 {
  return u32(iz * res * res + iy * res + ix);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let R = i32(params.fieldResolution);
  let vCount = R * (R + 1) * R;
  if (gid.x >= u32(vCount)) { return; }

  let idx = i32(gid.x);
  let iz = idx / ((R + 1) * R);
  let iy = (idx / R) % (R + 1);
  let ix = idx % R;

  if (iy == 0 || iy == R) {
    vVel[gid.x] = 0.0;
    return;
  }

  let cellBelow = densityMarker[idx3(ix, iy - 1, iz, R)];
  let cellAbove = densityMarker[idx3(ix, iy, iz, R)];
  let isFluid = cellBelow > 0.5 || cellAbove > 0.5;

  if (isFluid) {
    vVel[gid.x] += params.gravity * params.dt;
  }
}
