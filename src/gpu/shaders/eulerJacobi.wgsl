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
@group(0) @binding(2) var<storage, read> pressureIn: array<f32>;
@group(0) @binding(3) var<storage, read> divergence: array<f32>;
@group(0) @binding(4) var<storage, read_write> pressureOut: array<f32>;

fn idx3(ix: i32, iy: i32, iz: i32, res: i32) -> u32 {
  return u32(iz * res * res + iy * res + ix);
}

fn isFluid(ix: i32, iy: i32, iz: i32, res: i32) -> bool {
  if (ix < 0 || ix >= res || iy < 0 || iy >= res || iz < 0 || iz >= res) {
    return false;
  }
  return densityMarker[idx3(ix, iy, iz, res)] > 0.5;
}

fn pressureAt(ix: i32, iy: i32, iz: i32, res: i32) -> f32 {
  // Wall Neumann BC: mirror pressure from interior
  if (ix < 0) { return pressureIn[idx3(0, iy, iz, res)]; }
  if (ix >= res) { return pressureIn[idx3(res - 1, iy, iz, res)]; }
  if (iy < 0) { return pressureIn[idx3(ix, 0, iz, res)]; }
  if (iy >= res) { return pressureIn[idx3(ix, res - 1, iz, res)]; }
  if (iz < 0) { return pressureIn[idx3(ix, iy, 0, res)]; }
  if (iz >= res) { return pressureIn[idx3(ix, iy, res - 1, res)]; }

  // Air Dirichlet BC: p = 0
  if (!isFluid(ix, iy, iz, res)) {
    return 0.0;
  }
  return pressureIn[idx3(ix, iy, iz, res)];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let res = i32(params.fieldResolution);
  let total = u32(res * res * res);
  if (gid.x >= total) { return; }

  if (!isFluid(i32(gid.x) % res, (i32(gid.x) / res) % res, i32(gid.x) / (res * res), res)) {
    pressureOut[gid.x] = 0.0;
    return;
  }

  let ires = i32(gid.x);
  let iz = ires / (res * res);
  let iy = (ires / res) % res;
  let ix = ires % res;

  let dx2 = params.fieldCellSize * params.fieldCellSize;
  let pOld = pressureIn[gid.x];

  let sumP = pressureAt(ix - 1, iy, iz, res)
           + pressureAt(ix + 1, iy, iz, res)
           + pressureAt(ix, iy - 1, iz, res)
           + pressureAt(ix, iy + 1, iz, res)
           + pressureAt(ix, iy, iz - 1, res)
           + pressureAt(ix, iy, iz + 1, res);

  let pNew = (sumP - dx2 * divergence[gid.x]) / 6.0;
  pressureOut[gid.x] = pOld + 1.0 * (pNew - pOld);
}
