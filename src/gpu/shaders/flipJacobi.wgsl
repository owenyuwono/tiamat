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
@group(0) @binding(1) var<storage, read> gridVel: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> pressureIn: array<f32>;
@group(0) @binding(3) var<storage, read> divergence: array<f32>;
@group(0) @binding(4) var<storage, read_write> pressureOut: array<f32>;

const OMEGA: f32 = 1.0;

fn idx3(ix: i32, iy: i32, iz: i32, res: i32) -> u32 {
  return u32(iz * res * res + iy * res + ix);
}

fn isFluid(ix: i32, iy: i32, iz: i32, res: i32) -> bool {
  if (ix < 0 || ix >= res || iy < 0 || iy >= res || iz < 0 || iz >= res) {
    return false;
  }
  return gridVel[idx3(ix, iy, iz, res)].w > 0.01;
}

// Solid walls: Neumann (mirror pressure via boundary cell). Air cells: Dirichlet p=0.
fn pressureAt(ix: i32, iy: i32, iz: i32, res: i32) -> f32 {
  let cx = clamp(ix, 0, res - 1);
  let cy = clamp(iy, 0, res - 1);
  let cz = clamp(iz, 0, res - 1);

  if (!isFluid(cx, cy, cz, res)) {
    return 0.0;
  }
  return pressureIn[idx3(cx, cy, cz, res)];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let flatIdx = gid.x;
  let res = i32(params.fieldResolution);
  let total = u32(res * res * res);
  if (flatIdx >= total) { return; }

  let w = gridVel[flatIdx].w;
  if (w < 0.01) {
    pressureOut[flatIdx] = 0.0;
    return;
  }

  let ires = i32(flatIdx);
  let iz = ires / (res * res);
  let iy = (ires / res) % res;
  let ix = ires % res;

  let dx2 = params.fieldCellSize * params.fieldCellSize;
  let pOld = pressureIn[flatIdx];

  var sumP = 0.0;
  let neighbors = array<vec3i, 6>(
    vec3i(ix - 1, iy, iz), vec3i(ix + 1, iy, iz),
    vec3i(ix, iy - 1, iz), vec3i(ix, iy + 1, iz),
    vec3i(ix, iy, iz - 1), vec3i(ix, iy, iz + 1),
  );

  for (var n = 0u; n < 6u; n++) {
    let nb = neighbors[n];
    sumP += pressureAt(nb.x, nb.y, nb.z, res);
  }

  let pNew = (sumP - dx2 * divergence[flatIdx]) / 6.0;
  pressureOut[flatIdx] = pOld + OMEGA * (pNew - pOld);
}
