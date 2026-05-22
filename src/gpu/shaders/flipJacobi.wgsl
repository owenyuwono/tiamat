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
@group(0) @binding(2) var<storage, read> uVel: array<f32>;
@group(0) @binding(3) var<storage, read> vVel: array<f32>;
@group(0) @binding(4) var<storage, read> wVel: array<f32>;
@group(0) @binding(5) var<storage, read> pressureIn: array<f32>;
@group(0) @binding(6) var<storage, read> divergence: array<f32>;
@group(0) @binding(7) var<storage, read_write> pressureOut: array<f32>;

const OMEGA: f32 = 1.0;

fn idx3(ix: i32, iy: i32, iz: i32, res: i32) -> u32 {
  return u32(iz * res * res + iy * res + ix);
}

fn uIdx(ix: i32, iy: i32, iz: i32, res: i32) -> u32 {
  let ux = clamp(ix, 0, res);
  let uy = clamp(iy, 0, res - 1);
  let uz = clamp(iz, 0, res - 1);
  return u32(uz * res * (res + 1) + uy * (res + 1) + ux);
}

fn vIdx(ix: i32, iy: i32, iz: i32, res: i32) -> u32 {
  let vx = clamp(ix, 0, res - 1);
  let vy = clamp(iy, 0, res);
  let vz = clamp(iz, 0, res - 1);
  return u32(vz * (res + 1) * res + vy * res + vx);
}

fn wIdx(ix: i32, iy: i32, iz: i32, res: i32) -> u32 {
  let wx = clamp(ix, 0, res - 1);
  let wy = clamp(iy, 0, res - 1);
  let wz = clamp(iz, 0, res);
  return u32(wz * res * res + wy * res + wx);
}

fn isFluid(ix: i32, iy: i32, iz: i32, res: i32) -> bool {
  if (ix < 0 || ix >= res || iy < 0 || iy >= res || iz < 0 || iz >= res) {
    return false;
  }
  return gridVel[idx3(ix, iy, iz, res)].w > 0.01;
}

// Solid walls: Neumann (mirror pressure via boundary cell) decided by face velocity ~0.
// Air cells / non-solid: Dirichlet p=0.
fn pressureAt(ix: i32, iy: i32, iz: i32, res: i32) -> f32 {
  // X walls (use uVel faces)
  if (ix < 0) {
    let faceU = uVel[uIdx(0, iy, iz, res)];
    if (abs(faceU) < 0.001) {
      return pressureIn[idx3(0, iy, iz, res)]; // mirror for solid Neumann
    } else {
      return 0.0;
    }
  }
  if (ix >= res) {
    let faceU = uVel[uIdx(res, iy, iz, res)];
    if (abs(faceU) < 0.001) {
      return pressureIn[idx3(res - 1, iy, iz, res)];
    } else {
      return 0.0;
    }
  }

  // Y walls (use vVel faces)
  if (iy < 0) {
    let faceV = vVel[vIdx(ix, 0, iz, res)];
    if (abs(faceV) < 0.001) {
      return pressureIn[idx3(ix, 0, iz, res)];
    } else {
      return 0.0;
    }
  }
  if (iy >= res) {
    let faceV = vVel[vIdx(ix, res, iz, res)];
    if (abs(faceV) < 0.001) {
      return pressureIn[idx3(ix, res - 1, iz, res)];
    } else {
      return 0.0;
    }
  }

  // Z walls (use wVel faces)
  if (iz < 0) {
    let faceW = wVel[wIdx(ix, iy, 0, res)];
    if (abs(faceW) < 0.001) {
      return pressureIn[idx3(ix, iy, 0, res)];
    } else {
      return 0.0;
    }
  }
  if (iz >= res) {
    let faceW = wVel[wIdx(ix, iy, res, res)];
    if (abs(faceW) < 0.001) {
      return pressureIn[idx3(ix, iy, res - 1, res)];
    } else {
      return 0.0;
    }
  }

  // Interior: Dirichlet 0 for air/non-fluid
  if (!isFluid(ix, iy, iz, res)) {
    return 0.0;
  }
  return pressureIn[idx3(ix, iy, iz, res)];
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
