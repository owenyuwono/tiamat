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
@group(0) @binding(1) var<storage, read_write> uVel: array<f32>;
@group(0) @binding(2) var<storage, read_write> vVel: array<f32>;
@group(0) @binding(3) var<storage, read_write> wVel: array<f32>;
@group(0) @binding(4) var<storage, read> pressure: array<f32>;

fn idxU(ix: i32, iy: i32, iz: i32, res: i32) -> u32 {
  return u32(((iz * res + iy) * (res + 1)) + ix);
}
fn idxV(ix: i32, iy: i32, iz: i32, res: i32) -> u32 {
  return u32(((iz * (res + 1) + iy) * res) + ix);
}
fn idxW(ix: i32, iy: i32, iz: i32, res: i32) -> u32 {
  return u32(((iz * res + iy) * res) + ix);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let R = i32(params.fieldResolution);

  let uCount = (R + 1) * R * R;
  let vCount = R * (R + 1) * R;
  let wCount = R * R * (R + 1);
  let maxCount = max(uCount, max(vCount, wCount));

  if (idx >= u32(maxCount)) { return; }

  let h = params.fieldCellSize;
  let dt = params.dt;
  let invRho = 1.0 / params.restDensity;

  // U faces
  if (idx < u32(uCount)) {
    let iz = i32(idx / u32((R + 1) * R));
    let iy = i32((idx / u32(R + 1)) % u32(R));
    let ix = i32(idx % u32(R + 1));

    var vel = uVel[idx];
    if (ix > 0 && ix < R) {
      let pL = pressure[(iz * R + iy) * R + (ix - 1)];
      let pR = pressure[(iz * R + iy) * R + ix];
      vel -= dt * invRho * (pR - pL) / h;
    }
    if (ix == 0 || ix == R) { vel = 0.0; }
    uVel[idx] = vel;
  }

  // V faces
  if (idx < u32(vCount)) {
    let iz = i32(idx / u32(R * (R + 1)));
    let iy = i32((idx / u32(R)) % u32(R + 1));
    let ix = i32(idx % u32(R));

    var vel = vVel[idx];
    if (iy > 0 && iy < R) {
      let pD = pressure[((iz * R + (iy - 1)) * R) + ix];
      let pU = pressure[((iz * R + iy) * R) + ix];
      vel -= dt * invRho * (pU - pD) / h;
    }
    if (iy == 0 || iy == R) { vel = 0.0; }
    vVel[idx] = vel;
  }

  // W faces
  if (idx < u32(wCount)) {
    let iz = i32(idx / u32(R * R));
    let iy = i32((idx / u32(R)) % u32(R));
    let ix = i32(idx % u32(R));

    var vel = wVel[idx];
    if (iz > 0 && iz < R) {
      let pB = pressure[(((iz - 1) * R + iy) * R) + ix];
      let pT = pressure[((iz * R + iy) * R) + ix];
      vel -= dt * invRho * (pT - pB) / h;
    }
    if (iz == 0 || iz == R) { vel = 0.0; }
    wVel[idx] = vel;
  }
}
