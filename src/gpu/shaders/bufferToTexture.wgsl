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
@group(0) @binding(1) var<storage, read> densityField: array<u32>;
@group(0) @binding(2) var densityTexture: texture_storage_3d<rg32float, write>;

const INV_SCALE: f32 = 0.0001;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let res = params.fieldResolution;
  if (gid.x >= res || gid.y >= res || gid.z >= res) {
    return;
  }

  let idx = (gid.z * res * res + gid.y * res + gid.x) * 2u;
  let density = f32(densityField[idx]) * INV_SCALE;
  let impact = f32(densityField[idx + 1u]) * INV_SCALE;

  textureStore(densityTexture, gid, vec4<f32>(density, impact, 0.0, 0.0));
}
