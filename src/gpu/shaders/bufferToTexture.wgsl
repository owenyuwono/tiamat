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

// 3x3x3 box filter to remove grid-scale cellular artifacts from splatting while
// preserving overall fluid shape. This is the source of the "cellular / puffy"
// pattern the user sees on the settled pool (especially in the high-density middle).
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let res = params.fieldResolution;
  if (gid.x >= res || gid.y >= res || gid.z >= res) {
    return;
  }

  let resI = i32(res);
  let cx = i32(gid.x);
  let cy = i32(gid.y);
  let cz = i32(gid.z);

  var sumD: f32 = 0.0;
  var sumI: f32 = 0.0;
  var count: f32 = 0.0;

  for (var dz = -1; dz <= 1; dz++) {
    let iz = cz + dz;
    if (iz < 0 || iz >= resI) { continue; }
    for (var dy = -1; dy <= 1; dy++) {
      let iy = cy + dy;
      if (iy < 0 || iy >= resI) { continue; }
      for (var dx = -1; dx <= 1; dx++) {
        let ix = cx + dx;
        if (ix < 0 || ix >= resI) { continue; }
        let nidx = u32((iz * resI * resI + iy * resI + ix) * 2);
        sumD += f32(densityField[nidx]) * INV_SCALE;
        sumI += f32(densityField[nidx + 1u]) * INV_SCALE;
        count += 1.0;
      }
    }
  }

  let density = select(0.0, sumD / count, count > 0.0);
  let impact = select(0.0, sumI / count, count > 0.0);

  textureStore(densityTexture, gid, vec4<f32>(density, impact, 0.0, 0.0));
}
