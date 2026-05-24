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
@group(0) @binding(2) var<storage, read_write> densityPressure: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> cellCounts: array<u32>;
@group(0) @binding(4) var<storage, read> cellEntries: array<u32>;

fn hashCell(cx: i32, cy: i32, cz: i32, tableSize: u32) -> u32 {
  let h = (cx * 73856093) ^ (cy * 19349663) ^ (cz * 83492791);
  return u32(h & 0x7FFFFFFF) % tableSize;
}

fn mirrorDensityContrib(dist: f32, mass: f32, poly6Coeff: f32, H2: f32) -> f32 {
  let r2 = 4.0 * dist * dist;
  if (r2 >= H2) {
    return 0.0;
  }
  let d = H2 - r2;
  return mass * poly6Coeff * d * d * d;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.particleCount) {
    return;
  }

  let posI = positions[i].xyz;
  let cxi = i32(floor((posI.x + params.halfContainerX) * params.invCellSize));
  let cyi = i32(floor(posI.y * params.invCellSize));
  let czi = i32(floor((posI.z + params.halfContainerZ) * params.invCellSize));

  var rho = 0.0;

  for (var dx = -1; dx <= 1; dx++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dz = -1; dz <= 1; dz++) {
        let cellHash = hashCell(cxi + dx, cyi + dy, czi + dz, params.tableSize);
        let count = min(cellCounts[cellHash], params.maxPerCell);
        let base = cellHash * params.maxPerCell;
        for (var k = 0u; k < count; k++) {
          let j = cellEntries[base + k];
          let diff = posI - positions[j].xyz;
          let r2 = dot(diff, diff);
          if (r2 < params.H2) {
            let d = params.H2 - r2;
            rho += params.mass * params.poly6Coeff * d * d * d;
          }
        }
      }
    }
  }

  let xi = posI.x;
  let yi = posI.y;
  let zi = posI.z;

  // Mirror density at walls — scaled to avoid over-compensation
  // (especially at corners where multiple walls compound)
  var mirrorRho = 0.0;
  var wallCount = 0.0;

  if (yi < params.H) {
    mirrorRho += mirrorDensityContrib(yi, params.mass, params.poly6Coeff, params.H2);
    wallCount += 1.0;
  }
  if (params.containerMaxY - yi < params.H) {
    mirrorRho += mirrorDensityContrib(params.containerMaxY - yi, params.mass, params.poly6Coeff, params.H2);
    wallCount += 1.0;
  }
  if (xi + params.halfContainerX < params.H) {
    mirrorRho += mirrorDensityContrib(xi + params.halfContainerX, params.mass, params.poly6Coeff, params.H2);
    wallCount += 1.0;
  }
  if (params.halfContainerX - xi < params.H) {
    mirrorRho += mirrorDensityContrib(params.halfContainerX - xi, params.mass, params.poly6Coeff, params.H2);
    wallCount += 1.0;
  }
  if (zi + params.halfContainerZ < params.H) {
    mirrorRho += mirrorDensityContrib(zi + params.halfContainerZ, params.mass, params.poly6Coeff, params.H2);
    wallCount += 1.0;
  }
  if (params.halfContainerZ - zi < params.H) {
    mirrorRho += mirrorDensityContrib(params.halfContainerZ - zi, params.mass, params.poly6Coeff, params.H2);
    wallCount += 1.0;
  }

  if (wallCount > 0.0) {
    rho += mirrorRho * (0.4 / max(wallCount, 1.0));
  }

  let ratio = rho / params.restDensity;
  let r2_ = ratio * ratio;
  let r4 = r2_ * r2_;
  let r7 = r4 * r2_ * ratio;
  let pressure = max(0.0, params.stiffness * (r7 - 1.0));

  densityPressure[i] = vec2<f32>(rho, pressure);
}
