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
@group(0) @binding(2) var<storage, read> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> densityPressure: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> forces: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> xsph: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> cellCounts: array<u32>;
@group(0) @binding(7) var<storage, read> cellEntries: array<u32>;
@group(0) @binding(8) var<storage, read> sleepState: array<u32>;

const SLEEP_THRESHOLD: u32 = 120u;

fn hashCell(cx: i32, cy: i32, cz: i32, tableSize: u32) -> u32 {
  let h = (cx * 73856093) ^ (cy * 19349663) ^ (cz * 83492791);
  return u32(h & 0x7FFFFFFF) % tableSize;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.particleCount) {
    return;
  }

  if (sleepState[i] >= SLEEP_THRESHOLD) {
    forces[i] = vec4<f32>(0.0);
    xsph[i] = vec4<f32>(0.0);
    return;
  }

  let posI = positions[i].xyz;
  let velI = velocities[i].xyz;
  let dp = densityPressure[i];
  let densityI = dp.x;
  let pressureI = dp.y;

  let cxi = i32(floor((posI.x + params.halfContainerX) * params.invCellSize));
  let cyi = i32(floor(posI.y * params.invCellSize));
  let czi = i32(floor((posI.z + params.halfContainerZ) * params.invCellSize));

  var force = vec3<f32>(0.0, params.gravity * densityI, 0.0);
  var xsphAcc = vec3<f32>(0.0);

  for (var dx = -1; dx <= 1; dx++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dz = -1; dz <= 1; dz++) {
        let cellHash = hashCell(cxi + dx, cyi + dy, czi + dz, params.tableSize);
        let count = min(cellCounts[cellHash], params.maxPerCell);
        let base = cellHash * params.maxPerCell;
        for (var k = 0u; k < count; k++) {
          let j = cellEntries[base + k];
          if (j == i) {
            continue;
          }

          let diff = posI - positions[j].xyz;
          let r2 = dot(diff, diff);
          if (r2 >= params.H2 || r2 < 1e-8) {
            continue;
          }

          let invR = inverseSqrt(r2);
          let r = r2 * invR;
          let dir = diff * invR;
          let hr = params.H - r;
          let densityJ = densityPressure[j].x;
          let pressureJ = densityPressure[j].y;
          let invDensJ = 1.0 / densityJ;
          let velJ = velocities[j].xyz;

          let avgPressure = (pressureI + pressureJ) * 0.5;
          let pressureMag = -params.mass * avgPressure * invDensJ * params.spikyCoeff * hr * hr;
          force += dir * pressureMag;

          if (r < params.collisionRadius) {
            let overlap = params.collisionRadius - r;
            force += dir * params.collisionStiffness * overlap;
          }

          let viscMag = params.viscosity * params.mass * invDensJ * params.viscLapCoeff * hr;
          force += viscMag * (velJ - velI);

          let cohesion = 1.0 - r / params.H;
          let cohMag = params.surfaceTension * params.mass * params.mass * invDensJ * cohesion * cohesion;
          force -= dir * cohMag;

          let d2 = params.H2 - r2;
          let w = params.poly6Coeff * d2 * d2 * d2;
          let rhoAvg = (densityI + densityJ) * 0.5;
          let xsphW = params.mass / rhoAvg * w;
          xsphAcc += (velJ - velI) * xsphW;
        }
      }
    }
  }

  let wallDist = params.collisionRadius * 2.0;
  let wallK = params.collisionStiffness * 2.0;

  let dLeft = posI.x - (-params.halfContainerX);
  let dRight = params.halfContainerX - posI.x;
  let dBottom = posI.y;
  let dTop = params.containerMaxY - posI.y;
  let dBack = posI.z - (-params.halfContainerZ);
  let dFront = params.halfContainerZ - posI.z;

  if (dLeft < wallDist) {
    force.x += wallK * (wallDist - dLeft);
  }
  if (dRight < wallDist) {
    force.x -= wallK * (wallDist - dRight);
  }
  if (dBottom < wallDist) {
    force.y += wallK * (wallDist - dBottom);
  }
  if (dTop < wallDist) {
    force.y -= wallK * (wallDist - dTop);
  }
  if (dBack < wallDist) {
    force.z += wallK * (wallDist - dBack);
  }
  if (dFront < wallDist) {
    force.z -= wallK * (wallDist - dFront);
  }

  forces[i] = vec4<f32>(force, 0.0);
  xsph[i] = vec4<f32>(xsphAcc, 0.0);
}
