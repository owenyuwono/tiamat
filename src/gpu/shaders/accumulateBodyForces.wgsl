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

struct Obstacle {
  posRadius: vec4<f32>,
  velMass: vec4<f32>,
}

struct ObstacleData {
  count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  obstacles: array<Obstacle, 8>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<uniform> obstacleData: ObstacleData;
@group(0) @binding(2) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> densityPressure: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> obstacleForces: array<atomic<i32>>;

const FORCE_SCALE: f32 = 10.0;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.particleCount) {
    return;
  }

  let posI = positions[i].xyz;
  let dp = densityPressure[i];
  let densityI = dp.x;
  let pressureI = dp.y;

  for (var o = 0u; o < obstacleData.count; o++) {
    let obs = obstacleData.obstacles[o];
    let obsPos = obs.posRadius.xyz;
    let obsRadius = obs.posRadius.w;
    if (obsRadius < 1e-6) {
      continue;
    }

    let diff = posI - obsPos;
    let dist = length(diff);
    let surfaceDist = dist - obsRadius;

    if (surfaceDist >= params.H || surfaceDist < 0.0 || dist < 1e-6) {
      continue;
    }

    let normal = diff / dist;
    let mirrorDist = surfaceDist;
    if (mirrorDist >= params.H) {
      continue;
    }

    let hr = params.H - mirrorDist;
    let mirrorPressure = pressureI;
    let mirrorDensity = params.restDensity;
    let invMirrorDens = 1.0 / mirrorDensity;

    // Pressure force from mirror particle (reaction force on obstacle)
    let avgPressure = (pressureI + mirrorPressure) * 0.5;
    let pressureMag = params.mass * avgPressure * invMirrorDens * params.spikyCoeff * hr * hr;
    // Force on obstacle is opposite to force on particle
    let forceOnObstacle = normal * pressureMag;

    // Accumulate via fixed-point atomic add
    let baseIdx = o * 4u;
    atomicAdd(&obstacleForces[baseIdx + 0u], i32(forceOnObstacle.x * FORCE_SCALE));
    atomicAdd(&obstacleForces[baseIdx + 1u], i32(forceOnObstacle.y * FORCE_SCALE));
    atomicAdd(&obstacleForces[baseIdx + 2u], i32(forceOnObstacle.z * FORCE_SCALE));
  }
}
