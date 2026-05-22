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
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> forces: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> densityPressure: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> xsph: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> sprayParticles: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> sprayCounter: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.particleCount) {
    return;
  }

  let density = max(densityPressure[i].x, 10.0);
  let invRho = 1.0 / density;

  var vel = velocities[i].xyz;
  var pos = positions[i].xyz;

  vel += forces[i].xyz * invRho * params.dt;
  vel += params.xsphEpsilon * xsph[i].xyz;

  vel *= 1.0 - 0.5 * params.dt;

  let speed2 = dot(vel, vel);
  if (speed2 > params.maxVelocity * params.maxVelocity) {
    vel *= params.maxVelocity / sqrt(speed2);
  }

  pos += vel * params.dt;

  if (pos.x < -params.halfContainerX) {
    pos.x = -params.halfContainerX;
    vel.x *= params.boundaryDamping;
  }
  if (pos.x > params.halfContainerX) {
    pos.x = params.halfContainerX;
    vel.x *= params.boundaryDamping;
  }
  if (pos.y < 0.0) {
    pos.y = 0.0;
    vel.y *= params.boundaryDamping;
  }
  if (pos.y > params.containerMaxY) {
    pos.y = params.containerMaxY;
    vel.y *= params.boundaryDamping;
  }
  if (pos.z < -params.halfContainerZ) {
    pos.z = -params.halfContainerZ;
    vel.z *= params.boundaryDamping;
  }
  if (pos.z > params.halfContainerZ) {
    pos.z = params.halfContainerZ;
    vel.z *= params.boundaryDamping;
  }

  let densityRatio = densityPressure[i].x / params.restDensity;
  let surfaceness = max(0.0, 1.0 - densityRatio);
  let speed = length(vel);
  let kineticSurface = speed * speed * surfaceness;
  let trappedAir = xsph[i].w;

  // Spray emission — burst-emit multiple particles proportional to collision strength
  let emissionPotential = kineticSurface * (trappedAir + 0.1);
  if (densityRatio < 0.85 && emissionPotential > 1.0) {
    let burstCount = u32(clamp(emissionPotential * 0.7, 1.0, 4.0));
    var seed = i * 2654435761u;
    for (var b = 0u; b < burstCount; b++) {
      seed ^= seed << 13u; seed ^= seed >> 17u; seed ^= seed << 5u;
      let rx = (f32(seed & 0xFFFFu) / 65535.0 - 0.5) * 0.4;
      seed ^= seed << 13u; seed ^= seed >> 17u; seed ^= seed << 5u;
      let ry = f32(seed & 0xFFFFu) / 65535.0 * 0.3;
      seed ^= seed << 13u; seed ^= seed >> 17u; seed ^= seed << 5u;
      let rz = (f32(seed & 0xFFFFu) / 65535.0 - 0.5) * 0.4;
      let slot = atomicAdd(&sprayCounter[0], 1u);
      let sIdx = (slot % 32768u) * 2u;
      let lt = clamp(emissionPotential * 0.3, 0.5, 2.5);
      let jitter = vec3<f32>(rx, ry, rz);
      sprayParticles[sIdx] = vec4<f32>(pos + jitter * 0.02, lt);
      sprayParticles[sIdx + 1u] = vec4<f32>(vel * (0.6 + f32(b) * 0.15) + jitter * speed, 0.0);
    }
  }

  // Bubble emission — burst-emit proportional to trapped air
  if (densityRatio > 1.1 && trappedAir > 1.5) {
    let burstCount = u32(clamp(trappedAir * 0.5, 1.0, 3.0));
    var bSeed = i * 1664525u + 1013904223u;
    for (var b = 0u; b < burstCount; b++) {
      bSeed ^= bSeed << 13u; bSeed ^= bSeed >> 17u; bSeed ^= bSeed << 5u;
      let rx = (f32(bSeed & 0xFFFFu) / 65535.0 - 0.5) * 0.06;
      bSeed ^= bSeed << 13u; bSeed ^= bSeed >> 17u; bSeed ^= bSeed << 5u;
      let rz = (f32(bSeed & 0xFFFFu) / 65535.0 - 0.5) * 0.06;
      let slot = atomicAdd(&sprayCounter[0], 1u);
      let bIdx = (slot % 32768u) * 2u;
      sprayParticles[bIdx] = vec4<f32>(pos.x + rx, pos.y, pos.z + rz, 2.0);
      sprayParticles[bIdx + 1u] = vec4<f32>(0.0, 0.3, 0.0, 1.0);
    }
  }

  positions[i] = vec4<f32>(pos, positions[i].w);
  velocities[i] = vec4<f32>(vel, kineticSurface);
}
