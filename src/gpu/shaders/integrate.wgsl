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

  vel *= 1.0 - 1.5 * params.dt;

  let speed2 = dot(vel, vel);
  if (speed2 > params.maxVelocity * params.maxVelocity) {
    vel *= params.maxVelocity / sqrt(speed2);
  }

  pos += (vel + params.xsphEpsilon * xsph[i].xyz) * params.dt;

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

  positions[i] = vec4<f32>(pos, positions[i].w);
  velocities[i] = vec4<f32>(vel, velocities[i].w);
}
