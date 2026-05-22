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

// Step 1 (Müller FLIP): apply gravity, advect, enforce particle-wall collisions.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.particleCount) { return; }

  var pos = positions[i].xyz;
  var vel = velocities[i].xyz;

  vel.y += params.gravity * params.dt;
  pos += vel * params.dt;

  if (pos.x < -params.halfContainerX) {
    pos.x = -params.halfContainerX;
    vel.x = max(vel.x, 0.0);
  }
  if (pos.x > params.halfContainerX) {
    pos.x = params.halfContainerX;
    vel.x = min(vel.x, 0.0);
  }
  if (pos.y < 0.0) {
    pos.y = 0.0;
    vel.y = max(vel.y, 0.0);
  }
  if (pos.y > params.containerMaxY) {
    pos.y = params.containerMaxY;
    vel.y = min(vel.y, 0.0);
  }
  if (pos.z < -params.halfContainerZ) {
    pos.z = -params.halfContainerZ;
    vel.z = max(vel.z, 0.0);
  }
  if (pos.z > params.halfContainerZ) {
    pos.z = params.halfContainerZ;
    vel.z = min(vel.z, 0.0);
  }

  positions[i] = vec4f(pos, positions[i].w);
  velocities[i] = vec4f(vel, velocities[i].w);
}
