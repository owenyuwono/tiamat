struct SprayParams {
  dt: f32,
  gravity: f32,
  maxCount: u32,
  fieldResolution: u32,
  fieldDomainMinX: f32,
  fieldDomainMinY: f32,
  fieldDomainMinZ: f32,
  fieldInvCellSize: f32,
  threshold: f32,
  halfContainerX: f32,
  halfContainerZ: f32,
  containerMaxY: f32,
}

const FIXED_POINT_SCALE: f32 = 10000.0;

@group(0) @binding(0) var<uniform> params: SprayParams;
@group(0) @binding(1) var<storage, read_write> sprayParticles: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> densityField: array<atomic<u32>>;

fn getCellIdx(pos: vec3<f32>) -> u32 {
  let res = i32(params.fieldResolution);
  let ix = clamp(i32(floor((pos.x - params.fieldDomainMinX) * params.fieldInvCellSize)), 0, res - 1);
  let iy = clamp(i32(floor((pos.y - params.fieldDomainMinY) * params.fieldInvCellSize)), 0, res - 1);
  let iz = clamp(i32(floor((pos.z - params.fieldDomainMinZ) * params.fieldInvCellSize)), 0, res - 1);
  return u32((iz * res * res + iy * res + ix) * 2);
}

fn sampleDensityAt(pos: vec3<f32>) -> f32 {
  return f32(atomicLoad(&densityField[getCellIdx(pos)])) / FIXED_POINT_SCALE;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.maxCount) { return; }

  let posIdx = i * 2u;
  let velIdx = i * 2u + 1u;

  var posLife = sprayParticles[posIdx];
  var velAge = sprayParticles[velIdx];

  if (posLife.w <= 0.0) { return; }

  let dt = params.dt;
  let isBubble = velAge.w > 0.5;

  if (isBubble) {
    // Buoyancy — rise upward
    velAge.y += abs(params.gravity) * 0.4 * dt;

    // Drag — couple to surrounding fluid (simplified)
    let drag = 1.0 - 5.0 * dt;
    velAge = vec4(velAge.xyz * drag, velAge.w);

    posLife.x += velAge.x * dt;
    posLife.y += velAge.y * dt;
    posLife.z += velAge.z * dt;
    posLife.w -= dt;

    // Inject foam trail as bubble rises
    let trailIdx = getCellIdx(posLife.xyz) + 1u;
    atomicAdd(&densityField[trailIdx], u32(0.3 * FIXED_POINT_SCALE));

    // Pop at surface — burst of foam
    let density = sampleDensityAt(posLife.xyz);
    if (density < params.threshold * 0.5) {
      atomicAdd(&densityField[trailIdx], u32(2.0 * FIXED_POINT_SCALE));
      posLife.w = 0.0;
    }
  } else {
    // Spray — ballistic with gravity and air drag
    velAge.y += params.gravity * dt;
    posLife.x += velAge.x * dt;
    posLife.y += velAge.y * dt;
    posLife.z += velAge.z * dt;
    posLife.w -= dt;

    let drag = 1.0 - 2.0 * dt;
    velAge.x *= drag;
    velAge.z *= drag;

    // Floor bounce — splat and die quickly
    if (posLife.y < 0.0) {
      posLife.y = 0.0;
      velAge.y *= -0.3;
      velAge.x *= 0.5;
      velAge.z *= 0.5;
      posLife.w = min(posLife.w, 0.1);
    }

    // Water re-entry — inject foam and die
    if (posLife.w > 0.0) {
      let density = sampleDensityAt(posLife.xyz);
      if (density > params.threshold) {
        let spd = length(velAge.xyz);
        let foamIdx = getCellIdx(posLife.xyz) + 1u;
        atomicAdd(&densityField[foamIdx], u32(spd * 2.0 * FIXED_POINT_SCALE));
        posLife.w = 0.0;
      }
    }
  }

  // Container bounds kill
  if (posLife.x < -params.halfContainerX || posLife.x > params.halfContainerX ||
      posLife.y > params.containerMaxY ||
      posLife.z < -params.halfContainerZ || posLife.z > params.halfContainerZ) {
    posLife.w = 0.0;
  }

  sprayParticles[posIdx] = posLife;
  sprayParticles[velIdx] = velAge;
}
