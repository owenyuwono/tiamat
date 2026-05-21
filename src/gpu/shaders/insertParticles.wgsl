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
@group(0) @binding(2) var<storage, read_write> cellCounts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> cellEntries: array<u32>;

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

  let pos = positions[i].xyz;
  let cx = i32(floor((pos.x + params.halfContainerX) * params.invCellSize));
  let cy = i32(floor(pos.y * params.invCellSize));
  let cz = i32(floor((pos.z + params.halfContainerZ) * params.invCellSize));
  let cellHash = hashCell(cx, cy, cz, params.tableSize);

  let slot = atomicAdd(&cellCounts[cellHash], 1u);
  if (slot < params.maxPerCell) {
    cellEntries[cellHash * params.maxPerCell + slot] = i;
  }
}
