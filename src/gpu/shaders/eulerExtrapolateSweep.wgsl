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
@group(0) @binding(1) var<storage, read_write> uVel: array<f32>;
@group(0) @binding(2) var<storage, read_write> vVel: array<f32>;
@group(0) @binding(3) var<storage, read_write> wVel: array<f32>;
@group(0) @binding(4) var<storage, read> validIn: array<u32>;
@group(0) @binding(5) var<storage, read_write> validOut: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let R = i32(params.fieldResolution);

  let uCount = u32((R + 1) * R * R);
  let vCount = u32(R * (R + 1) * R);
  let wCount = u32(R * R * (R + 1));
  let maxCount = max(uCount, max(vCount, wCount));
  if (idx >= maxCount) { return; }

  let vOffset = uCount;
  let wOffset = uCount + vCount;

  if (idx < uCount) {
    if (validIn[idx] == 1u) {
      validOut[idx] = 1u;
    } else {
      let iz = i32(idx / u32((R + 1) * R));
      let iy = i32((idx / u32(R + 1)) % u32(R));

      var sum = 0.0;
      var cnt = 0u;
      if (iy > 0)     { let n = idx - u32(R + 1);      if (validIn[n] == 1u) { sum += uVel[n]; cnt += 1u; } }
      if (iy < R - 1) { let n = idx + u32(R + 1);      if (validIn[n] == 1u) { sum += uVel[n]; cnt += 1u; } }
      if (iz > 0)     { let n = idx - u32(R * (R + 1)); if (validIn[n] == 1u) { sum += uVel[n]; cnt += 1u; } }
      if (iz < R - 1) { let n = idx + u32(R * (R + 1)); if (validIn[n] == 1u) { sum += uVel[n]; cnt += 1u; } }

      if (cnt > 0u) {
        uVel[idx] = sum / f32(cnt);
        validOut[idx] = 1u;
      } else {
        validOut[idx] = 0u;
      }
    }
  }

  if (idx < vCount) {
    let gIdx = vOffset + idx;
    if (validIn[gIdx] == 1u) {
      validOut[gIdx] = 1u;
    } else {
      let iz = i32(idx / u32((R + 1) * R));
      let ix = i32(idx % u32(R));

      var sum = 0.0;
      var cnt = 0u;
      if (ix > 0)     { let n = idx - 1u;               if (validIn[vOffset + n] == 1u) { sum += vVel[n]; cnt += 1u; } }
      if (ix < R - 1) { let n = idx + 1u;               if (validIn[vOffset + n] == 1u) { sum += vVel[n]; cnt += 1u; } }
      if (iz > 0)     { let n = idx - u32((R + 1) * R); if (validIn[vOffset + n] == 1u) { sum += vVel[n]; cnt += 1u; } }
      if (iz < R - 1) { let n = idx + u32((R + 1) * R); if (validIn[vOffset + n] == 1u) { sum += vVel[n]; cnt += 1u; } }

      if (cnt > 0u) {
        vVel[idx] = sum / f32(cnt);
        validOut[gIdx] = 1u;
      } else {
        validOut[gIdx] = 0u;
      }
    }
  }

  if (idx < wCount) {
    let gIdx = wOffset + idx;
    if (validIn[gIdx] == 1u) {
      validOut[gIdx] = 1u;
    } else {
      let ix = i32(idx % u32(R));
      let iy = i32((idx / u32(R)) % u32(R));

      var sum = 0.0;
      var cnt = 0u;
      if (ix > 0)     { let n = idx - 1u;    if (validIn[wOffset + n] == 1u) { sum += wVel[n]; cnt += 1u; } }
      if (ix < R - 1) { let n = idx + 1u;    if (validIn[wOffset + n] == 1u) { sum += wVel[n]; cnt += 1u; } }
      if (iy > 0)     { let n = idx - u32(R); if (validIn[wOffset + n] == 1u) { sum += wVel[n]; cnt += 1u; } }
      if (iy < R - 1) { let n = idx + u32(R); if (validIn[wOffset + n] == 1u) { sum += wVel[n]; cnt += 1u; } }

      if (cnt > 0u) {
        wVel[idx] = sum / f32(cnt);
        validOut[gIdx] = 1u;
      } else {
        validOut[gIdx] = 0u;
      }
    }
  }
}
