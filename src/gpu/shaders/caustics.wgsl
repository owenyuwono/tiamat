struct CausticsParams {
  domainMin: vec3<f32>,
  threshold: f32,
  domainMax: vec3<f32>,
  strength: f32,
}

@group(0) @binding(0) var densityTex: texture_3d<f32>;
@group(0) @binding(1) var densitySamp: sampler;
@group(0) @binding(2) var causticsOut: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<uniform> params: CausticsParams;

const CAUSTICS_RES: f32 = 256.0;
const Y_STEPS: i32 = 50;

fn worldToUV(p: vec3<f32>) -> vec3<f32> {
  return (p - params.domainMin) / (params.domainMax - params.domainMin);
}

fn findSurfaceHeight(wx: f32, wz: f32) -> f32 {
  let yTop = params.domainMax.y;
  let yBot = params.domainMin.y;
  let dy = (yTop - yBot) / f32(Y_STEPS);

  for (var i = 0; i < Y_STEPS; i++) {
    let y = yTop - f32(i) * dy;
    let uvw = worldToUV(vec3<f32>(wx, y, wz));
    if uvw.x < 0.0 || uvw.x > 1.0 || uvw.z < 0.0 || uvw.z > 1.0 {
      continue;
    }
    let d = textureSampleLevel(densityTex, densitySamp, uvw, 0.0).r;
    if d > params.threshold {
      return y;
    }
  }
  return yBot;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if gid.x >= u32(CAUSTICS_RES) || gid.y >= u32(CAUSTICS_RES) {
    return;
  }

  let domainSize = params.domainMax - params.domainMin;
  let texelWorld = vec2<f32>(domainSize.x, domainSize.z) / CAUSTICS_RES;

  let u = (f32(gid.x) + 0.5) / CAUSTICS_RES;
  let v = (f32(gid.y) + 0.5) / CAUSTICS_RES;
  let wx = params.domainMin.x + u * domainSize.x;
  let wz = params.domainMin.z + v * domainSize.z;

  let hC = findSurfaceHeight(wx, wz);
  let hL = findSurfaceHeight(wx - texelWorld.x, wz);
  let hR = findSurfaceHeight(wx + texelWorld.x, wz);
  let hF = findSurfaceHeight(wx, wz - texelWorld.y);
  let hB = findSurfaceHeight(wx, wz + texelWorld.y);

  let laplacian = hL + hR + hF + hB - 4.0 * hC;
  let intensity = clamp(1.0 + params.strength * laplacian, 0.0, 3.0);

  textureStore(causticsOut, vec2<u32>(gid.x, gid.y), vec4<f32>(intensity, 0.0, 0.0, 0.0));
}
