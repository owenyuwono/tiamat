struct RenderParams {
  invViewProjection: mat4x4<f32>,
  domainMin: vec3<f32>,
  threshold: f32,
  domainMax: vec3<f32>,
  _pad0: f32,
  lightDir: vec3<f32>,
  _pad1: f32,
  lightColor: vec3<f32>,
  _pad2: f32,
  bgColor: vec3<f32>,
  _pad3: f32,
  resolution: vec2<f32>,
  _pad4: vec2<f32>,
  cameraPosition: vec3<f32>,
  _pad5: f32,
}

@group(0) @binding(0) var<uniform> params: RenderParams;
@group(0) @binding(1) var densityTex: texture_3d<f32>;
@group(0) @binding(2) var densitySampler: sampler;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
  let x = f32(i32(vi) / 2) * 4.0 - 1.0;
  let y = f32(i32(vi) % 2) * 4.0 - 1.0;
  var out: VertexOutput;
  out.position = vec4<f32>(x, y, 0.0, 1.0);
  return out;
}

fn worldToUV(p: vec3<f32>) -> vec3<f32> {
  return (p - params.domainMin) / (params.domainMax - params.domainMin);
}

fn sampleField(p: vec3<f32>) -> vec2<f32> {
  return textureSampleLevel(densityTex, densitySampler, worldToUV(p), 0.0).rg;
}

fn sampleDensity(p: vec3<f32>) -> f32 {
  return sampleField(p).r;
}

fn intersectAABB(ro: vec3<f32>, rd: vec3<f32>) -> vec2<f32> {
  let invDir = 1.0 / rd;
  let t0 = (params.domainMin - ro) * invDir;
  let t1 = (params.domainMax - ro) * invDir;
  let tmin = min(t0, t1);
  let tmax = max(t0, t1);
  let tNear = max(max(tmin.x, tmin.y), tmin.z);
  let tFar = min(min(tmax.x, tmax.y), tmax.z);
  return vec2<f32>(tNear, tFar);
}

fn estimateNormalRaw(p: vec3<f32>) -> vec3<f32> {
  let e = 0.02;
  return
    vec3<f32>(1.0, -1.0, -1.0) * sampleDensity(p + vec3<f32>(e, -e, -e)) +
    vec3<f32>(-1.0, -1.0, 1.0) * sampleDensity(p + vec3<f32>(-e, -e, e)) +
    vec3<f32>(-1.0, 1.0, -1.0) * sampleDensity(p + vec3<f32>(-e, e, -e)) +
    vec3<f32>(1.0, 1.0, 1.0)   * sampleDensity(p + vec3<f32>(e, e, e));
}

@fragment
fn fs_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
  let ro = params.cameraPosition;
  let ndcRaw = (fragCoord.xy / params.resolution) * 2.0 - 1.0;
  let ndc = vec2<f32>(ndcRaw.x, -ndcRaw.y);
  let farClip = params.invViewProjection * vec4<f32>(ndc, 1.0, 1.0);
  let rd = normalize(farClip.xyz / farClip.w - ro);

  let tHit = intersectAABB(ro, rd);
  if (tHit.x > tHit.y) {
    discard;
  }

  let tNear = max(tHit.x, 0.0);
  let tFar = tHit.y;

  let stepSize = 0.025;
  var t = tNear + stepSize * 0.5;
  var hit = false;
  var hitPos = vec3<f32>(0.0);

  for (var i = 0; i < 400; i++) {
    if (t > tFar) { break; }
    let p = ro + rd * t;
    let d = sampleDensity(p);

    if (d > params.threshold) {
      var tLo = t - stepSize;
      var tHi = t;
      for (var j = 0; j < 6; j++) {
        let tMid = (tLo + tHi) * 0.5;
        if (sampleDensity(ro + rd * tMid) > params.threshold) {
          tHi = tMid;
        } else {
          tLo = tMid;
        }
      }
      hitPos = ro + rd * tHi;
      hit = true;
      break;
    }
    t += stepSize;
  }

  if (!hit) {
    discard;
  }

  let grad = estimateNormalRaw(hitPos);
  let gradLen = length(grad);
  let N = grad / max(gradLen, 0.001);
  let V = -rd;
  let L = normalize(params.lightDir);
  let H = normalize(L + V);

  let rg = sampleField(hitPos);
  var avgImpact = 0.0;
  if (rg.r > 0.001) {
    avgImpact = rg.g / rg.r;
  }
  let impactFoam = smoothstep(0.2, 1.5, avgImpact);
  let topSurface = smoothstep(-0.1, 0.5, N.y);
  let foam = clamp(impactFoam * topSurface, 0.0, 1.0);

  let cosTheta = max(dot(N, V), 0.0);
  let fresnel = 0.02 + 0.98 * pow(1.0 - cosTheta, 5.0);

  var depth = 0.0;
  var tD = t;
  for (var i = 0; i < 16; i++) {
    tD += stepSize * 2.0;
    if (tD > tFar) { depth = tD - t; break; }
    if (sampleDensity(ro + rd * tD) < params.threshold) {
      depth = tD - t;
      break;
    }
  }
  if (depth == 0.0) { depth = tD - t; }
  let depthFactor = 1.0 - exp(-depth * 3.0);

  let deepColor = vec3<f32>(0.02, 0.12, 0.42);
  let shallowColor = vec3<f32>(0.1, 0.4, 0.9);
  var waterColor = mix(shallowColor, deepColor, depthFactor);
  waterColor *= vec3<f32>(0.8, 0.88, 1.0);
  waterColor = mix(waterColor, vec3<f32>(1.0), foam);

  let NdotL = max(dot(N, L), 0.0);
  let spec = pow(max(dot(N, H), 0.0), mix(128.0, 16.0, foam));

  let ambient = waterColor * mix(0.5, 0.9, foam);
  let diffuse = waterColor * NdotL * 0.6;
  let specular = params.lightColor * spec * mix(2.0, 0.5, foam);
  var color = ambient + diffuse + specular;

  let reflColor = mix(mix(params.bgColor, vec3<f32>(0.3, 0.5, 0.9), 0.4), params.bgColor, foam);
  color = mix(color, reflColor, fresnel * 0.4);

  let alpha = clamp(depthFactor * 0.85 + 0.15 + foam, 0.0, 1.0);

  return vec4<f32>(color, alpha);
}
