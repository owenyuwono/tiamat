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
@group(0) @binding(3) var sandTex: texture_2d<f32>;
@group(0) @binding(4) var sandSamp: sampler;
@group(0) @binding(5) var foamNoiseTex: texture_2d<f32>;
@group(0) @binding(6) var foamNoiseSamp: sampler;

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
  return vec3<f32>(
    sampleDensity(p + vec3<f32>(e, 0.0, 0.0)) - sampleDensity(p - vec3<f32>(e, 0.0, 0.0)),
    sampleDensity(p + vec3<f32>(0.0, e, 0.0)) - sampleDensity(p - vec3<f32>(0.0, e, 0.0)),
    sampleDensity(p + vec3<f32>(0.0, 0.0, e)) - sampleDensity(p - vec3<f32>(0.0, 0.0, e))
  );
}

fn distributionGGX(NdotH: f32, roughness: f32) -> f32 {
  let a = roughness * roughness;
  let a2 = a * a;
  let d = NdotH * NdotH * (a2 - 1.0) + 1.0;
  return a2 / (3.14159265 * d * d);
}

fn geometrySchlickGGX(NdotX: f32, roughness: f32) -> f32 {
  let r = roughness + 1.0;
  let k = (r * r) / 8.0;
  return NdotX / (NdotX * (1.0 - k) + k);
}

fn geometrySmith(NdotV: f32, NdotL: f32, roughness: f32) -> f32 {
  return geometrySchlickGGX(NdotV, roughness) * geometrySchlickGGX(NdotL, roughness);
}

fn shadowMarch(origin: vec3<f32>, lightDir: vec3<f32>, maxDist: f32) -> f32 {
  var accumDensity = 0.0;
  let steps = 16;
  let shadowStepSize = maxDist / f32(steps);
  for (var i = 1; i <= steps; i++) {
    let p = origin + lightDir * (f32(i) * shadowStepSize);
    let uv = worldToUV(p);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || uv.z < 0.0 || uv.z > 1.0) {
      break;
    }
    let d = sampleDensity(p);
    accumDensity += max(d - params.threshold, 0.0);
  }
  return exp(-accumDensity * 3.0 * shadowStepSize);
}

fn sampleAO(origin: vec3<f32>, normal: vec3<f32>) -> f32 {
  var occlusion = 0.0;
  let offsets = array<f32, 4>(0.04, 0.08, 0.16, 0.32);
  let weights = array<f32, 4>(0.4, 0.3, 0.2, 0.1);
  for (var i = 0u; i < 4u; i++) {
    let p = origin + normal * offsets[i];
    let d = sampleDensity(p);
    occlusion += weights[i] * smoothstep(0.0, params.threshold * 2.0, d);
  }
  return 1.0 - occlusion * 0.6;
}

fn acesFilm(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
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
  let impactFoam = smoothstep(0.002, 0.1, avgImpact);
  let topSurface = smoothstep(-0.2, 0.3, N.y);
  let curvatureFoam = smoothstep(0.3, 0.8, gradLen / max(sampleDensity(hitPos), 0.01));

  // Spray: thin fluid edges just above threshold with steep gradient
  let thinness = 1.0 - smoothstep(params.threshold, params.threshold * 2.5, rg.r);
  let spray = thinness * smoothstep(0.1, 0.5, gradLen) * topSurface;

  let rawFoam = max(max(impactFoam, curvatureFoam * 0.4), spray * 0.7) * topSurface;
  let foamUV = hitPos.xz * 2.5;
  let foamNoise = textureSampleLevel(foamNoiseTex, foamNoiseSamp, foamUV, 0.0).r;
  let foamDetail = mix(0.3, 1.0, foamNoise);
  let foam = clamp(rawFoam * foamDetail, 0.0, 1.0);

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
  let depthFactor = 1.0 - exp(-depth * 2.5);

  // Refraction — sample sand floor through water with Snell's law distortion
  let refrDir = refract(rd, N, 1.0 / 1.33);
  var floorCol = params.bgColor;
  if (refrDir.y < -0.001) {
    let tFloor = -hitPos.y / refrDir.y;
    let floorPos = hitPos + refrDir * tFloor;
    let floorUV = floorPos.xz * 0.5;
    let sandRGB = textureSampleLevel(sandTex, sandSamp, floorUV, 0.0).rgb;
    let floorNdotL = max(dot(vec3<f32>(0.0, 1.0, 0.0), L), 0.0);
    floorCol = sandRGB * (0.55 + 0.45 * floorNdotL);
  }

  let absorption = vec3<f32>(3.0, 1.0, 0.4);
  let transmittance = exp(-absorption * depth);
  var waterColor = transmittance * vec3<f32>(0.2, 0.7, 0.65) + (vec3<f32>(1.0) - transmittance) * vec3<f32>(0.01, 0.15, 0.4);
  waterColor = mix(floorCol, waterColor, depthFactor);
  waterColor = mix(waterColor, vec3<f32>(1.0), foam);

  let shadow = shadowMarch(hitPos + N * 0.03, L, 2.0);
  let ao = sampleAO(hitPos, N);

  let NdotL = max(dot(N, L), 0.001);
  let NdotV = max(dot(N, V), 0.001);
  let NdotH = max(dot(N, H), 0.0);

  let roughness = mix(0.06, 0.5, foam);
  let F0 = mix(0.02, 0.04, foam);
  let D = distributionGGX(NdotH, roughness);
  let G = geometrySmith(NdotV, NdotL, roughness);
  let F = F0 + (1.0 - F0) * pow(1.0 - max(dot(H, V), 0.0), 5.0);
  let specBRDF = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);

  let ambient = waterColor * mix(0.65, 0.9, foam) * ao;
  let diffuse = waterColor * NdotL * 0.6 * shadow;
  let specular = params.lightColor * specBRDF * NdotL * shadow;
  var color = ambient + diffuse + specular;

  // Sky gradient reflection
  let reflDir = reflect(rd, N);
  let skyT = clamp(reflDir.y * 0.5 + 0.5, 0.0, 1.0);
  let horizon = vec3<f32>(0.85, 0.92, 0.97);
  let zenith = vec3<f32>(0.4, 0.7, 0.95);
  let reflColor = mix(horizon, zenith, skyT);
  color = mix(color, reflColor, fresnel * 0.6);

  // Foam edge glow
  let foamGlow = smoothstep(0.0, 0.3, foam) * 0.15;
  color += vec3<f32>(foamGlow);

  let mapped = acesFilm(color);
  let alpha = clamp(depthFactor * 0.75 + 0.3 + foam, 0.0, 1.0);

  return vec4<f32>(mapped, alpha);
}
