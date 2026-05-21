precision highp float;
precision highp sampler3D;

uniform sampler3D uDensityTex;
uniform vec3 uDomainMin;
uniform vec3 uDomainMax;
uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform float uThreshold;
uniform vec3 uBgColor;
uniform mat4 uInvViewProjection;
uniform vec2 uResolution;

vec3 worldToUV(vec3 p) {
  return (p - uDomainMin) / (uDomainMax - uDomainMin);
}

vec2 sampleField(vec3 p) {
  return texture(uDensityTex, worldToUV(p)).rg;
}

float sampleDensity(vec3 p) {
  return sampleField(p).r;
}

vec2 intersectAABB(vec3 ro, vec3 rd) {
  vec3 invDir = 1.0 / rd;
  vec3 t0 = (uDomainMin - ro) * invDir;
  vec3 t1 = (uDomainMax - ro) * invDir;
  vec3 tmin = min(t0, t1);
  vec3 tmax = max(t0, t1);
  float tNear = max(max(tmin.x, tmin.y), tmin.z);
  float tFar = min(min(tmax.x, tmax.y), tmax.z);
  return vec2(tNear, tFar);
}

vec3 estimateNormalRaw(vec3 p) {
  float e = 0.02;
  return vec3(
    sampleDensity(p + vec3(e, 0, 0)) - sampleDensity(p - vec3(e, 0, 0)),
    sampleDensity(p + vec3(0, e, 0)) - sampleDensity(p - vec3(0, e, 0)),
    sampleDensity(p + vec3(0, 0, e)) - sampleDensity(p - vec3(0, 0, e))
  );
}

void main() {
  vec3 ro = cameraPosition;
  vec2 ndc = (gl_FragCoord.xy / uResolution) * 2.0 - 1.0;
  vec4 farClip = uInvViewProjection * vec4(ndc, 1.0, 1.0);
  vec3 rd = normalize(farClip.xyz / farClip.w - ro);

  vec2 tHit = intersectAABB(ro, rd);
  if (tHit.x > tHit.y) discard;

  float tNear = max(tHit.x, 0.0);
  float tFar = tHit.y;

  float stepSize = 0.025;
  float t = tNear + stepSize * 0.5;
  bool hit = false;
  vec3 hitPos;

  for (int i = 0; i < 400; i++) {
    if (t > tFar) break;
    vec3 p = ro + rd * t;
    float d = sampleDensity(p);

    if (d > uThreshold) {
      float tLo = t - stepSize;
      float tHi = t;
      for (int j = 0; j < 6; j++) {
        float tMid = (tLo + tHi) * 0.5;
        if (sampleDensity(ro + rd * tMid) > uThreshold) tHi = tMid;
        else tLo = tMid;
      }
      hitPos = ro + rd * tHi;
      hit = true;
      break;
    }
    t += stepSize;
  }

  if (!hit) discard;

  // Normal + gradient magnitude (reuse gradient for foam)
  vec3 grad = estimateNormalRaw(hitPos);
  float gradLen = length(grad);
  vec3 N = grad / max(gradLen, 0.001);
  vec3 V = -rd;
  vec3 L = normalize(uLightDir);
  vec3 H = normalize(L + V);

  // Impact foam from XSPH divergence (high when particles collide, zero during free fall)
  vec2 rg = sampleField(hitPos);
  float avgImpact = rg.r > 0.001 ? rg.g / rg.r : 0.0;
  float impactFoam = smoothstep(0.2, 1.5, avgImpact);

  // Restrict foam to upward-facing surfaces (water top, not side walls or bottom)
  float topSurface = smoothstep(-0.1, 0.5, N.y);

  float foam = clamp(impactFoam * topSurface, 0.0, 1.0);

  // Fresnel (Schlick, water F0 ~ 0.02)
  float cosTheta = max(dot(N, V), 0.0);
  float fresnel = 0.02 + 0.98 * pow(1.0 - cosTheta, 5.0);

  // Depth estimation via secondary march
  float depth = 0.0;
  float tD = t;
  for (int i = 0; i < 32; i++) {
    tD += stepSize * 2.0;
    if (tD > tFar) { depth = tD - t; break; }
    if (sampleDensity(ro + rd * tD) < uThreshold) {
      depth = tD - t;
      break;
    }
  }
  float depthFactor = 1.0 - exp(-depth * 3.0);

  vec3 deepColor = vec3(0.02, 0.12, 0.42);
  vec3 shallowColor = vec3(0.1, 0.4, 0.9);
  vec3 waterColor = mix(shallowColor, deepColor, depthFactor);

  // Blue absorption — water absorbs red/green more than blue
  waterColor *= vec3(0.8, 0.88, 1.0);

  // Blend foam into water color — bright white foam
  vec3 foamColor = vec3(1.0);
  waterColor = mix(waterColor, foamColor, foam);

  // Lighting
  float NdotL = max(dot(N, L), 0.0);
  float spec = pow(max(dot(N, H), 0.0), mix(128.0, 16.0, foam));

  vec3 ambient = waterColor * mix(0.5, 0.9, foam);
  vec3 diffuse = waterColor * NdotL * 0.6;
  vec3 specular = uLightColor * spec * mix(2.0, 0.5, foam);
  vec3 color = ambient + diffuse + specular;

  // Blue-tinted fresnel for water, neutral for foam
  vec3 reflColor = mix(mix(uBgColor, vec3(0.3, 0.5, 0.9), 0.4), uBgColor, foam);
  color = mix(color, reflColor, fresnel * 0.4);

  float alpha = clamp(depthFactor * 0.85 + 0.15 + foam, 0.0, 1.0);

  gl_FragColor = vec4(color, alpha);
}
