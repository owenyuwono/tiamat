struct FloorParams {
  viewProjection: mat4x4<f32>,
  lightDir: vec3<f32>,
  time: f32,
  domainMinXZ: vec2<f32>,
  domainSizeXZ: vec2<f32>,
}

@group(0) @binding(0) var<uniform> params: FloorParams;
@group(0) @binding(1) var sandTex: texture_2d<f32>;
@group(0) @binding(2) var sandSampler: sampler;
@group(0) @binding(3) var causticsTex: texture_2d<f32>;
@group(0) @binding(4) var causticsSampler: sampler;
@group(0) @binding(5) var noiseTex: texture_2d<f32>;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) normal: vec3<f32>,
}

@vertex
fn vs_main(
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
) -> VertexOutput {
  var out: VertexOutput;
  out.position = params.viewProjection * vec4<f32>(pos, 1.0);
  out.worldPos = pos;
  out.normal = normal;
  return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let uv = in.worldPos.xz * 0.5;
  let sandColor = textureSampleLevel(sandTex, sandSampler, uv, 0.0).rgb;
  let sideColor = vec3<f32>(0.72, 0.65, 0.48);
  let isTop = in.normal.y > 0.5;
  let baseColor = select(sideColor, sandColor, isTop);

  let L = normalize(params.lightDir);
  let NdotL = max(dot(in.normal, L), 0.0);
  var color = baseColor * (0.6 + 0.4 * NdotL);

  if isTop {
    let causticsUV = (in.worldPos.xz - params.domainMinXZ) / params.domainSizeXZ;
    let causticPhys = textureSampleLevel(causticsTex, causticsSampler, causticsUV, 0.0).r;

    let t = params.time;
    let noiseUV1 = in.worldPos.xz * 1.5 + vec2<f32>(t * 0.03, t * 0.02);
    let noiseUV2 = in.worldPos.xz * 2.3 + vec2<f32>(-t * 0.025, t * 0.035);
    let n1 = textureSampleLevel(noiseTex, sandSampler, noiseUV1, 0.0).r;
    let n2 = textureSampleLevel(noiseTex, sandSampler, noiseUV2, 0.0).r;
    let shimmer = (n1 + n2) * 0.5;

    let causticFinal = causticPhys * (0.7 + 0.3 * shimmer);
    color *= causticFinal;
  }

  return vec4<f32>(color, 1.0);
}
