struct FloorParams {
  viewProjection: mat4x4<f32>,
  lightDir: vec3<f32>,
  _pad: f32,
}

@group(0) @binding(0) var<uniform> params: FloorParams;
@group(0) @binding(1) var sandTex: texture_2d<f32>;
@group(0) @binding(2) var sandSampler: sampler;

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
  let baseColor = select(sideColor, sandColor, in.normal.y > 0.5);

  let L = normalize(params.lightDir);
  let NdotL = max(dot(in.normal, L), 0.0);
  let color = baseColor * (0.6 + 0.4 * NdotL);

  return vec4<f32>(color, 1.0);
}
