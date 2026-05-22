struct SprayRenderParams {
  viewProjection: mat4x4<f32>,
  cameraRight: vec3<f32>,
  pointSize: f32,
  cameraUp: vec3<f32>,
  _pad: f32,
}

@group(0) @binding(0) var<uniform> params: SprayRenderParams;
@group(0) @binding(1) var<storage, read> sprayParticles: array<vec4<f32>>;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) alpha: f32,
  @location(1) uv: vec2<f32>,
}

@vertex
fn vs_main(
  @builtin(vertex_index) vertexIdx: u32,
  @builtin(instance_index) instanceIdx: u32,
) -> VSOut {
  let posLife = sprayParticles[instanceIdx * 2u];
  var out: VSOut;

  let isBubble = sprayParticles[instanceIdx * 2u + 1u].w > 0.5;
  if (posLife.w <= 0.0 || isBubble) {
    out.position = vec4<f32>(0.0, 0.0, -2.0, 1.0);
    out.alpha = 0.0;
    out.uv = vec2<f32>(0.0);
    return out;
  }

  let offsets = array<vec2<f32>, 4>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, 1.0),
  );
  let uv = offsets[vertexIdx];

  let worldPos = posLife.xyz
    + params.cameraRight * uv.x * params.pointSize
    + params.cameraUp * uv.y * params.pointSize;

  out.position = params.viewProjection * vec4<f32>(worldPos, 1.0);
  out.alpha = smoothstep(0.0, 0.8, posLife.w);
  out.uv = uv * 0.5 + 0.5;
  return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let dist = length(in.uv - vec2<f32>(0.5));
  let soft = exp(-dist * dist * 8.0);
  return vec4<f32>(0.9, 0.95, 1.0, soft * in.alpha * 0.4);
}
