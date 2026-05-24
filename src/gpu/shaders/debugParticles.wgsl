struct Uniforms {
  viewProjection: mat4x4<f32>,
  camRight: vec3<f32>,
  pointSize: f32,
  camUp: vec3<f32>,
  particleCount: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> densityPressure: array<vec2<f32>>;

struct VsOut {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec3<f32>,
  @location(1) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VsOut {
  var out: VsOut;
  if (iid >= uniforms.particleCount) {
    out.position = vec4(0.0, 0.0, -2.0, 1.0);
    return out;
  }

  let pos = positions[iid].xyz;
  let density = densityPressure[iid].x;

  let corners = array<vec2<f32>, 4>(
    vec2(-1.0, -1.0),
    vec2(1.0, -1.0),
    vec2(-1.0, 1.0),
    vec2(1.0, 1.0),
  );
  let c = corners[vid];
  out.uv = c;

  let worldPos = pos + (uniforms.camRight * c.x + uniforms.camUp * c.y) * uniforms.pointSize;
  out.position = uniforms.viewProjection * vec4(worldPos, 1.0);

  let restDensity = 1000.0;
  let ratio = clamp(density / restDensity, 0.0, 2.0);
  if (ratio < 1.0) {
    out.color = mix(vec3(0.2, 0.2, 1.0), vec3(0.2, 1.0, 0.2), ratio);
  } else {
    out.color = mix(vec3(0.2, 1.0, 0.2), vec3(1.0, 0.2, 0.2), ratio - 1.0);
  }

  return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  let d = dot(in.uv, in.uv);
  if (d > 1.0) { discard; }
  return vec4(in.color, 1.0);
}
