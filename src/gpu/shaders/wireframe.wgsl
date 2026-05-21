struct WireUniforms {
  viewProjection: mat4x4<f32>,
  color: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: WireUniforms;

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
  return uniforms.viewProjection * vec4<f32>(position, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
  return uniforms.color;
}
