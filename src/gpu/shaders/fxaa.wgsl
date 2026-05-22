struct FxaaUniforms {
  texelSize: vec2<f32>,
}

@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var inputSamp: sampler;
@group(0) @binding(2) var<uniform> uniforms: FxaaUniforms;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
  let x = f32(i32(vi) / 2) * 4.0 - 1.0;
  let y = f32(i32(vi) % 2) * 4.0 - 1.0;
  var out: VertexOutput;
  out.position = vec4<f32>(x, y, 0.0, 1.0);
  out.uv = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
  return out;
}

const EDGE_THRESHOLD: f32 = 0.125;
const EDGE_THRESHOLD_MIN: f32 = 0.0312;
const SUBPIX_QUALITY: f32 = 0.75;

fn luma(c: vec3<f32>) -> f32 {
  return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

fn samp(uv: vec2<f32>) -> vec4<f32> {
  return textureSampleLevel(inputTex, inputSamp, uv, 0.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let uv = in.uv;
  let tx = uniforms.texelSize;

  let rgbM = samp(uv).rgb;
  let lumaM = luma(rgbM);

  let lumaN = luma(samp(uv + vec2<f32>(0.0, -tx.y)).rgb);
  let lumaS = luma(samp(uv + vec2<f32>(0.0, tx.y)).rgb);
  let lumaE = luma(samp(uv + vec2<f32>(tx.x, 0.0)).rgb);
  let lumaW = luma(samp(uv + vec2<f32>(-tx.x, 0.0)).rgb);

  let lumaMin = min(lumaM, min(min(lumaN, lumaS), min(lumaE, lumaW)));
  let lumaMax = max(lumaM, max(max(lumaN, lumaS), max(lumaE, lumaW)));
  let lumaRange = lumaMax - lumaMin;

  if lumaRange < max(EDGE_THRESHOLD_MIN, lumaMax * EDGE_THRESHOLD) {
    return vec4<f32>(rgbM, 1.0);
  }

  let lumaNE = luma(samp(uv + vec2<f32>(tx.x, -tx.y)).rgb);
  let lumaSW = luma(samp(uv + vec2<f32>(-tx.x, tx.y)).rgb);
  let lumaNW = luma(samp(uv + vec2<f32>(-tx.x, -tx.y)).rgb);
  let lumaSE = luma(samp(uv + vec2<f32>(tx.x, tx.y)).rgb);

  let lumaNS = lumaN + lumaS;
  let lumaEW = lumaE + lumaW;
  let lumaNESE = lumaNE + lumaSE;
  let lumaNWNE = lumaNW + lumaNE;
  let lumaNWSW = lumaNW + lumaSW;
  let lumaSWSE = lumaSW + lumaSE;

  let lumaAvg = (lumaNS + lumaEW) * 0.25;
  let subpixA = abs(lumaAvg - lumaM);
  let subpixB = clamp(subpixA / lumaRange, 0.0, 1.0);
  let subpixC = (-2.0 * subpixB + 3.0) * subpixB * subpixB;
  let subpixOffset = subpixC * subpixC * SUBPIX_QUALITY;

  let edgeH = abs(lumaNWSW - 2.0 * lumaW) + abs(lumaNS - 2.0 * lumaM) * 2.0 + abs(lumaNESE - 2.0 * lumaE);
  let edgeV = abs(lumaNWNE - 2.0 * lumaN) + abs(lumaEW - 2.0 * lumaM) * 2.0 + abs(lumaSWSE - 2.0 * lumaS);
  let isHorizontal = edgeH >= edgeV;

  var stepLength: f32;
  var lumaPos: f32;
  var lumaNeg: f32;
  if isHorizontal {
    stepLength = tx.y;
    lumaPos = lumaS;
    lumaNeg = lumaN;
  } else {
    stepLength = tx.x;
    lumaPos = lumaE;
    lumaNeg = lumaW;
  }

  let gradPos = abs(lumaPos - lumaM);
  let gradNeg = abs(lumaNeg - lumaM);

  var pStep: f32;
  var gradNearest: f32;
  if gradPos >= gradNeg {
    pStep = stepLength;
    gradNearest = gradPos;
  } else {
    pStep = -stepLength;
    gradNearest = gradNeg;
  }

  var uvEdge = uv;
  if isHorizontal {
    uvEdge.y += pStep * 0.5;
  } else {
    uvEdge.x += pStep * 0.5;
  }

  let gradThreshold = gradNearest * 0.25;
  let lumaEdge = (lumaM + select(lumaNeg, lumaPos, gradPos >= gradNeg)) * 0.5;

  var uvPos = uvEdge;
  var uvNegDir = uvEdge;
  var lumaEndPos: f32;
  var lumaEndNeg: f32;
  var donePos = false;
  var doneNeg = false;

  let quality = array<f32, 12>(1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 4.0, 8.0);

  for (var i = 0u; i < 12u; i++) {
    let q = quality[i];
    if !donePos {
      if isHorizontal {
        uvPos.x += tx.x * q;
      } else {
        uvPos.y += tx.y * q;
      }
      lumaEndPos = luma(samp(uvPos).rgb) - lumaEdge;
      donePos = abs(lumaEndPos) >= gradThreshold;
    }
    if !doneNeg {
      if isHorizontal {
        uvNegDir.x -= tx.x * q;
      } else {
        uvNegDir.y -= tx.y * q;
      }
      lumaEndNeg = luma(samp(uvNegDir).rgb) - lumaEdge;
      doneNeg = abs(lumaEndNeg) >= gradThreshold;
    }
    if donePos && doneNeg { break; }
  }

  var distPos: f32;
  var distNeg: f32;
  if isHorizontal {
    distPos = uvPos.x - uv.x;
    distNeg = uv.x - uvNegDir.x;
  } else {
    distPos = uvPos.y - uv.y;
    distNeg = uv.y - uvNegDir.y;
  }

  let distMin = min(distPos, distNeg);
  let spanLength = distPos + distNeg;
  let edgeOffset = -distMin / spanLength + 0.5;

  let lumaEndNearest = select(lumaEndNeg, lumaEndPos, distPos < distNeg);
  let goodSpan = (lumaEndNearest < 0.0) != (lumaM - lumaEdge < 0.0);

  let pixelOffset = select(0.0, edgeOffset, goodSpan);
  let finalOffset = max(pixelOffset, subpixOffset);

  var finalUv = uv;
  if isHorizontal {
    finalUv.y += finalOffset * pStep;
  } else {
    finalUv.x += finalOffset * pStep;
  }

  return vec4<f32>(samp(finalUv).rgb, 1.0);
}
