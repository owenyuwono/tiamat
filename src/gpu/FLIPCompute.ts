import * as THREE from 'three';
import { SPH } from '../sph/constants';
import type { SimConfig } from '../ui/SimConfig';
import type { GPUProfiler } from './GPUProfiler';

import flipP2GShader from './shaders/flipP2G.wgsl?raw';
import flipNormalizeAShader from './shaders/flipNormalizeA.wgsl?raw';
import flipNormalizeBShader from './shaders/flipNormalizeB.wgsl?raw';
import flipDivergenceShader from './shaders/flipDivergence.wgsl?raw';
import flipJacobiShader from './shaders/flipJacobi.wgsl?raw';
import flipProjectShader from './shaders/flipProject.wgsl?raw';
import flipProjectStaggeredShader from './shaders/flipProjectStaggered.wgsl?raw';
import flipG2PShader from './shaders/flipG2P.wgsl?raw';
import clearDensityFieldShader from './shaders/clearDensityField.wgsl?raw';
import flipSplatDensityShader from './shaders/flipSplatDensity.wgsl?raw';

const JACOBI_ITERATIONS = 40;
const FLIP_MAX_SUBSTEPS = 2;
const MAX_PER_CELL = 32;

function nextPowerOfTwo(n: number): number {
  let v = n - 1;
  v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
  return v + 1;
}

export class FLIPCompute {
  private device: GPUDevice;
  private particleCount: number;
  private fieldResolution: number;
  private fieldSize: number;

  private positionsBuffer: GPUBuffer;
  private velocitiesBuffer: GPUBuffer;
  private densityFieldBuffer: GPUBuffer;
  private paramsBuffer: GPUBuffer;

  // Staggered MAC accumulators (P2G)
  private accumU: GPUBuffer;
  private accumV: GPUBuffer;
  private accumW: GPUBuffer;
  private accumWeightU: GPUBuffer;
  private accumWeightV: GPUBuffer;
  private accumWeightW: GPUBuffer;

  // Staggered MAC grid face velocities (proper FLIP)
  private uVelBuffer: GPUBuffer;      // x-faces: (R+1) x R x R
  private vVelBuffer: GPUBuffer;      // y-faces: R x (R+1) x R
  private wVelBuffer: GPUBuffer;      // z-faces: R x R x (R+1)
  private uOldVelBuffer: GPUBuffer;
  private vOldVelBuffer: GPUBuffer;
  private wOldVelBuffer: GPUBuffer;

  private gridVelBuffer: GPUBuffer;

  private pressureBuffer: GPUBuffer;
  private pressureAltBuffer: GPUBuffer;
  private divergenceBuffer: GPUBuffer;

  private flipP2GPipeline: GPUComputePipeline;
  private flipP2GBindGroup: GPUBindGroup;
  private flipNormalizeAPipeline: GPUComputePipeline;
  private flipNormalizeABindGroup: GPUBindGroup;
  private flipNormalizeBPipeline: GPUComputePipeline;
  private flipNormalizeBBindGroup: GPUBindGroup;
  private flipDivergencePipeline: GPUComputePipeline;
  private flipDivergenceBindGroup: GPUBindGroup;
  private flipJacobiPipeline: GPUComputePipeline;
  private flipJacobiBindGroupA: GPUBindGroup;
  private flipJacobiBindGroupB: GPUBindGroup;
  private flipProjectPipeline: GPUComputePipeline;
  private flipProjectBindGroupA: GPUBindGroup;
  private flipProjectBindGroupB: GPUBindGroup;
  private flipProjectStaggeredPipeline: GPUComputePipeline;
  private flipProjectStaggeredBindGroup: GPUBindGroup;
  private flipG2PPipeline: GPUComputePipeline;
  private flipG2PBindGroup: GPUBindGroup;
  private clearDensityFieldPipeline: GPUComputePipeline;
  private clearDensityFieldBindGroup: GPUBindGroup;
  private splatDensityPipeline: GPUComputePipeline;
  private splatDensityBindGroup: GPUBindGroup;

  private paramsArrayBuffer: ArrayBuffer;
  private paramsF32: Float32Array;
  private paramsU32: Uint32Array;

  constructor(
    device: GPUDevice,
    particleCount: number,
    containerSize: THREE.Vector3,
    fieldResolution: number,
    domainMin: THREE.Vector3,
    domainMax: THREE.Vector3,
    splatRadius: number,
  ) {
    this.device = device;
    this.particleCount = particleCount;
    this.fieldResolution = fieldResolution;
    this.fieldSize = fieldResolution * fieldResolution * fieldResolution * 2 * 4;

    const N = particleCount;
    const res = fieldResolution;
    const res3 = res * res * res;

    this.positionsBuffer = device.createBuffer({
      size: N * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.velocitiesBuffer = device.createBuffer({
      size: N * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.densityFieldBuffer = device.createBuffer({
      size: this.fieldSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // 6 staggered MAC P2G accumulators (face-sized)
    const uCount = (res + 1) * res * res;
    const vCount = res * (res + 1) * res;
    const wCount = res * res * (res + 1);
    const SCD = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.accumU = device.createBuffer({ size: uCount * 4, usage: SCD });
    this.accumV = device.createBuffer({ size: vCount * 4, usage: SCD });
    this.accumW = device.createBuffer({ size: wCount * 4, usage: SCD });
    this.accumWeightU = device.createBuffer({ size: uCount * 4, usage: SCD });
    this.accumWeightV = device.createBuffer({ size: vCount * 4, usage: SCD });
    this.accumWeightW = device.createBuffer({ size: wCount * 4, usage: SCD });

    // Staggered MAC face velocity buffers (float per face)
    this.uVelBuffer = device.createBuffer({ size: uCount * 4, usage: SCD });
    this.vVelBuffer = device.createBuffer({ size: vCount * 4, usage: SCD });
    this.wVelBuffer = device.createBuffer({ size: wCount * 4, usage: SCD });
    this.uOldVelBuffer = device.createBuffer({ size: uCount * 4, usage: SCD });
    this.vOldVelBuffer = device.createBuffer({ size: vCount * 4, usage: SCD });
    this.wOldVelBuffer = device.createBuffer({ size: wCount * 4, usage: SCD });

    this.gridVelBuffer = device.createBuffer({ size: res3 * 16, usage: SCD });

    this.pressureBuffer = device.createBuffer({ size: res3 * 4, usage: SCD });
    this.pressureAltBuffer = device.createBuffer({ size: res3 * 4, usage: SCD });
    this.divergenceBuffer = device.createBuffer({ size: res3 * 4, usage: SCD });

    this.paramsBuffer = device.createBuffer({
      size: 128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.paramsArrayBuffer = new ArrayBuffer(128);
    this.paramsF32 = new Float32Array(this.paramsArrayBuffer);
    this.paramsU32 = new Uint32Array(this.paramsArrayBuffer);
    const H = SPH.smoothingRadius;
    const H2 = H * H;
    const spacing = H * 0.5;
    const mass = SPH.restDensity * spacing * spacing * spacing;
    const tableSize = nextPowerOfTwo(particleCount * 3);
    const domainSize = domainMax.x - domainMin.x;
    const fieldCellSize = domainSize / fieldResolution;
    const fieldInvCellSize = 1 / fieldCellSize;
    const splatRadius2 = splatRadius * splatRadius;
    const splatRadiusCells = Math.max(2, Math.ceil(splatRadius / fieldCellSize));

    this.paramsU32[0] = particleCount;
    this.paramsU32[1] = tableSize;
    this.paramsU32[2] = MAX_PER_CELL;
    this.paramsU32[3] = fieldResolution;
    this.paramsF32[4] = H;
    this.paramsF32[5] = H2;
    this.paramsF32[6] = mass;
    this.paramsF32[7] = SPH.restDensity;
    this.paramsF32[8] = SPH.stiffness;
    this.paramsF32[9] = SPH.viscosity;
    this.paramsF32[10] = SPH.gravity;
    this.paramsF32[11] = SPH.boundaryDamping;
    this.paramsF32[12] = SPH.maxVelocity;
    this.paramsF32[13] = SPH.collisionRadius;
    this.paramsF32[14] = SPH.collisionStiffness;
    this.paramsF32[15] = SPH.xsphEpsilon;
    this.paramsF32[16] = SPH.surfaceTension;
    this.paramsF32[17] = 0.008;
    this.paramsF32[18] = 315 / (64 * Math.PI * Math.pow(H, 9));
    this.paramsF32[19] = -45 / (Math.PI * Math.pow(H, 6));
    this.paramsF32[20] = 45 / (Math.PI * Math.pow(H, 6));
    this.paramsF32[21] = 1 / H;
    this.paramsF32[22] = containerSize.x / 2;
    this.paramsF32[23] = containerSize.z / 2;
    this.paramsF32[24] = containerSize.y;
    this.paramsF32[25] = domainMin.x;
    this.paramsF32[26] = domainMin.y;
    this.paramsF32[27] = domainMin.z;
    this.paramsF32[28] = fieldCellSize;
    this.paramsF32[29] = fieldInvCellSize;
    this.paramsF32[30] = splatRadius2;
    this.paramsU32[31] = splatRadiusCells;

    device.queue.writeBuffer(this.paramsBuffer, 0, this.paramsArrayBuffer);

    // Zero staggered face velocities
    const zeroU = new Float32Array((res + 1) * res * res);
    const zeroV = new Float32Array(res * (res + 1) * res);
    const zeroW = new Float32Array(res * res * (res + 1));
    device.queue.writeBuffer(this.uVelBuffer, 0, zeroU);
    device.queue.writeBuffer(this.vVelBuffer, 0, zeroV);
    device.queue.writeBuffer(this.wVelBuffer, 0, zeroW);
    device.queue.writeBuffer(this.uOldVelBuffer, 0, zeroU);
    device.queue.writeBuffer(this.vOldVelBuffer, 0, zeroV);
    device.queue.writeBuffer(this.wOldVelBuffer, 0, zeroW);

    // Zero the 6 staggered accumulators
    device.queue.writeBuffer(this.accumU, 0, zeroU);
    device.queue.writeBuffer(this.accumV, 0, zeroV);
    device.queue.writeBuffer(this.accumW, 0, zeroW);
    device.queue.writeBuffer(this.accumWeightU, 0, zeroU);
    device.queue.writeBuffer(this.accumWeightV, 0, zeroV);
    device.queue.writeBuffer(this.accumWeightW, 0, zeroW);

    const zeroGrid = new Float32Array(res3 * 4);
    device.queue.writeBuffer(this.gridVelBuffer, 0, zeroGrid);

    const S = GPUShaderStage.COMPUTE;
    const uniform = (b: number) => ({ binding: b, visibility: S, buffer: { type: 'uniform' as const } });
    const storage = (b: number) => ({ binding: b, visibility: S, buffer: { type: 'storage' as const } });
    const readOnly = (b: number) => ({ binding: b, visibility: S, buffer: { type: 'read-only-storage' as const } });

    // flipP2G: params, pos(r), vel(r), accumU/V/W(rw), weightU/V/W(rw)
    const p2g = this.createPipeline(device, flipP2GShader, [
      uniform(0), readOnly(1), readOnly(2),
      storage(3), storage(4), storage(5),
      storage(6), storage(7), storage(8),
    ], 'flipP2G');
    this.flipP2GPipeline = p2g.pipeline;
    this.flipP2GBindGroup = device.createBindGroup({
      layout: p2g.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.positionsBuffer } },
        { binding: 2, resource: { buffer: this.velocitiesBuffer } },
        { binding: 3, resource: { buffer: this.accumU } },
        { binding: 4, resource: { buffer: this.accumV } },
        { binding: 5, resource: { buffer: this.accumW } },
        { binding: 6, resource: { buffer: this.accumWeightU } },
        { binding: 7, resource: { buffer: this.accumWeightV } },
        { binding: 8, resource: { buffer: this.accumWeightW } },
      ],
    });

    const normalizeA = this.createPipeline(device, flipNormalizeAShader, [
      uniform(0), storage(1), storage(2), storage(3), storage(4),
      storage(5), storage(6), storage(7), storage(8),
    ], 'flipNormalizeA');
    this.flipNormalizeAPipeline = normalizeA.pipeline;
    this.flipNormalizeABindGroup = device.createBindGroup({
      layout: normalizeA.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.accumU } },
        { binding: 2, resource: { buffer: this.accumWeightU } },
        { binding: 3, resource: { buffer: this.accumV } },
        { binding: 4, resource: { buffer: this.accumWeightV } },
        { binding: 5, resource: { buffer: this.uVelBuffer } },
        { binding: 6, resource: { buffer: this.vVelBuffer } },
        { binding: 7, resource: { buffer: this.uOldVelBuffer } },
        { binding: 8, resource: { buffer: this.vOldVelBuffer } },
      ],
    });

    const normalizeB = this.createPipeline(device, flipNormalizeBShader, [
      uniform(0), storage(1), storage(2), storage(3), storage(4),
      readOnly(5), readOnly(6), storage(7),
    ], 'flipNormalizeB');
    this.flipNormalizeBPipeline = normalizeB.pipeline;
    this.flipNormalizeBBindGroup = device.createBindGroup({
      layout: normalizeB.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.accumW } },
        { binding: 2, resource: { buffer: this.accumWeightW } },
        { binding: 3, resource: { buffer: this.wVelBuffer } },
        { binding: 4, resource: { buffer: this.wOldVelBuffer } },
        { binding: 5, resource: { buffer: this.accumWeightU } },
        { binding: 6, resource: { buffer: this.accumWeightV } },
        { binding: 7, resource: { buffer: this.gridVelBuffer } },
      ],
    });

    // flipDivergence: params, uVel(r), vVel(r), wVel(r), divergence(w)
    const divergence = this.createPipeline(device, flipDivergenceShader, [
      uniform(0), readOnly(1), readOnly(2), readOnly(3), storage(4),
    ], 'flipDivergence');
    this.flipDivergencePipeline = divergence.pipeline;
    this.flipDivergenceBindGroup = device.createBindGroup({
      layout: divergence.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.uVelBuffer } },
        { binding: 2, resource: { buffer: this.vVelBuffer } },
        { binding: 3, resource: { buffer: this.wVelBuffer } },
        { binding: 4, resource: { buffer: this.divergenceBuffer } },
      ],
    });

    // flipJacobi: params, gridVel(r), u/v/wVel(r for Neumann BC), pressureIn(r), divergence(r), pressureOut(rw)
    const jacobi = this.createPipeline(device, flipJacobiShader, [
      uniform(0), readOnly(1), readOnly(2), readOnly(3), readOnly(4), readOnly(5), readOnly(6), storage(7),
    ], 'flipJacobi');
    this.flipJacobiPipeline = jacobi.pipeline;
    this.flipJacobiBindGroupA = device.createBindGroup({
      layout: jacobi.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.gridVelBuffer } },
        { binding: 2, resource: { buffer: this.uVelBuffer } },
        { binding: 3, resource: { buffer: this.vVelBuffer } },
        { binding: 4, resource: { buffer: this.wVelBuffer } },
        { binding: 5, resource: { buffer: this.pressureBuffer } },
        { binding: 6, resource: { buffer: this.divergenceBuffer } },
        { binding: 7, resource: { buffer: this.pressureAltBuffer } },
      ],
    });
    this.flipJacobiBindGroupB = device.createBindGroup({
      layout: jacobi.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.gridVelBuffer } },
        { binding: 2, resource: { buffer: this.uVelBuffer } },
        { binding: 3, resource: { buffer: this.vVelBuffer } },
        { binding: 4, resource: { buffer: this.wVelBuffer } },
        { binding: 5, resource: { buffer: this.pressureAltBuffer } },
        { binding: 6, resource: { buffer: this.divergenceBuffer } },
        { binding: 7, resource: { buffer: this.pressureBuffer } },
      ],
    });

    // flipProject: params, gridVel(rw), pressure(r)
    const project = this.createPipeline(device, flipProjectShader, [
      uniform(0), storage(1), readOnly(2),
    ], 'flipProject');
    this.flipProjectPipeline = project.pipeline;
    this.flipProjectBindGroupA = device.createBindGroup({
      layout: project.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.gridVelBuffer } },
        { binding: 2, resource: { buffer: this.pressureBuffer } },
      ],
    });
    this.flipProjectBindGroupB = device.createBindGroup({
      layout: project.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.gridVelBuffer } },
        { binding: 2, resource: { buffer: this.pressureAltBuffer } },
      ],
    });

    // flipProjectStaggered: params, uVel(rw), vVel(rw), wVel(rw), pressure(r)
    const projectStaggered = this.createPipeline(device, flipProjectStaggeredShader, [
      uniform(0), storage(1), storage(2), storage(3), readOnly(4),
    ], 'flipProjectStaggered');
    this.flipProjectStaggeredPipeline = projectStaggered.pipeline;
    this.flipProjectStaggeredBindGroup = device.createBindGroup({
      layout: projectStaggered.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.uVelBuffer } },
        { binding: 2, resource: { buffer: this.vVelBuffer } },
        { binding: 3, resource: { buffer: this.wVelBuffer } },
        { binding: 4, resource: { buffer: this.pressureBuffer } },
      ],
    });

    const g2p = this.createPipeline(device, flipG2PShader, [
      uniform(0), storage(1), storage(2),
      readOnly(3), readOnly(4), readOnly(5), readOnly(6), readOnly(7), readOnly(8),
    ], 'flipG2P');
    this.flipG2PPipeline = g2p.pipeline;
    this.flipG2PBindGroup = device.createBindGroup({
      layout: g2p.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.positionsBuffer } },
        { binding: 2, resource: { buffer: this.velocitiesBuffer } },
        { binding: 3, resource: { buffer: this.uVelBuffer } },
        { binding: 4, resource: { buffer: this.vVelBuffer } },
        { binding: 5, resource: { buffer: this.wVelBuffer } },
        { binding: 6, resource: { buffer: this.uOldVelBuffer } },
        { binding: 7, resource: { buffer: this.vOldVelBuffer } },
        { binding: 8, resource: { buffer: this.wOldVelBuffer } },
      ],
    });

    // clearDensityField (reused)
    const clearDf = this.createPipeline(device, clearDensityFieldShader, [
      uniform(0), storage(1),
    ], 'flipClearDensityField');
    this.clearDensityFieldPipeline = clearDf.pipeline;
    this.clearDensityFieldBindGroup = device.createBindGroup({
      layout: clearDf.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.densityFieldBuffer } },
      ],
    });

    // Jittered splat breaks axis-aligned density grid artifacts in the raymarch.
    const splat = this.createPipeline(device, flipSplatDensityShader, [
      uniform(0), readOnly(1), readOnly(2), storage(3),
    ], 'flipSplatDensity');
    this.splatDensityPipeline = splat.pipeline;
    this.splatDensityBindGroup = device.createBindGroup({
      layout: splat.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.positionsBuffer } },
        { binding: 2, resource: { buffer: this.velocitiesBuffer } },
        { binding: 3, resource: { buffer: this.densityFieldBuffer } },
      ],
    });
  }

  private createPipeline(
    device: GPUDevice,
    shader: string,
    layout: GPUBindGroupLayoutEntry[],
    label: string,
  ): { pipeline: GPUComputePipeline; bindGroupLayout: GPUBindGroupLayout } {
    const bindGroupLayout = device.createBindGroupLayout({ entries: layout, label: `${label}_layout` });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout], label: `${label}_pipelineLayout` });
    const module = device.createShaderModule({ code: shader, label: `${label}_shader` });
    const pipeline = device.createComputePipeline({
      label,
      layout: pipelineLayout,
      compute: { module, entryPoint: 'main' },
    });
    return { pipeline, bindGroupLayout };
  }

  uploadInitialPositions(posX: Float32Array, posY: Float32Array, posZ: Float32Array) {
    // Fully random placement inside the initial block bounds for FLIP.
    // This eliminates any lattice/grid imprint from the start (no cubic lattice at all).
    // Compute AABB from the (lattice) positions passed in, then inset slightly.
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;
    for (let i = 0; i < this.particleCount; i++) {
      if (posX[i] < minX) minX = posX[i];
      if (posX[i] > maxX) maxX = posX[i];
      if (posY[i] < minY) minY = posY[i];
      if (posY[i] > maxY) maxY = posY[i];
      if (posZ[i] < minZ) minZ = posZ[i];
      if (posZ[i] > maxZ) maxZ = posZ[i];
    }
    const margin = (maxX - minX) * 0.02; // small inset so particles stay well inside on first frame
    const sx = minX + margin;
    const ex = maxX - margin;
    const sy = minY + margin * 0.5;
    const ey = maxY;
    const sz = minZ + margin;
    const ez = maxZ - margin;

    const packed = new Float32Array(this.particleCount * 4);
    for (let i = 0; i < this.particleCount; i++) {
      // Per-particle deterministic "random" in [0,1) using large primes.
      const rx = ((i * 73856093) % 100000) / 100000;
      const ry = ((i * 19349663) % 100000) / 100000;
      const rz = ((i * 83492791) % 100000) / 100000;

      const px = sx + rx * (ex - sx);
      const py = sy + ry * (ey - sy);
      const pz = sz + rz * (ez - sz);

      packed[i * 4] = px;
      packed[i * 4 + 1] = py;
      packed[i * 4 + 2] = pz;
    }
    this.device.queue.writeBuffer(this.positionsBuffer, 0, packed);
  }

  getDevice(): GPUDevice { return this.device; }
  getDensityFieldBuffer(): GPUBuffer { return this.densityFieldBuffer; }
  getParamsBuffer(): GPUBuffer { return this.paramsBuffer; }
  getFieldResolution(): number { return this.fieldResolution; }

  encodeStep(encoder: GPUCommandEncoder, substeps: number, profiler?: GPUProfiler | null) {
    const fixedDt = 0.008;
    substeps = Math.min(substeps, FLIP_MAX_SUBSTEPS);
    this.paramsF32[17] = fixedDt;
    this.device.queue.writeBuffer(this.paramsBuffer, 0, this.paramsArrayBuffer);

    const res = this.fieldResolution;
    const res3 = res * res * res;
    const gridWG = Math.ceil(res3 / 256);
    const particleWG = Math.ceil(this.particleCount / 256);
    const fieldElements = res3 * 2;
    const faceWG = Math.ceil(((res + 1) * res * res) / 256);

    for (let s = 0; s < substeps; s++) {
      encoder.clearBuffer(this.uVelBuffer);
      encoder.clearBuffer(this.vVelBuffer);
      encoder.clearBuffer(this.wVelBuffer);
      encoder.clearBuffer(this.uOldVelBuffer);
      encoder.clearBuffer(this.vOldVelBuffer);
      encoder.clearBuffer(this.wOldVelBuffer);
      encoder.clearBuffer(this.pressureBuffer);
      encoder.clearBuffer(this.divergenceBuffer);
      encoder.clearBuffer(this.accumU);
      encoder.clearBuffer(this.accumV);
      encoder.clearBuffer(this.accumW);
      encoder.clearBuffer(this.accumWeightU);
      encoder.clearBuffer(this.accumWeightV);
      encoder.clearBuffer(this.accumWeightW);
      encoder.clearBuffer(this.gridVelBuffer);
      this.dispatch(encoder, this.flipP2GPipeline, this.flipP2GBindGroup,
        particleWG, profiler?.timestampWrites('flipP2G'));
      this.dispatch(encoder, this.flipNormalizeAPipeline, this.flipNormalizeABindGroup,
        faceWG, profiler?.timestampWrites('flipNormalizeA'));
      this.dispatch(encoder, this.flipNormalizeBPipeline, this.flipNormalizeBBindGroup,
        faceWG, profiler?.timestampWrites('flipNormalizeB'));
      this.dispatch(encoder, this.flipDivergencePipeline, this.flipDivergenceBindGroup,
        gridWG, profiler?.timestampWrites('flipDivergence'));
      for (let j = 0; j < JACOBI_ITERATIONS; j++) {
        const jacobiBindGroup = j % 2 === 0 ? this.flipJacobiBindGroupA : this.flipJacobiBindGroupB;
        this.dispatch(encoder, this.flipJacobiPipeline, jacobiBindGroup, gridWG);
      }
      const projectBindGroup = JACOBI_ITERATIONS % 2 === 0
        ? this.flipProjectBindGroupA
        : this.flipProjectBindGroupB;
      this.dispatch(encoder, this.flipProjectPipeline, projectBindGroup,
        gridWG, profiler?.timestampWrites('flipProject'));
      this.dispatch(encoder, this.flipProjectStaggeredPipeline, this.flipProjectStaggeredBindGroup,
        faceWG, profiler?.timestampWrites('flipProjectStaggered'));
      this.dispatch(encoder, this.flipG2PPipeline, this.flipG2PBindGroup,
        particleWG, profiler?.timestampWrites('flipG2P'));
    }

    this.dispatch(encoder, this.clearDensityFieldPipeline, this.clearDensityFieldBindGroup,
      Math.ceil(fieldElements / 256), profiler?.timestampWrites('clearDensityField'));
    this.dispatch(encoder, this.splatDensityPipeline, this.splatDensityBindGroup,
      particleWG, profiler?.timestampWrites('splatDensity'));
  }

  private dispatch(
    encoder: GPUCommandEncoder,
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    workgroupCount: number,
    timestampWrites?: GPUComputePassTimestampWrites,
  ) {
    const pass = encoder.beginComputePass(
      timestampWrites ? { timestampWrites } : undefined,
    );
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroupCount);
    pass.end();
  }

  updateSimConfig(config: SimConfig) {
    this.paramsF32[8] = config.stiffness;
    this.paramsF32[9] = config.viscosity;
    this.paramsF32[10] = config.gravity;
    this.paramsF32[11] = config.boundaryDamping;
    this.paramsF32[12] = config.maxVelocity;
    this.paramsF32[15] = config.xsphEpsilon;
    this.paramsF32[16] = config.surfaceTension;
    this.paramsF32[30] = config.splatRadius * config.splatRadius;
    const fieldCellSize = this.paramsF32[28];
    this.paramsU32[31] = Math.ceil(config.splatRadius / fieldCellSize);
    this.device.queue.writeBuffer(this.paramsBuffer, 0, this.paramsArrayBuffer);
  }

  resetVelocities() {
    const vel = new Float32Array(this.particleCount * 4);
    for (let i = 0; i < this.particleCount; i++) {
      const hx = ((i * 73856093) % 1000) / 1000 - 0.5;
      const hz = ((i * 83492791) % 1000) / 1000 - 0.5;
      vel[i * 4] = hx * 0.015;
      vel[i * 4 + 2] = hz * 0.015;
    }
    this.device.queue.writeBuffer(this.velocitiesBuffer, 0, vel);
    // Zero staggered face velocities on reset
    const R = this.fieldResolution;
    const zeroU = new Float32Array((R + 1) * R * R);
    const zeroV = new Float32Array(R * (R + 1) * R);
    const zeroW = new Float32Array(R * R * (R + 1));
    this.device.queue.writeBuffer(this.uVelBuffer, 0, zeroU);
    this.device.queue.writeBuffer(this.vVelBuffer, 0, zeroV);
    this.device.queue.writeBuffer(this.wVelBuffer, 0, zeroW);
    this.device.queue.writeBuffer(this.uOldVelBuffer, 0, zeroU);
    this.device.queue.writeBuffer(this.vOldVelBuffer, 0, zeroV);
    this.device.queue.writeBuffer(this.wOldVelBuffer, 0, zeroW);

    const zeroGrid = new Float32Array(this.fieldResolution ** 3 * 4);
    this.device.queue.writeBuffer(this.gridVelBuffer, 0, zeroGrid);
  }

  dispose(destroyDevice = false) {
    this.positionsBuffer.destroy();
    this.velocitiesBuffer.destroy();
    this.densityFieldBuffer.destroy();
    this.accumU.destroy();
    this.accumV.destroy();
    this.accumW.destroy();
    this.accumWeightU.destroy();
    this.accumWeightV.destroy();
    this.accumWeightW.destroy();
    this.uVelBuffer.destroy();
    this.vVelBuffer.destroy();
    this.wVelBuffer.destroy();
    this.uOldVelBuffer.destroy();
    this.vOldVelBuffer.destroy();
    this.wOldVelBuffer.destroy();
    this.gridVelBuffer.destroy();
    this.pressureBuffer.destroy();
    this.pressureAltBuffer.destroy();
    this.divergenceBuffer.destroy();
    this.paramsBuffer.destroy();
    if (destroyDevice) this.device.destroy();
  }
}
