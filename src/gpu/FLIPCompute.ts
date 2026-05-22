import * as THREE from 'three';
import { SPH } from '../sph/constants';
import type { SimConfig } from '../ui/SimConfig';
import type { GPUProfiler } from './GPUProfiler';

import flipClearGridShader from './shaders/flipClearGrid.wgsl?raw';
import flipP2GShader from './shaders/flipP2G.wgsl?raw';
import flipNormalizeShader from './shaders/flipNormalize.wgsl?raw';
import flipDivergenceShader from './shaders/flipDivergence.wgsl?raw';
import flipJacobiShader from './shaders/flipJacobi.wgsl?raw';
import flipProjectShader from './shaders/flipProject.wgsl?raw';
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
  private xsphBuffer: GPUBuffer;
  private densityFieldBuffer: GPUBuffer;
  private paramsBuffer: GPUBuffer;

  private accumVelXBuffer: GPUBuffer;
  private accumVelYBuffer: GPUBuffer;
  private accumVelZBuffer: GPUBuffer;
  private accumWeightBuffer: GPUBuffer;
  private gridVelBuffer: GPUBuffer;
  private gridOldVelBuffer: GPUBuffer;
  private pressureBuffer: GPUBuffer;
  private pressureAltBuffer: GPUBuffer;
  private divergenceBuffer: GPUBuffer;

  private flipClearGridPipeline: GPUComputePipeline;
  private flipClearGridBindGroup: GPUBindGroup;
  private flipP2GPipeline: GPUComputePipeline;
  private flipP2GBindGroup: GPUBindGroup;
  private flipNormalizePipeline: GPUComputePipeline;
  private flipNormalizeBindGroup: GPUBindGroup;
  private flipDivergencePipeline: GPUComputePipeline;
  private flipDivergenceBindGroup: GPUBindGroup;
  private flipJacobiPipeline: GPUComputePipeline;
  private flipJacobiBindGroupA: GPUBindGroup;
  private flipJacobiBindGroupB: GPUBindGroup;
  private flipProjectPipeline: GPUComputePipeline;
  private flipProjectBindGroupA: GPUBindGroup;
  private flipProjectBindGroupB: GPUBindGroup;
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
    const res3 = fieldResolution * fieldResolution * fieldResolution;

    this.positionsBuffer = device.createBuffer({
      size: N * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.velocitiesBuffer = device.createBuffer({
      size: N * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.xsphBuffer = device.createBuffer({
      size: N * 16,
      usage: GPUBufferUsage.STORAGE,
    });
    this.densityFieldBuffer = device.createBuffer({
      size: this.fieldSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    this.accumVelXBuffer = device.createBuffer({ size: res3 * 4, usage: GPUBufferUsage.STORAGE });
    this.accumVelYBuffer = device.createBuffer({ size: res3 * 4, usage: GPUBufferUsage.STORAGE });
    this.accumVelZBuffer = device.createBuffer({ size: res3 * 4, usage: GPUBufferUsage.STORAGE });
    this.accumWeightBuffer = device.createBuffer({ size: res3 * 4, usage: GPUBufferUsage.STORAGE });
    this.gridVelBuffer = device.createBuffer({ size: res3 * 16, usage: GPUBufferUsage.STORAGE });
    this.gridOldVelBuffer = device.createBuffer({ size: res3 * 16, usage: GPUBufferUsage.STORAGE });
    this.pressureBuffer = device.createBuffer({ size: res3 * 4, usage: GPUBufferUsage.STORAGE });
    this.pressureAltBuffer = device.createBuffer({ size: res3 * 4, usage: GPUBufferUsage.STORAGE });
    this.divergenceBuffer = device.createBuffer({ size: res3 * 4, usage: GPUBufferUsage.STORAGE });

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
    const splatRadiusCells = Math.ceil(splatRadius / fieldCellSize);

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

    const zeroGrid = new Float32Array(res3 * 4);
    device.queue.writeBuffer(this.gridVelBuffer, 0, zeroGrid);
    device.queue.writeBuffer(this.gridOldVelBuffer, 0, zeroGrid);

    const S = GPUShaderStage.COMPUTE;
    const uniform = (b: number) => ({ binding: b, visibility: S, buffer: { type: 'uniform' as const } });
    const storage = (b: number) => ({ binding: b, visibility: S, buffer: { type: 'storage' as const } });
    const readOnly = (b: number) => ({ binding: b, visibility: S, buffer: { type: 'read-only-storage' as const } });

    // flipClearGrid: params, accumVelX/Y/Z(rw), accumWeight(rw), pressure(rw)
    const clearGrid = this.createPipeline(device, flipClearGridShader, [
      uniform(0), storage(1), storage(2), storage(3), storage(4), storage(5),
    ]);
    this.flipClearGridPipeline = clearGrid.pipeline;
    this.flipClearGridBindGroup = device.createBindGroup({
      layout: clearGrid.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.accumVelXBuffer } },
        { binding: 2, resource: { buffer: this.accumVelYBuffer } },
        { binding: 3, resource: { buffer: this.accumVelZBuffer } },
        { binding: 4, resource: { buffer: this.accumWeightBuffer } },
        { binding: 5, resource: { buffer: this.pressureBuffer } },
      ],
    });

    // flipP2G: params, positions(r), velocities(r), accumVelX/Y/Z(rw), accumWeight(rw)
    const p2g = this.createPipeline(device, flipP2GShader, [
      uniform(0), readOnly(1), readOnly(2), storage(3), storage(4), storage(5), storage(6),
    ]);
    this.flipP2GPipeline = p2g.pipeline;
    this.flipP2GBindGroup = device.createBindGroup({
      layout: p2g.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.positionsBuffer } },
        { binding: 2, resource: { buffer: this.velocitiesBuffer } },
        { binding: 3, resource: { buffer: this.accumVelXBuffer } },
        { binding: 4, resource: { buffer: this.accumVelYBuffer } },
        { binding: 5, resource: { buffer: this.accumVelZBuffer } },
        { binding: 6, resource: { buffer: this.accumWeightBuffer } },
      ],
    });

    // flipNormalize: params, accumVelX/Y/Z(rw), accumWeight(rw), gridVel(rw), gridOldVel(rw)
    const normalize = this.createPipeline(device, flipNormalizeShader, [
      uniform(0), storage(1), storage(2), storage(3), storage(4), storage(5), storage(6),
    ]);
    this.flipNormalizePipeline = normalize.pipeline;
    this.flipNormalizeBindGroup = device.createBindGroup({
      layout: normalize.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.accumVelXBuffer } },
        { binding: 2, resource: { buffer: this.accumVelYBuffer } },
        { binding: 3, resource: { buffer: this.accumVelZBuffer } },
        { binding: 4, resource: { buffer: this.accumWeightBuffer } },
        { binding: 5, resource: { buffer: this.gridVelBuffer } },
        { binding: 6, resource: { buffer: this.gridOldVelBuffer } },
      ],
    });

    // flipDivergence: params, gridVel(r), divergence(rw)
    const divergence = this.createPipeline(device, flipDivergenceShader, [
      uniform(0), readOnly(1), storage(2),
    ]);
    this.flipDivergencePipeline = divergence.pipeline;
    this.flipDivergenceBindGroup = device.createBindGroup({
      layout: divergence.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.gridVelBuffer } },
        { binding: 2, resource: { buffer: this.divergenceBuffer } },
      ],
    });

    // flipJacobi: params, gridVel(r), pressureIn(r), divergence(r), pressureOut(rw)
    const jacobi = this.createPipeline(device, flipJacobiShader, [
      uniform(0), readOnly(1), readOnly(2), readOnly(3), storage(4),
    ]);
    this.flipJacobiPipeline = jacobi.pipeline;
    this.flipJacobiBindGroupA = device.createBindGroup({
      layout: jacobi.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.gridVelBuffer } },
        { binding: 2, resource: { buffer: this.pressureBuffer } },
        { binding: 3, resource: { buffer: this.divergenceBuffer } },
        { binding: 4, resource: { buffer: this.pressureAltBuffer } },
      ],
    });
    this.flipJacobiBindGroupB = device.createBindGroup({
      layout: jacobi.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.gridVelBuffer } },
        { binding: 2, resource: { buffer: this.pressureAltBuffer } },
        { binding: 3, resource: { buffer: this.divergenceBuffer } },
        { binding: 4, resource: { buffer: this.pressureBuffer } },
      ],
    });

    // flipProject: params, gridVel(rw), pressure(r)
    const project = this.createPipeline(device, flipProjectShader, [
      uniform(0), storage(1), readOnly(2),
    ]);
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

    // flipG2P: params, pos(rw), vel(rw), xsph(rw), gridVel(r), gridOldVel(r)  -- for FLIP delta
    const g2p = this.createPipeline(device, flipG2PShader, [
      uniform(0), storage(1), storage(2), storage(3), readOnly(4), readOnly(5),
    ]);
    this.flipG2PPipeline = g2p.pipeline;
    this.flipG2PBindGroup = device.createBindGroup({
      layout: g2p.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.positionsBuffer } },
        { binding: 2, resource: { buffer: this.velocitiesBuffer } },
        { binding: 3, resource: { buffer: this.xsphBuffer } },
        { binding: 4, resource: { buffer: this.gridVelBuffer } },
        { binding: 5, resource: { buffer: this.gridOldVelBuffer } },
      ],
    });

    // clearDensityField (reused)
    const clearDf = this.createPipeline(device, clearDensityFieldShader, [
      uniform(0), storage(1),
    ]);
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
    ]);
    this.splatDensityPipeline = splat.pipeline;
    this.splatDensityBindGroup = device.createBindGroup({
      layout: splat.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.positionsBuffer } },
        { binding: 2, resource: { buffer: this.xsphBuffer } },
        { binding: 3, resource: { buffer: this.densityFieldBuffer } },
      ],
    });
  }

  private createPipeline(
    device: GPUDevice,
    shader: string,
    layout: GPUBindGroupLayoutEntry[],
  ): { pipeline: GPUComputePipeline; bindGroupLayout: GPUBindGroupLayout } {
    const bindGroupLayout = device.createBindGroupLayout({ entries: layout });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
    const pipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: {
        module: device.createShaderModule({ code: shader }),
        entryPoint: 'main',
      },
    });
    return { pipeline, bindGroupLayout };
  }

  uploadInitialPositions(posX: Float32Array, posY: Float32Array, posZ: Float32Array) {
    const halfCell = this.paramsF32[28] * 0.5;
    const spacing = SPH.smoothingRadius * 0.5;
    const jitter = spacing * 0.45;
    // Rotate ~22.5° so lattice columns aren't parallel to container walls.
    const rotY = Math.PI / 8;
    const cosR = Math.cos(rotY);
    const sinR = Math.sin(rotY);
    const packed = new Float32Array(this.particleCount * 4);
    for (let i = 0; i < this.particleCount; i++) {
      const hx = ((i * 73856093) % 1000) / 1000 - 0.5;
      const hy = ((i * 19349663) % 1000) / 1000 - 0.5;
      const hz = ((i * 83492791) % 1000) / 1000 - 0.5;

      const lx = posX[i] + halfCell + hx * jitter;
      const ly = posY[i] + halfCell + hy * jitter * 0.35;
      const lz = posZ[i] + halfCell + hz * jitter;

      packed[i * 4] = lx * cosR - lz * sinR;
      packed[i * 4 + 1] = ly;
      packed[i * 4 + 2] = lx * sinR + lz * cosR;
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

    for (let s = 0; s < substeps; s++) {
      this.dispatch(encoder, this.flipClearGridPipeline, this.flipClearGridBindGroup,
        gridWG, profiler?.timestampWrites('flipClearGrid'));
      this.dispatch(encoder, this.flipP2GPipeline, this.flipP2GBindGroup,
        particleWG, profiler?.timestampWrites('flipP2G'));
      this.dispatch(encoder, this.flipNormalizePipeline, this.flipNormalizeBindGroup,
        gridWG, profiler?.timestampWrites('flipNormalize'));
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
    const res3 = this.fieldResolution ** 3;
    const zeroGrid = new Float32Array(res3 * 4);
    this.device.queue.writeBuffer(this.gridVelBuffer, 0, zeroGrid);
    this.device.queue.writeBuffer(this.gridOldVelBuffer, 0, zeroGrid);
  }

  dispose(destroyDevice = false) {
    this.positionsBuffer.destroy();
    this.velocitiesBuffer.destroy();
    this.xsphBuffer.destroy();
    this.densityFieldBuffer.destroy();
    this.accumVelXBuffer.destroy();
    this.accumVelYBuffer.destroy();
    this.accumVelZBuffer.destroy();
    this.accumWeightBuffer.destroy();
    this.gridVelBuffer.destroy();
    this.gridOldVelBuffer.destroy();
    this.pressureBuffer.destroy();
    this.pressureAltBuffer.destroy();
    this.divergenceBuffer.destroy();
    this.paramsBuffer.destroy();
    if (destroyDevice) this.device.destroy();
  }
}
