import * as THREE from 'three';
import { SPH } from '../sph/constants';
import clearGridShader from './shaders/clearGrid.wgsl?raw';
import insertParticlesShader from './shaders/insertParticles.wgsl?raw';
import computeDensityShader from './shaders/computeDensity.wgsl?raw';
import computeForcesShader from './shaders/computeForces.wgsl?raw';
import integrateShader from './shaders/integrate.wgsl?raw';
import clearDensityFieldShader from './shaders/clearDensityField.wgsl?raw';
import splatDensityShader from './shaders/splatDensity.wgsl?raw';

const MAX_PER_CELL = 32;
const FIXED_POINT_SCALE = 10000.0;

function nextPowerOfTwo(n: number): number {
  let v = n - 1;
  v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
  return v + 1;
}

export class GPUCompute {
  private device: GPUDevice;
  private particleCount: number;
  private tableSize: number;
  private fieldResolution: number;
  private fieldSize: number;

  private positionsBuffer: GPUBuffer;
  private velocitiesBuffer: GPUBuffer;
  private forcesBuffer: GPUBuffer;
  private densityPressureBuffer: GPUBuffer;
  private xsphBuffer: GPUBuffer;
  private cellCountsBuffer: GPUBuffer;
  private cellEntriesBuffer: GPUBuffer;
  private densityFieldBuffer: GPUBuffer;
  private paramsBuffer: GPUBuffer;
  private stagingBuffers: [GPUBuffer, GPUBuffer];
  private currentStaging: number = 0;
  private firstFrame: boolean = true;

  private clearGridPipeline: GPUComputePipeline;
  private clearGridBindGroup: GPUBindGroup;
  private insertParticlesPipeline: GPUComputePipeline;
  private insertParticlesBindGroup: GPUBindGroup;
  private computeDensityPipeline: GPUComputePipeline;
  private computeDensityBindGroup: GPUBindGroup;
  private computeForcesPipeline: GPUComputePipeline;
  private computeForcesBindGroup: GPUBindGroup;
  private integratePipeline: GPUComputePipeline;
  private integrateBindGroup: GPUBindGroup;
  private clearDensityFieldPipeline: GPUComputePipeline;
  private clearDensityFieldBindGroup: GPUBindGroup;
  private splatDensityPipeline: GPUComputePipeline;
  private splatDensityBindGroup: GPUBindGroup;

  private paramsArrayBuffer: ArrayBuffer;
  private paramsF32: Float32Array;
  private paramsU32: Uint32Array;

  static async create(
    particleCount: number,
    containerSize: THREE.Vector3,
    fieldResolution: number,
    domainMin: THREE.Vector3,
    domainMax: THREE.Vector3,
    splatRadius: number,
  ): Promise<GPUCompute | null> {
    if (!navigator.gpu) return null;
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    const device = await adapter.requestDevice();
    return new GPUCompute(device, particleCount, containerSize, fieldResolution, domainMin, domainMax, splatRadius);
  }

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
    this.tableSize = nextPowerOfTwo(particleCount * 3);
    this.fieldResolution = fieldResolution;
    this.fieldSize = fieldResolution * fieldResolution * fieldResolution * 2 * 4;

    const N = particleCount;

    this.positionsBuffer = device.createBuffer({
      size: N * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.velocitiesBuffer = device.createBuffer({
      size: N * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.forcesBuffer = device.createBuffer({
      size: N * 16,
      usage: GPUBufferUsage.STORAGE,
    });
    this.densityPressureBuffer = device.createBuffer({
      size: N * 8,
      usage: GPUBufferUsage.STORAGE,
    });
    this.xsphBuffer = device.createBuffer({
      size: N * 16,
      usage: GPUBufferUsage.STORAGE,
    });
    this.cellCountsBuffer = device.createBuffer({
      size: this.tableSize * 4,
      usage: GPUBufferUsage.STORAGE,
    });
    this.cellEntriesBuffer = device.createBuffer({
      size: this.tableSize * MAX_PER_CELL * 4,
      usage: GPUBufferUsage.STORAGE,
    });
    this.densityFieldBuffer = device.createBuffer({
      size: this.fieldSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.paramsBuffer = device.createBuffer({
      size: 128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.stagingBuffers = [
      device.createBuffer({
        size: this.fieldSize,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      }),
      device.createBuffer({
        size: this.fieldSize,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      }),
    ];

    this.paramsArrayBuffer = new ArrayBuffer(128);
    this.paramsF32 = new Float32Array(this.paramsArrayBuffer);
    this.paramsU32 = new Uint32Array(this.paramsArrayBuffer);

    const H = SPH.smoothingRadius;
    const H2 = H * H;
    const spacing = H * 0.5;
    const mass = SPH.restDensity * spacing * spacing * spacing;
    const domainSize = domainMax.x - domainMin.x;
    const fieldCellSize = domainSize / fieldResolution;
    const fieldInvCellSize = 1 / fieldCellSize;
    const splatRadius2 = splatRadius * splatRadius;
    const splatRadiusCells = Math.ceil(splatRadius / fieldCellSize);

    this.paramsU32[0] = particleCount;
    this.paramsU32[1] = this.tableSize;
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

    const clearGridResult = this.createPipeline(device, clearGridShader, [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ]);
    this.clearGridPipeline = clearGridResult.pipeline;
    this.clearGridBindGroup = device.createBindGroup({
      layout: clearGridResult.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.cellCountsBuffer } },
      ],
    });

    const insertParticlesResult = this.createPipeline(device, insertParticlesShader, [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ]);
    this.insertParticlesPipeline = insertParticlesResult.pipeline;
    this.insertParticlesBindGroup = device.createBindGroup({
      layout: insertParticlesResult.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.positionsBuffer } },
        { binding: 2, resource: { buffer: this.cellCountsBuffer } },
        { binding: 3, resource: { buffer: this.cellEntriesBuffer } },
      ],
    });

    const computeDensityResult = this.createPipeline(device, computeDensityShader, [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    ]);
    this.computeDensityPipeline = computeDensityResult.pipeline;
    this.computeDensityBindGroup = device.createBindGroup({
      layout: computeDensityResult.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.positionsBuffer } },
        { binding: 2, resource: { buffer: this.densityPressureBuffer } },
        { binding: 3, resource: { buffer: this.cellCountsBuffer } },
        { binding: 4, resource: { buffer: this.cellEntriesBuffer } },
      ],
    });

    const computeForcesResult = this.createPipeline(device, computeForcesShader, [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    ]);
    this.computeForcesPipeline = computeForcesResult.pipeline;
    this.computeForcesBindGroup = device.createBindGroup({
      layout: computeForcesResult.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.positionsBuffer } },
        { binding: 2, resource: { buffer: this.velocitiesBuffer } },
        { binding: 3, resource: { buffer: this.densityPressureBuffer } },
        { binding: 4, resource: { buffer: this.forcesBuffer } },
        { binding: 5, resource: { buffer: this.xsphBuffer } },
        { binding: 6, resource: { buffer: this.cellCountsBuffer } },
        { binding: 7, resource: { buffer: this.cellEntriesBuffer } },
      ],
    });

    const integrateResult = this.createPipeline(device, integrateShader, [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    ]);
    this.integratePipeline = integrateResult.pipeline;
    this.integrateBindGroup = device.createBindGroup({
      layout: integrateResult.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.positionsBuffer } },
        { binding: 2, resource: { buffer: this.velocitiesBuffer } },
        { binding: 3, resource: { buffer: this.forcesBuffer } },
        { binding: 4, resource: { buffer: this.densityPressureBuffer } },
        { binding: 5, resource: { buffer: this.xsphBuffer } },
      ],
    });

    const clearDensityFieldResult = this.createPipeline(device, clearDensityFieldShader, [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ]);
    this.clearDensityFieldPipeline = clearDensityFieldResult.pipeline;
    this.clearDensityFieldBindGroup = device.createBindGroup({
      layout: clearDensityFieldResult.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.densityFieldBuffer } },
      ],
    });

    const splatDensityResult = this.createPipeline(device, splatDensityShader, [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ]);
    this.splatDensityPipeline = splatDensityResult.pipeline;
    this.splatDensityBindGroup = device.createBindGroup({
      layout: splatDensityResult.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.positionsBuffer } },
        { binding: 2, resource: { buffer: this.xsphBuffer } },
        { binding: 3, resource: { buffer: this.densityFieldBuffer } },
      ],
    });

    device.lost.then((info) => {
      console.error('WebGPU device lost:', info.message);
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
    const packed = new Float32Array(this.particleCount * 4);
    for (let i = 0; i < this.particleCount; i++) {
      packed[i * 4] = posX[i];
      packed[i * 4 + 1] = posY[i];
      packed[i * 4 + 2] = posZ[i];
    }
    this.device.queue.writeBuffer(this.positionsBuffer, 0, packed);
  }

  async step(dt: number, substeps: number) {
    void dt;
    const fixedDt = 0.005;
    this.paramsF32[17] = fixedDt;
    this.device.queue.writeBuffer(this.paramsBuffer, 0, this.paramsArrayBuffer);

    const encoder = this.device.createCommandEncoder();

    this.dispatch(encoder, this.clearGridPipeline, this.clearGridBindGroup, Math.ceil(this.tableSize / 256));
    this.dispatch(encoder, this.insertParticlesPipeline, this.insertParticlesBindGroup, Math.ceil(this.particleCount / 64));

    for (let s = 0; s < substeps; s++) {
      this.dispatch(encoder, this.computeDensityPipeline, this.computeDensityBindGroup, Math.ceil(this.particleCount / 64));
      this.dispatch(encoder, this.computeForcesPipeline, this.computeForcesBindGroup, Math.ceil(this.particleCount / 64));
      this.dispatch(encoder, this.integratePipeline, this.integrateBindGroup, Math.ceil(this.particleCount / 64));

      if (s < substeps - 1) {
        this.dispatch(encoder, this.clearGridPipeline, this.clearGridBindGroup, Math.ceil(this.tableSize / 256));
        this.dispatch(encoder, this.insertParticlesPipeline, this.insertParticlesBindGroup, Math.ceil(this.particleCount / 64));
      }
    }

    const fieldElements = this.fieldResolution * this.fieldResolution * this.fieldResolution * 2;
    this.dispatch(encoder, this.clearDensityFieldPipeline, this.clearDensityFieldBindGroup, Math.ceil(fieldElements / 256));
    this.dispatch(encoder, this.splatDensityPipeline, this.splatDensityBindGroup, Math.ceil(this.particleCount / 64));

    encoder.copyBufferToBuffer(this.densityFieldBuffer, 0, this.stagingBuffers[this.currentStaging], 0, this.fieldSize);

    this.device.queue.submit([encoder.finish()]);
  }

  private dispatch(
    encoder: GPUCommandEncoder,
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    workgroupCount: number,
  ) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroupCount);
    pass.end();
  }

  async readDensityField(target: Float32Array) {
    const readIndex = this.firstFrame ? this.currentStaging : 1 - this.currentStaging;
    if (this.firstFrame) {
      await this.device.queue.onSubmittedWorkDone();
      this.firstFrame = false;
    }

    await this.stagingBuffers[readIndex].mapAsync(GPUMapMode.READ);
    const data = new Uint32Array(this.stagingBuffers[readIndex].getMappedRange());
    const invScale = 1.0 / FIXED_POINT_SCALE;
    for (let i = 0; i < data.length; i++) {
      target[i] = data[i] * invScale;
    }
    this.stagingBuffers[readIndex].unmap();
    this.currentStaging = 1 - this.currentStaging;
  }

  dispose() {
    this.positionsBuffer.destroy();
    this.velocitiesBuffer.destroy();
    this.forcesBuffer.destroy();
    this.densityPressureBuffer.destroy();
    this.xsphBuffer.destroy();
    this.cellCountsBuffer.destroy();
    this.cellEntriesBuffer.destroy();
    this.densityFieldBuffer.destroy();
    this.paramsBuffer.destroy();
    this.stagingBuffers[0].destroy();
    this.stagingBuffers[1].destroy();
    this.device.destroy();
  }
}
