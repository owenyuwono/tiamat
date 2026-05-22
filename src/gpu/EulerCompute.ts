import * as THREE from 'three';
import { SPH } from '../sph/constants';
import type { SimConfig } from '../ui/SimConfig';
import type { GPUProfiler } from './GPUProfiler';

import eulerClearGridShader from './shaders/eulerClearGrid.wgsl?raw';
import eulerAdvectVelocityShader from './shaders/eulerAdvectVelocity.wgsl?raw';
import eulerApplyForcesShader from './shaders/eulerApplyForces.wgsl?raw';
import eulerDivergenceShader from './shaders/eulerDivergence.wgsl?raw';
import eulerJacobiShader from './shaders/eulerJacobi.wgsl?raw';
import eulerProjectShader from './shaders/eulerProject.wgsl?raw';
import eulerAdvectDensityShader from './shaders/eulerAdvectDensity.wgsl?raw';
import eulerWriteDensityShader from './shaders/eulerWriteDensity.wgsl?raw';
import eulerExtrapolateInitShader from './shaders/eulerExtrapolateInit.wgsl?raw';
import eulerExtrapolateSweepShader from './shaders/eulerExtrapolateSweep.wgsl?raw';

const JACOBI_ITERATIONS = 80;
const N_EXTRAP_LAYERS = 3;

function nextPowerOfTwo(n: number): number {
  let v = n - 1;
  v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
  return v + 1;
}

export class EulerCompute {
  private device: GPUDevice;
  private fieldResolution: number;

  // Staggered MAC face velocities
  private uVelBuffer: GPUBuffer;
  private vVelBuffer: GPUBuffer;
  private wVelBuffer: GPUBuffer;
  private uVelTempBuffer: GPUBuffer;
  private vVelTempBuffer: GPUBuffer;
  private wVelTempBuffer: GPUBuffer;

  // Pressure solve
  private pressureBuffer: GPUBuffer;
  private pressureAltBuffer: GPUBuffer;
  private divergenceBuffer: GPUBuffer;

  // Density marker (fluid vs air)
  private densityMarkerBuffer: GPUBuffer;
  private densityMarkerTempBuffer: GPUBuffer;

  // Rendering output
  private densityFieldBuffer: GPUBuffer;
  private paramsBuffer: GPUBuffer;

  // Pipelines
  private clearGridPipeline: GPUComputePipeline;
  private clearGridBindGroup: GPUBindGroup;
  private advectVelocityPipeline: GPUComputePipeline;
  private advectVelocityBindGroup: GPUBindGroup;
  private applyForcesPipeline: GPUComputePipeline;
  private applyForcesBindGroup: GPUBindGroup;
  private divergencePipeline: GPUComputePipeline;
  private divergenceBindGroup: GPUBindGroup;
  private jacobiPipeline: GPUComputePipeline;
  private jacobiBindGroupA: GPUBindGroup;
  private jacobiBindGroupB: GPUBindGroup;
  private projectPipeline: GPUComputePipeline;
  private projectBindGroup: GPUBindGroup;
  private advectDensityPipeline: GPUComputePipeline;
  private advectDensityBindGroupA: GPUBindGroup;
  private advectDensityBindGroupB: GPUBindGroup;
  private extrapInitPipeline: GPUComputePipeline;
  private extrapInitBindGroup: GPUBindGroup;
  private extrapSweepPipeline: GPUComputePipeline;
  private extrapSweepBindGroupA: GPUBindGroup;
  private extrapSweepBindGroupB: GPUBindGroup;
  private writeDensityPipeline: GPUComputePipeline;
  private writeDensityBindGroup: GPUBindGroup;

  private paramsArrayBuffer: ArrayBuffer;
  private paramsF32: Float32Array;
  private paramsU32: Uint32Array;

  private validFaceBuffer: GPUBuffer;
  private validFaceTempBuffer: GPUBuffer;

  // Buffer sizes for copyBufferToBuffer
  private uFaceBytes: number;
  private vFaceBytes: number;
  private wFaceBytes: number;
  private cellBytes: number;

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
    this.fieldResolution = fieldResolution;

    const res = fieldResolution;
    const res3 = res * res * res;
    const uCount = (res + 1) * res * res;
    const vCount = res * (res + 1) * res;
    const wCount = res * res * (res + 1);

    this.uFaceBytes = uCount * 4;
    this.vFaceBytes = vCount * 4;
    this.wFaceBytes = wCount * 4;
    this.cellBytes = res3 * 4;

    // Staggered face velocity buffers (main + temp for advection copy)
    this.uVelBuffer = device.createBuffer({ size: this.uFaceBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.vVelBuffer = device.createBuffer({ size: this.vFaceBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.wVelBuffer = device.createBuffer({ size: this.wFaceBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.uVelTempBuffer = device.createBuffer({ size: this.uFaceBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.vVelTempBuffer = device.createBuffer({ size: this.vFaceBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.wVelTempBuffer = device.createBuffer({ size: this.wFaceBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

    this.pressureBuffer = device.createBuffer({ size: this.cellBytes, usage: GPUBufferUsage.STORAGE });
    this.pressureAltBuffer = device.createBuffer({ size: this.cellBytes, usage: GPUBufferUsage.STORAGE });
    this.divergenceBuffer = device.createBuffer({ size: this.cellBytes, usage: GPUBufferUsage.STORAGE });

    this.densityMarkerBuffer = device.createBuffer({ size: this.cellBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.densityMarkerTempBuffer = device.createBuffer({ size: this.cellBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

    const totalFaces = uCount + vCount + wCount;
    this.validFaceBuffer = device.createBuffer({ size: totalFaces * 4, usage: GPUBufferUsage.STORAGE });
    this.validFaceTempBuffer = device.createBuffer({ size: totalFaces * 4, usage: GPUBufferUsage.STORAGE });

    this.densityFieldBuffer = device.createBuffer({
      size: res3 * 2 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    this.paramsBuffer = device.createBuffer({
      size: 128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Initialize params (same layout as SPH/FLIP)
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
    this.paramsU32[2] = 16;
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
    this.paramsF32[17] = 0.016;
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

    // Zero all grid buffers
    const zeroU = new Float32Array(uCount);
    const zeroV = new Float32Array(vCount);
    const zeroW = new Float32Array(wCount);
    const zeroCell = new Float32Array(res3);
    device.queue.writeBuffer(this.uVelBuffer, 0, zeroU);
    device.queue.writeBuffer(this.vVelBuffer, 0, zeroV);
    device.queue.writeBuffer(this.wVelBuffer, 0, zeroW);
    device.queue.writeBuffer(this.uVelTempBuffer, 0, zeroU);
    device.queue.writeBuffer(this.vVelTempBuffer, 0, zeroV);
    device.queue.writeBuffer(this.wVelTempBuffer, 0, zeroW);
    device.queue.writeBuffer(this.densityMarkerBuffer, 0, zeroCell);
    device.queue.writeBuffer(this.densityMarkerTempBuffer, 0, zeroCell);

    // Create pipelines and bind groups
    const S = GPUShaderStage.COMPUTE;
    const uniform = (b: number) => ({ binding: b, visibility: S, buffer: { type: 'uniform' as const } });
    const storage = (b: number) => ({ binding: b, visibility: S, buffer: { type: 'storage' as const } });
    const readOnly = (b: number) => ({ binding: b, visibility: S, buffer: { type: 'read-only-storage' as const } });

    // clearGrid: params + pressure + pressureAlt + divergence
    const clearGrid = this.createPipeline(device, eulerClearGridShader, [
      uniform(0), storage(1), storage(2), storage(3),
    ]);
    this.clearGridPipeline = clearGrid.pipeline;
    this.clearGridBindGroup = device.createBindGroup({
      layout: clearGrid.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.pressureBuffer } },
        { binding: 2, resource: { buffer: this.pressureAltBuffer } },
        { binding: 3, resource: { buffer: this.divergenceBuffer } },
      ],
    });

    // advectVelocity: params + uVel/vVel/wVel (read) + uTemp/vTemp/wTemp (write)
    const advectVel = this.createPipeline(device, eulerAdvectVelocityShader, [
      uniform(0), readOnly(1), readOnly(2), readOnly(3), storage(4), storage(5), storage(6),
    ]);
    this.advectVelocityPipeline = advectVel.pipeline;
    this.advectVelocityBindGroup = device.createBindGroup({
      layout: advectVel.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.uVelBuffer } },
        { binding: 2, resource: { buffer: this.vVelBuffer } },
        { binding: 3, resource: { buffer: this.wVelBuffer } },
        { binding: 4, resource: { buffer: this.uVelTempBuffer } },
        { binding: 5, resource: { buffer: this.vVelTempBuffer } },
        { binding: 6, resource: { buffer: this.wVelTempBuffer } },
      ],
    });

    // applyForces: params + vVel (rw) + densityMarker (read)
    const applyForces = this.createPipeline(device, eulerApplyForcesShader, [
      uniform(0), storage(1), readOnly(2),
    ]);
    this.applyForcesPipeline = applyForces.pipeline;
    this.applyForcesBindGroup = device.createBindGroup({
      layout: applyForces.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.vVelBuffer } },
        { binding: 2, resource: { buffer: this.densityMarkerBuffer } },
      ],
    });

    // divergence: params + uVel/vVel/wVel (read) + divergence (write)
    const div = this.createPipeline(device, eulerDivergenceShader, [
      uniform(0), readOnly(1), readOnly(2), readOnly(3), storage(4),
    ]);
    this.divergencePipeline = div.pipeline;
    this.divergenceBindGroup = device.createBindGroup({
      layout: div.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.uVelBuffer } },
        { binding: 2, resource: { buffer: this.vVelBuffer } },
        { binding: 3, resource: { buffer: this.wVelBuffer } },
        { binding: 4, resource: { buffer: this.divergenceBuffer } },
      ],
    });

    // jacobi: params + densityMarker (read) + pressureIn (read) + divergence (read) + pressureOut (rw)
    const jacobi = this.createPipeline(device, eulerJacobiShader, [
      uniform(0), readOnly(1), readOnly(2), readOnly(3), storage(4),
    ]);
    this.jacobiPipeline = jacobi.pipeline;
    this.jacobiBindGroupA = device.createBindGroup({
      layout: jacobi.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.densityMarkerBuffer } },
        { binding: 2, resource: { buffer: this.pressureBuffer } },
        { binding: 3, resource: { buffer: this.divergenceBuffer } },
        { binding: 4, resource: { buffer: this.pressureAltBuffer } },
      ],
    });
    this.jacobiBindGroupB = device.createBindGroup({
      layout: jacobi.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.densityMarkerBuffer } },
        { binding: 2, resource: { buffer: this.pressureAltBuffer } },
        { binding: 3, resource: { buffer: this.divergenceBuffer } },
        { binding: 4, resource: { buffer: this.pressureBuffer } },
      ],
    });

    // project: params + uVel/vVel/wVel (rw) + pressure (read)
    // After 40 (even) Jacobi iterations: A writes to Alt, B writes to pressure
    // Iteration 0: A (read pressure, write Alt), 1: B (read Alt, write pressure), ...
    // Iteration 39 (odd): B writes to pressureBuffer — final result is in pressureBuffer
    const proj = this.createPipeline(device, eulerProjectShader, [
      uniform(0), storage(1), storage(2), storage(3), readOnly(4),
    ]);
    this.projectPipeline = proj.pipeline;
    this.projectBindGroup = device.createBindGroup({
      layout: proj.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.uVelBuffer } },
        { binding: 2, resource: { buffer: this.vVelBuffer } },
        { binding: 3, resource: { buffer: this.wVelBuffer } },
        { binding: 4, resource: { buffer: this.pressureBuffer } },
      ],
    });

    // extrapInit: params + densityMarker (read) + validFlags (rw)
    const extrapInit = this.createPipeline(device, eulerExtrapolateInitShader, [
      uniform(0), readOnly(1), storage(2),
    ]);
    this.extrapInitPipeline = extrapInit.pipeline;
    this.extrapInitBindGroup = device.createBindGroup({
      layout: extrapInit.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.densityMarkerBuffer } },
        { binding: 2, resource: { buffer: this.validFaceBuffer } },
      ],
    });

    // extrapSweep: params + uVel/vVel/wVel (rw) + validIn (read) + validOut (rw)
    const extrapSweep = this.createPipeline(device, eulerExtrapolateSweepShader, [
      uniform(0), storage(1), storage(2), storage(3), readOnly(4), storage(5),
    ]);
    this.extrapSweepPipeline = extrapSweep.pipeline;
    this.extrapSweepBindGroupA = device.createBindGroup({
      layout: extrapSweep.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.uVelBuffer } },
        { binding: 2, resource: { buffer: this.vVelBuffer } },
        { binding: 3, resource: { buffer: this.wVelBuffer } },
        { binding: 4, resource: { buffer: this.validFaceBuffer } },
        { binding: 5, resource: { buffer: this.validFaceTempBuffer } },
      ],
    });
    this.extrapSweepBindGroupB = device.createBindGroup({
      layout: extrapSweep.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.uVelBuffer } },
        { binding: 2, resource: { buffer: this.vVelBuffer } },
        { binding: 3, resource: { buffer: this.wVelBuffer } },
        { binding: 4, resource: { buffer: this.validFaceTempBuffer } },
        { binding: 5, resource: { buffer: this.validFaceBuffer } },
      ],
    });

    // advectDensity: params + uVel/vVel/wVel (read) + densityIn (read) + densityOut (write)
    const advDens = this.createPipeline(device, eulerAdvectDensityShader, [
      uniform(0), readOnly(1), readOnly(2), readOnly(3), readOnly(4), storage(5),
    ]);
    this.advectDensityPipeline = advDens.pipeline;
    this.advectDensityBindGroupA = device.createBindGroup({
      layout: advDens.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.uVelBuffer } },
        { binding: 2, resource: { buffer: this.vVelBuffer } },
        { binding: 3, resource: { buffer: this.wVelBuffer } },
        { binding: 4, resource: { buffer: this.densityMarkerBuffer } },
        { binding: 5, resource: { buffer: this.densityMarkerTempBuffer } },
      ],
    });
    this.advectDensityBindGroupB = device.createBindGroup({
      layout: advDens.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.uVelBuffer } },
        { binding: 2, resource: { buffer: this.vVelBuffer } },
        { binding: 3, resource: { buffer: this.wVelBuffer } },
        { binding: 4, resource: { buffer: this.densityMarkerTempBuffer } },
        { binding: 5, resource: { buffer: this.densityMarkerBuffer } },
      ],
    });

    // writeDensity: params + densityMarker (read) + densityField (rw)
    const writeDens = this.createPipeline(device, eulerWriteDensityShader, [
      uniform(0), readOnly(1), storage(2),
    ]);
    this.writeDensityPipeline = writeDens.pipeline;
    this.writeDensityBindGroup = device.createBindGroup({
      layout: writeDens.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.densityMarkerBuffer } },
        { binding: 2, resource: { buffer: this.densityFieldBuffer } },
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
    const R = this.fieldResolution;
    const h = this.paramsF32[28];
    const minX = this.paramsF32[25];
    const minY = this.paramsF32[26];
    const minZ = this.paramsF32[27];
    const density = new Float32Array(R * R * R);

    for (let i = 0; i < posX.length; i++) {
      const ix = Math.floor((posX[i] - minX) / h);
      const iy = Math.floor((posY[i] - minY) / h);
      const iz = Math.floor((posZ[i] - minZ) / h);
      if (ix >= 0 && ix < R && iy >= 0 && iy < R && iz >= 0 && iz < R) {
        density[iz * R * R + iy * R + ix] += 1.0;
      }
    }

    for (let i = 0; i < density.length; i++) {
      density[i] = density[i] > 0 ? 1.0 : 0.0;
    }

    this.device.queue.writeBuffer(this.densityMarkerBuffer, 0, density);
    this.device.queue.writeBuffer(this.densityMarkerTempBuffer, 0, density);
  }

  getDevice(): GPUDevice { return this.device; }
  getDensityFieldBuffer(): GPUBuffer { return this.densityFieldBuffer; }
  getParamsBuffer(): GPUBuffer { return this.paramsBuffer; }
  getFieldResolution(): number { return this.fieldResolution; }

  encodeStep(encoder: GPUCommandEncoder, _substeps: number, profiler?: GPUProfiler | null) {
    this.paramsF32[17] = 0.016;
    this.device.queue.writeBuffer(this.paramsBuffer, 0, this.paramsArrayBuffer);

    const res = this.fieldResolution;
    const res3 = res * res * res;
    const gridWG = Math.ceil(res3 / 256);
    const faceWG = Math.ceil(((res + 1) * res * res) / 256);
    const vFaceWG = Math.ceil((res * (res + 1) * res) / 256);

    // 1. Clear pressure and divergence
    this.dispatch(encoder, this.clearGridPipeline, this.clearGridBindGroup,
      gridWG, profiler?.timestampWrites('eulerClearGrid'));

    // 2. Semi-Lagrangian velocity advection → temp buffers
    this.dispatch(encoder, this.advectVelocityPipeline, this.advectVelocityBindGroup,
      faceWG, profiler?.timestampWrites('eulerAdvectVelocity'));

    // 3. Copy temp → main velocity buffers
    encoder.copyBufferToBuffer(this.uVelTempBuffer, 0, this.uVelBuffer, 0, this.uFaceBytes);
    encoder.copyBufferToBuffer(this.vVelTempBuffer, 0, this.vVelBuffer, 0, this.vFaceBytes);
    encoder.copyBufferToBuffer(this.wVelTempBuffer, 0, this.wVelBuffer, 0, this.wFaceBytes);

    // 4. Apply gravity to v-faces
    this.dispatch(encoder, this.applyForcesPipeline, this.applyForcesBindGroup,
      vFaceWG, profiler?.timestampWrites('eulerApplyForces'));

    // 5. Compute divergence
    this.dispatch(encoder, this.divergencePipeline, this.divergenceBindGroup,
      gridWG, profiler?.timestampWrites('eulerDivergence'));

    // 6. Jacobi pressure solve (80 iterations, ping-pong A/B)
    for (let j = 0; j < JACOBI_ITERATIONS; j++) {
      const bg = j % 2 === 0 ? this.jacobiBindGroupA : this.jacobiBindGroupB;
      this.dispatch(encoder, this.jacobiPipeline, bg, gridWG);
    }

    // 7. Pressure projection (final pressure is in pressureBuffer after 80 even iterations)
    this.dispatch(encoder, this.projectPipeline, this.projectBindGroup,
      faceWG, profiler?.timestampWrites('eulerProject'));

    // 8. Classify face validity from density markers
    this.dispatch(encoder, this.extrapInitPipeline, this.extrapInitBindGroup,
      faceWG, profiler?.timestampWrites('eulerExtrapInit'));

    // 9. Extrapolate velocity into air cells (3 layers, ping-pong valid buffers)
    for (let e = 0; e < N_EXTRAP_LAYERS; e++) {
      const bg = e % 2 === 0 ? this.extrapSweepBindGroupA : this.extrapSweepBindGroupB;
      this.dispatch(encoder, this.extrapSweepPipeline, bg, faceWG);
    }

    // 10. Upwind density advection (4 substeps, ping-pong A/B)
    const DENSITY_SUBSTEPS = 4;
    for (let s = 0; s < DENSITY_SUBSTEPS; s++) {
      const bg = s % 2 === 0 ? this.advectDensityBindGroupA : this.advectDensityBindGroupB;
      this.dispatch(encoder, this.advectDensityPipeline, bg,
        gridWG, s === 0 ? profiler?.timestampWrites('eulerAdvectDensity') : undefined);
    }

    // 11. Convert density marker to rendering format
    this.dispatch(encoder, this.writeDensityPipeline, this.writeDensityBindGroup,
      gridWG, profiler?.timestampWrites('eulerWriteDensity'));
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
    const R = this.fieldResolution;
    const zeroU = new Float32Array((R + 1) * R * R);
    const zeroV = new Float32Array(R * (R + 1) * R);
    const zeroW = new Float32Array(R * R * (R + 1));
    this.device.queue.writeBuffer(this.uVelBuffer, 0, zeroU);
    this.device.queue.writeBuffer(this.vVelBuffer, 0, zeroV);
    this.device.queue.writeBuffer(this.wVelBuffer, 0, zeroW);
    this.device.queue.writeBuffer(this.uVelTempBuffer, 0, zeroU);
    this.device.queue.writeBuffer(this.vVelTempBuffer, 0, zeroV);
    this.device.queue.writeBuffer(this.wVelTempBuffer, 0, zeroW);
  }

  dispose(destroyDevice = false) {
    this.uVelBuffer.destroy();
    this.vVelBuffer.destroy();
    this.wVelBuffer.destroy();
    this.uVelTempBuffer.destroy();
    this.vVelTempBuffer.destroy();
    this.wVelTempBuffer.destroy();
    this.pressureBuffer.destroy();
    this.pressureAltBuffer.destroy();
    this.divergenceBuffer.destroy();
    this.densityMarkerBuffer.destroy();
    this.densityMarkerTempBuffer.destroy();
    this.validFaceBuffer.destroy();
    this.validFaceTempBuffer.destroy();
    this.densityFieldBuffer.destroy();
    this.paramsBuffer.destroy();
    if (destroyDevice) this.device.destroy();
  }
}
