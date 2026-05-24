import * as THREE from 'three';
import type { GPUProfiler } from './GPUProfiler';
import bufferToTextureShader from './shaders/bufferToTexture.wgsl?raw';
import waterRaymarchShader from './shaders/waterRaymarch.wgsl?raw';
import floorShader from './shaders/floor.wgsl?raw';
import sprayRenderShader from './shaders/sprayRender.wgsl?raw';
import fxaaShader from './shaders/fxaa.wgsl?raw';
import causticsShader from './shaders/caustics.wgsl?raw';
import debugParticlesShader from './shaders/debugParticles.wgsl?raw';
import { generateWorleyNoise } from './generateWorleyNoise';

export class WebGPURenderer {
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private format: GPUTextureFormat;
  readonly canvas: HTMLCanvasElement;

  private densityTexture: GPUTexture;
  private densityTextureView: GPUTextureView;
  private linearSampler: GPUSampler;
  private repeatSampler: GPUSampler;

  private sandTexture: GPUTexture;
  private sandTextureView: GPUTextureView;
  private foamNoiseTexture: GPUTexture;
  private foamNoiseTextureView: GPUTextureView;

  private bufferToTexPipeline: GPUComputePipeline;
  private bufferToTexBindGroupLayout: GPUBindGroupLayout;
  private bufferToTexBindGroup: GPUBindGroup;
  private fieldResolution: number;

  private waterPipeline: GPURenderPipeline;
  private waterBindGroupLayout: GPUBindGroupLayout;
  private waterBindGroup: GPUBindGroup;
  private renderUniformBuffer: GPUBuffer;
  private renderUniformData: Float32Array;


  private floorPipeline: GPURenderPipeline;
  private floorBindGroupLayout: GPUBindGroupLayout;
  private floorBindGroup: GPUBindGroup;
  private floorVertexBuffer: GPUBuffer;
  private floorUniformBuffer: GPUBuffer;
  private floorUniformData: Float32Array;
  private floorVertexCount: number;

  private sprayRenderPipeline: GPURenderPipeline | null = null;
  private sprayRenderBindGroupLayout: GPUBindGroupLayout | null = null;
  private sprayRenderBindGroup: GPUBindGroup | null = null;
  private sprayRenderUniformBuffer: GPUBuffer;
  private sprayRenderUniformData: Float32Array;
  private obstaclesUniformBuffer: GPUBuffer | null = null;

  private depthTexture: GPUTexture;
  private width = 0;
  private height = 0;

  private causticsTexture: GPUTexture;
  private causticsTextureView: GPUTextureView;
  private causticsPipeline: GPUComputePipeline;
  private causticsBindGroupLayout: GPUBindGroupLayout;
  private causticsBindGroup: GPUBindGroup;
  private causticsUniformBuffer: GPUBuffer;
  private causticsUniformData: Float32Array;

  private fxaaEnabled = true;
  private fxaaColorTexture: GPUTexture;
  private fxaaColorTextureView: GPUTextureView;
  private fxaaPipeline: GPURenderPipeline;
  private fxaaBindGroupLayout: GPUBindGroupLayout;
  private fxaaBindGroup: GPUBindGroup;
  private fxaaUniformBuffer: GPUBuffer;
  private fxaaUniformData: Float32Array;

  private _invVP = new THREE.Matrix4();
  private _vp = new THREE.Matrix4();
  private _camPos = new THREE.Vector3();
  private _camRight = new THREE.Vector3();
  private _camUp = new THREE.Vector3();
  private _camFwd = new THREE.Vector3();

  private debugMode = false;
  private debugPipeline: GPURenderPipeline | null = null;
  private debugBindGroupLayout: GPUBindGroupLayout | null = null;
  private debugBindGroup: GPUBindGroup | null = null;
  private debugUniformBuffer: GPUBuffer;
  private debugUniformData: Float32Array;
  private debugParticleCount = 0;

  private static readonly LIGHT_DIR = new THREE.Vector3(5, 8, 5).normalize();

  constructor(
    device: GPUDevice,
    densityFieldBuffer: GPUBuffer,
    paramsBuffer: GPUBuffer,
    fieldResolution: number,
    domainMin: THREE.Vector3,
    domainMax: THREE.Vector3,
    containerSize: THREE.Vector3,
  ) {
    this.device = device;
    this.fieldResolution = fieldResolution;

    this.canvas = document.createElement('canvas');
    this.canvas.style.position = 'fixed';
    this.canvas.style.top = '0';
    this.canvas.style.left = '0';
    this.canvas.style.width = '100%';
    this.canvas.style.height = '100%';
    document.body.appendChild(this.canvas);

    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context = this.canvas.getContext('webgpu') as unknown as GPUCanvasContext;
    this.context.configure({ device, format: this.format, alphaMode: 'opaque' });

    this.densityTexture = device.createTexture({
      size: [fieldResolution, fieldResolution, fieldResolution],
      format: 'rg32float',
      dimension: '3d',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.densityTextureView = this.densityTexture.createView();

    this.linearSampler = device.createSampler({
      minFilter: 'linear',
      magFilter: 'linear',
      mipmapFilter: 'nearest',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      addressModeW: 'clamp-to-edge',
    });

    this.repeatSampler = device.createSampler({
      minFilter: 'linear',
      magFilter: 'linear',
      mipmapFilter: 'nearest',
      addressModeU: 'repeat',
      addressModeV: 'repeat',
    });

    this.sandTexture = device.createTexture({
      size: [1, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.sandTextureView = this.sandTexture.createView();
    device.queue.writeTexture(
      { texture: this.sandTexture },
      new Uint8Array([194, 178, 128, 255]),
      { bytesPerRow: 4 },
      [1, 1],
    );

    const noiseData = generateWorleyNoise(256, 12);
    this.foamNoiseTexture = device.createTexture({
      size: [256, 256],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.foamNoiseTextureView = this.foamNoiseTexture.createView();
    device.queue.copyExternalImageToTexture(
      { source: noiseData },
      { texture: this.foamNoiseTexture },
      [256, 256],
    );

    // Spray render uniform buffer (96 bytes: mat4 VP + vec3 camRight + f32 pointSize + vec3 camUp + pad)
    this.sprayRenderUniformData = new Float32Array(24);
    this.sprayRenderUniformBuffer = device.createBuffer({
      size: 96,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Debug particles uniform buffer (96 bytes: mat4 VP + vec3 camRight + f32 pointSize + vec3 camUp + u32 particleCount)
    this.debugUniformData = new Float32Array(24);
    this.debugUniformBuffer = device.createBuffer({
      size: 96,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Buffer-to-texture compute pipeline
    const b2tModule = device.createShaderModule({ code: bufferToTextureShader });
    this.bufferToTexBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rg32float', viewDimension: '3d' } },
      ],
    });
    this.bufferToTexPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.bufferToTexBindGroupLayout] }),
      compute: { module: b2tModule, entryPoint: 'main' },
    });
    this.bufferToTexBindGroup = device.createBindGroup({
      layout: this.bufferToTexBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: densityFieldBuffer } },
        { binding: 2, resource: this.densityTextureView },
      ],
    });

    // Render uniform buffer (192 bytes)
    this.renderUniformData = new Float32Array(48);
    this.renderUniformBuffer = device.createBuffer({
      size: 192,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Set static uniforms
    // domainMin (offset 16) + threshold (offset 19)
    this.renderUniformData[16] = domainMin.x;
    this.renderUniformData[17] = domainMin.y;
    this.renderUniformData[18] = domainMin.z;
    this.renderUniformData[19] = 0.3; // threshold
    // domainMax (offset 20)
    this.renderUniformData[20] = domainMax.x;
    this.renderUniformData[21] = domainMax.y;
    this.renderUniformData[22] = domainMax.z;
    // lightDir (offset 24)
    const ld = WebGPURenderer.LIGHT_DIR;
    this.renderUniformData[24] = ld.x;
    this.renderUniformData[25] = ld.y;
    this.renderUniformData[26] = ld.z;
    // lightColor (offset 28)
    this.renderUniformData[28] = 1.0;
    this.renderUniformData[29] = 1.0;
    this.renderUniformData[30] = 1.0;
    // bgColor (offset 32) — 0x87CEEB (sky blue)
    this.renderUniformData[32] = 0x87 / 255;
    this.renderUniformData[33] = 0xCE / 255;
    this.renderUniformData[34] = 0xEB / 255;

    // Water render pipeline
    const waterModule = device.createShaderModule({ code: waterRaymarchShader });
    this.waterBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 7, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
    this.waterPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.waterBindGroupLayout] }),
      vertex: { module: waterModule, entryPoint: 'vs_main' },
      fragment: {
        module: waterModule,
        entryPoint: 'fs_main',
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'always' },
    });
    this.waterBindGroup = this.createWaterBindGroup();

    // Floor pipeline
    const floorModule = device.createShaderModule({ code: floorShader });
    this.floorUniformData = new Float32Array(24);
    this.floorUniformData[16] = ld.x;
    this.floorUniformData[17] = ld.y;
    this.floorUniformData[18] = ld.z;
    // [19] = time (written per frame)
    this.floorUniformData[20] = domainMin.x;
    this.floorUniformData[21] = domainMin.z;
    this.floorUniformData[22] = domainMax.x - domainMin.x;
    this.floorUniformData[23] = domainMax.z - domainMin.z;

    this.floorUniformBuffer = device.createBuffer({
      size: 96,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.floorBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
      ],
    });
    this.floorPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.floorBindGroupLayout] }),
      vertex: {
        module: floorModule,
        entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 24,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
            { shaderLocation: 1, offset: 12, format: 'float32x3' },
          ],
        }],
      },
      fragment: {
        module: floorModule,
        entryPoint: 'fs_main',
        targets: [{ format: this.format }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });
    // Caustics compute pipeline + texture
    this.causticsTexture = device.createTexture({
      size: [256, 256],
      format: 'r32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.causticsTextureView = this.causticsTexture.createView();
    this.causticsUniformData = new Float32Array(8);
    this.causticsUniformData[0] = domainMin.x;
    this.causticsUniformData[1] = domainMin.y;
    this.causticsUniformData[2] = domainMin.z;
    this.causticsUniformData[3] = 0.3; // threshold
    this.causticsUniformData[4] = domainMax.x;
    this.causticsUniformData[5] = domainMax.y;
    this.causticsUniformData[6] = domainMax.z;
    this.causticsUniformData[7] = 25.0; // strength
    this.causticsUniformBuffer = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.causticsUniformBuffer, 0, this.causticsUniformData);

    this.causticsBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'r32float', viewDimension: '2d' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });
    const causticsModule = device.createShaderModule({ code: causticsShader });
    this.causticsPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.causticsBindGroupLayout] }),
      compute: { module: causticsModule, entryPoint: 'main' },
    });
    this.causticsBindGroup = device.createBindGroup({
      layout: this.causticsBindGroupLayout,
      entries: [
        { binding: 0, resource: this.densityTextureView },
        { binding: 1, resource: this.linearSampler },
        { binding: 2, resource: this.causticsTextureView },
        { binding: 3, resource: { buffer: this.causticsUniformBuffer } },
      ],
    });

    this.floorBindGroup = this.createFloorBindGroup();

    // Floor geometry — matches container footprint
    const fhx = containerSize.x / 2, fhz = containerSize.z / 2, fd = 0.12;
    const FA = [-fhx, 0, -fhz], FB = [fhx, 0, -fhz], FC = [fhx, 0, fhz], FD = [-fhx, 0, fhz];
    const FE = [-fhx, -fd, -fhz], FF = [fhx, -fd, -fhz], FG = [fhx, -fd, fhz], FH = [-fhx, -fd, fhz];
    const floorVerts: number[] = [];
    const pushV = (p: number[], n: number[]) => floorVerts.push(p[0], p[1], p[2], n[0], n[1], n[2]);
    const topN = [0,1,0], frontN = [0,0,1], backN = [0,0,-1], rightN = [1,0,0], leftN = [-1,0,0];
    for (const v of [FA,FB,FC, FA,FC,FD]) pushV(v, topN);
    for (const v of [FD,FC,FG, FD,FG,FH]) pushV(v, frontN);
    for (const v of [FB,FA,FE, FB,FE,FF]) pushV(v, backN);
    for (const v of [FC,FB,FF, FC,FF,FG]) pushV(v, rightN);
    for (const v of [FA,FD,FH, FA,FH,FE]) pushV(v, leftN);
    this.floorVertexCount = floorVerts.length / 6;
    this.floorVertexBuffer = device.createBuffer({
      size: floorVerts.length * 4,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.floorVertexBuffer, 0, new Float32Array(floorVerts));

    // Initial depth texture (will be recreated on resize)
    this.depthTexture = device.createTexture({
      size: [1, 1],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    // FXAA post-process
    this.fxaaUniformData = new Float32Array(4);
    this.fxaaUniformBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.fxaaBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
    const fxaaModule = device.createShaderModule({ code: fxaaShader });
    this.fxaaPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.fxaaBindGroupLayout] }),
      vertex: { module: fxaaModule, entryPoint: 'vs_main' },
      fragment: {
        module: fxaaModule,
        entryPoint: 'fs_main',
        targets: [{ format: this.format }],
      },
      primitive: { topology: 'triangle-list' },
    });
    this.fxaaColorTexture = device.createTexture({
      size: [1, 1],
      format: this.format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.fxaaColorTextureView = this.fxaaColorTexture.createView();
    this.fxaaBindGroup = this.createFxaaBindGroup();
  }

  private createFloorBindGroup(): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.floorBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.floorUniformBuffer } },
        { binding: 1, resource: this.sandTextureView },
        { binding: 2, resource: this.repeatSampler },
        { binding: 3, resource: this.causticsTextureView },
        { binding: 4, resource: this.linearSampler },
        { binding: 5, resource: this.foamNoiseTextureView },
      ],
    });
  }

  private createFxaaBindGroup(): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.fxaaBindGroupLayout,
      entries: [
        { binding: 0, resource: this.fxaaColorTextureView },
        { binding: 1, resource: this.linearSampler },
        { binding: 2, resource: { buffer: this.fxaaUniformBuffer } },
      ],
    });
  }

  resize(width: number, height: number, pixelRatio: number, renderScale = 1.0) {
    const w = Math.floor(width * pixelRatio * renderScale);
    const h = Math.floor(height * pixelRatio * renderScale);
    if (w === this.width && h === this.height) return;
    this.width = w;
    this.height = h;

    this.canvas.width = w;
    this.canvas.height = h;

    this.depthTexture.destroy();
    this.depthTexture = this.device.createTexture({
      size: [w, h],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.fxaaColorTexture.destroy();
    this.fxaaColorTexture = this.device.createTexture({
      size: [w, h],
      format: this.format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.fxaaColorTextureView = this.fxaaColorTexture.createView();
    this.fxaaUniformData[0] = 1.0 / w;
    this.fxaaUniformData[1] = 1.0 / h;
    this.device.queue.writeBuffer(this.fxaaUniformBuffer, 0, this.fxaaUniformData);
    this.fxaaBindGroup = this.createFxaaBindGroup();
  }

  encodeFrame(encoder: GPUCommandEncoder, camera: THREE.PerspectiveCamera, profiler?: GPUProfiler | null, time = 0) {
    this._vp.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    camera.getWorldPosition(this._camPos);
    camera.matrixWorld.extractBasis(this._camRight, this._camUp, this._camFwd);

    if (this.debugMode && this.debugPipeline && this.debugBindGroup) {
      // Debug mode: skip raymarching, render particles as colored dots
      const vpE = this._vp.elements;
      for (let i = 0; i < 16; i++) this.debugUniformData[i] = vpE[i];
      this.debugUniformData[16] = this._camRight.x;
      this.debugUniformData[17] = this._camRight.y;
      this.debugUniformData[18] = this._camRight.z;
      this.debugUniformData[19] = 0.008;
      this.debugUniformData[20] = this._camUp.x;
      this.debugUniformData[21] = this._camUp.y;
      this.debugUniformData[22] = this._camUp.z;
      const u32View = new Uint32Array(this.debugUniformData.buffer);
      u32View[23] = this.debugParticleCount;
      this.device.queue.writeBuffer(this.debugUniformBuffer, 0, this.debugUniformData);

      // Floor uniforms
      for (let i = 0; i < 16; i++) this.floorUniformData[i] = vpE[i];
      this.floorUniformData[19] = time;
      this.device.queue.writeBuffer(this.floorUniformBuffer, 0, this.floorUniformData);

      const canvasView = this.context.getCurrentTexture().createView();
      const depthView = this.depthTexture.createView();
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: canvasView,
          clearValue: { r: 0x87 / 255, g: 0xCE / 255, b: 0xEB / 255, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
        depthStencilAttachment: {
          view: depthView,
          depthClearValue: 1.0,
          depthLoadOp: 'clear',
          depthStoreOp: 'store',
        },
      });

      pass.setPipeline(this.floorPipeline);
      pass.setBindGroup(0, this.floorBindGroup);
      pass.setVertexBuffer(0, this.floorVertexBuffer);
      pass.draw(this.floorVertexCount);

      pass.setPipeline(this.debugPipeline);
      pass.setBindGroup(0, this.debugBindGroup);
      pass.draw(4, this.debugParticleCount);

      pass.end();
      return;
    }

    // Buffer-to-texture compute pass
    const res = this.fieldResolution;
    const wg = Math.ceil(res / 4);
    const b2tDesc: GPUComputePassDescriptor = {};
    const b2tTs = profiler?.timestampWrites('bufferToTexture');
    if (b2tTs) b2tDesc.timestampWrites = b2tTs;
    const b2tPass = encoder.beginComputePass(b2tDesc);
    b2tPass.setPipeline(this.bufferToTexPipeline);
    b2tPass.setBindGroup(0, this.bufferToTexBindGroup);
    b2tPass.dispatchWorkgroups(wg, wg, wg);
    b2tPass.end();

    // Caustics compute pass
    const causticsDesc: GPUComputePassDescriptor = {};
    const causticsTs = profiler?.timestampWrites('causticsPass');
    if (causticsTs) causticsDesc.timestampWrites = causticsTs;
    this.causticsUniformData[3] = this.renderUniformData[19]; // sync threshold
    this.device.queue.writeBuffer(this.causticsUniformBuffer, 0, this.causticsUniformData);
    const causticsPass = encoder.beginComputePass(causticsDesc);
    causticsPass.setPipeline(this.causticsPipeline);
    causticsPass.setBindGroup(0, this.causticsBindGroup);
    causticsPass.dispatchWorkgroups(32, 32); // 256/8 = 32
    causticsPass.end();

    // Update render uniforms
    this._invVP.copy(camera.projectionMatrix).multiply(camera.matrixWorldInverse).invert();
    const e = this._invVP.elements;
    for (let i = 0; i < 16; i++) {
      this.renderUniformData[i] = e[i];
    }
    this.renderUniformData[36] = this.width;
    this.renderUniformData[37] = this.height;
    this.renderUniformData[40] = this._camPos.x;
    this.renderUniformData[41] = this._camPos.y;
    this.renderUniformData[42] = this._camPos.z;

    this.device.queue.writeBuffer(this.renderUniformBuffer, 0, this.renderUniformData);

    // Update floor uniforms (viewProjection matrix + time)
    const vpE = this._vp.elements;
    for (let i = 0; i < 16; i++) {
      this.floorUniformData[i] = vpE[i];
    }
    this.floorUniformData[19] = time;
    this.device.queue.writeBuffer(this.floorUniformBuffer, 0, this.floorUniformData);

    // Render pass
    const canvasView = this.context.getCurrentTexture().createView();
    const colorView = this.fxaaEnabled ? this.fxaaColorTextureView : canvasView;
    const depthView = this.depthTexture.createView();

    const renderPassDesc: GPURenderPassDescriptor = {
      colorAttachments: [{
        view: colorView,
        clearValue: { r: 0x87 / 255, g: 0xCE / 255, b: 0xEB / 255, a: 1.0 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: depthView,
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    };
    const rpTs = profiler?.timestampWrites('renderPass');
    if (rpTs) renderPassDesc.timestampWrites = rpTs;
    const pass = encoder.beginRenderPass(renderPassDesc);

    // Draw floor
    pass.setPipeline(this.floorPipeline);
    pass.setBindGroup(0, this.floorBindGroup);
    pass.setVertexBuffer(0, this.floorVertexBuffer);
    pass.draw(this.floorVertexCount);

    // Draw water (fullscreen triangle, alpha-blends over floor)
    pass.setPipeline(this.waterPipeline);
    pass.setBindGroup(0, this.waterBindGroup);
    pass.draw(3);

    pass.end();

    if (this.fxaaEnabled) {
      const fxaaPassDesc: GPURenderPassDescriptor = {
        colorAttachments: [{
          view: canvasView,
          loadOp: 'clear' as const,
          storeOp: 'store' as const,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        }],
      };
      const fxaaTs = profiler?.timestampWrites('fxaaPass');
      if (fxaaTs) fxaaPassDesc.timestampWrites = fxaaTs;
      const fxaaPass = encoder.beginRenderPass(fxaaPassDesc);
      fxaaPass.setPipeline(this.fxaaPipeline);
      fxaaPass.setBindGroup(0, this.fxaaBindGroup);
      fxaaPass.draw(3);
      fxaaPass.end();
    }

    // Draw spray AFTER FXAA — small additive particles get eaten by FXAA's
    // edge detection, so they render directly to the canvas with scene depth
    if (this.sprayRenderPipeline && this.sprayRenderBindGroup) {
      const vpE2 = this._vp.elements;
      for (let i = 0; i < 16; i++) {
        this.sprayRenderUniformData[i] = vpE2[i];
      }
      camera.matrixWorld.extractBasis(this._camRight, this._camUp, this._camFwd);
      this.sprayRenderUniformData[16] = this._camRight.x;
      this.sprayRenderUniformData[17] = this._camRight.y;
      this.sprayRenderUniformData[18] = this._camRight.z;
      this.sprayRenderUniformData[19] = 0.008;
      this.sprayRenderUniformData[20] = this._camUp.x;
      this.sprayRenderUniformData[21] = this._camUp.y;
      this.sprayRenderUniformData[22] = this._camUp.z;
      this.device.queue.writeBuffer(this.sprayRenderUniformBuffer, 0, this.sprayRenderUniformData);

      const sprayPassDesc: GPURenderPassDescriptor = {
        colorAttachments: [{
          view: canvasView,
          loadOp: 'load' as const,
          storeOp: 'store' as const,
        }],
        depthStencilAttachment: {
          view: depthView,
          depthLoadOp: 'load',
          depthStoreOp: 'store',
        },
      };
      const sprayPass = encoder.beginRenderPass(sprayPassDesc);
      sprayPass.setPipeline(this.sprayRenderPipeline);
      sprayPass.setBindGroup(0, this.sprayRenderBindGroup);
      sprayPass.draw(4, 32768);
      sprayPass.end();
    }
  }

  setFxaaEnabled(enabled: boolean) {
    this.fxaaEnabled = enabled;
  }

  setThreshold(value: number) {
    this.renderUniformData[19] = value;
  }

  setLightEnabled(enabled: boolean) {
    const ld = enabled ? WebGPURenderer.LIGHT_DIR : new THREE.Vector3(0, 0, 0);
    this.renderUniformData[24] = ld.x;
    this.renderUniformData[25] = ld.y;
    this.renderUniformData[26] = ld.z;
    this.floorUniformData[16] = ld.x;
    this.floorUniformData[17] = ld.y;
    this.floorUniformData[18] = ld.z;
  }

  setObstaclesBuffer(buffer: GPUBuffer) {
    this.obstaclesUniformBuffer = buffer;
    this.waterBindGroup = this.createWaterBindGroup();
  }

  private createWaterBindGroup(): GPUBindGroup {
    if (!this.obstaclesUniformBuffer) {
      this.obstaclesUniformBuffer = this.device.createBuffer({
        size: 272,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
    return this.device.createBindGroup({
      layout: this.waterBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.renderUniformBuffer } },
        { binding: 1, resource: this.densityTextureView },
        { binding: 2, resource: this.linearSampler },
        { binding: 3, resource: this.sandTextureView },
        { binding: 4, resource: this.repeatSampler },
        { binding: 5, resource: this.foamNoiseTextureView },
        { binding: 6, resource: this.repeatSampler },
        { binding: 7, resource: { buffer: this.obstaclesUniformBuffer } },
      ],
    });
  }

  async loadFloorTexture(url: string) {
    const response = await fetch(url);
    const blob = await response.blob();
    const bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none' });

    this.sandTexture.destroy();
    this.sandTexture = this.device.createTexture({
      size: [bitmap.width, bitmap.height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.sandTextureView = this.sandTexture.createView();

    this.device.queue.copyExternalImageToTexture(
      { source: bitmap },
      { texture: this.sandTexture },
      [bitmap.width, bitmap.height],
    );

    this.floorBindGroup = this.createFloorBindGroup();
    this.waterBindGroup = this.createWaterBindGroup();
  }

  initSprayRenderer(sprayBuffer: GPUBuffer) {
    const sprayModule = this.device.createShaderModule({ code: sprayRenderShader });

    this.sprayRenderBindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });

    this.sprayRenderPipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.sprayRenderBindGroupLayout] }),
      vertex: { module: sprayModule, entryPoint: 'vs_main' },
      fragment: {
        module: sprayModule,
        entryPoint: 'fs_main',
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one' },
            alpha: { srcFactor: 'one', dstFactor: 'one' },
          },
        }],
      },
      primitive: { topology: 'triangle-strip' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less' },
    });

    this.sprayRenderBindGroup = this.device.createBindGroup({
      layout: this.sprayRenderBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.sprayRenderUniformBuffer } },
        { binding: 1, resource: { buffer: sprayBuffer } },
      ],
    });
  }

  initDebugRenderer(positionsBuffer: GPUBuffer, densityPressureBuffer: GPUBuffer, particleCount: number) {
    this.debugParticleCount = particleCount;
    const module = this.device.createShaderModule({ code: debugParticlesShader });
    this.debugBindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    this.debugPipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.debugBindGroupLayout] }),
      vertex: { module, entryPoint: 'vs_main' },
      fragment: {
        module,
        entryPoint: 'fs_main',
        targets: [{ format: this.format }],
      },
      primitive: { topology: 'triangle-strip' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });
    this.debugBindGroup = this.device.createBindGroup({
      layout: this.debugBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.debugUniformBuffer } },
        { binding: 1, resource: { buffer: positionsBuffer } },
        { binding: 2, resource: { buffer: densityPressureBuffer } },
      ],
    });
  }

  invalidateDebugBindings() {
    this.debugBindGroup = null;
  }

  rebindDebugBuffers(positionsBuffer: GPUBuffer, densityPressureBuffer: GPUBuffer, particleCount: number) {
    this.debugParticleCount = particleCount;
    if (!this.debugBindGroupLayout) return;
    this.debugBindGroup = this.device.createBindGroup({
      layout: this.debugBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.debugUniformBuffer } },
        { binding: 1, resource: { buffer: positionsBuffer } },
        { binding: 2, resource: { buffer: densityPressureBuffer } },
      ],
    });
  }

  setDebugMode(enabled: boolean) {
    this.debugMode = enabled;
  }

  getDebugMode(): boolean {
    return this.debugMode;
  }

  rebindComputeBuffers(densityFieldBuffer: GPUBuffer, paramsBuffer: GPUBuffer, sprayBuffer?: GPUBuffer) {
    this.bufferToTexBindGroup = this.device.createBindGroup({
      layout: this.bufferToTexBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: paramsBuffer } },
        { binding: 1, resource: { buffer: densityFieldBuffer } },
        { binding: 2, resource: this.densityTextureView },
      ],
    });
    if (sprayBuffer && this.sprayRenderBindGroupLayout) {
      this.sprayRenderBindGroup = this.device.createBindGroup({
        layout: this.sprayRenderBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.sprayRenderUniformBuffer } },
          { binding: 1, resource: { buffer: sprayBuffer } },
        ],
      });
    }
  }

  dispose() {
    this.canvas.remove();
    this.densityTexture.destroy();
    this.sandTexture.destroy();
    this.foamNoiseTexture.destroy();
    this.depthTexture.destroy();
    this.renderUniformBuffer.destroy();
    this.sprayRenderUniformBuffer.destroy();
    this.floorUniformBuffer.destroy();
    this.floorVertexBuffer.destroy();
    this.causticsTexture.destroy();
    this.causticsUniformBuffer.destroy();
    this.fxaaColorTexture.destroy();
    this.fxaaUniformBuffer.destroy();
    this.debugUniformBuffer.destroy();
  }
}
