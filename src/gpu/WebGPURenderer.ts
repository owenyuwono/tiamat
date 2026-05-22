import * as THREE from 'three';
import type { GPUProfiler } from './GPUProfiler';
import bufferToTextureShader from './shaders/bufferToTexture.wgsl?raw';
import waterRaymarchShader from './shaders/waterRaymarch.wgsl?raw';
import floorShader from './shaders/floor.wgsl?raw';
import sprayRenderShader from './shaders/sprayRender.wgsl?raw';
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

  private _invVP = new THREE.Matrix4();
  private _vp = new THREE.Matrix4();
  private _camPos = new THREE.Vector3();
  private _camRight = new THREE.Vector3();
  private _camUp = new THREE.Vector3();
  private _camFwd = new THREE.Vector3();

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
    this.floorUniformData = new Float32Array(20);
    this.floorUniformData[16] = ld.x;
    this.floorUniformData[17] = ld.y;
    this.floorUniformData[18] = ld.z;

    this.floorUniformBuffer = device.createBuffer({
      size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.floorBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '2d' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
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
    this.floorBindGroup = device.createBindGroup({
      layout: this.floorBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.floorUniformBuffer } },
        { binding: 1, resource: this.sandTextureView },
        { binding: 2, resource: this.repeatSampler },
      ],
    });

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
  }

  encodeFrame(encoder: GPUCommandEncoder, camera: THREE.PerspectiveCamera, profiler?: GPUProfiler | null) {
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

    // Update render uniforms
    this._invVP.copy(camera.projectionMatrix).multiply(camera.matrixWorldInverse).invert();
    const e = this._invVP.elements;
    for (let i = 0; i < 16; i++) {
      this.renderUniformData[i] = e[i];
    }
    this.renderUniformData[36] = this.width;
    this.renderUniformData[37] = this.height;
    camera.getWorldPosition(this._camPos);
    this.renderUniformData[40] = this._camPos.x;
    this.renderUniformData[41] = this._camPos.y;
    this.renderUniformData[42] = this._camPos.z;

    this.device.queue.writeBuffer(this.renderUniformBuffer, 0, this.renderUniformData);

    // Update floor uniforms (viewProjection matrix)
    this._vp.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    const vpE = this._vp.elements;
    for (let i = 0; i < 16; i++) {
      this.floorUniformData[i] = vpE[i];
    }
    this.device.queue.writeBuffer(this.floorUniformBuffer, 0, this.floorUniformData);

    // Render pass
    const colorView = this.context.getCurrentTexture().createView();
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

    // Draw spray particles
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

      pass.setPipeline(this.sprayRenderPipeline);
      pass.setBindGroup(0, this.sprayRenderBindGroup);
      pass.draw(4, 32768);
    }

    pass.end();
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

    this.floorBindGroup = this.device.createBindGroup({
      layout: this.floorBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.floorUniformBuffer } },
        { binding: 1, resource: this.sandTextureView },
        { binding: 2, resource: this.repeatSampler },
      ],
    });

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
  }
}
