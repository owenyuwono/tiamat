import * as THREE from 'three';
import type { GPUProfiler } from './GPUProfiler';
import bufferToTextureShader from './shaders/bufferToTexture.wgsl?raw';
import wireframeShader from './shaders/wireframe.wgsl?raw';
import waterRaymarchShader from './shaders/waterRaymarch.wgsl?raw';

export class WebGPURenderer {
  private device: GPUDevice;
  private context: GPUCanvasContext;
  private format: GPUTextureFormat;
  readonly canvas: HTMLCanvasElement;

  private densityTexture: GPUTexture;
  private densityTextureView: GPUTextureView;
  private linearSampler: GPUSampler;

  private bufferToTexPipeline: GPUComputePipeline;
  private bufferToTexBindGroup: GPUBindGroup;
  private fieldResolution: number;

  private waterPipeline: GPURenderPipeline;
  private waterBindGroup: GPUBindGroup;
  private renderUniformBuffer: GPUBuffer;
  private renderUniformData: Float32Array;

  private wireframePipeline: GPURenderPipeline;
  private wireframeBindGroup: GPUBindGroup;
  private wireframeVertexBuffer: GPUBuffer;
  private wireframeUniformBuffer: GPUBuffer;
  private wireframeUniformData: Float32Array;

  private depthTexture: GPUTexture;
  private width = 0;
  private height = 0;

  private _invVP = new THREE.Matrix4();
  private _vp = new THREE.Matrix4();
  private _camPos = new THREE.Vector3();

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

    // Buffer-to-texture compute pipeline
    const b2tModule = device.createShaderModule({ code: bufferToTextureShader });
    const b2tBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rg32float', viewDimension: '3d' } },
      ],
    });
    this.bufferToTexPipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [b2tBindGroupLayout] }),
      compute: { module: b2tModule, entryPoint: 'main' },
    });
    this.bufferToTexBindGroup = device.createBindGroup({
      layout: b2tBindGroupLayout,
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
    this.renderUniformData[19] = 0.75; // threshold
    // domainMax (offset 20)
    this.renderUniformData[20] = domainMax.x;
    this.renderUniformData[21] = domainMax.y;
    this.renderUniformData[22] = domainMax.z;
    // lightDir (offset 24)
    const ld = new THREE.Vector3(5, 8, 5).normalize();
    this.renderUniformData[24] = ld.x;
    this.renderUniformData[25] = ld.y;
    this.renderUniformData[26] = ld.z;
    // lightColor (offset 28)
    this.renderUniformData[28] = 1.0;
    this.renderUniformData[29] = 1.0;
    this.renderUniformData[30] = 1.0;
    // bgColor (offset 32) — 0xd0d8e8
    this.renderUniformData[32] = 0xd0 / 255;
    this.renderUniformData[33] = 0xd8 / 255;
    this.renderUniformData[34] = 0xe8 / 255;

    // Water render pipeline
    const waterModule = device.createShaderModule({ code: waterRaymarchShader });
    const waterBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float', viewDimension: '3d' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });
    this.waterPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [waterBindGroupLayout] }),
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
    this.waterBindGroup = device.createBindGroup({
      layout: waterBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.renderUniformBuffer } },
        { binding: 1, resource: this.densityTextureView },
        { binding: 2, resource: this.linearSampler },
      ],
    });

    // Wireframe pipeline
    const wireModule = device.createShaderModule({ code: wireframeShader });
    this.wireframeUniformData = new Float32Array(20);
    // color (0x8090a0)
    this.wireframeUniformData[16] = 0x80 / 255;
    this.wireframeUniformData[17] = 0x90 / 255;
    this.wireframeUniformData[18] = 0xa0 / 255;
    this.wireframeUniformData[19] = 1.0;

    this.wireframeUniformBuffer = device.createBuffer({
      size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const wireBindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
    this.wireframePipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [wireBindGroupLayout] }),
      vertex: {
        module: wireModule,
        entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 12,
          attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }],
        }],
      },
      fragment: {
        module: wireModule,
        entryPoint: 'fs_main',
        targets: [{ format: this.format }],
      },
      primitive: { topology: 'line-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });
    this.wireframeBindGroup = device.createBindGroup({
      layout: wireBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.wireframeUniformBuffer } },
      ],
    });

    // Wireframe vertex buffer — 12 edges of a box centered at (0, cy, 0)
    const hx = containerSize.x / 2, hy = containerSize.y, hz = containerSize.z / 2;
    const lo = [-hx, 0, -hz] as const;
    const hi = [hx, hy, hz] as const;
    const edges = [
      // bottom
      lo[0],lo[1],lo[2], hi[0],lo[1],lo[2],
      hi[0],lo[1],lo[2], hi[0],lo[1],hi[2],
      hi[0],lo[1],hi[2], lo[0],lo[1],hi[2],
      lo[0],lo[1],hi[2], lo[0],lo[1],lo[2],
      // top
      lo[0],hi[1],lo[2], hi[0],hi[1],lo[2],
      hi[0],hi[1],lo[2], hi[0],hi[1],hi[2],
      hi[0],hi[1],hi[2], lo[0],hi[1],hi[2],
      lo[0],hi[1],hi[2], lo[0],hi[1],lo[2],
      // verticals
      lo[0],lo[1],lo[2], lo[0],hi[1],lo[2],
      hi[0],lo[1],lo[2], hi[0],hi[1],lo[2],
      hi[0],lo[1],hi[2], hi[0],hi[1],hi[2],
      lo[0],lo[1],hi[2], lo[0],hi[1],hi[2],
    ];
    this.wireframeVertexBuffer = device.createBuffer({
      size: edges.length * 4,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.wireframeVertexBuffer, 0, new Float32Array(edges));

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

    // Update wireframe uniforms (viewProjection matrix)
    this._vp.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    const vpE = this._vp.elements;
    for (let i = 0; i < 16; i++) {
      this.wireframeUniformData[i] = vpE[i];
    }
    this.device.queue.writeBuffer(this.wireframeUniformBuffer, 0, this.wireframeUniformData);

    // Render pass
    const colorView = this.context.getCurrentTexture().createView();
    const depthView = this.depthTexture.createView();

    const renderPassDesc: GPURenderPassDescriptor = {
      colorAttachments: [{
        view: colorView,
        clearValue: { r: 0xd0 / 255, g: 0xd8 / 255, b: 0xe8 / 255, a: 1.0 },
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

    // Draw wireframe
    pass.setPipeline(this.wireframePipeline);
    pass.setBindGroup(0, this.wireframeBindGroup);
    pass.setVertexBuffer(0, this.wireframeVertexBuffer);
    pass.draw(24);

    // Draw water (fullscreen triangle)
    pass.setPipeline(this.waterPipeline);
    pass.setBindGroup(0, this.waterBindGroup);
    pass.draw(3);

    pass.end();
  }

  dispose() {
    this.densityTexture.destroy();
    this.depthTexture.destroy();
    this.renderUniformBuffer.destroy();
    this.wireframeUniformBuffer.destroy();
    this.wireframeVertexBuffer.destroy();
  }
}
