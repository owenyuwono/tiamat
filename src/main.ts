import './style.css';
import './ui/panels.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { SPHSimulation } from './sph/simulation';
import { WaterRenderer } from './rendering/WaterRenderer';
import { GPUCompute } from './gpu/GPUCompute';
import { FLIPCompute } from './gpu/FLIPCompute';
import { EulerCompute } from './gpu/EulerCompute';
import { GPUProfiler } from './gpu/GPUProfiler';
import { WebGPURenderer } from './gpu/WebGPURenderer';
import { createDefaultConfig } from './ui/SimConfig';
import type { Algorithm } from './ui/SimConfig';
import { AlgorithmPicker } from './ui/AlgorithmPicker';
import { ControlPanel } from './ui/ControlPanel';
import { StatsPanel } from './ui/StatsPanel';
import { RigidBodySystem } from './gpu/RigidBodySystem';

const CONTAINER_SIZE = new THREE.Vector3(4, 4, 4);
const FIELD_RESOLUTION = 100;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87CEEB);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(6, 4.5, 6);
camera.lookAt(0, 2, 0);

async function init() {
  const config = createDefaultConfig();
  const containerSize = CONTAINER_SIZE;
  const fieldResolution = FIELD_RESOLUTION;
  const splatPad = config.splatRadius;
  const domainMin = new THREE.Vector3(-containerSize.x / 2 - splatPad, -splatPad, -containerSize.z / 2 - splatPad);
  const domainMax = new THREE.Vector3(containerSize.x / 2 + splatPad, containerSize.y + splatPad, containerSize.z / 2 + splatPad);

  let gpuCompute: GPUCompute | null = null;
  let flipCompute: FLIPCompute | null = null;
  let eulerCompute: EulerCompute | null = null;
  let activeCompute: GPUCompute | FLIPCompute | EulerCompute | null = null;
  let glRenderer: THREE.WebGLRenderer | null = null;
  let webgpuRenderer: WebGPURenderer | null = null;
  let waterRenderer: WaterRenderer | null = null;
  let profiler: GPUProfiler | null = null;
  let controls: OrbitControls;

  let savedPosX = new Float32Array(0);
  let savedPosY = new Float32Array(0);
  let savedPosZ = new Float32Array(0);

  function generatePositions(count: number) {
    const sim = new SPHSimulation(scene, count, containerSize);
    sim.setInstancedRendering(false);
    const p = sim.getParticlePositions();
    savedPosX = new Float32Array(p.posX);
    savedPosY = new Float32Array(p.posY);
    savedPosZ = new Float32Array(p.posZ);
    return sim;
  }

  let sim = generatePositions(config.particleCount);

  gpuCompute = await GPUCompute.create(
    config.particleCount, containerSize, fieldResolution, domainMin, domainMax, config.splatRadius
  );

  if (gpuCompute) {
    const device = gpuCompute.getDevice();
    device.onuncapturederror = (e) => console.error('WebGPU uncaptured error:', e.error);

    device.pushErrorScope('validation');
    flipCompute = new FLIPCompute(
      device, config.particleCount, containerSize, fieldResolution, domainMin, domainMax, config.splatRadius,
    );
    device.popErrorScope().then(err => { if (err) console.error('FLIP construction validation error:', err.message); });

    eulerCompute = new EulerCompute(
      device, config.particleCount, containerSize, fieldResolution, domainMin, domainMax, config.splatRadius,
    );
    activeCompute = gpuCompute;

    webgpuRenderer = new WebGPURenderer(
      device,
      gpuCompute.getDensityFieldBuffer(),
      gpuCompute.getParamsBuffer(),
      gpuCompute.getFieldResolution(),
      domainMin, domainMax, containerSize,
    );
    webgpuRenderer.resize(window.innerWidth, window.innerHeight, Math.min(window.devicePixelRatio, 2), config.renderScale);
    controls = new OrbitControls(camera, webgpuRenderer.canvas);

    if (device.features.has('timestamp-query')) {
      profiler = new GPUProfiler(device);
      profiler.setParticleCount(config.particleCount);
    }

    gpuCompute.uploadInitialPositions(savedPosX, savedPosY, savedPosZ);
    webgpuRenderer.initDebugRenderer(gpuCompute.getPositionsBuffer(), gpuCompute.getDensityPressureBuffer(), config.particleCount);
    webgpuRenderer.initSprayRenderer(gpuCompute.getSprayBuffer());
    webgpuRenderer.setObstaclesBuffer(gpuCompute.getObstaclesUniformBuffer());
    webgpuRenderer.loadFloorTexture('/sand_diff.jpg');
    console.log('WebGPU render + compute enabled');
  } else {
    glRenderer = new THREE.WebGLRenderer({ antialias: true });
    glRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    glRenderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(glRenderer.domElement);
    controls = new OrbitControls(camera, glRenderer.domElement);

    scene.add(new THREE.AmbientLight(0x8090b0, 1.0));
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
    dirLight.position.set(5, 8, 5);
    scene.add(dirLight);

    const boxGeo = new THREE.BoxGeometry(containerSize.x, containerSize.y, containerSize.z);
    const edges = new THREE.EdgesGeometry(boxGeo);
    const wireframe = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x8090a0 }));
    wireframe.position.set(0, containerSize.y / 2, 0);
    scene.add(wireframe);

    waterRenderer = new WaterRenderer(scene, domainMin, domainMax, {
      resolution: fieldResolution, splatRadius: config.splatRadius, threshold: config.threshold,
    });

    console.log('WebGPU unavailable, using CPU SPH fallback');
  }

  const rigidBodies = new RigidBodySystem(containerSize, -9.81);

  if (webgpuRenderer) {
    const canvas = webgpuRenderer.canvas;
    let isDragging = false;
    let pointerDownPos: { x: number; y: number } | null = null;

    function ndcFromEvent(e: PointerEvent) {
      const rect = canvas.getBoundingClientRect();
      return {
        x: ((e.clientX - rect.left) / rect.width) * 2 - 1,
        y: -((e.clientY - rect.top) / rect.height) * 2 + 1,
      };
    }

    canvas.addEventListener('pointerdown', (e) => {
      pointerDownPos = { x: e.clientX, y: e.clientY };
      camera.updateMatrixWorld();
      const ndc = ndcFromEvent(e);
      const { origin, direction } = rigidBodies.getRayFromNDC(ndc.x, ndc.y, camera);

      if (rigidBodies.hasBody() && rigidBodies.hitTest(origin, direction)) {
        isDragging = true;
        controls.enabled = false;
        rigidBodies.startDrag();
        canvas.setPointerCapture(e.pointerId);
      }
    });

    canvas.addEventListener('pointermove', (e) => {
      if (!isDragging) return;
      camera.updateMatrixWorld();
      const ndc = ndcFromEvent(e);
      const { origin, direction } = rigidBodies.getRayFromNDC(ndc.x, ndc.y, camera);
      const now = performance.now();
      const dt = Math.max((now - lastTime) / 1000, 1 / 120);
      rigidBodies.updateDrag(origin, direction, dt);
    });

    canvas.addEventListener('pointerup', (e) => {
      if (isDragging) {
        isDragging = false;
        controls.enabled = true;
        rigidBodies.endDrag();
        canvas.releasePointerCapture(e.pointerId);
        pointerDownPos = null;
        return;
      }

      if (!pointerDownPos) return;
      const dx = e.clientX - pointerDownPos.x;
      const dy = e.clientY - pointerDownPos.y;
      pointerDownPos = null;
      if (dx * dx + dy * dy > 25) return;

      camera.updateMatrixWorld();
      const ndc = ndcFromEvent(e);
      rigidBodies.raycastSpawn(ndc.x, ndc.y, camera, 0.3);
    });
  }

  controls.enableDamping = true;
  controls.target.set(0, 2, 0);

  function switchAlgorithm(algo: Algorithm) {
    if (!gpuCompute || !flipCompute || !eulerCompute || !webgpuRenderer) return;
    if (algo === 'sph') activeCompute = gpuCompute;
    else if (algo === 'flip') activeCompute = flipCompute;
    else activeCompute = eulerCompute;
    activeCompute.uploadInitialPositions(savedPosX, savedPosY, savedPosZ);
    activeCompute.resetVelocities();
    webgpuRenderer.rebindComputeBuffers(
      activeCompute.getDensityFieldBuffer(),
      activeCompute.getParamsBuffer(),
      algo === 'sph' ? gpuCompute.getSprayBuffer() : undefined,
    );
  }

  async function reinitGPU(count: number) {
    if (!gpuCompute || !webgpuRenderer) return;
    const device = gpuCompute.getDevice();

    webgpuRenderer.invalidateDebugBindings();
    gpuCompute.dispose();
    if (flipCompute) flipCompute.dispose();
    if (eulerCompute) eulerCompute.dispose();

    gpuCompute = new GPUCompute(
      device, count, containerSize, fieldResolution, domainMin, domainMax, config.splatRadius,
    );
    flipCompute = new FLIPCompute(
      device, count, containerSize, fieldResolution, domainMin, domainMax, config.splatRadius,
    );
    eulerCompute = new EulerCompute(
      device, count, containerSize, fieldResolution, domainMin, domainMax, config.splatRadius,
    );
    if (config.algorithm === 'sph') activeCompute = gpuCompute;
    else if (config.algorithm === 'flip') activeCompute = flipCompute;
    else activeCompute = eulerCompute;

    sim = generatePositions(count);
    activeCompute.uploadInitialPositions(savedPosX, savedPosY, savedPosZ);

    webgpuRenderer.rebindComputeBuffers(
      activeCompute.getDensityFieldBuffer(),
      activeCompute.getParamsBuffer(),
      gpuCompute.getSprayBuffer(),
    );
    webgpuRenderer.rebindDebugBuffers(gpuCompute.getPositionsBuffer(), gpuCompute.getDensityPressureBuffer(), count);

    profiler?.setParticleCount(count);
  }

  const isGPU = !!gpuCompute;
  const statsPanel = new StatsPanel();

  const algorithmPicker = isGPU
    ? new AlgorithmPicker(config.algorithm, (algo) => {
        config.algorithm = algo;
        switchAlgorithm(algo);
      })
    : undefined;

  new ControlPanel(config, {
    gpuMode: isGPU,
    algorithmPicker,
    onReset: () => {
      if (activeCompute) {
        activeCompute.uploadInitialPositions(savedPosX, savedPosY, savedPosZ);
        activeCompute.resetVelocities();
      }
      rigidBodies.reset();
    },
    onRenderScaleChange: (scale: number) => {
      if (webgpuRenderer) {
        webgpuRenderer.resize(window.innerWidth, window.innerHeight, Math.min(window.devicePixelRatio, 2), scale);
      }
    },
    onParticleCountChange: (count: number) => {
      reinitGPU(count);
    },
  });

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    if (webgpuRenderer) {
      webgpuRenderer.resize(window.innerWidth, window.innerHeight, Math.min(window.devicePixelRatio, 2), config.renderScale);
    }
    if (glRenderer) {
      glRenderer.setSize(window.innerWidth, window.innerHeight);
    }
  });

  const rendererSize = new THREE.Vector2();
  const lightRef = new THREE.DirectionalLight(0xffffff, 1.0);
  lightRef.position.set(5, 8, 5);
  let lastTime = performance.now();
  let elapsedTime = 0;

  async function animate() {
    const now = performance.now();
    const dtMs = Math.min(now - lastTime, 16);
    const dt = dtMs / 1000;
    lastTime = now;
    elapsedTime += dt;

    const fixedDt = 0.008;
    const substeps = config.paused ? 0 : Math.min(Math.ceil(dt / fixedDt), config.substepLimit);

    controls.update();
    camera.updateMatrixWorld();

    if (activeCompute && webgpuRenderer) {
      webgpuRenderer.setLightEnabled(config.lightEnabled);
      webgpuRenderer.setFxaaEnabled(config.fxaaEnabled);
      webgpuRenderer.setDebugMode(config.debugMode);
      activeCompute.updateSimConfig(config);
      webgpuRenderer.setThreshold(config.threshold);

      if (gpuCompute && !config.paused) {
        rigidBodies.writeUniform(gpuCompute.getDevice(), gpuCompute.getObstaclesUniformBuffer());
      }

      profiler?.beginFrame();
      profiler?.setSubsteps(substeps);
      const device = activeCompute.getDevice();
      device.pushErrorScope('validation');
      const encoder = device.createCommandEncoder();
      if (!config.paused) {
        activeCompute.encodeStep(encoder, substeps, profiler);
      }
      webgpuRenderer.encodeFrame(encoder, camera, profiler, elapsedTime);
      profiler?.resolve(encoder);
      device.queue.submit([encoder.finish()]);
      device.popErrorScope().then(err => { if (err) console.error('Frame validation error:', err.message); });
      await device.queue.onSubmittedWorkDone();

      if (!config.paused) {
        rigidBodies.integrate(substeps, fixedDt);
      }

      await profiler?.readback();
    } else if (glRenderer && waterRenderer) {
      if (!config.paused) {
        sim.step(dt);
      }
      const particles = sim.getParticlePositions();
      const xs = sim.getXSPH();
      glRenderer.getDrawingBufferSize(rendererSize);
      waterRenderer.update(
        particles.posX, particles.posY, particles.posZ,
        xs.xsphX, xs.xsphY, xs.xsphZ,
        particles.count, lightRef, camera, rendererSize
      );
      glRenderer.render(scene, camera);
    }

    statsPanel.update(dtMs, profiler?.getSnapshot() ?? null, config.particleCount, substeps);

    requestAnimationFrame(animate);
  }

  animate();
}

init();
