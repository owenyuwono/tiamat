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

const CONTAINER_SIZE = new THREE.Vector3(6, 4, 6);
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
  const domainMin = new THREE.Vector3(-containerSize.x / 2, 0, -containerSize.z / 2);
  const domainMax = new THREE.Vector3(containerSize.x / 2, containerSize.y, containerSize.z / 2);

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
    );
  }

  async function reinitGPU(count: number) {
    if (!gpuCompute || !webgpuRenderer) return;
    const device = gpuCompute.getDevice();

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
    );

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

  async function animate() {
    const now = performance.now();
    const dtMs = Math.min(now - lastTime, 16);
    const dt = dtMs / 1000;
    lastTime = now;

    const fixedDt = 0.004;
    const substeps = config.paused ? 0 : Math.min(Math.ceil(dt / fixedDt), config.substepLimit);

    controls.update();
    camera.updateMatrixWorld();

    if (activeCompute && webgpuRenderer) {
      webgpuRenderer.setLightEnabled(config.lightEnabled);
      activeCompute.updateSimConfig(config);
      webgpuRenderer.setThreshold(config.threshold);

      profiler?.beginFrame();
      profiler?.setSubsteps(substeps);
      const device = activeCompute.getDevice();
      device.pushErrorScope('validation');
      const encoder = device.createCommandEncoder();
      if (!config.paused) {
        activeCompute.encodeStep(encoder, substeps, profiler);
      }
      webgpuRenderer.encodeFrame(encoder, camera, profiler);
      profiler?.resolve(encoder);
      device.queue.submit([encoder.finish()]);
      device.popErrorScope().then(err => { if (err) console.error('Frame validation error:', err.message); });
      await device.queue.onSubmittedWorkDone();
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
