import './style.css';
import './ui/panels.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { SPHSimulation } from './sph/simulation';
import { WaterRenderer } from './rendering/WaterRenderer';
import { GPUCompute } from './gpu/GPUCompute';
import { GPUProfiler } from './gpu/GPUProfiler';
import { WebGPURenderer } from './gpu/WebGPURenderer';
import { createDefaultConfig } from './ui/SimConfig';
import { ControlPanel } from './ui/ControlPanel';
import { StatsPanel } from './ui/StatsPanel';

const CONTAINER_SIZE = new THREE.Vector3(4, 4, 4);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xd0d8e8);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(6, 4.5, 6);
camera.lookAt(0, 2, 0);

async function init() {
  const domainMin = new THREE.Vector3(-CONTAINER_SIZE.x / 2, 0, -CONTAINER_SIZE.z / 2);
  const domainMax = new THREE.Vector3(CONTAINER_SIZE.x / 2, CONTAINER_SIZE.y, CONTAINER_SIZE.z / 2);
  const config = createDefaultConfig();

  let gpuCompute: GPUCompute | null = null;
  let glRenderer: THREE.WebGLRenderer | null = null;
  let webgpuRenderer: WebGPURenderer | null = null;
  let waterRenderer: WaterRenderer | null = null;
  let profiler: GPUProfiler | null = null;
  let controls: OrbitControls;

  let savedPosX = new Float32Array(0);
  let savedPosY = new Float32Array(0);
  let savedPosZ = new Float32Array(0);

  function generatePositions(count: number) {
    const sim = new SPHSimulation(scene, count, CONTAINER_SIZE);
    sim.setInstancedRendering(false);
    const p = sim.getParticlePositions();
    savedPosX = new Float32Array(p.posX);
    savedPosY = new Float32Array(p.posY);
    savedPosZ = new Float32Array(p.posZ);
    return sim;
  }

  let sim = generatePositions(config.particleCount);

  gpuCompute = await GPUCompute.create(
    config.particleCount, CONTAINER_SIZE, 80, domainMin, domainMax, 0.1
  );

  if (gpuCompute) {
    webgpuRenderer = new WebGPURenderer(
      gpuCompute.getDevice(),
      gpuCompute.getDensityFieldBuffer(),
      gpuCompute.getParamsBuffer(),
      gpuCompute.getFieldResolution(),
      domainMin, domainMax, CONTAINER_SIZE,
    );
    webgpuRenderer.resize(window.innerWidth, window.innerHeight, Math.min(window.devicePixelRatio, 2), config.renderScale);
    controls = new OrbitControls(camera, webgpuRenderer.canvas);

    const device = gpuCompute.getDevice();
    if (device.features.has('timestamp-query')) {
      profiler = new GPUProfiler(device);
      profiler.setParticleCount(config.particleCount);
    }

    gpuCompute.uploadInitialPositions(savedPosX, savedPosY, savedPosZ);
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

    const boxGeo = new THREE.BoxGeometry(CONTAINER_SIZE.x, CONTAINER_SIZE.y, CONTAINER_SIZE.z);
    const edges = new THREE.EdgesGeometry(boxGeo);
    const wireframe = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x8090a0 }));
    wireframe.position.set(0, CONTAINER_SIZE.y / 2, 0);
    scene.add(wireframe);

    waterRenderer = new WaterRenderer(scene, domainMin, domainMax, {
      resolution: 80, splatRadius: 0.1, threshold: 0.75,
    });

    console.log('WebGPU unavailable, using CPU SPH fallback');
  }

  controls.enableDamping = true;
  controls.target.set(0, 2, 0);

  async function reinitGPU(count: number) {
    if (!gpuCompute || !webgpuRenderer) return;
    const device = gpuCompute.getDevice();

    gpuCompute.dispose();
    gpuCompute = new GPUCompute(
      device, count, CONTAINER_SIZE, 80, domainMin, domainMax, 0.1,
    );

    sim = generatePositions(count);
    gpuCompute.uploadInitialPositions(savedPosX, savedPosY, savedPosZ);
    webgpuRenderer.rebindComputeBuffers(
      gpuCompute.getDensityFieldBuffer(),
      gpuCompute.getParamsBuffer(),
    );
    profiler?.setParticleCount(count);
  }

  const isGPU = !!gpuCompute;
  const statsPanel = new StatsPanel();
  // @ts-expect-error kept for dispose reference
  const _controlPanel = new ControlPanel(config, {
    gpuMode: isGPU,
    onReset: () => {
      if (gpuCompute) {
        gpuCompute.uploadInitialPositions(savedPosX, savedPosY, savedPosZ);
        gpuCompute.resetVelocities();
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

    const fixedDt = 0.008;
    const substeps = config.paused ? 0 : Math.min(Math.ceil(dt / fixedDt), config.substepLimit);

    controls.update();
    camera.updateMatrixWorld();

    if (gpuCompute && webgpuRenderer) {
      gpuCompute.updateSimConfig(config);
      webgpuRenderer.setThreshold(config.threshold);

      profiler?.beginFrame();
      profiler?.setSubsteps(substeps);
      const device = gpuCompute.getDevice();
      const encoder = device.createCommandEncoder();
      if (!config.paused) {
        gpuCompute.encodeStep(encoder, substeps, profiler);
      }
      webgpuRenderer.encodeFrame(encoder, camera, profiler);
      profiler?.resolve(encoder);
      device.queue.submit([encoder.finish()]);
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
