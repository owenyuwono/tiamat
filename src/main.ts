import './style.css';
import * as THREE from 'three';
import Stats from 'stats.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { SPHSimulation } from './sph/simulation';
import { WaterRenderer } from './rendering/WaterRenderer';
import { GPUCompute } from './gpu/GPUCompute';

const CONTAINER_SIZE = new THREE.Vector3(4, 4, 4);
const PARTICLE_COUNT = 70000;

const stats = new Stats();
stats.showPanel(0);
document.body.appendChild(stats.dom);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xd0d8e8);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(6, 4.5, 6);
camera.lookAt(0, 2, 0);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 2, 0);

scene.add(new THREE.AmbientLight(0x8090b0, 1.0));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
dirLight.position.set(5, 8, 5);
scene.add(dirLight);

const boxGeo = new THREE.BoxGeometry(CONTAINER_SIZE.x, CONTAINER_SIZE.y, CONTAINER_SIZE.z);
const edges = new THREE.EdgesGeometry(boxGeo);
const wireframe = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x8090a0 }));
wireframe.position.set(0, CONTAINER_SIZE.y / 2, 0);
scene.add(wireframe);

async function init() {
  const sim = new SPHSimulation(scene, PARTICLE_COUNT, CONTAINER_SIZE);
  sim.setInstancedRendering(false);

  const domainMin = new THREE.Vector3(-CONTAINER_SIZE.x / 2, 0, -CONTAINER_SIZE.z / 2);
  const domainMax = new THREE.Vector3(CONTAINER_SIZE.x / 2, CONTAINER_SIZE.y, CONTAINER_SIZE.z / 2);

  const waterRenderer = new WaterRenderer(scene, domainMin, domainMax, {
    resolution: 80,
    splatRadius: 0.1,
    threshold: 0.75,
  });

  const gpuCompute = await GPUCompute.create(
    PARTICLE_COUNT, CONTAINER_SIZE, 80, domainMin, domainMax, 0.1
  );

  if (gpuCompute) {
    const p = sim.getParticlePositions();
    gpuCompute.uploadInitialPositions(p.posX, p.posY, p.posZ);
    console.log('WebGPU SPH enabled');
  } else {
    console.log('WebGPU unavailable, using CPU SPH fallback');
  }

  const rendererSize = new THREE.Vector2();
  let lastTime = performance.now();

  async function animate() {
    stats.begin();
    const now = performance.now();
    const dt = Math.min((now - lastTime) / 1000, 0.016);
    lastTime = now;

    const fixedDt = 0.005;
    const substeps = Math.min(Math.ceil(dt / fixedDt), 3);

    controls.update();
    camera.updateMatrixWorld();
    renderer.getDrawingBufferSize(rendererSize);

    if (gpuCompute) {
      await gpuCompute.step(dt, substeps);
      await gpuCompute.readDensityField(waterRenderer.getTextureData());
      waterRenderer.markTextureNeedsUpdate();
      waterRenderer.updateUniforms(dirLight, camera, rendererSize);
    } else {
      sim.step(dt);
      const particles = sim.getParticlePositions();
      const xs = sim.getXSPH();
      waterRenderer.update(
        particles.posX, particles.posY, particles.posZ,
        xs.xsphX, xs.xsphY, xs.xsphZ,
        particles.count, dirLight, camera, rendererSize
      );
    }
    renderer.render(scene, camera);

    stats.end();
    requestAnimationFrame(animate);
  }

  animate();
}

init();
