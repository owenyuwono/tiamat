import './style.css';
import * as THREE from 'three';
import Stats from 'stats.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { SPHSimulation } from './sph/simulation';
import { WaterRenderer } from './rendering/WaterRenderer';

const CONTAINER_SIZE = new THREE.Vector3(2, 2, 2);
const PARTICLE_COUNT = 8000;

const stats = new Stats();
stats.showPanel(0);
document.body.appendChild(stats.dom);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xd0d8e8);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(3, 2.5, 3);
camera.lookAt(0, 0, 0);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 0, 0);

scene.add(new THREE.AmbientLight(0x8090b0, 1.0));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
dirLight.position.set(5, 8, 5);
scene.add(dirLight);

const boxGeo = new THREE.BoxGeometry(CONTAINER_SIZE.x, CONTAINER_SIZE.y, CONTAINER_SIZE.z);
const edges = new THREE.EdgesGeometry(boxGeo);
const wireframe = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x8090a0 }));
wireframe.position.set(0, CONTAINER_SIZE.y / 2, 0);
scene.add(wireframe);

const sim = new SPHSimulation(scene, PARTICLE_COUNT, CONTAINER_SIZE);
sim.setInstancedRendering(false);

const waterRenderer = new WaterRenderer(
  scene,
  new THREE.Vector3(-CONTAINER_SIZE.x / 2, 0, -CONTAINER_SIZE.z / 2),
  new THREE.Vector3(CONTAINER_SIZE.x / 2, CONTAINER_SIZE.y, CONTAINER_SIZE.z / 2),
  { resolution: 64, splatRadius: 0.05, threshold: 0.75 },
);

const rendererSize = new THREE.Vector2();
let lastTime = performance.now();

function animate() {
  stats.begin();

  const now = performance.now();
  const dt = Math.min((now - lastTime) / 1000, 0.016);
  lastTime = now;

  sim.step(dt);
  const particles = sim.getParticlePositions();
  const vel = sim.getParticleVelocities();
  controls.update();
  camera.updateMatrixWorld();
  renderer.getDrawingBufferSize(rendererSize);
  waterRenderer.update(particles.posX, particles.posY, particles.posZ, vel.velX, vel.velY, vel.velZ, particles.count, dirLight, camera, rendererSize);
  renderer.render(scene, camera);

  stats.end();
  requestAnimationFrame(animate);
}

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

animate();
