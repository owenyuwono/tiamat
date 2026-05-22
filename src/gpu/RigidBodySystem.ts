import * as THREE from 'three';

export const MAX_OBSTACLES = 8;
export const OBSTACLE_UNIFORM_SIZE = 16 + MAX_OBSTACLES * 32; // 272 bytes
export const OBSTACLE_FORCES_SIZE = MAX_OBSTACLES * 16; // 128 bytes (4 × i32 per body)
const FORCE_FIXED_SCALE = 1000.0;

interface RigidBody {
  position: THREE.Vector3;
  velocity: THREE.Vector3;
  radius: number;
  mass: number;
  active: boolean;
}

export class RigidBodySystem {
  private bodies: RigidBody[] = [];
  private nextSlot = 0;
  private uniformData: Float32Array;
  private uniformU32: Uint32Array;
  private containerHalfX: number;
  private containerHalfZ: number;
  private containerMaxY: number;
  private gravity: number;

  constructor(containerSize: THREE.Vector3, gravity: number) {
    this.containerHalfX = containerSize.x / 2;
    this.containerHalfZ = containerSize.z / 2;
    this.containerMaxY = containerSize.y;
    this.gravity = gravity;
    const buf = new ArrayBuffer(OBSTACLE_UNIFORM_SIZE);
    this.uniformData = new Float32Array(buf);
    this.uniformU32 = new Uint32Array(buf);
    for (let i = 0; i < MAX_OBSTACLES; i++) {
      this.bodies.push({
        position: new THREE.Vector3(),
        velocity: new THREE.Vector3(),
        radius: 0,
        mass: 1,
        active: false,
      });
    }
  }

  spawn(x: number, y: number, z: number, radius: number, density: number): number {
    const volume = (4 / 3) * Math.PI * radius * radius * radius;
    const mass = density * volume;
    const idx = this.nextSlot % MAX_OBSTACLES;
    const body = this.bodies[idx];
    body.position.set(x, y, z);
    body.velocity.set(0, 0, 0);
    body.radius = radius;
    body.mass = mass;
    body.active = true;
    this.nextSlot++;
    return idx;
  }

  getActiveCount(): number {
    let count = 0;
    for (const b of this.bodies) if (b.active) count++;
    return count;
  }

  writeUniform(device: GPUDevice, buffer: GPUBuffer) {
    for (let i = 0; i < MAX_OBSTACLES; i++) {
      const b = this.bodies[i];
      const offset = 4 + i * 8; // 4 floats header + 8 floats per body
      if (b.active) {
        this.uniformData[offset + 0] = b.position.x;
        this.uniformData[offset + 1] = b.position.y;
        this.uniformData[offset + 2] = b.position.z;
        this.uniformData[offset + 3] = b.radius;
        this.uniformData[offset + 4] = b.velocity.x;
        this.uniformData[offset + 5] = b.velocity.y;
        this.uniformData[offset + 6] = b.velocity.z;
        this.uniformData[offset + 7] = b.mass;
      } else {
        this.uniformData[offset + 3] = 0; // radius 0 — shaders skip via guard
      }
    }
    this.uniformU32[0] = MAX_OBSTACLES; // always iterate all slots; radius guard handles inactive
    device.queue.writeBuffer(buffer, 0, this.uniformData);
  }

  integrateFromForces(forcesData: Int32Array, substeps: number, fixedDt: number) {
    const effectiveDt = fixedDt * substeps;
    const invSubsteps = substeps > 0 ? 1.0 / substeps : 0;

    for (let i = 0; i < MAX_OBSTACLES; i++) {
      const body = this.bodies[i];
      if (!body.active) continue;

      // Forces accumulated across all substeps — normalize by substep count
      const fx = forcesData[i * 4 + 0] / FORCE_FIXED_SCALE * invSubsteps;
      const fy = forcesData[i * 4 + 1] / FORCE_FIXED_SCALE * invSubsteps;
      const fz = forcesData[i * 4 + 2] / FORCE_FIXED_SCALE * invSubsteps;

      const invMass = 1.0 / body.mass;
      body.velocity.x += (fx * invMass) * effectiveDt;
      body.velocity.y += (this.gravity + fy * invMass) * effectiveDt;
      body.velocity.z += (fz * invMass) * effectiveDt;

      body.velocity.multiplyScalar(0.998);

      body.position.addScaledVector(body.velocity, effectiveDt);

      this.collideContainer(body);
    }
    this.collideBodies();
  }

  private collideContainer(body: RigidBody) {
    const r = body.radius;
    const damp = -0.3;
    if (body.position.x - r < -this.containerHalfX) {
      body.position.x = -this.containerHalfX + r;
      body.velocity.x *= damp;
    }
    if (body.position.x + r > this.containerHalfX) {
      body.position.x = this.containerHalfX - r;
      body.velocity.x *= damp;
    }
    if (body.position.y - r < 0) {
      body.position.y = r;
      body.velocity.y *= damp;
    }
    if (body.position.y + r > this.containerMaxY) {
      body.position.y = this.containerMaxY - r;
      body.velocity.y *= damp;
    }
    if (body.position.z - r < -this.containerHalfZ) {
      body.position.z = -this.containerHalfZ + r;
      body.velocity.z *= damp;
    }
    if (body.position.z + r > this.containerHalfZ) {
      body.position.z = this.containerHalfZ - r;
      body.velocity.z *= damp;
    }
  }

  private collideBodies() {
    for (let i = 0; i < MAX_OBSTACLES; i++) {
      if (!this.bodies[i].active) continue;
      for (let j = i + 1; j < MAX_OBSTACLES; j++) {
        if (!this.bodies[j].active) continue;
        const a = this.bodies[i], b = this.bodies[j];
        const dx = b.position.x - a.position.x;
        const dy = b.position.y - a.position.y;
        const dz = b.position.z - a.position.z;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const minDist = a.radius + b.radius;
        if (dist < minDist && dist > 1e-6) {
          const nx = dx / dist, ny = dy / dist, nz = dz / dist;
          const overlap = minDist - dist;
          const totalMass = a.mass + b.mass;
          a.position.x -= nx * overlap * (b.mass / totalMass);
          a.position.y -= ny * overlap * (b.mass / totalMass);
          a.position.z -= nz * overlap * (b.mass / totalMass);
          b.position.x += nx * overlap * (a.mass / totalMass);
          b.position.y += ny * overlap * (a.mass / totalMass);
          b.position.z += nz * overlap * (a.mass / totalMass);
          const relVn = (b.velocity.x - a.velocity.x) * nx +
                        (b.velocity.y - a.velocity.y) * ny +
                        (b.velocity.z - a.velocity.z) * nz;
          if (relVn < 0) {
            const restitution = 0.5;
            const impulse = -(1 + restitution) * relVn / totalMass;
            a.velocity.x -= impulse * b.mass * nx;
            a.velocity.y -= impulse * b.mass * ny;
            a.velocity.z -= impulse * b.mass * nz;
            b.velocity.x += impulse * a.mass * nx;
            b.velocity.y += impulse * a.mass * ny;
            b.velocity.z += impulse * a.mass * nz;
          }
        }
      }
    }
  }

  reset() {
    for (const b of this.bodies) b.active = false;
    this.nextSlot = 0;
  }

  raycastSpawn(
    ndcX: number, ndcY: number,
    camera: THREE.PerspectiveCamera,
    radius: number, density: number,
  ): number | null {
    const ray = new THREE.Vector3(ndcX, ndcY, 0.5).unproject(camera);
    ray.sub(camera.position).normalize();
    const origin = camera.position;

    const spawnY = this.containerMaxY + radius;
    if (Math.abs(ray.y) < 1e-6) return null;
    const t = (spawnY - origin.y) / ray.y;
    if (t < 0) return null;
    const x = origin.x + ray.x * t;
    const z = origin.z + ray.z * t;

    if (Math.abs(x) > this.containerHalfX + radius ||
        Math.abs(z) > this.containerHalfZ + radius) return null;

    return this.spawn(x, spawnY, z, radius, density);
  }
}
