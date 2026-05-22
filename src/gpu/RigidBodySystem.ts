import * as THREE from 'three';

export const MAX_OBSTACLES = 8;
export const OBSTACLE_UNIFORM_SIZE = 16 + MAX_OBSTACLES * 32; // 272 bytes
export const OBSTACLE_FORCES_SIZE = MAX_OBSTACLES * 16; // 128 bytes

interface RigidBody {
  position: THREE.Vector3;
  velocity: THREE.Vector3;
  radius: number;
  mass: number;
  active: boolean;
  dragging: boolean;
}

export class RigidBodySystem {
  private body: RigidBody;
  private uniformData: Float32Array;
  private uniformU32: Uint32Array;
  private containerHalfX: number;
  private containerHalfZ: number;
  private containerMaxY: number;
  private gravity: number;
  private prevPos = new THREE.Vector3();

  constructor(containerSize: THREE.Vector3, gravity: number) {
    this.containerHalfX = containerSize.x / 2;
    this.containerHalfZ = containerSize.z / 2;
    this.containerMaxY = containerSize.y;
    this.gravity = gravity;
    const buf = new ArrayBuffer(OBSTACLE_UNIFORM_SIZE);
    this.uniformData = new Float32Array(buf);
    this.uniformU32 = new Uint32Array(buf);
    this.body = {
      position: new THREE.Vector3(),
      velocity: new THREE.Vector3(),
      radius: 0,
      mass: 1,
      active: false,
      dragging: false,
    };
  }

  hasBody(): boolean {
    return this.body.active;
  }

  getActiveCount(): number {
    return this.body.active ? 1 : 0;
  }

  hitTest(origin: THREE.Vector3, direction: THREE.Vector3): boolean {
    if (!this.body.active) return false;
    const oc = origin.clone().sub(this.body.position);
    const b = oc.dot(direction);
    const c = oc.dot(oc) - this.body.radius * this.body.radius;
    return b * b - c > 0;
  }

  startDrag() {
    if (!this.body.active) return;
    this.body.dragging = true;
    this.body.velocity.set(0, 0, 0);
    this.prevPos.copy(this.body.position);
  }

  updateDrag(origin: THREE.Vector3, direction: THREE.Vector3, dt: number) {
    if (!this.body.active || !this.body.dragging) return;

    if (Math.abs(direction.y) < 1e-6) return;
    const t = (this.body.position.y - origin.y) / direction.y;
    if (t <= 0) return;

    const r = this.body.radius;
    const x = Math.max(-this.containerHalfX + r, Math.min(this.containerHalfX - r, origin.x + direction.x * t));
    const z = Math.max(-this.containerHalfZ + r, Math.min(this.containerHalfZ - r, origin.z + direction.z * t));

    this.prevPos.copy(this.body.position);
    this.body.position.x = x;
    this.body.position.z = z;

    if (dt > 1e-6) {
      this.body.velocity.set(
        (this.body.position.x - this.prevPos.x) / dt,
        0,
        (this.body.position.z - this.prevPos.z) / dt,
      );
    }
  }

  endDrag() {
    if (!this.body.active) return;
    this.body.dragging = false;
  }

  integrate(substeps: number, fixedDt: number) {
    if (substeps <= 0) return;
    const body = this.body;
    if (!body.active || body.dragging) return;

    for (let s = 0; s < substeps; s++) {
      body.velocity.y += this.gravity * fixedDt;
      body.velocity.multiplyScalar(1.0 - 2.0 * fixedDt);
      body.position.addScaledVector(body.velocity, fixedDt);
    }

    this.collideContainer(body);
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

  writeUniform(device: GPUDevice, buffer: GPUBuffer) {
    const b = this.body;
    const offset = 4;
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
      this.uniformData[offset + 3] = 0;
    }
    for (let i = 1; i < MAX_OBSTACLES; i++) {
      this.uniformData[4 + i * 8 + 3] = 0;
    }
    this.uniformU32[0] = MAX_OBSTACLES;
    device.queue.writeBuffer(buffer, 0, this.uniformData);
  }

  reset() {
    this.body.active = false;
    this.body.dragging = false;
    this.body.velocity.set(0, 0, 0);
  }

  raycastSpawn(
    ndcX: number, ndcY: number,
    camera: THREE.PerspectiveCamera,
    radius: number,
  ): boolean {
    if (this.body.active) return false;

    const ray = new THREE.Vector3(ndcX, ndcY, 0.5).unproject(camera);
    ray.sub(camera.position).normalize();
    const origin = camera.position;

    const planeY = this.containerMaxY;
    if (Math.abs(ray.y) < 1e-6) return false;
    const t = (planeY - origin.y) / ray.y;
    if (t < 0) return false;

    const r = radius;
    const x = Math.max(-this.containerHalfX + r, Math.min(this.containerHalfX - r, origin.x + ray.x * t));
    const z = Math.max(-this.containerHalfZ + r, Math.min(this.containerHalfZ - r, origin.z + ray.z * t));

    this.body.position.set(x, planeY + radius, z);
    this.body.velocity.set(0, 0, 0);
    this.body.radius = radius;
    this.body.mass = 1;
    this.body.active = true;
    this.body.dragging = false;
    this.prevPos.copy(this.body.position);
    return true;
  }

  getRayFromNDC(ndcX: number, ndcY: number, camera: THREE.PerspectiveCamera): { origin: THREE.Vector3; direction: THREE.Vector3 } {
    const far = new THREE.Vector3(ndcX, ndcY, 0.5).unproject(camera);
    const direction = far.sub(camera.position).normalize();
    return { origin: camera.position.clone(), direction };
  }
}
