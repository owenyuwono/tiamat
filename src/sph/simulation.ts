import * as THREE from 'three';
import { SPH } from './constants';

const H = SPH.smoothingRadius;
const H2 = H * H;
const POLY6_COEFF = 315 / (64 * Math.PI * Math.pow(H, 9));
const SPIKY_COEFF = -45 / (Math.PI * Math.pow(H, 6));
const VISC_LAP_COEFF = 45 / (Math.PI * Math.pow(H, 6));
const CELL_SIZE = H;
const INV_CELL = 1 / CELL_SIZE;

const P1 = 73856093;
const P2 = 19349663;
const P3 = 83492791;

export class SPHSimulation {
  private n: number;
  private mass: number;

  // SoA particle data
  private posX: Float32Array;
  private posY: Float32Array;
  private posZ: Float32Array;
  private velX: Float32Array;
  private velY: Float32Array;
  private velZ: Float32Array;
  private forceX: Float32Array;
  private forceY: Float32Array;
  private forceZ: Float32Array;
  private density: Float32Array;
  private pressure: Float32Array;
  private xsphX: Float32Array;
  private xsphY: Float32Array;
  private xsphZ: Float32Array;

  // Spatial hash (flat array, prefix-sum approach)
  private tableSize: number;
  private cellCount: Int32Array;
  private cellStart: Int32Array;
  private cellEntries: Int32Array;
  private particleHash: Int32Array;

  private mesh: THREE.InstancedMesh;
  private containerSize: THREE.Vector3;
  private useInstancedRendering = true;

  constructor(scene: THREE.Scene, particleCount: number, containerSize: THREE.Vector3) {
    this.n = particleCount;
    this.containerSize = containerSize;

    const spacing = H * 0.5;
    this.mass = SPH.restDensity * spacing * spacing * spacing;

    // Allocate SoA arrays
    this.posX = new Float32Array(this.n);
    this.posY = new Float32Array(this.n);
    this.posZ = new Float32Array(this.n);
    this.velX = new Float32Array(this.n);
    this.velY = new Float32Array(this.n);
    this.velZ = new Float32Array(this.n);
    this.forceX = new Float32Array(this.n);
    this.forceY = new Float32Array(this.n);
    this.forceZ = new Float32Array(this.n);
    this.density = new Float32Array(this.n);
    this.pressure = new Float32Array(this.n);
    this.xsphX = new Float32Array(this.n);
    this.xsphY = new Float32Array(this.n);
    this.xsphZ = new Float32Array(this.n);

    // Spatial hash tables
    this.tableSize = this.nextPowerOfTwo(this.n * 3);
    this.cellCount = new Int32Array(this.tableSize);
    this.cellStart = new Int32Array(this.tableSize);
    this.cellEntries = new Int32Array(this.n);
    this.particleHash = new Int32Array(this.n);

    // Init positions (centered block, drops from above)
    const countPerAxis = Math.ceil(Math.cbrt(particleCount));
    const blockWidth = (countPerAxis - 1) * spacing;
    const offsetX = -blockWidth / 2;
    const offsetZ = -blockWidth / 2;
    for (let i = 0; i < this.n; i++) {
      this.posX[i] = (i % countPerAxis) * spacing + offsetX;
      this.posY[i] = (Math.floor(i / countPerAxis) % countPerAxis) * spacing + spacing;
      this.posZ[i] = Math.floor(i / (countPerAxis * countPerAxis)) * spacing + offsetZ;
    }

    // Create mesh with identity matrices
    const geo = new THREE.SphereGeometry(spacing * 0.35, 6, 4);
    const mat = new THREE.MeshStandardMaterial({ color: 0x3399ff, roughness: 0.2, metalness: 0.1 });
    this.mesh = new THREE.InstancedMesh(geo, mat, this.n);

    const identity = new THREE.Matrix4();
    for (let i = 0; i < this.n; i++) {
      this.mesh.setMatrixAt(i, identity);
    }
    this.updateMesh();
    scene.add(this.mesh);
  }

  getParticlePositions() {
    return { posX: this.posX, posY: this.posY, posZ: this.posZ, count: this.n };
  }

  getParticleVelocities() {
    return { velX: this.velX, velY: this.velY, velZ: this.velZ };
  }

  getXSPH() {
    return { xsphX: this.xsphX, xsphY: this.xsphY, xsphZ: this.xsphZ };
  }

  setInstancedRendering(enabled: boolean) {
    this.useInstancedRendering = enabled;
    this.mesh.visible = enabled;
  }

  step(_dt: number) {
    const fixedDt = 0.005;
    const substeps = Math.min(Math.ceil(_dt / fixedDt), 2);

    this.buildGrid();

    for (let s = 0; s < substeps; s++) {
      this.computeDensityPressure();
      this.computeForces();
      this.integrate(fixedDt);
    }
    if (this.useInstancedRendering) this.updateMesh();
  }

  private hashCell(cx: number, cy: number, cz: number): number {
    return (((cx * P1) ^ (cy * P2) ^ (cz * P3)) & 0x7FFFFFFF) % this.tableSize;
  }

  private buildGrid() {
    const n = this.n;
    this.cellCount.fill(0);

    const halfX = this.containerSize.x / 2;
    const halfZ = this.containerSize.z / 2;

    // Count particles per cell
    for (let i = 0; i < n; i++) {
      const cx = Math.floor((this.posX[i] + halfX) * INV_CELL);
      const cy = Math.floor(this.posY[i] * INV_CELL);
      const cz = Math.floor((this.posZ[i] + halfZ) * INV_CELL);
      const h = this.hashCell(cx, cy, cz);
      this.particleHash[i] = h;
      this.cellCount[h]++;
    }

    // Prefix sum
    this.cellStart[0] = 0;
    for (let i = 1; i < this.tableSize; i++) {
      this.cellStart[i] = this.cellStart[i - 1] + this.cellCount[i - 1];
    }

    // Reset counts for insertion pass
    this.cellCount.fill(0);

    // Insert particles
    for (let i = 0; i < n; i++) {
      const h = this.particleHash[i];
      const idx = this.cellStart[h] + this.cellCount[h];
      this.cellEntries[idx] = i;
      this.cellCount[h]++;
    }
  }

  private computeDensityPressure() {
    const n = this.n;
    const mass = this.mass;
    const px = this.posX, py = this.posY, pz = this.posZ;
    const halfX = this.containerSize.x / 2;
    const halfZ = this.containerSize.z / 2;
    const maxY = this.containerSize.y;

    for (let i = 0; i < n; i++) {
      const xi = px[i], yi = py[i], zi = pz[i];
      let rho = 0;

      const cxi = Math.floor((xi + halfX) * INV_CELL);
      const cyi = Math.floor(yi * INV_CELL);
      const czi = Math.floor((zi + halfZ) * INV_CELL);

      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          for (let dz = -1; dz <= 1; dz++) {
            const h = this.hashCell(cxi + dx, cyi + dy, czi + dz);
            const start = this.cellStart[h];
            const end = start + this.cellCount[h];
            for (let k = start; k < end; k++) {
              const j = this.cellEntries[k];
              const diffX = xi - px[j];
              const diffY = yi - py[j];
              const diffZ = zi - pz[j];
              const r2 = diffX * diffX + diffY * diffY + diffZ * diffZ;
              if (r2 < H2) {
                const d = H2 - r2;
                rho += mass * POLY6_COEFF * d * d * d;
              }
            }
          }
        }
      }

      // Mirror particle contributions at boundaries — scaled to avoid over-compensation
      let mirrorRho = 0;
      let wallCount = 0;
      const addMirror = (dist: number) => {
        const r2 = 4 * dist * dist;
        if (r2 < H2) {
          const d = H2 - r2;
          mirrorRho += mass * POLY6_COEFF * d * d * d;
          wallCount++;
        }
      };
      if (yi < H) addMirror(yi);
      if (maxY - yi < H) addMirror(maxY - yi);
      if (xi + halfX < H) addMirror(xi + halfX);
      if (halfX - xi < H) addMirror(halfX - xi);
      if (zi + halfZ < H) addMirror(zi + halfZ);
      if (halfZ - zi < H) addMirror(halfZ - zi);
      if (wallCount > 0) {
        rho += mirrorRho * (0.4 / Math.max(wallCount, 1));
      }

      this.density[i] = rho;
      // Tait equation with negative pressure clamping (gamma=7)
      const ratio = rho / SPH.restDensity;
      const r2 = ratio * ratio;
      const r4 = r2 * r2;
      const r7 = r4 * r2 * ratio;
      this.pressure[i] = Math.max(0, SPH.stiffness * (r7 - 1));
    }
  }

  private computeForces() {
    const n = this.n;
    const mass = this.mass;
    const px = this.posX, py = this.posY, pz = this.posZ;
    const vx = this.velX, vy = this.velY, vz = this.velZ;
    const dens = this.density, press = this.pressure;
    const fx = this.forceX, fy = this.forceY, fz = this.forceZ;

    const halfX = this.containerSize.x / 2;
    const halfZ = this.containerSize.z / 2;

    const xsX = this.xsphX, xsY = this.xsphY, xsZ = this.xsphZ;

    for (let i = 0; i < n; i++) {
      const xi = px[i], yi = py[i], zi = pz[i];
      const vxi = vx[i], vyi = vy[i], vzi = vz[i];
      const rhoi = dens[i], pressi = press[i];
      let fxi = 0, fyi = SPH.gravity * rhoi, fzi = 0;
      let xsxi = 0, xsyi = 0, xszi = 0;

      const cxi = Math.floor((xi + halfX) * INV_CELL);
      const cyi = Math.floor(yi * INV_CELL);
      const czi = Math.floor((zi + halfZ) * INV_CELL);

      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          for (let dz = -1; dz <= 1; dz++) {
            const h = this.hashCell(cxi + dx, cyi + dy, czi + dz);
            const start = this.cellStart[h];
            const end = start + this.cellCount[h];
            for (let k = start; k < end; k++) {
              const j = this.cellEntries[k];
              if (j === i) continue;
              const diffX = xi - px[j];
              const diffY = yi - py[j];
              const diffZ = zi - pz[j];
              const r2 = diffX * diffX + diffY * diffY + diffZ * diffZ;
              if (r2 >= H2 || r2 < 1e-8) continue;
              const r = Math.sqrt(r2);
              const invR = 1 / r;
              const dirX = diffX * invR;
              const dirY = diffY * invR;
              const dirZ = diffZ * invR;

              const hr = H - r;
              const invDensJ = 1 / dens[j];

              // Pressure
              const avgP = (pressi + press[j]) * 0.5;
              const pMag = -mass * avgP * invDensJ * SPIKY_COEFF * hr * hr;
              fxi += dirX * pMag;
              fyi += dirY * pMag;
              fzi += dirZ * pMag;

              // Collision repulsion
              if (r < SPH.collisionRadius) {
                const overlap = SPH.collisionRadius - r;
                const cMag = SPH.collisionStiffness * overlap;
                fxi += dirX * cMag;
                fyi += dirY * cMag;
                fzi += dirZ * cMag;
              }

              // Viscosity
              const viscMag = SPH.viscosity * mass * invDensJ * VISC_LAP_COEFF * hr;
              fxi += viscMag * (vx[j] - vxi);
              fyi += viscMag * (vy[j] - vyi);
              fzi += viscMag * (vz[j] - vzi);

              // Surface tension (cohesion)
              const cohesion = (1 - r / H);
              const cohMag = SPH.surfaceTension * mass * mass * invDensJ * cohesion * cohesion;
              fxi -= dirX * cohMag;
              fyi -= dirY * cohMag;
              fzi -= dirZ * cohMag;

              // XSPH velocity smoothing accumulation
              const d2 = H2 - r2;
              const w = POLY6_COEFF * d2 * d2 * d2;
              const rhoAvg = (rhoi + dens[j]) * 0.5;
              const xsphW = mass / rhoAvg * w;
              xsxi += (vx[j] - vxi) * xsphW;
              xsyi += (vy[j] - vyi) * xsphW;
              xszi += (vz[j] - vzi) * xsphW;
            }
          }
        }
      }


      fx[i] = fxi;
      fy[i] = fyi;
      fz[i] = fzi;
      xsX[i] = xsxi;
      xsY[i] = xsyi;
      xsZ[i] = xszi;
    }
  }

  private integrate(dt: number) {
    const n = this.n;
    const halfX = this.containerSize.x / 2;
    const halfZ = this.containerSize.z / 2;
    const maxY = this.containerSize.y;
    const damp = SPH.boundaryDamping;
    const maxV = SPH.maxVelocity;

    const eps = SPH.xsphEpsilon;

    for (let i = 0; i < n; i++) {
      const invRho = 1 / this.density[i];
      this.velX[i] += this.forceX[i] * invRho * dt;
      this.velY[i] += this.forceY[i] * invRho * dt;
      this.velZ[i] += this.forceZ[i] * invRho * dt;

      // XSPH velocity smoothing
      this.velX[i] += eps * this.xsphX[i];
      this.velY[i] += eps * this.xsphY[i];
      this.velZ[i] += eps * this.xsphZ[i];

      // Clamp velocity
      const speed2 = this.velX[i] * this.velX[i] + this.velY[i] * this.velY[i] + this.velZ[i] * this.velZ[i];
      if (speed2 > maxV * maxV) {
        const s = maxV / Math.sqrt(speed2);
        this.velX[i] *= s;
        this.velY[i] *= s;
        this.velZ[i] *= s;
      }

      this.posX[i] += this.velX[i] * dt;
      this.posY[i] += this.velY[i] * dt;
      this.posZ[i] += this.velZ[i] * dt;

      if (this.posX[i] < -halfX) { this.posX[i] = -halfX; this.velX[i] *= damp; }
      if (this.posX[i] > halfX) { this.posX[i] = halfX; this.velX[i] *= damp; }
      if (this.posY[i] < 0) { this.posY[i] = 0; this.velY[i] *= damp; }
      if (this.posY[i] > maxY) { this.posY[i] = maxY; this.velY[i] *= damp; }
      if (this.posZ[i] < -halfZ) { this.posZ[i] = -halfZ; this.velZ[i] *= damp; }
      if (this.posZ[i] > halfZ) { this.posZ[i] = halfZ; this.velZ[i] *= damp; }
    }
  }

  private updateMesh() {
    const arr = this.mesh.instanceMatrix.array as Float32Array;
    for (let i = 0; i < this.n; i++) {
      const base = i * 16;
      arr[base] = 1;      // scale x
      arr[base + 5] = 1;  // scale y
      arr[base + 10] = 1; // scale z
      arr[base + 12] = this.posX[i];
      arr[base + 13] = this.posY[i];
      arr[base + 14] = this.posZ[i];
      arr[base + 15] = 1;
    }
    this.mesh.instanceMatrix.needsUpdate = true;
  }

  private nextPowerOfTwo(n: number): number {
    let v = n - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
  }
}
