export class DensityField {
  readonly resolution: number;
  readonly data: Float32Array;
  private readonly domainMin: [number, number, number];
  private readonly domainSize: [number, number, number];
  private readonly cellSize: number;
  private readonly invCellSize: number;
  private readonly splatRadius2: number;
  private readonly splatRadiusCells: number;
  private dirtyMin: [number, number, number] = [0, 0, 0];
  private dirtyMax: [number, number, number] = [0, 0, 0];
  private hasDirty = false;

  constructor(
    resolution: number,
    domainMin: [number, number, number],
    domainMax: [number, number, number],
    splatRadius: number,
    externalData?: Float32Array,
  ) {
    this.resolution = resolution;
    this.domainMin = domainMin;
    this.domainSize = [
      domainMax[0] - domainMin[0],
      domainMax[1] - domainMin[1],
      domainMax[2] - domainMin[2],
    ];
    this.cellSize = this.domainSize[0] / resolution;
    this.invCellSize = 1 / this.cellSize;
    this.splatRadius2 = splatRadius * splatRadius;
    this.splatRadiusCells = Math.ceil(splatRadius / this.cellSize);
    this.data = externalData ?? new Float32Array(resolution * resolution * resolution * 2);
  }

  splatParticles(
    posX: Float32Array, posY: Float32Array, posZ: Float32Array,
    velX: Float32Array, velY: Float32Array, velZ: Float32Array,
    count: number,
  ) {
    const data = this.data;
    const res = this.resolution;
    const res2 = res * res;
    const invCell = this.invCellSize;
    const r = this.splatRadiusCells;
    const sR2 = this.splatRadius2;
    const minX = this.domainMin[0];
    const minY = this.domainMin[1];
    const minZ = this.domainMin[2];
    const cellSz = this.cellSize;
    const resM1 = res - 1;

    if (this.hasDirty) {
      const [dxMin, dyMin, dzMin] = this.dirtyMin;
      const [dxMax, dyMax, dzMax] = this.dirtyMax;
      for (let iz = dzMin; iz <= dzMax; iz++) {
        const zOff = iz * res2;
        for (let iy = dyMin; iy <= dyMax; iy++) {
          const off = (zOff + iy * res) * 2;
          for (let ix = dxMin; ix <= dxMax; ix++) {
            const idx = off + ix * 2;
            data[idx] = 0;
            data[idx + 1] = 0;
          }
        }
      }
    }

    let gxMin = res, gyMin = res, gzMin = res;
    let gxMax = 0, gyMax = 0, gzMax = 0;

    for (let p = 0; p < count; p++) {
      const px = posX[p], py = posY[p], pz = posZ[p];
      const speed = Math.sqrt(velX[p] * velX[p] + velY[p] * velY[p] + velZ[p] * velZ[p]);

      const cx = Math.floor((px - minX) * invCell);
      const cy = Math.floor((py - minY) * invCell);
      const cz = Math.floor((pz - minZ) * invCell);

      const x0 = Math.max(0, cx - r);
      const x1 = Math.min(resM1, cx + r);
      const y0 = Math.max(0, cy - r);
      const y1 = Math.min(resM1, cy + r);
      const z0 = Math.max(0, cz - r);
      const z1 = Math.min(resM1, cz + r);

      if (x0 < gxMin) gxMin = x0;
      if (x1 > gxMax) gxMax = x1;
      if (y0 < gyMin) gyMin = y0;
      if (y1 > gyMax) gyMax = y1;
      if (z0 < gzMin) gzMin = z0;
      if (z1 > gzMax) gzMax = z1;

      for (let iz = z0; iz <= z1; iz++) {
        const wz = (iz + 0.5) * cellSz + minZ - pz;
        const wz2 = wz * wz;
        if (wz2 >= sR2) continue;
        const zOff = iz * res2;
        for (let iy = y0; iy <= y1; iy++) {
          const wy = (iy + 0.5) * cellSz + minY - py;
          const wyz2 = wy * wy + wz2;
          if (wyz2 >= sR2) continue;
          const yzOff = (zOff + iy * res) * 2;
          for (let ix = x0; ix <= x1; ix++) {
            const wx = (ix + 0.5) * cellSz + minX - px;
            const dist2 = wx * wx + wyz2;
            if (dist2 < sR2) {
              const t = 1 - dist2 / sR2;
              const w = t * t * t;
              const idx = yzOff + ix * 2;
              data[idx] += w;
              data[idx + 1] += w * speed;
            }
          }
        }
      }
    }

    this.dirtyMin = [gxMin, gyMin, gzMin];
    this.dirtyMax = [gxMax, gyMax, gzMax];
    this.hasDirty = true;
  }
}
