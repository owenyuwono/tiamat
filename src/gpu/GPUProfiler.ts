const MAX_TIMESTAMPS = 64;
const SAMPLE_INTERVAL = 60;

export interface GPUTimingEntry {
  label: string;
  totalMs: number;
  count: number;
}

export interface GPUTimingSnapshot {
  passes: GPUTimingEntry[];
  totalMs: number;
  particleCount: number;
  substeps: number;
}

export class GPUProfiler {
  private querySet: GPUQuerySet;
  private resolveBuffer: GPUBuffer;
  private stagingBuffer: GPUBuffer;
  private nextIndex = 0;
  private passLabels: string[] = [];
  private savedLabels: string[] = [];
  private frameCount = 0;
  private needsReadback = false;
  private substeps = 0;
  private particleCount = 0;
  private lastSnapshot: GPUTimingSnapshot | null = null;

  constructor(device: GPUDevice) {
    this.querySet = device.createQuerySet({
      type: 'timestamp',
      count: MAX_TIMESTAMPS,
    });

    const size = MAX_TIMESTAMPS * 8;
    this.resolveBuffer = device.createBuffer({
      size,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    this.stagingBuffer = device.createBuffer({
      size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  beginFrame() {
    this.nextIndex = 0;
    this.passLabels = [];
  }

  setSubsteps(n: number) {
    this.substeps = n;
  }

  setParticleCount(n: number) {
    this.particleCount = n;
  }

  timestampWrites(label: string): GPUComputePassTimestampWrites {
    const begin = this.nextIndex++;
    const end = this.nextIndex++;
    this.passLabels.push(label);
    return {
      querySet: this.querySet,
      beginningOfPassWriteIndex: begin,
      endOfPassWriteIndex: end,
    };
  }

  resolve(encoder: GPUCommandEncoder) {
    this.frameCount++;
    if (this.nextIndex === 0 || this.frameCount % SAMPLE_INTERVAL !== 0) return;

    encoder.resolveQuerySet(this.querySet, 0, this.nextIndex, this.resolveBuffer, 0);
    encoder.copyBufferToBuffer(
      this.resolveBuffer, 0,
      this.stagingBuffer, 0,
      this.nextIndex * 8,
    );
    this.savedLabels = this.passLabels.slice();
    this.needsReadback = true;
  }

  async readback() {
    if (!this.needsReadback) return;
    this.needsReadback = false;

    try {
      await this.stagingBuffer.mapAsync(GPUMapMode.READ);
      const data = new BigInt64Array(this.stagingBuffer.getMappedRange());

      const ordered: GPUTimingEntry[] = [];
      const map = new Map<string, GPUTimingEntry>();

      for (let i = 0; i < this.savedLabels.length; i++) {
        const label = this.savedLabels[i];
        const begin = data[i * 2];
        const end = data[i * 2 + 1];
        const ms = Number(end - begin) / 1_000_000;

        const existing = map.get(label);
        if (existing) {
          existing.totalMs += ms;
          existing.count++;
        } else {
          const entry = { label, totalMs: ms, count: 1 };
          map.set(label, entry);
          ordered.push(entry);
        }
      }

      this.stagingBuffer.unmap();

      let total = 0;
      for (const { totalMs } of ordered) total += totalMs;

      this.lastSnapshot = {
        passes: ordered,
        totalMs: total,
        particleCount: this.particleCount,
        substeps: this.substeps,
      };
    } catch {
      // buffer destroyed or device lost
    }
  }

  getSnapshot(): GPUTimingSnapshot | null {
    return this.lastSnapshot;
  }

  dispose() {
    this.querySet.destroy();
    this.resolveBuffer.destroy();
    this.stagingBuffer.destroy();
  }
}
