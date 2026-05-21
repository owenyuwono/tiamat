const MAX_TIMESTAMPS = 40;
const SAMPLE_INTERVAL = 60;

export class GPUProfiler {
  private querySet: GPUQuerySet;
  private resolveBuffer: GPUBuffer;
  private stagingBuffer: GPUBuffer;
  private nextIndex = 0;
  private passLabels: string[] = [];
  private savedLabels: string[] = [];
  private frameCount = 0;
  private needsReadback = false;
  private overlay: HTMLDivElement;
  private substeps = 0;
  private particleCount = 0;

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

    this.overlay = document.createElement('div');
    this.overlay.style.cssText =
      'position:fixed;top:48px;left:0;padding:6px 8px;' +
      'background:rgba(0,0,0,0.75);color:#0f0;font:11px/1.4 monospace;' +
      'pointer-events:none;z-index:10000;white-space:pre;';
    document.body.appendChild(this.overlay);
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

      const ordered: { label: string; totalMs: number; count: number }[] = [];
      const map = new Map<string, { label: string; totalMs: number; count: number }>();

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

      let text = `particles: ${this.particleCount.toLocaleString()}\nsubsteps: ${this.substeps}\n`;
      let total = 0;
      for (const { label, totalMs, count } of ordered) {
        const tag = count > 1 ? ` ×${count}` : '';
        text += `${label}${tag}`.padEnd(22) + totalMs.toFixed(3) + 'ms\n';
        total += totalMs;
      }
      text += 'GPU total'.padEnd(22) + total.toFixed(3) + 'ms';
      this.overlay.textContent = text;
    } catch {
      // buffer destroyed or device lost
    }
  }

  dispose() {
    this.querySet.destroy();
    this.resolveBuffer.destroy();
    this.stagingBuffer.destroy();
    this.overlay.remove();
  }
}
