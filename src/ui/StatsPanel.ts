import type { GPUTimingSnapshot } from '../gpu/GPUProfiler';

interface PassRow {
  el: HTMLDivElement;
  labelEl: HTMLSpanElement;
  valueEl: HTMLSpanElement;
}

export class StatsPanel {
  private el: HTMLDivElement;
  private fpsEl: HTMLSpanElement;
  private frameTimeEl: HTMLSpanElement;
  private particlesEl: HTMLSpanElement;
  private substepsEl: HTMLSpanElement;
  private gpuHeaderEl: HTMLDivElement;
  private gpuSeparator: HTMLHRElement;
  private gpuTotalRow: HTMLDivElement;
  private gpuTotalValueEl: HTMLSpanElement;
  private passRows: PassRow[] = [];
  private smoothedFps = 60;
  private smoothedFrameTime = 16;

  constructor() {
    this.el = document.createElement('div');
    this.el.className = 'panel panel-stats';

    this.fpsEl = document.createElement('span');
    this.frameTimeEl = document.createElement('span');
    this.particlesEl = document.createElement('span');
    this.substepsEl = document.createElement('span');

    this.el.appendChild(this.buildRow('FPS', this.fpsEl));
    this.el.appendChild(this.buildRow('Frame', this.frameTimeEl));
    this.el.appendChild(this.buildSeparator());
    this.el.appendChild(this.buildRow('Particles', this.particlesEl));
    this.el.appendChild(this.buildRow('Substeps', this.substepsEl));

    this.gpuSeparator = this.buildSeparator();
    this.gpuSeparator.style.display = 'none';
    this.el.appendChild(this.gpuSeparator);

    this.gpuHeaderEl = document.createElement('div');
    this.gpuHeaderEl.className = 'stats-gpu-header';
    this.gpuHeaderEl.textContent = 'GPU Passes';
    this.gpuHeaderEl.style.display = 'none';
    this.el.appendChild(this.gpuHeaderEl);

    this.gpuTotalValueEl = document.createElement('span');
    this.gpuTotalRow = this.buildRow('GPU Total', this.gpuTotalValueEl);
    this.gpuTotalRow.style.display = 'none';
    this.gpuTotalRow.style.marginTop = '2px';
    this.el.appendChild(this.gpuTotalRow);

    document.body.appendChild(this.el);
  }

  private buildRow(label: string, valueEl: HTMLSpanElement): HTMLDivElement {
    const row = document.createElement('div');
    row.className = 'stats-row';
    const labelEl = document.createElement('span');
    labelEl.className = 'stats-label';
    labelEl.textContent = label;
    valueEl.className = 'stats-value';
    row.appendChild(labelEl);
    row.appendChild(valueEl);
    return row;
  }

  private buildSeparator(): HTMLHRElement {
    const hr = document.createElement('hr');
    hr.className = 'panel-separator';
    return hr;
  }

  private ensurePassRows(count: number) {
    while (this.passRows.length < count) {
      const el = document.createElement('div');
      el.className = 'stats-sub-row';
      const labelEl = document.createElement('span');
      labelEl.className = 'stats-sub-label';
      const valueEl = document.createElement('span');
      valueEl.className = 'stats-sub-value';
      el.appendChild(labelEl);
      el.appendChild(valueEl);
      this.el.insertBefore(el, this.gpuTotalRow);
      this.passRows.push({ el, labelEl, valueEl });
    }
    for (let i = 0; i < this.passRows.length; i++) {
      this.passRows[i].el.style.display = i < count ? '' : 'none';
    }
  }

  update(dtMs: number, gpuSnapshot: GPUTimingSnapshot | null, particleCount: number, substeps: number) {
    if (dtMs > 0) {
      const instantFps = 1000 / dtMs;
      this.smoothedFps += (instantFps - this.smoothedFps) * 0.2;
      this.smoothedFrameTime += (dtMs - this.smoothedFrameTime) * 0.2;
    }

    this.fpsEl.textContent = this.smoothedFps.toFixed(1);
    this.frameTimeEl.textContent = this.smoothedFrameTime.toFixed(1) + ' ms';
    this.particlesEl.textContent = particleCount.toLocaleString();
    this.substepsEl.textContent = String(substeps);

    if (!gpuSnapshot) {
      this.gpuSeparator.style.display = 'none';
      this.gpuHeaderEl.style.display = 'none';
      this.gpuTotalRow.style.display = 'none';
      this.ensurePassRows(0);
      return;
    }

    this.gpuSeparator.style.display = '';
    this.gpuHeaderEl.style.display = '';
    this.gpuTotalRow.style.display = '';

    this.ensurePassRows(gpuSnapshot.passes.length);
    for (let i = 0; i < gpuSnapshot.passes.length; i++) {
      const { label, totalMs, count } = gpuSnapshot.passes[i];
      this.passRows[i].labelEl.textContent = count > 1 ? `${label} x${count}` : label;
      this.passRows[i].valueEl.textContent = totalMs.toFixed(2) + ' ms';
    }

    this.gpuTotalValueEl.textContent = gpuSnapshot.totalMs.toFixed(2) + ' ms';
  }

  dispose() {
    this.el.remove();
  }
}
