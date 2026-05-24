export class StatsPanel {
  private el: HTMLDivElement;
  private fpsEl: HTMLSpanElement;
  private smoothedFps = 60;
  private lastTime = performance.now();

  constructor() {
    this.el = document.createElement('div');
    this.el.className = 'panel panel-stats';

    const row = document.createElement('div');
    row.className = 'stats-row';
    const label = document.createElement('span');
    label.className = 'stats-label';
    label.textContent = 'FPS';
    this.fpsEl = document.createElement('span');
    this.fpsEl.className = 'stats-value';
    row.appendChild(label);
    row.appendChild(this.fpsEl);
    this.el.appendChild(row);

    document.body.appendChild(this.el);
  }

  update() {
    const now = performance.now();
    const dt = now - this.lastTime;
    this.lastTime = now;
    if (dt > 0) {
      const instantFps = 1000 / dt;
      this.smoothedFps += (instantFps - this.smoothedFps) * 0.05;
    }
    this.fpsEl.textContent = Math.round(this.smoothedFps).toString();
  }

  dispose() {
    this.el.remove();
  }
}
