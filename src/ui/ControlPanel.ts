import type { SimConfig } from './SimConfig';
import type { AlgorithmPicker } from './AlgorithmPicker';

export interface ControlPanelCallbacks {
  onReset: () => void;
  algorithmPicker?: AlgorithmPicker;
}

export class ControlPanel {
  private el: HTMLDivElement;
  private config: SimConfig;
  private pauseBtn!: HTMLButtonElement;
  private keydownHandler: (e: KeyboardEvent) => void;

  constructor(config: SimConfig, callbacks: ControlPanelCallbacks) {
    this.config = config;

    this.el = document.createElement('div');
    this.el.className = 'panel panel-controls';

    if (callbacks.algorithmPicker) {
      this.el.appendChild(callbacks.algorithmPicker.el);
      const hr = document.createElement('hr');
      hr.className = 'panel-separator';
      this.el.appendChild(hr);
    }

    const btnRow = document.createElement('div');
    btnRow.className = 'panel-btn-row';

    this.pauseBtn = document.createElement('button');
    this.pauseBtn.className = 'panel-btn';
    this.pauseBtn.textContent = 'Pause';
    this.pauseBtn.addEventListener('click', () => this.togglePause());

    const resetBtn = document.createElement('button');
    resetBtn.className = 'panel-btn';
    resetBtn.textContent = 'Reset';
    resetBtn.addEventListener('click', () => callbacks.onReset());

    const fxaaBtn = document.createElement('button');
    fxaaBtn.className = 'panel-btn';
    fxaaBtn.textContent = 'FXAA: On';
    fxaaBtn.addEventListener('click', () => {
      this.config.fxaaEnabled = !this.config.fxaaEnabled;
      fxaaBtn.textContent = this.config.fxaaEnabled ? 'FXAA: On' : 'FXAA: Off';
    });

    btnRow.appendChild(this.pauseBtn);
    btnRow.appendChild(resetBtn);
    btnRow.appendChild(fxaaBtn);

    const hint = document.createElement('div');
    hint.style.cssText = 'font-size:10px;color:oklch(50% 0.01 260);text-align:center;margin-top:6px;';
    hint.textContent = 'Space to pause';

    this.el.appendChild(btnRow);
    this.el.appendChild(hint);

    document.body.appendChild(this.el);

    this.keydownHandler = (e: KeyboardEvent) => {
      if (e.code === 'Space' && e.target === document.body) {
        e.preventDefault();
        this.togglePause();
      }
    };
    document.addEventListener('keydown', this.keydownHandler);
  }

  private togglePause() {
    this.config.paused = !this.config.paused;
    this.pauseBtn.textContent = this.config.paused ? 'Resume' : 'Pause';
  }

  dispose() {
    document.removeEventListener('keydown', this.keydownHandler);
    this.el.remove();
  }
}
