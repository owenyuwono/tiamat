import type { SimConfig } from './SimConfig';
import type { AlgorithmPicker } from './AlgorithmPicker';

interface SliderDef {
  key: keyof SimConfig;
  label: string;
  min: number;
  max: number;
  step: number;
  format?: (v: number) => string;
}

export interface ControlPanelCallbacks {
  onReset: () => void;
  onRenderScaleChange: (scale: number) => void;
  onParticleCountChange: (count: number) => void;
  gpuMode: boolean;
  algorithmPicker?: AlgorithmPicker;
}

const PHYSICS_SLIDERS: SliderDef[] = [
  { key: 'gravity', label: 'Gravity', min: -20, max: 0, step: 0.1 },
  { key: 'stiffness', label: 'Stiffness', min: 10, max: 500, step: 5 },
  { key: 'viscosity', label: 'Viscosity', min: 0, max: 10, step: 0.1 },
  { key: 'surfaceTension', label: 'Surface Tension', min: 0, max: 2, step: 0.05 },
  { key: 'boundaryDamping', label: 'Wall Damping', min: -1, max: 0, step: 0.05 },
  { key: 'maxVelocity', label: 'Max Velocity', min: 1, max: 20, step: 0.5 },
  { key: 'xsphEpsilon', label: 'XSPH Epsilon', min: 0, max: 0.5, step: 0.01 },
];

const RENDER_SLIDERS: SliderDef[] = [
  { key: 'splatRadius', label: 'Particle Size', min: 0.04, max: 0.3, step: 0.01 },
  { key: 'threshold', label: 'Density Threshold', min: 0.1, max: 3.0, step: 0.05 },
  { key: 'renderScale', label: 'Render Scale', min: 0.25, max: 1.0, step: 0.05, format: v => (v * 100).toFixed(0) + '%' },
];


export class ControlPanel {
  private el: HTMLDivElement;
  private config: SimConfig;
  private callbacks: ControlPanelCallbacks;
  private pauseBtn!: HTMLButtonElement;
  private keydownHandler: (e: KeyboardEvent) => void;

  constructor(config: SimConfig, callbacks: ControlPanelCallbacks) {
    this.config = config;
    this.callbacks = callbacks;

    this.el = document.createElement('div');
    this.el.className = 'panel panel-controls';

    if (callbacks.algorithmPicker) {
      this.el.appendChild(callbacks.algorithmPicker.el);
      this.el.appendChild(this.buildSeparator());
    }

    const particleSlider: SliderDef = {
      key: 'particleCount',
      label: 'Particles',
      min: 10000,
      max: 1000000,
      step: 50000,
      format: v => (v / 1000).toFixed(0) + 'k',
    };
    this.el.appendChild(this.buildSlider(particleSlider, true));
    this.el.appendChild(this.buildSeparator());

    this.el.appendChild(this.buildSimSection());

    if (callbacks.gpuMode) {
      this.el.appendChild(this.buildSeparator());
      const advancedSliders = [...PHYSICS_SLIDERS, ...RENDER_SLIDERS];
      const advanced = this.buildSection('Advanced', advancedSliders);
      advanced.classList.add('collapsed');
      this.el.appendChild(advanced);

    }

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

  private buildSimSection(): HTMLDivElement {
    const btnRow = document.createElement('div');
    btnRow.className = 'panel-btn-row';

    this.pauseBtn = document.createElement('button');
    this.pauseBtn.className = 'panel-btn';
    this.pauseBtn.textContent = 'Pause';
    this.pauseBtn.addEventListener('click', () => this.togglePause());

    const resetBtn = document.createElement('button');
    resetBtn.className = 'panel-btn';
    resetBtn.textContent = 'Reset';
    resetBtn.addEventListener('click', () => this.callbacks.onReset());

    const lightBtn = document.createElement('button');
    lightBtn.className = 'panel-btn';
    lightBtn.textContent = 'Light: On';
    lightBtn.addEventListener('click', () => {
      this.config.lightEnabled = !this.config.lightEnabled;
      lightBtn.textContent = this.config.lightEnabled ? 'Light: On' : 'Light: Off';
    });

    btnRow.appendChild(this.pauseBtn);
    btnRow.appendChild(resetBtn);
    btnRow.appendChild(lightBtn);

    const hint = document.createElement('div');
    hint.style.cssText = 'font-size:10px;color:oklch(50% 0.01 260);text-align:center;margin-top:6px;';
    hint.textContent = 'Space to pause';

    const wrapper = document.createElement('div');
    wrapper.appendChild(btnRow);
    wrapper.appendChild(hint);
    return wrapper;
  }

  private buildSection(title: string, sliders: SliderDef[]): HTMLDivElement {
    const section = document.createElement('div');
    section.className = 'panel-section';

    const header = this.buildSectionHeader(title, section);
    section.appendChild(header);

    const body = document.createElement('div');
    body.className = 'panel-section-body';

    for (const def of sliders) {
      body.appendChild(this.buildSlider(def));
    }

    section.appendChild(body);
    return section;
  }

  private buildSectionHeader(title: string, section: HTMLDivElement): HTMLDivElement {
    const header = document.createElement('div');
    header.className = 'panel-section-header';

    const label = document.createElement('span');
    label.textContent = title;

    const chevron = document.createElement('span');
    chevron.className = 'panel-section-chevron';
    chevron.textContent = '▼';

    header.appendChild(label);
    header.appendChild(chevron);

    header.addEventListener('click', () => {
      section.classList.toggle('collapsed');
    });

    return header;
  }

  private buildSlider(def: SliderDef, commitOnRelease = false): HTMLDivElement {
    const row = document.createElement('div');
    row.className = 'slider-row';

    const labelRow = document.createElement('div');
    labelRow.className = 'slider-label';

    const labelText = document.createElement('span');
    labelText.className = 'slider-label-text';
    labelText.textContent = def.label;

    const valueText = document.createElement('span');
    valueText.className = 'slider-value';
    const currentVal = this.config[def.key] as number;
    valueText.textContent = def.format ? def.format(currentVal) : this.formatValue(currentVal, def.step);

    labelRow.appendChild(labelText);
    labelRow.appendChild(valueText);

    const input = document.createElement('input');
    input.type = 'range';
    input.min = String(def.min);
    input.max = String(def.max);
    input.step = String(def.step);
    input.value = String(currentVal);

    input.addEventListener('input', () => {
      const v = parseFloat(input.value);
      if (!commitOnRelease) {
        (this.config as unknown as Record<string, number>)[def.key] = v;
      }
      valueText.textContent = def.format ? def.format(v) : this.formatValue(v, def.step);
    });

    if (commitOnRelease) {
      input.addEventListener('change', () => {
        const v = parseFloat(input.value);
        (this.config as unknown as Record<string, number>)[def.key] = v;
        if (def.key === 'particleCount') {
          this.callbacks.onParticleCountChange(v);
        }
      });
    }

    if (def.key === 'renderScale') {
      input.addEventListener('change', () => {
        this.callbacks.onRenderScaleChange(this.config.renderScale);
      });
    }

    row.appendChild(labelRow);
    row.appendChild(input);
    return row;
  }

  private formatValue(value: number, step: number): string {
    const decimals = step < 0.01 ? 3 : step < 0.1 ? 2 : step < 1 ? 1 : 0;
    return value.toFixed(decimals);
  }

  private buildSeparator(): HTMLHRElement {
    const hr = document.createElement('hr');
    hr.className = 'panel-separator';
    return hr;
  }

  dispose() {
    document.removeEventListener('keydown', this.keydownHandler);
    this.el.remove();
  }
}
