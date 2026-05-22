import type { Algorithm } from './SimConfig';

interface AlgorithmDef {
  value: Algorithm;
  name: string;
  description: string;
  disabled?: boolean;
}

const ALGORITHMS: AlgorithmDef[] = [
  {
    value: 'sph',
    name: 'SPH',
    description: 'Lagrangian solver. Pairwise kernel density estimation with Tait pressure, viscosity, and XSPH velocity smoothing.',
  },
  {
    value: 'flip',
    name: 'FLIP',
    description: 'Hybrid PIC/FLIP. Particles transfer momentum to a MAC grid, pressure projection via Jacobi iteration, then grid-to-particle interpolation.',
    disabled: true,
  },
  {
    value: 'euler',
    name: 'Euler',
    description: 'Pure grid Navier-Stokes. Semi-Lagrangian advection on a staggered MAC grid with Jacobi pressure projection. Fixed O(n³) cost, no particles.',
    disabled: true,
  },
];

export class AlgorithmPicker {
  readonly el: HTMLDivElement;
  private current: Algorithm;
  private onChange: (algo: Algorithm) => void;
  private descriptionEl: HTMLDivElement;
  private buttons: Map<Algorithm, HTMLButtonElement> = new Map();

  constructor(initial: Algorithm, onChange: (algo: Algorithm) => void) {
    this.current = initial;
    this.onChange = onChange;

    this.el = document.createElement('div');
    this.el.className = 'algo-picker';

    const toggle = document.createElement('div');
    toggle.className = 'algo-picker-toggle';

    for (const algo of ALGORITHMS) {
      const btn = document.createElement('button');
      btn.className = 'algo-picker-btn' + (this.current === algo.value ? ' active' : '') + (algo.disabled ? ' disabled' : '');
      btn.textContent = algo.name;
      if (algo.disabled) {
        btn.title = 'Coming soon';
      } else {
        btn.addEventListener('click', () => this.select(algo.value));
      }
      this.buttons.set(algo.value, btn);
      toggle.appendChild(btn);
    }

    this.descriptionEl = document.createElement('div');
    this.descriptionEl.className = 'algo-picker-desc';
    this.descriptionEl.textContent = this.getDescription(this.current);

    this.el.appendChild(toggle);
    this.el.appendChild(this.descriptionEl);
  }

  private select(algo: Algorithm) {
    if (this.current === algo) return;
    this.current = algo;

    for (const [value, btn] of this.buttons) {
      btn.classList.toggle('active', value === algo);
    }

    this.descriptionEl.textContent = this.getDescription(algo);
    this.onChange(algo);
  }

  private getDescription(algo: Algorithm): string {
    return ALGORITHMS.find(a => a.value === algo)!.description;
  }
}
