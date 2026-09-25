import { FRAME_MS, isStill } from './gl';

// Light / Dark. The switch is a view transition revealed through a 12-frame grain threshold on the
// shared 24 fps clock: the new theme is sampled into place rather than faded.

const FRAMES = 12;
let styleReady = false;

function buildDissolve() {
  if (styleReady) return;
  styleReady = true;
  const N = 128;
  const c = document.createElement('canvas');
  c.width = c.height = N;
  const ctx = c.getContext('2d')!;
  // Value field = a soft low-frequency layer plus white grain, so the reveal has a little structure.
  const coarse = new Float32Array(16 * 16).map(() => Math.random());
  const field = new Float32Array(N * N);
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const gx = (x / N) * 16, gy = (y / N) * 16;
      const x0 = Math.floor(gx), y0 = Math.floor(gy);
      const fx = gx - x0, fy = gy - y0;
      const at = (i: number, j: number) => coarse[((j & 15) << 4) | (i & 15)];
      const top = at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx;
      const bot = at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx;
      field[y * N + x] = 0.4 * (top * (1 - fy) + bot * fy) + 0.6 * Math.random();
    }
  }
  const frames: string[] = [];
  const img = ctx.createImageData(N, N);
  for (let k = 1; k <= FRAMES; k++) {
    const th = k / FRAMES;
    for (let i = 0; i < N * N; i++) {
      const on = field[i] < th ? 255 : 0;
      img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = 0;
      img.data[i * 4 + 3] = on;
    }
    ctx.putImageData(img, 0, 0);
    frames.push(c.toDataURL('image/png'));
  }
  const kf = frames
    .map((f, i) => `${((i / FRAMES) * 100).toFixed(3)}% { -webkit-mask-image: url(${f}); mask-image: url(${f}); }`)
    .join('\n');
  const dur = Math.round(FRAMES * FRAME_MS);
  const css = `
@keyframes grain-in { ${kf} 100% { -webkit-mask-image: none; mask-image: none; } }
::view-transition-old(root), ::view-transition-new(root) { animation: none; mix-blend-mode: normal; }
::view-transition-new(root) {
  -webkit-mask-size: 256px 256px; mask-size: 256px 256px;
  -webkit-mask-repeat: repeat; mask-repeat: repeat;
  animation: grain-in ${dur}ms step-end both;
}`;
  const s = document.createElement('style');
  s.textContent = css;
  document.head.appendChild(s);
}

function current(): 'light' | 'dark' {
  return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light';
}

function apply(theme: 'light' | 'dark') {
  document.documentElement.dataset.theme = theme;
  document.querySelectorAll<HTMLButtonElement>('[data-theme-set]').forEach((b) => {
    b.setAttribute('aria-pressed', String(b.dataset.themeSet === theme));
  });
  document.dispatchEvent(new CustomEvent('themechange', { detail: theme }));
}

export function initTheme() {
  apply(current());
  document.querySelectorAll<HTMLButtonElement>('[data-theme-set]').forEach((b) => {
    b.addEventListener('click', () => {
      const next = b.dataset.themeSet as 'light' | 'dark';
      if (next === current()) return;
      try { localStorage.setItem('theme', next); } catch {}
      const doc = document as Document & { startViewTransition?: (cb: () => void) => unknown };
      if (!doc.startViewTransition || isStill()) {
        apply(next);
        return;
      }
      buildDissolve();
      doc.startViewTransition(() => apply(next));
    });
  });
  // Follow the system until the visitor has chosen.
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
    let stored: string | null = null;
    try { stored = localStorage.getItem('theme'); } catch {}
    if (!stored) apply(e.matches ? 'dark' : 'light');
  });
}
