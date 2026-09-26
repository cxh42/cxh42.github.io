import { isStill, onFrame, fmtAb, tReadout } from './gl';

// The name is never hidden or redrawn: it is plain text from the first paint. What samples in on load is
// the particle portrait behind it; this module only runs the readout on the shared clock alongside it
// (40 reverse steps, about 1.7 s) and marks the moment it settles, which starts the 陈星昊 typing loop.
const STEPS = 40;

export function initHero() {
  const root = document.documentElement;
  const outT = document.querySelector<HTMLElement>('[data-readout-t]');
  const outAb = document.querySelector<HTMLElement>('[data-readout-ab]');

  const setT = tReadout(outT);
  const readout = (t: number) =>
    setT(t, () => {
      if (!outAb) return;
      outAb.textContent = fmtAb(t);
      outAb.classList.toggle('live', t > 0);
    });

  if (isStill()) {
    readout(0);
    root.classList.add('hero-done');
    return;
  }
  let step = 0;
  readout(1);
  const stop = onFrame(() => {
    step++;
    readout(Math.max(0, 1 - step / STEPS));
    if (step >= STEPS) {
      root.classList.add('hero-done');
      stop();
    }
  });
}
