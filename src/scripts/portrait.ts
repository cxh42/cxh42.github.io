import { alphaBar, compile, getGL, isStill, onFrame, token } from './gl';

// The particle portrait: a silhouette sampled out of Gaussian noise together with the name, then
// sent back into noise (and drifting upward, fading) as the reader scrolls on.
// Positions come from public/data/silhouette.bin: int16 pairs, bust height normalised to 1, y up.

const STEPS = 44;

const VS = /* glsl */ `#version 300 es
in vec2 aPos;
in vec2 aEps;
in float aSeed;
uniform float uAb;
uniform vec2 uCenter;
uniform float uScale;
uniform vec2 uRes;
uniform float uPx;
uniform float uTime;
uniform float uRise;
uniform float uSpread;
out float vA;
void main() {
  vec2 p = sqrt(uAb) * aPos + sqrt(1.0 - uAb) * aEps * uSpread;
  p += 0.0018 * vec2(sin(uTime * 0.5 + aSeed * 6.2832), cos(uTime * 0.41 + aSeed * 9.1));
  p.y += uRise;
  vec2 px = uCenter + p * uScale;
  gl_Position = vec4(px / uRes * 2.0 - 1.0, 0.0, 1.0);
  gl_PointSize = uPx * (0.75 + 0.5 * fract(aSeed * 7.13));
  vA = 0.5 + 0.5 * fract(aSeed * 3.71);
}`;

const FS = /* glsl */ `#version 300 es
precision highp float;
in float vA;
out vec4 o;
uniform vec3 uInk;
uniform float uAlpha;
uniform float uFloor;
uniform float uSoft;
void main() {
  float d = length(gl_PointCoord - 0.5);
  float a = smoothstep(0.5, 0.3, d) * vA * uAlpha;
  // Nothing below the hero's rule: the foot and everything after it stay on a clean ground.
  a *= smoothstep(uFloor, uFloor + uSoft, gl_FragCoord.y);
  o = vec4(uInk * a, a);
}`;

function gauss(seed: number) {
  let s = seed >>> 0;
  const r = () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296;
  return () => Math.sqrt(-2 * Math.log(Math.max(r(), 1e-7))) * Math.cos(2 * Math.PI * r());
}

export function initPortrait() {
  const canvas = document.querySelector<HTMLCanvasElement>('[data-portrait]');
  if (!canvas) return;
  const gl = getGL(canvas);
  let prog: ReturnType<typeof compile> | null = null;
  try {
    if (gl) prog = compile(gl, VS, FS);
  } catch {
    prog = null;
  }
  if (!gl || !prog) return;
  const { program, u } = prog;
  const still = isStill();
  const foot = document.querySelector<HTMLElement>('.hero-foot');

  let count = 0;
  const vao = gl.createVertexArray();
  let ink = token('--ink');
  let dark = document.documentElement.dataset.theme === 'dark';
  document.addEventListener('themechange', () => {
    ink = token('--ink');
    dark = document.documentElement.dataset.theme === 'dark';
    if (still) render(0);
  });

  let dpr = 1;
  let W = 1;
  let H = 1;
  function size() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = canvas!.clientWidth;
    H = canvas!.clientHeight;
    canvas!.width = Math.max(1, Math.round(W * dpr));
    canvas!.height = Math.max(1, Math.round(H * dpr));
  }

  // Where the bust sits, in CSS px with y measured upward from the bottom of the canvas.
  function layout() {
    const narrow = W < 760;
    if (narrow) {
      const h = Math.min(H * 0.5, W * 1.14);
      return { h, cx: W * 0.43, cy: H - (64 + H * 0.03) - h / 2 };
    }
    const h = Math.min(H * 0.94, W * 0.6);
    return { h, cx: W * 0.6, cy: h / 2 - H * 0.01 };
  }

  let step = still ? STEPS : 0;
  let start = performance.now();

  function render(frame: number) {
    void frame;
    const { h, cx, cy } = layout();
    const vh = window.innerHeight;
    const s = still ? 0 : Math.min(1, Math.max(0, window.scrollY / (vh * 0.9)));
    const tIntro = 1 - step / STEPS;
    const t = Math.max(tIntro, 0.9 * Math.pow(s, 1.2));
    gl!.viewport(0, 0, canvas!.width, canvas!.height);
    gl!.clearColor(0, 0, 0, 0);
    gl!.clear(gl!.COLOR_BUFFER_BIT);
    if (!count) return;
    gl!.useProgram(program);
    gl!.enable(gl!.BLEND);
    gl!.blendFunc(gl!.ONE, gl!.ONE_MINUS_SRC_ALPHA);
    gl!.uniform1f(u('uAb'), alphaBar(t));
    gl!.uniform2f(u('uCenter'), cx * dpr, cy * dpr);
    gl!.uniform1f(u('uScale'), h * dpr);
    gl!.uniform2f(u('uRes'), canvas!.width, canvas!.height);
    gl!.uniform1f(u('uPx'), 1.7 * dpr);
    gl!.uniform1f(u('uTime'), still ? 0 : (performance.now() - start) / 1000);
    gl!.uniform1f(u('uRise'), 0.14 * s);
    gl!.uniform1f(u('uSpread'), 0.55);
    gl!.uniform3fv(u('uInk'), ink);
    const rule = foot ? foot.getBoundingClientRect().top : H;
    gl!.uniform1f(u('uFloor'), (H - rule) * dpr);
    gl!.uniform1f(u('uSoft'), 56 * dpr);
    // Hold density while it scatters, then let go over the last half of the hero.
    const fade = 1 - Math.min(1, Math.max(0, (s - 0.12) / 0.43));
    gl!.uniform1f(u('uAlpha'), (dark ? 0.62 : 0.5) * fade * fade * (3 - 2 * fade));
    gl!.bindVertexArray(vao);
    gl!.drawArrays(gl!.POINTS, 0, count);
    gl!.bindVertexArray(null);
  }

  function tick(frame: number) {
    if (step < STEPS) step++;
    // Past the hero the portrait has fully dispersed: nothing left to draw.
    if (window.scrollY > window.innerHeight * 1.05 && step >= STEPS) return;
    render(frame);
  }

  size();
  new ResizeObserver(() => {
    size();
    if (still) render(0);
  }).observe(canvas);

  fetch('/data/silhouette.bin')
    .then((r) => r.arrayBuffer())
    .then((buf) => {
      const raw = new Int16Array(buf);
      count = raw.length / 2;
      const pos = new Float32Array(count * 2);
      for (let i = 0; i < raw.length; i++) pos[i] = raw[i] / 20000;
      const g = gauss(11);
      const eps = new Float32Array(count * 2).map(() => g());
      const seed = new Float32Array(count).map((_, i) => ((i * 0.618034) % 1));
      gl.bindVertexArray(vao);
      const bind = (data: Float32Array, name: string, n: number) => {
        gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
        gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
        const loc = gl.getAttribLocation(program, name);
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, n, gl.FLOAT, false, 0, 0);
      };
      bind(pos, 'aPos', 2);
      bind(eps, 'aEps', 2);
      bind(seed, 'aSeed', 1);
      gl.bindVertexArray(null);
      start = performance.now();
      if (still) render(0);
      else onFrame(tick);
    })
    .catch(() => {});
}
