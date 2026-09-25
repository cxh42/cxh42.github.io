import { QUAD_VS, NOISE_GLSL, compile, fullscreenQuad, getGL, isStill, onFrame, token, fmtT, fmtAb } from './gl';

// Sampling the name: 40 reverse steps on the shared 24 fps clock (about 1.7 s). Once the DOM text takes
// over, the canvas is cleared and stops: the settled page carries no grain.
const STEPS = 40;

const FS = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 o;
uniform sampler2D uSig;
uniform float uT;
uniform uint uFrame;
uniform float uGrain;
uniform float uLod;
uniform float uSigOn;
uniform vec3 uInk;
${NOISE_GLSL}
void main() {
  vec2 cell = floor(gl_FragCoord.xy / uGrain);
  float eps = gauss(uvec3(uvec2(cell), uFrame));
  float ab = alphaBar(uT);
  float nz = sqrt(1.0 - ab);
  // Coarse to fine: at high t the sampler only sees the low frequencies of the name.
  float cov = textureLod(uSig, vec2(vUv.x, 1.0 - vUv.y), uLod * nz).a;
  float x0 = mix(-1.0, 1.0, cov);
  float xt = sqrt(ab) * x0 + nz * eps;
  // Stochastic screen: a rising threshold turns pure noise into sparse grain rather than static.
  float th = 1.65 * nz;
  float w = mix(1.0, 0.16, nz);
  float v = smoothstep(th - w, th + w, xt) * uSigOn * mix(0.55, 1.0, sqrt(ab));
  o = vec4(uInk * v, v);
}`;

export function initHero() {
  const root = document.documentElement;
  const hero = document.querySelector<HTMLElement>('[data-hero]');
  const canvas = document.querySelector<HTMLCanvasElement>('[data-hero-canvas]');
  const nameEl = document.querySelector<HTMLElement>('[data-hero-name]');
  const outT = document.querySelector<HTMLElement>('[data-readout-t]');
  const outAb = document.querySelector<HTMLElement>('[data-readout-ab]');
  if (!hero || !canvas || !nameEl) return;

  const still = isStill();
  const gl = getGL(canvas);
  let prog: ReturnType<typeof compile> | null = null;
  try {
    if (gl) prog = compile(gl, QUAD_VS, FS);
  } catch {
    prog = null;
  }
  if (!gl || !prog) {
    root.classList.add('no-gl', 'hero-done', 'hero-settle');
    return;
  }
  const { program, u } = prog;
  const draw = fullscreenQuad(gl, program);
  const tex = gl.createTexture();

  let dpr = 1;
  let lodMax = 6;
  let t = still ? 0 : 1;
  let step = 0;
  let sampling = !still;
  let sigOn = still ? 0 : 1;
  let ink = token('--ink');

  function paintSignal() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const box = canvas!.getBoundingClientRect();
    const W = Math.max(1, Math.round(box.width * dpr));
    const H = Math.max(1, Math.round(box.height * dpr));
    canvas!.width = W;
    canvas!.height = H;
    lodMax = Math.log2(dpr * 44);

    const c2 = document.createElement('canvas');
    c2.width = W;
    c2.height = H;
    const ctx = c2.getContext('2d')!;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = '#fff';
    const range = document.createRange();
    nameEl!.querySelectorAll<HTMLElement>('[data-w]').forEach((w) => {
      const node = w.firstChild;
      if (!node || node.nodeType !== Node.TEXT_NODE) return;
      const cs = getComputedStyle(w);
      ctx.font = `${cs.fontStyle} ${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
      const text = node.textContent || '';
      // Draw glyph by glyph at the DOM's own positions so the hand-over at t = 0 is pixel-exact.
      for (let i = 0; i < text.length; i++) {
        range.setStart(node, i);
        range.setEnd(node, i + 1);
        const r = range.getBoundingClientRect();
        const m = ctx.measureText(text[i]);
        const asc = m.fontBoundingBoxAscent;
        const desc = m.fontBoundingBoxDescent;
        const base = r.top - box.top + r.height * (asc / (asc + desc));
        ctx.fillText(text[i], r.left - box.left, base);
      }
    });
    gl!.bindTexture(gl!.TEXTURE_2D, tex);
    gl!.pixelStorei(gl!.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl!.texImage2D(gl!.TEXTURE_2D, 0, gl!.RGBA, gl!.RGBA, gl!.UNSIGNED_BYTE, c2);
    gl!.generateMipmap(gl!.TEXTURE_2D);
    gl!.texParameteri(gl!.TEXTURE_2D, gl!.TEXTURE_MIN_FILTER, gl!.LINEAR_MIPMAP_LINEAR);
    gl!.texParameteri(gl!.TEXTURE_2D, gl!.TEXTURE_MAG_FILTER, gl!.LINEAR);
    gl!.texParameteri(gl!.TEXTURE_2D, gl!.TEXTURE_WRAP_S, gl!.CLAMP_TO_EDGE);
    gl!.texParameteri(gl!.TEXTURE_2D, gl!.TEXTURE_WRAP_T, gl!.CLAMP_TO_EDGE);
    gl!.viewport(0, 0, W, H);
  }

  function render(frame: number) {
    gl!.clearColor(0, 0, 0, 0);
    gl!.clear(gl!.COLOR_BUFFER_BIT);
    gl!.useProgram(program);
    gl!.activeTexture(gl!.TEXTURE0);
    gl!.bindTexture(gl!.TEXTURE_2D, tex);
    gl!.uniform1i(u('uSig'), 0);
    gl!.uniform1f(u('uT'), t);
    gl!.uniform1ui(u('uFrame'), frame >>> 0);
    gl!.uniform1f(u('uGrain'), Math.max(1, Math.round(dpr * 1.25)));
    gl!.uniform1f(u('uLod'), lodMax);
    gl!.uniform1f(u('uSigOn'), sigOn);
    gl!.uniform3fv(u('uInk'), ink);
    draw();
  }

  function readout() {
    if (outT) { outT.textContent = fmtT(t); outT.classList.toggle('live', t > 0); }
    if (outAb) { outAb.textContent = fmtAb(t); outAb.classList.toggle('live', t > 0); }
  }

  function tick(frame: number) {
    if (sampling) {
      step++;
      t = Math.max(0, 1 - step / STEPS);
      readout();
      if (t < 0.34) root.classList.add('hero-settle');
      if (step >= STEPS) {
        sampling = false;
        root.classList.add('hero-done');
        root.classList.remove('hero-live');
        // Two frames later the DOM text is painted; the sampler lets go and clears.
        requestAnimationFrame(() =>
          requestAnimationFrame(() => {
            sigOn = 0;
            render(frame);
            stop();
          }),
        );
      }
    }
    if (sampling) render(frame);
  }

  let stop = () => {};
  const start = () => {
    stop();
    if (still) return;
    stop = onFrame(tick);
  };

  document.addEventListener('themechange', () => {
    ink = token('--ink');
  });

  let resizeTimer = 0;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(() => {
      if (sampling) paintSignal();
    }, 120);
  });

  const fontsReady = Promise.race([
    document.fonts ? document.fonts.ready : Promise.resolve(),
    new Promise((r) => setTimeout(r, 1500)),
  ]);
  fontsReady.then(() => {
    if (!still) root.classList.add('hero-live');
    paintSignal();
    readout();
    start();
  });
}
