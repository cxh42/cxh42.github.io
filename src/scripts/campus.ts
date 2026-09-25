import { QUAD_VS, NOISE_GLSL, compile, fullscreenQuad, getGL, isStill, onFrame, watchVisible, fmtT } from './gl';

// Education: one sticky dark screen tinted in each school's colour. Scroll picks the school; crossing a
// threshold starts a timed diffusion hand-over on the shared 24 fps clock. The current campus is noised
// forward (coarse, grainy, never white), the next campus takes over under the noise, and it is sampled
// back to clean. Every school settles at t = 0: nothing grains at rest.

const PEAK = 0.72; // how far forward the hand-over noises (t in 0..1)
const FRAMES = 28; // about 1.17 s per hand-over
const INTRO = 22; // the first campus sampling in as the band arrives

const FS = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 o;
uniform sampler2D uA;
uniform sampler2D uB;
uniform float uHasA;
uniform float uHasB;
uniform vec2 uImgA;
uniform vec2 uImgB;
uniform vec2 uFocusA;
uniform vec2 uFocusB;
uniform float uMix;
uniform float uT;
uniform vec3 uTint;
uniform float uGain;
uniform uint uFrame;
uniform float uGrain;
uniform float uLod;
uniform vec2 uSize;
uniform vec3 uScreen;
${NOISE_GLSL}
vec2 cover(vec2 p, vec2 img, vec2 focus) {
  float ra = uSize.x / uSize.y;
  float ia = img.x / img.y;
  vec2 sc = ra > ia ? vec2(1.0, ia / ra) : vec2(ra / ia, 1.0);
  return (1.0 - sc) * focus + p * sc;
}
void main() {
  vec2 p = vec2(vUv.x, 1.0 - vUv.y);
  float ab = alphaBar(uT);
  float nz = sqrt(1.0 - ab);
  float lod = uLod * nz;
  vec3 a = mix(uScreen, textureLod(uA, cover(p, uImgA, uFocusA), lod).rgb, uHasA);
  vec3 b = mix(uScreen, textureLod(uB, cover(p, uImgB, uFocusB), lod).rgb, uHasB);
  vec3 x0 = mix(a, b, uMix) * 2.0 - 1.0;
  vec2 cell = floor(gl_FragCoord.xy / uGrain);
  float eps = gauss(uvec3(uvec2(cell), uFrame));
  vec3 xt = sqrt(ab) * x0 + nz * eps * 0.55;
  vec3 c = clamp(xt * 0.5 + 0.5, 0.0, 1.0);
  // Screening-room tone, applied after the noise so pure noise reads as dark grain.
  float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
  vec3 col = pow(mix(vec3(l), c, 0.62), vec3(1.18)) * mix(0.56, 0.42, nz);
  col *= uGain;
  // The school's colour as a filter over the photograph.
  float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
  col = mix(col, lum * (uTint * 2.4 + 0.1), 0.6);
  // Scrims: lower left under the degree, top left under the heading and index, right edge under the rail.
  col = mix(col, uScreen, smoothstep(1.25, 0.1, length(vUv * vec2(0.9, 1.6))) * 0.7);
  col = mix(col, uScreen, smoothstep(1.1, 0.2, length((vUv - vec2(0.0, 1.0)) * vec2(1.2, 1.5))) * 0.55);
  col = mix(col, uScreen, smoothstep(0.8, 1.0, vUv.x) * 0.7);
  o = vec4(col, 1.0);
}`;

type Item = {
  el: HTMLElement;
  lg: string;
  sm: string;
  focus: [number, number];
  focusNarrow: [number, number];
  tint: [number, number, number];
  gain: number;
  incoming: boolean;
  img?: HTMLImageElement;
};

const pair = (v: string | undefined): [number, number] => {
  const [x, y] = (v || '50% 50%').split(' ').map((n) => parseFloat(n) / 100);
  return [x, y];
};
const hex = (h: string | undefined): [number, number, number] => {
  const n = parseInt((h || '#808080').slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
};
const smooth = (x: number) => {
  const c = Math.min(1, Math.max(0, x));
  return c * c * (3 - 2 * c);
};

export function initCampus() {
  const root = document.documentElement;
  const section = document.querySelector<HTMLElement>('[data-edu]');
  const canvas = document.querySelector<HTMLCanvasElement>('[data-edu-canvas]');
  const out = document.querySelector<HTMLElement>('[data-edu-t]');
  const index = document.querySelector<HTMLElement>('[data-edu-index]');
  const goButtons = Array.from(document.querySelectorAll<HTMLButtonElement>('[data-edu-go]'));
  if (!section || !canvas || isStill()) return;

  const gl = getGL(canvas, { alpha: false });
  let prog: ReturnType<typeof compile> | null = null;
  try {
    if (gl) prog = compile(gl, QUAD_VS, FS);
  } catch {
    prog = null;
  }
  if (!gl || !prog) return;
  root.classList.add('edu-gl');

  const items: Item[] = Array.from(section.querySelectorAll<HTMLElement>('[data-edu-item]')).map((el) => ({
    el,
    lg: el.dataset.srcLg!,
    sm: el.dataset.srcSm!,
    focus: pair(el.dataset.focus),
    focusNarrow: pair(el.dataset.focusNarrow || el.dataset.focus),
    tint: hex(el.dataset.tint),
    gain: Number(el.dataset.exposure || 1),
    incoming: el.dataset.incoming === '1',
  }));
  const n = items.length;

  const { program, u } = prog;
  const draw = fullscreenQuad(gl, program);
  const textures = items.map(() => gl.createTexture()!);
  const loaded = items.map(() => false);
  const blank = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, blank);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([11, 12, 14, 255]));
  const screen: [number, number, number] = [0x0b / 255, 0x0c / 255, 0x0e / 255];

  let dpr = 1;
  function size() {
    dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    const r = canvas!.getBoundingClientRect();
    canvas!.width = Math.max(1, Math.round(r.width * dpr));
    canvas!.height = Math.max(1, Math.round(r.height * dpr));
  }
  size();
  new ResizeObserver(size).observe(canvas);

  let requested = false;
  function load() {
    if (requested) return;
    requested = true;
    const big = window.innerWidth * (window.devicePixelRatio || 1) > 1400;
    items.forEach((it, i) => {
      const img = new Image();
      img.decoding = 'async';
      img.src = big ? it.lg : it.sm;
      img
        .decode()
        .then(() => {
          it.img = img;
          gl!.bindTexture(gl!.TEXTURE_2D, textures[i]);
          gl!.pixelStorei(gl!.UNPACK_FLIP_Y_WEBGL, false);
          gl!.texImage2D(gl!.TEXTURE_2D, 0, gl!.RGBA, gl!.RGBA, gl!.UNSIGNED_BYTE, img);
          gl!.generateMipmap(gl!.TEXTURE_2D);
          gl!.texParameteri(gl!.TEXTURE_2D, gl!.TEXTURE_MIN_FILTER, gl!.LINEAR_MIPMAP_LINEAR);
          gl!.texParameteri(gl!.TEXTURE_2D, gl!.TEXTURE_MAG_FILTER, gl!.LINEAR);
          gl!.texParameteri(gl!.TEXTURE_2D, gl!.TEXTURE_WRAP_S, gl!.CLAMP_TO_EDGE);
          gl!.texParameteri(gl!.TEXTURE_2D, gl!.TEXTURE_WRAP_T, gl!.CLAMP_TO_EDGE);
          loaded[i] = true;
        })
        .catch(() => {});
    });
  }

  const hold = (_i: number) => 0; // nothing grains at rest, the incoming school included

  // Which school the scroll position asks for: equal thirds of the pinned range.
  function target() {
    const r = section!.getBoundingClientRect();
    const p = Math.min(1, Math.max(0, -r.top / Math.max(1, r.height - window.innerHeight)));
    return Math.min(n - 1, Math.floor(p * n));
  }

  let shown = 0; // the school on screen (or leaving, mid hand-over)
  let next = 0; // the school arriving
  let frame0 = -1; // frame the current hand-over started, -1 when idle
  let entered = false;
  let introFrame = -1;
  let visible = false;
  let lastText = -2;

  let lastMark = -1;
  function mark(i: number) {
    if (i === lastMark) return;
    lastMark = i;
    index?.style.setProperty('--edu-i', String(i));
    goButtons.forEach((b, k) => b.setAttribute('aria-current', String(k === i)));
  }
  function showText(i: number) {
    if (i === lastText) return;
    lastText = i;
    items.forEach((it, k) => it.el.classList.toggle('is-on', k === i));
  }

  function bindItem(unit: number, idx: number, narrow: boolean) {
    gl!.activeTexture(gl!.TEXTURE0 + unit);
    gl!.bindTexture(gl!.TEXTURE_2D, loaded[idx] ? textures[idx] : blank);
    const pre = unit === 0 ? 'A' : 'B';
    const img = items[idx].img;
    const f = narrow ? items[idx].focusNarrow : items[idx].focus;
    gl!.uniform1i(u(`u${pre}`), unit);
    gl!.uniform1f(u(`uHas${pre}`), loaded[idx] ? 1 : 0);
    gl!.uniform2f(u(`uImg${pre}`), img ? img.naturalWidth : 16, img ? img.naturalHeight : 9);
    gl!.uniform2f(u(`uFocus${pre}`), f[0], f[1]);
  }

  function tick(frame: number) {
    if (!visible) return;
    const r = section!.getBoundingClientRect();
    const vh = window.innerHeight;

    // Arrival: the band opens noisy; once it is well in view, the first campus is sampled clean.
    if (r.top > vh * 0.98) {
      entered = false;
      introFrame = -1;
    } else if (!entered && r.top < vh * 0.55) {
      entered = true;
      introFrame = frame;
    }

    // A threshold crossed while idle starts a hand-over toward the requested school.
    const want = target();
    if (frame0 < 0 && want !== shown && entered) {
      next = want;
      frame0 = frame;
    }

    let t: number;
    let mix = 0;
    let tint = items[shown].tint;
    if (frame0 >= 0) {
      const k = Math.min(1, (frame - frame0) / FRAMES);
      // Forward quickly to the peak, then sample back slowly; the next campus takes over under the noise.
      const bump = k < 0.4 ? smooth(k / 0.4) : 1 - smooth((k - 0.4) / 0.6);
      mix = smooth((k - 0.3) / 0.25);
      t = Math.min(1, hold(shown) * (1 - mix) + hold(next) * mix + (PEAK - 0.06) * bump);
      const tm = smooth((k - 0.15) / 0.6);
      tint = [0, 1, 2].map((c) => items[shown].tint[c] * (1 - tm) + items[next].tint[c] * tm) as [number, number, number];
      showText(mix < 0.5 ? shown : next);
      mark(mix < 0.5 ? shown : next);
      if (k >= 1) {
        shown = next;
        frame0 = -1;
        mix = 0;
        tint = items[shown].tint;
      }
    } else if (!entered) {
      t = PEAK;
      showText(-1);
    } else {
      const k = introFrame < 0 ? 1 : Math.min(1, (frame - introFrame) / INTRO);
      t = hold(shown) + (PEAK - hold(shown)) * (1 - smooth(k));
      showText(k > 0.5 ? shown : -1);
    }

    if (out) {
      out.textContent = fmtT(t);
      out.classList.toggle('live', t > 0.0005);
    }
    const narrow = canvas!.width / canvas!.height < 1;
    gl!.viewport(0, 0, canvas!.width, canvas!.height);
    gl!.useProgram(program);
    bindItem(0, shown, narrow);
    bindItem(1, frame0 >= 0 ? next : shown, narrow);
    gl!.uniform1f(u('uMix'), mix);
    gl!.uniform1f(u('uT'), t);
    gl!.uniform3fv(u('uTint'), tint);
    const g = frame0 >= 0 ? items[shown].gain * (1 - mix) + items[next].gain * mix : items[shown].gain;
    gl!.uniform1f(u('uGain'), g);
    gl!.uniform1ui(u('uFrame'), frame >>> 0);
    gl!.uniform1f(u('uGrain'), Math.max(1, Math.round(1.6 * dpr)));
    gl!.uniform1f(u('uLod'), 7.0);
    gl!.uniform2f(u('uSize'), canvas!.width, canvas!.height);
    gl!.uniform3fv(u('uScreen'), screen);
    draw();
  }

  // The index is a way in as well as a readout: a row scrolls to that school's stretch of the band.
  goButtons.forEach((b) =>
    b.addEventListener('click', () => {
      const i = Number(b.dataset.eduGo);
      const r = section!.getBoundingClientRect();
      const y = window.scrollY + r.top + ((i + 0.5) / n) * (r.height - window.innerHeight);
      window.scrollTo({ top: y, behavior: 'smooth' });
    }),
  );

  mark(0);
  watchVisible(section, (on) => {
    visible = on;
    if (on) load();
  }, '100% 0px');
  onFrame(tick);
}
