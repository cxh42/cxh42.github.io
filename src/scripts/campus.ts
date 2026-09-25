import { QUAD_VS, NOISE_GLSL, compile, fullscreenQuad, getGL, isStill, onFrame, watchVisible, fmtT } from './gl';

// Education: one sticky dark screen, driven by scroll. Campuses never pass through an empty frame:
// each hand-over is a grain-threshold dissolve (coarse patches first, then fine grain) with a small
// bump of forward noise at its midpoint. The incoming school holds at a residual t (still being sampled).
// One pass, two textures, no render targets.

const FS = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 o;
uniform sampler2D uA;
uniform sampler2D uB;
uniform float uHasA;
uniform float uCoarseA;
uniform float uHasB;
uniform vec2 uImgA;
uniform vec2 uImgB;
uniform vec2 uFocusA;
uniform vec2 uFocusB;
uniform float uMix;
uniform float uT;
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
float lattice(vec2 q) {
  vec2 i = floor(q);
  vec2 f = fract(q);
  f = f * f * (3.0 - 2.0 * f);
  uvec2 k = uvec2(ivec2(i) + 4096);
  float a = rnd(uvec3(k, 7u));
  float b = rnd(uvec3(k + uvec2(1u, 0u), 7u));
  float c = rnd(uvec3(k + uvec2(0u, 1u), 7u));
  float d = rnd(uvec3(k + uvec2(1u, 1u), 7u));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
void main() {
  vec2 p = vec2(vUv.x, 1.0 - vUv.y);
  float ab = alphaBar(uT);
  float nz = sqrt(1.0 - ab);
  float lod = uLod * nz;
  // On entry, side A is the incoming photo itself at its coarsest mip: the band opens on the campus's
  // own mean colour and resolves coarse to fine, never through black patches.
  vec3 a = mix(uScreen, textureLod(uA, cover(p, uImgA, uFocusA), max(lod, uCoarseA * 9.0)).rgb, uHasA);
  vec3 b = mix(uScreen, textureLod(uB, cover(p, uImgB, uFocusB), lod).rgb, uHasB);
  // Dissolve field: two octaves of value noise plus pixel grain, so the new campus arrives in
  // soft patches that sharpen into grain rather than as a flat cross-fade.
  vec2 q = p * vec2(uSize.x / uSize.y, 1.0);
  vec2 cell = floor(gl_FragCoord.xy / uGrain);
  float field = 0.6 * lattice(q * 5.0) + 0.25 * lattice(q * 17.0) + 0.15 * rnd(uvec3(uvec2(cell), 3u));
  float edge = uMix * 1.16 - 0.08;
  float m = smoothstep(field - 0.05, field + 0.05, edge);
  vec3 x0 = mix(a, b, m) * 2.0 - 1.0;
  float eps = gauss(uvec3(uvec2(cell), uFrame));
  vec3 xt = sqrt(ab) * x0 + nz * eps * 0.55;
  vec3 c = clamp(xt * 0.5 + 0.5, 0.0, 1.0);
  float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
  c = mix(vec3(l), c, 0.62);
  vec3 col = pow(c, vec3(1.18)) * mix(0.54, 0.42, nz);
  // Scrims: lower left under the degree, the top under the heading, the right edge under the rail.
  float s = smoothstep(1.25, 0.1, length(vUv * vec2(0.9, 1.6)));
  col = mix(col, uScreen, s * 0.72);
  col = mix(col, uScreen, smoothstep(0.78, 1.0, vUv.y) * 0.55);
  col = mix(col, uScreen, smoothstep(0.8, 1.0, vUv.x) * 0.7);
  o = vec4(col, 1.0);
}`;

type Item = {
  el: HTMLElement;
  lg: string;
  sm: string;
  focus: [number, number];
  focusNarrow: [number, number];
  incoming: boolean;
  img?: HTMLImageElement;
};

const pair = (v: string | undefined): [number, number] => {
  const [x, y] = (v || '50% 50%').split(' ').map((n) => parseFloat(n) / 100);
  return [x, y];
};

export function initCampus() {
  const root = document.documentElement;
  const section = document.querySelector<HTMLElement>('[data-edu]');
  const canvas = document.querySelector<HTMLCanvasElement>('[data-edu-canvas]');
  const out = document.querySelector<HTMLElement>('[data-edu-t]');
  const ticks = Array.from(document.querySelectorAll<HTMLElement>('[data-edu-tick]'));
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
    incoming: el.dataset.incoming === '1',
  }));

  const { program, u } = prog;
  const draw = fullscreenQuad(gl, program);
  const textures = items.map(() => gl.createTexture()!);
  const loaded = items.map(() => false);
  // Bound in place of a campus that has not decoded yet, so no sampler ever reads an empty unit.
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

  const hold = (i: number) => (i >= 0 && items[i].incoming ? 0.14 : 0);
  const ease = (x: number) => x * x * (3 - 2 * x);

  // Scroll -> { from, to, mix }. -1 is the empty screen the band opens on.
  // While the band scrolls in, the first campus dissolves in; once pinned, each school holds for a
  // unit and each hand-over dissolves across a unit.
  function state() {
    const r = section!.getBoundingClientRect();
    const vh = window.innerHeight;
    const n = items.length;
    if (r.top > 0) {
      const e = Math.min(1, Math.max(0, (vh - r.top) / (vh * 0.8)));
      return { from: -1, to: 0, mix: ease(e) };
    }
    const p = Math.min(1, Math.max(0, -r.top / Math.max(1, r.height - vh)));
    const units = 2 * n - 1;
    const x = p * units;
    const k = Math.min(units - 1, Math.floor(x));
    if (k % 2 === 0) return { from: k / 2, to: k / 2, mix: 0 };
    const from = (k - 1) / 2;
    return { from, to: from + 1, mix: ease(x - k) };
  }

  let visible = false;
  let lastOn = -2;
  function bindItem(unit: number, idx: number, narrow: boolean, coarse = false) {
    gl!.activeTexture(gl!.TEXTURE0 + unit);
    gl!.bindTexture(gl!.TEXTURE_2D, idx >= 0 && loaded[idx] ? textures[idx] : blank);
    const pre = unit === 0 ? 'A' : 'B';
    const img = idx >= 0 ? items[idx].img : undefined;
    const f = idx >= 0 ? (narrow ? items[idx].focusNarrow : items[idx].focus) : [0.5, 0.5];
    gl!.uniform1i(u(`u${pre}`), unit);
    gl!.uniform1f(u(`uHas${pre}`), idx >= 0 && loaded[idx] ? 1 : 0);
    gl!.uniform2f(u(`uImg${pre}`), img ? img.naturalWidth : 16, img ? img.naturalHeight : 9);
    gl!.uniform2f(u(`uFocus${pre}`), f[0], f[1]);
    if (unit === 0) gl!.uniform1f(u('uCoarseA'), coarse ? 1 : 0);
  }

  function tick(frame: number) {
    if (!visible) return;
    const { from, to, mix } = state();
    const cur = mix < 0.5 ? from : to;
    const t = hold(from) + (hold(to) - hold(from)) * mix + 0.2 * Math.sin(Math.PI * mix);
    if (cur !== lastOn) {
      lastOn = cur;
      items.forEach((it, i) => it.el.classList.toggle('is-on', i === cur));
      ticks.forEach((k, i) => k.classList.toggle('is-on', i === cur));
    }
    if (out) {
      out.textContent = fmtT(t);
      out.classList.toggle('live', t > 0.0005);
    }
    const narrow = canvas!.width / canvas!.height < 1;
    gl!.viewport(0, 0, canvas!.width, canvas!.height);
    gl!.useProgram(program);
    bindItem(0, from < 0 ? to : from, narrow, from < 0);
    bindItem(1, to, narrow);
    gl!.uniform1f(u('uMix'), from === to ? 0 : mix);
    gl!.uniform1f(u('uT'), t);
    gl!.uniform1ui(u('uFrame'), frame >>> 0);
    gl!.uniform1f(u('uGrain'), Math.max(1, Math.round(1.6 * dpr)));
    gl!.uniform1f(u('uLod'), 6.5);
    gl!.uniform2f(u('uSize'), canvas!.width, canvas!.height);
    gl!.uniform3fv(u('uScreen'), screen);
    draw();
  }

  watchVisible(section, (on) => {
    visible = on;
    if (on) load();
  }, '100% 0px');
  onFrame(tick);
}
