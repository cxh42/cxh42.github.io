import { alphaBar, compile, getGL, isStill, onFrame, token, watchVisible, tReadout } from './gl';

// Visitors: the land is a point cloud that diffuses out of a Gaussian blob onto the sphere
// (p_t = sqrt(ab) p0 + sqrt(1 - ab) eps). Places with visits are marked in the accent, sized by area and
// toned along one cobalt ramp by how many visits came from there (deeper on light, brighter on dark).

const VS = /* glsl */ `#version 300 es
in vec3 aPos;
in vec3 aEps;
in float aSize;
in float aW;
uniform mat3 uRot;
uniform float uAb;
uniform float uPx;
uniform float uScale;
uniform int uHover;
out float vDepth;
out float vW;
out float vSize;
out float vHover;
void main() {
  vec3 p = sqrt(uAb) * aPos + sqrt(1.0 - uAb) * aEps * 0.62;
  vec3 q = uRot * p;
  gl_Position = vec4(q.x * uScale, q.y * uScale, 0.0, 1.0);
  vDepth = q.z;
  vW = aW;
  vHover = gl_VertexID == uHover ? 1.0 : 0.0;
  vSize = uPx * aSize * (0.78 + 0.22 * clamp(q.z, -1.0, 1.0)) * (1.0 + 0.3 * vHover);
  gl_PointSize = vSize;
}`;

const FS = /* glsl */ `#version 300 es
precision highp float;
in float vDepth;
in float vW;
in float vSize;
in float vHover;
out vec4 o;
uniform vec3 uLow;
uniform vec3 uHigh;
uniform vec3 uSurface;
uniform vec3 uInk;
uniform float uRing;
uniform float uFront;
uniform float uBack;
void main() {
  float d = length(gl_PointCoord - 0.5);
  float px = 1.0 / max(vSize, 1.0);
  // Land grains stay soft; marks get a crisp edge and a thin ring of the ground (ink when hovered), so
  // neighbouring marks never merge.
  float disc = smoothstep(0.5, uRing > 0.0 ? 0.5 - 1.5 * px : 0.32, d);
  vec3 fill = mix(uLow, uHigh, vW);
  float inner = smoothstep(0.5 - uRing * px, 0.5 - (uRing + 1.0) * px, d);
  vec3 col = uRing > 0.0 ? mix(mix(uSurface, uInk, vHover), fill, inner) : fill;
  float face = mix(uBack, uFront, smoothstep(-0.12, 0.12, vDepth));
  float a = disc * face;
  o = vec4(col * a, a);
}`;

function gaussPair(seed: number) {
  // Deterministic Box-Muller from a small LCG so the blob is the same on every visit.
  let s = seed;
  const r = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  return () => {
    const u1 = Math.max(r(), 1e-7);
    const u2 = r();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };
}

const toXYZ = (lat: number, lon: number): [number, number, number] => {
  const la = (lat * Math.PI) / 180;
  const lo = (lon * Math.PI) / 180;
  return [Math.cos(la) * Math.sin(lo), Math.sin(la), Math.cos(la) * Math.cos(lo)];
};

// Mark scale, in units of the land grain. Diameter follows the square root of the count, so a mark's area
// is proportional to its visits; the smallest stays a clear dot. Tone runs low -> high on the same root.
const D_MIN = 2.8;
const D_MAX = 10;
const mixRGB = (a: number[], b: number[], k: number) => a.map((v, i) => v * (1 - k) + b[i] * k) as [number, number, number];

export function initGlobe() {
  const root = document.documentElement;
  const canvas = document.querySelector<HTMLCanvasElement>('[data-globe]');
  const setT = tReadout(document.querySelector<HTMLElement>('[data-globe-t]'));
  if (!canvas) return;
  const gl = getGL(canvas);
  let prog: ReturnType<typeof compile> | null = null;
  try {
    if (gl) prog = compile(gl, VS, FS);
  } catch {
    prog = null;
  }
  if (!gl || !prog) {
    root.classList.add('no-gl');
    return;
  }
  root.classList.add('globe-gl');
  const still = isStill();
  const { program, u } = prog;
  type Mark = { lat: number; lon: number; n: number; label: string; p: [number, number, number] };
  // Busiest first, so smaller marks draw on top of larger neighbours; the index is also the vertex id.
  const marks: Mark[] = (JSON.parse(canvas.dataset.marks || '[]') as Omit<Mark, 'p'>[])
    .sort((a, b) => b.n - a.n)
    .map((m) => ({ ...m, p: toXYZ(m.lat, m.lon).map((v) => v * 1.012) as [number, number, number] }));
  const nRef = Math.max(8, ...marks.map((m) => m.n));
  const markSize = (n: number) => Math.max(D_MIN, D_MAX * Math.sqrt(n / nRef));
  const markTone = (n: number) => Math.min(1, Math.max(0, (Math.sqrt(n) - 1) / (Math.sqrt(nRef) - 1)));
  const wrap = canvas.closest<HTMLElement>('.globe-wrap');
  const tip = wrap?.querySelector<HTMLElement>('[data-globe-tip]') ?? null;
  const keyDots = Array.from(wrap?.querySelectorAll<HTMLElement>('[data-globe-key] i[data-n]') ?? []);

  let land: { vao: WebGLVertexArrayObject; count: number } | null = null;
  let dots: { vao: WebGLVertexArrayObject; count: number } | null = null;

  function makeCloud(pos: Float32Array, size: Float32Array, seed: number, tone?: Float32Array) {
    const g = gaussPair(seed);
    const eps = new Float32Array(pos.length);
    for (let i = 0; i < eps.length; i++) eps[i] = g();
    const vao = gl!.createVertexArray()!;
    gl!.bindVertexArray(vao);
    const bind = (data: Float32Array, name: string, n: number) => {
      const b = gl!.createBuffer();
      gl!.bindBuffer(gl!.ARRAY_BUFFER, b);
      gl!.bufferData(gl!.ARRAY_BUFFER, data, gl!.STATIC_DRAW);
      const loc = gl!.getAttribLocation(program, name);
      gl!.enableVertexAttribArray(loc);
      gl!.vertexAttribPointer(loc, n, gl!.FLOAT, false, 0, 0);
    };
    bind(pos, 'aPos', 3);
    bind(eps, 'aEps', 3);
    bind(size, 'aSize', 1);
    if (tone) bind(tone, 'aW', 1);
    gl!.bindVertexArray(null);
    return { vao, count: pos.length / 3 };
  }

  let dpr = 1;
  // Device pixels per grain unit, as the shader uses it.
  const unitPx = () => Math.max(1.6, (canvas!.width / 620) * 2.4);
  function size() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = canvas!.getBoundingClientRect();
    canvas!.width = Math.max(1, Math.round(r.width * dpr));
    canvas!.height = Math.max(1, Math.round(r.height * dpr));
    paintKey();
  }

  let ink = token('--ink');
  let surface = token('--plate');
  let low = ink;
  let high = ink;
  function readColors() {
    ink = token('--ink');
    surface = token('--plate');
    const accent = token('--accent');
    const dark = root.dataset.theme === 'dark';
    // One cobalt ramp: a pale (light) or dim (dark) step for a single visit, rising past the accent toward
    // the ink, so the busiest places read deeper on the light plate and brighter on the dark one.
    low = mixRGB(accent, surface, dark ? 0.38 : 0.42);
    high = mixRGB(accent, ink, dark ? 0.45 : 0.2);
  }
  const css = (c: number[]) => `rgb(${c.map((v) => Math.round(v * 255)).join(' ')})`;
  // The size key under the globe uses the globe's own scale and ramp, at the front of the sphere.
  function paintKey() {
    const u = unitPx() / dpr;
    keyDots.forEach((el) => {
      const n = Number(el.dataset.n);
      const d = markSize(n) * u;
      el.style.width = el.style.height = `${d.toFixed(1)}px`;
      el.style.background = css(mixRGB(low, high, markTone(n)));
    });
  }
  readColors();
  size();
  new ResizeObserver(size).observe(canvas);
  document.addEventListener('themechange', () => {
    readColors();
    paintKey();
    if (still) render(0);
  });

  // Start over the Atlantic, tilted to show the northern hemisphere.
  let yaw = (30 * Math.PI) / 180;
  let pitch = 0.36;
  let vYaw = 0;
  let vPitch = 0;
  let dragging = false;
  let t = still ? 0 : 1;
  let intro: number[] = [];
  let started = false;
  let visible = false;
  let hover = -1;
  let inside = false;
  let pinned = false;

  function rot(): Float32Array {
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    // q = Rx(pitch) * Ry(yaw) * p, column-major for GLSL.
    return new Float32Array([cy, sp * sy, -cp * sy, 0, cp, sp, sy, -sp * cy, cp * cy]);
  }

  function render(frame: number) {
    void frame;
    const W = canvas!.width;
    const H = canvas!.height;
    gl!.viewport(0, 0, W, H);
    gl!.clearColor(0, 0, 0, 0);
    gl!.clear(gl!.COLOR_BUFFER_BIT);
    if (!land) return;
    gl!.useProgram(program);
    gl!.enable(gl!.BLEND);
    gl!.blendFunc(gl!.ONE, gl!.ONE_MINUS_SRC_ALPHA);
    gl!.uniformMatrix3fv(u('uRot'), false, rot());
    gl!.uniform1f(u('uAb'), alphaBar(t));
    gl!.uniform1f(u('uScale'), 0.9);
    gl!.uniform1f(u('uPx'), unitPx());
    gl!.uniform3fv(u('uLow'), ink);
    gl!.uniform3fv(u('uHigh'), ink);
    gl!.uniform3fv(u('uSurface'), surface);
    gl!.uniform3fv(u('uInk'), ink);
    gl!.uniform1f(u('uRing'), 0);
    gl!.uniform1i(u('uHover'), -1);
    gl!.uniform1f(u('uFront'), 0.86);
    gl!.uniform1f(u('uBack'), 0.12);
    gl!.bindVertexArray(land.vao);
    gl!.drawArrays(gl!.POINTS, 0, land.count);
    if (dots && dots.count) {
      gl!.uniform3fv(u('uLow'), low);
      gl!.uniform3fv(u('uHigh'), high);
      gl!.uniform1f(u('uRing'), Math.max(1, Math.round(1.25 * dpr)));
      gl!.uniform1i(u('uHover'), hover);
      gl!.uniform1f(u('uFront'), 1.0);
      gl!.uniform1f(u('uBack'), 0.22);
      gl!.bindVertexArray(dots.vao);
      gl!.drawArrays(gl!.POINTS, 0, dots.count);
    }
    gl!.bindVertexArray(null);
  }

  function tick(frame: number) {
    if (!visible || !land) return;
    if (intro.length) t = intro.shift()!;
    setT(t);
    if (!dragging) {
      if (!still && !inside && !pinned) yaw += 0.0036;
      yaw += vYaw;
      pitch = Math.max(-1.1, Math.min(1.1, pitch + vPitch));
      vYaw *= 0.9;
      vPitch *= 0.9;
    }
    render(frame);
    if (hover >= 0) placeTip();
  }

  async function load() {
    const res = await fetch('/data/land.bin');
    const raw = new Int16Array(await res.arrayBuffer());
    const n = raw.length / 2;
    const pos = new Float32Array(n * 3);
    const size = new Float32Array(n).fill(1);
    for (let i = 0; i < n; i++) pos.set(toXYZ(raw[i * 2] / 100, raw[i * 2 + 1] / 100), i * 3);
    land = makeCloud(pos, size, 42);
    if (marks.length) {
      const mp = new Float32Array(marks.length * 3);
      const ms = new Float32Array(marks.length);
      const mw = new Float32Array(marks.length);
      marks.forEach((m, i) => {
        mp.set(m.p, i * 3);
        ms[i] = markSize(m.n);
        mw[i] = markTone(m.n);
      });
      dots = makeCloud(mp, ms, 7, mw);
    }
    if (still) render(0);
  }

  watchVisible(canvas, (on) => {
    visible = on;
    if (!on) setHover(-1);
    if (on && !started) {
      started = true;
      load().then(() => {
        if (!still) intro = Array.from({ length: 36 }, (_, k) => 1 - (k + 1) / 36);
      });
    }
  }, '200px');
  if (!still) onFrame(tick);

  // Hover (or tap) a mark for its count and place. Rotation holds while the pointer is over the globe or a
  // readout is open, so a mark can be aimed at; the hit area is wider than the mark.
  function project(m: Mark) {
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const [x, y, z] = m.p;
    const qx = cy * x + sy * z;
    const qy = sp * sy * x + cp * y - sp * cy * z;
    const qz = -cp * sy * x + sp * y + cp * cy * z;
    const r = canvas!.getBoundingClientRect();
    return {
      x: r.left + (qx * 0.9 * 0.5 + 0.5) * r.width,
      y: r.top + (0.5 - qy * 0.9 * 0.5) * r.height,
      z: qz,
      rad: (markSize(m.n) * unitPx() * (0.78 + 0.22 * Math.min(1, qz))) / dpr / 2,
    };
  }
  function pick(cx: number, cy: number) {
    if (t > 0.001) return -1;
    let best = -1;
    let bestD = Infinity;
    marks.forEach((m, i) => {
      const q = project(m);
      if (q.z < 0.05) return;
      const d = Math.hypot(q.x - cx, q.y - cy);
      if (d <= Math.max(12, q.rad + 6) && d < bestD) {
        best = i;
        bestD = d;
      }
    });
    return best;
  }
  function placeTip() {
    if (!tip || !wrap || hover < 0) return;
    const q = project(marks[hover]);
    if (q.z < 0.05) return setHover(-1);
    const w = wrap.getBoundingClientRect();
    const half = tip.offsetWidth / 2;
    const x = Math.min(Math.max(q.x - w.left, half), w.width - half);
    const above = q.y - w.top - q.rad > tip.offsetHeight + 16;
    tip.style.left = `${x}px`;
    tip.style.top = `${q.y - w.top + (above ? -q.rad : q.rad)}px`;
    tip.style.transform = above ? 'translate(-50%, calc(-100% - 10px))' : 'translate(-50%, 10px)';
  }
  function setHover(i: number) {
    if (i < 0) pinned = false;
    if (i === hover) return;
    hover = i;
    canvas!.style.cursor = i >= 0 ? 'pointer' : '';
    if (tip) {
      tip.hidden = i < 0;
      if (i >= 0) {
        const m = marks[i];
        tip.querySelector('[data-tip-n]')!.textContent = `${m.n.toLocaleString('en-US')} ${m.n === 1 ? 'visit' : 'visits'}`;
        tip.querySelector('[data-tip-place]')!.textContent = m.label;
        placeTip();
      }
    }
    if (still) render(0);
  }
  canvas.addEventListener('pointerenter', (e) => {
    if (e.pointerType === 'mouse') inside = true;
  });
  canvas.addEventListener('pointerleave', (e) => {
    if (e.pointerType !== 'mouse') return;
    inside = false;
    if (!pinned) setHover(-1);
  });

  // Drag to turn, with a little inertia. A press that barely moves is a tap.
  let lx = 0, ly = 0, sx = 0, sy = 0;
  canvas.addEventListener('pointerdown', (e) => {
    dragging = true;
    lx = sx = e.clientX;
    ly = sy = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!dragging) {
      if (e.pointerType === 'mouse' && !pinned) setHover(pick(e.clientX, e.clientY));
      return;
    }
    const dx = e.clientX - lx;
    const dy = e.clientY - ly;
    lx = e.clientX;
    ly = e.clientY;
    vYaw = dx * 0.006;
    vPitch = dy * 0.005;
    yaw += vYaw;
    pitch = Math.max(-1.1, Math.min(1.1, pitch + vPitch));
    if (still) render(0);
  });
  canvas.addEventListener('pointerup', (e) => {
    dragging = false;
    if (Math.hypot(e.clientX - sx, e.clientY - sy) > 6) return;
    const i = pick(e.clientX, e.clientY);
    setHover(i);
    pinned = i >= 0 && e.pointerType !== 'mouse';
  });
  canvas.addEventListener('pointercancel', () => (dragging = false));
}
