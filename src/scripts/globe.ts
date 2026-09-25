import { alphaBar, compile, getGL, isStill, onFrame, token, watchVisible, fmtT } from './gl';

// Visitors: the land is a point cloud that diffuses out of a Gaussian blob onto the sphere
// (p_t = sqrt(ab) p0 + sqrt(1 - ab) eps). Countries with visits are marked in the accent.

const VS = /* glsl */ `#version 300 es
in vec3 aPos;
in vec3 aEps;
in float aSize;
uniform mat3 uRot;
uniform float uAb;
uniform float uPx;
uniform float uScale;
out float vDepth;
void main() {
  vec3 p = sqrt(uAb) * aPos + sqrt(1.0 - uAb) * aEps * 0.62;
  vec3 q = uRot * p;
  gl_Position = vec4(q.x * uScale, q.y * uScale, 0.0, 1.0);
  vDepth = q.z;
  gl_PointSize = uPx * aSize * (0.78 + 0.22 * clamp(q.z, -1.0, 1.0));
}`;

const FS = /* glsl */ `#version 300 es
precision highp float;
in float vDepth;
out vec4 o;
uniform vec3 uColor;
uniform float uFront;
uniform float uBack;
void main() {
  float d = length(gl_PointCoord - 0.5);
  float disc = smoothstep(0.5, 0.32, d);
  float face = mix(uBack, uFront, smoothstep(-0.12, 0.12, vDepth));
  float a = disc * face;
  o = vec4(uColor * a, a);
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

export function initGlobe() {
  const root = document.documentElement;
  const canvas = document.querySelector<HTMLCanvasElement>('[data-globe]');
  const out = document.querySelector<HTMLElement>('[data-globe-t]');
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
  const marks: { lat: number; lon: number; n: number }[] = JSON.parse(canvas.dataset.marks || '[]');

  let land: { vao: WebGLVertexArrayObject; count: number } | null = null;
  let dots: { vao: WebGLVertexArrayObject; count: number } | null = null;

  function makeCloud(pos: Float32Array, size: Float32Array, seed: number) {
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
    gl!.bindVertexArray(null);
    return { vao, count: pos.length / 3 };
  }

  let dpr = 1;
  function size() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = canvas!.getBoundingClientRect();
    canvas!.width = Math.max(1, Math.round(r.width * dpr));
    canvas!.height = Math.max(1, Math.round(r.height * dpr));
  }
  size();
  new ResizeObserver(size).observe(canvas);

  let ink = token('--ink');
  let accent = token('--accent');
  document.addEventListener('themechange', () => {
    ink = token('--ink');
    accent = token('--accent');
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
    gl!.uniform1f(u('uPx'), Math.max(1.6, (W / 620) * 2.4));
    gl!.uniform3fv(u('uColor'), ink);
    gl!.uniform1f(u('uFront'), 0.86);
    gl!.uniform1f(u('uBack'), 0.12);
    gl!.bindVertexArray(land.vao);
    gl!.drawArrays(gl!.POINTS, 0, land.count);
    if (dots && dots.count) {
      gl!.uniform3fv(u('uColor'), accent);
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
    if (out) {
      out.textContent = fmtT(t);
      out.classList.toggle('live', t > 0.0005);
    }
    if (!dragging) {
      if (!still) yaw += 0.0036;
      yaw += vYaw;
      pitch = Math.max(-1.1, Math.min(1.1, pitch + vPitch));
      vYaw *= 0.9;
      vPitch *= 0.9;
    }
    render(frame);
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
      marks.forEach((m, i) => {
        const [x, y, z] = toXYZ(m.lat, m.lon);
        mp.set([x * 1.012, y * 1.012, z * 1.012], i * 3);
        ms[i] = 2.4 + 1.5 * Math.log10(1 + m.n);
      });
      dots = makeCloud(mp, ms, 7);
    }
    if (still) render(0);
  }

  watchVisible(canvas, (on) => {
    visible = on;
    if (on && !started) {
      started = true;
      load().then(() => {
        if (!still) intro = Array.from({ length: 36 }, (_, k) => 1 - (k + 1) / 36);
        if (out) out.textContent = fmtT(t);
      });
    }
  }, '200px');
  if (!still) onFrame(tick);

  // Drag to turn, with a little inertia.
  let lx = 0, ly = 0;
  canvas.addEventListener('pointerdown', (e) => {
    dragging = true;
    lx = e.clientX;
    ly = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!dragging) return;
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
  const end = () => (dragging = false);
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
}
