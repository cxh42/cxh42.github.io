// Shared WebGL2 plumbing, the noise schedule, and the page's single 24 fps clock.

export const FPS = 24;
export const FRAME_MS = 1000 / FPS;

export const isStill = () => document.documentElement.classList.contains('still');

export function getGL(canvas: HTMLCanvasElement, opts: WebGLContextAttributes = {}) {
  try {
    return canvas.getContext('webgl2', {
      alpha: true,
      antialias: false,
      premultipliedAlpha: true,
      depth: false,
      stencil: false,
      powerPreference: 'default',
      ...opts,
    }) as WebGL2RenderingContext | null;
  } catch {
    return null;
  }
}

export const QUAD_VS = /* glsl */ `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

// PCG hash -> uniform -> Box-Muller Gaussian. Integer hashing keeps the grain free of lattice artefacts.
// alphaBar is the cosine schedule (Nichol & Dhariwal 2021) over normalised t in [0, 1].
export const NOISE_GLSL = /* glsl */ `
uint pcg(uint v) {
  uint s = v * 747796405u + 2891336453u;
  uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
// Keep 24 bits before converting: some desktop drivers (NVIDIA via ANGLE/GL) return 0 for float() of a
// full 32-bit uint, which flattens the noise to a constant.
float rnd(uvec3 p) { return float(pcg(p.x + pcg(p.y + pcg(p.z))) >> 8u) * (1.0 / 16777216.0); }
float gauss(uvec3 p) {
  float u1 = max(rnd(p), 1e-7);
  float u2 = rnd(p + uvec3(7u, 13u, 101u));
  return sqrt(-2.0 * log(u1)) * cos(6.28318530718 * u2);
}
float alphaBar(float t) {
  const float s = 0.008;
  float f = cos(((t + s) / (1.0 + s)) * 1.57079632679);
  float f0 = cos((s / (1.0 + s)) * 1.57079632679);
  return clamp((f * f) / (f0 * f0), 0.0, 1.0);
}
`;

export function alphaBar(t: number) {
  const s = 0.008;
  const f = Math.cos(((t + s) / (1 + s)) * (Math.PI / 2));
  const f0 = Math.cos((s / (1 + s)) * (Math.PI / 2));
  return Math.min(1, Math.max(0, (f * f) / (f0 * f0)));
}

export function compile(gl: WebGL2RenderingContext, vs: string, fs: string) {
  const make = (type: number, src: string) => {
    const sh = gl.createShader(type)!;
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(sh);
      gl.deleteShader(sh);
      throw new Error(log || 'shader compile failed');
    }
    return sh;
  };
  const p = gl.createProgram()!;
  gl.attachShader(p, make(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, make(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) || 'link failed');
  const uniforms = new Map<string, WebGLUniformLocation | null>();
  const u = (name: string) => {
    if (!uniforms.has(name)) uniforms.set(name, gl.getUniformLocation(p, name));
    return uniforms.get(name)!;
  };
  return { program: p, u };
}

export function fullscreenQuad(gl: WebGL2RenderingContext, program: WebGLProgram) {
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(program, 'aPos');
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
  return () => {
    gl.bindVertexArray(vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  };
}

/** Reads a hex colour token (e.g. --ink) as linear-ish 0..1 RGB. */
export function token(name: string, el: Element = document.documentElement): [number, number, number] {
  const raw = getComputedStyle(el).getPropertyValue(name).trim();
  const m = raw.match(/^#([0-9a-f]{6})$/i);
  if (!m) return [0.5, 0.5, 0.5];
  const n = parseInt(m[1], 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

/**
 * The one clock. Callbacks run at most 24 times a second, on shared frame boundaries,
 * so every sampler on the page ticks together like frames of the same film.
 */
type Tick = (frame: number) => void;
const ticks = new Set<Tick>();
let raf = 0;
let lastFrame = -1;
function loop(now: number) {
  raf = requestAnimationFrame(loop);
  const frame = Math.floor(now / FRAME_MS);
  if (frame === lastFrame) return;
  lastFrame = frame;
  ticks.forEach((fn) => fn(frame));
}
export function onFrame(fn: Tick) {
  ticks.add(fn);
  if (!raf) raf = requestAnimationFrame(loop);
  return () => {
    ticks.delete(fn);
    if (ticks.size === 0 && raf) {
      cancelAnimationFrame(raf);
      raf = 0;
    }
  };
}

/** Fires with true/false as the element enters or leaves the viewport (with a margin). */
export function watchVisible(el: Element, cb: (on: boolean) => void, margin = '120px') {
  const io = new IntersectionObserver((es) => es.forEach((e) => cb(e.isIntersecting)), { rootMargin: margin });
  io.observe(el);
  return io;
}

export const fmtT = (t: number) => String(Math.round(t * 1000));
export const fmtAb = (t: number) => alphaBar(t).toFixed(3);
