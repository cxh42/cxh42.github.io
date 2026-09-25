import { QUAD_VS, NOISE_GLSL, compile, fullscreenQuad, getGL, isStill, onFrame, watchVisible, fmtT } from './gl';

// The ViTeX figure: the real edits play through the same sampler. Switching scenes runs the forward
// process up to t = 720, swaps the clip, then samples back to t = 0.

const FS = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 o;
uniform sampler2D uTex;
uniform float uHasTex;
uniform float uT;
uniform uint uFrame;
uniform float uGrain;
uniform float uLod;
uniform vec2 uSize;     // canvas size in device px
uniform float uGap;     // gap between the two panes, device px
uniform float uStack;   // 0: side by side, 1: stacked
${NOISE_GLSL}
void main() {
  vec2 px = vec2(gl_FragCoord.x, uSize.y - gl_FragCoord.y);
  float along = mix(px.x, px.y, uStack);
  float span = mix(uSize.x, uSize.y, uStack);
  float pane = (span - uGap) * 0.5;
  float side = along < pane ? 0.0 : 1.0;
  float local = along - side * (pane + uGap);
  if (local < 0.0 || local > pane) { o = vec4(0.0); return; }
  vec2 wh = mix(vec2(pane, uSize.y), vec2(uSize.x, pane), uStack);
  vec2 p = mix(vec2(local, px.y), vec2(px.x, local), uStack) / wh;
  // Cover-fit each 426x240 half of the clip into its pane.
  float ta = 426.0 / 240.0;
  float ra = wh.x / wh.y;
  vec2 sc = ra > ta ? vec2(1.0, ta / ra) : vec2(ra / ta, 1.0);
  p = (p - 0.5) * sc + 0.5;
  vec2 uv = vec2(side * 0.5 + p.x * 0.5, p.y);
  vec2 cell = floor(gl_FragCoord.xy / uGrain);
  float eps = gauss(uvec3(uvec2(cell), uFrame));
  float ab = alphaBar(uT);
  float nz = sqrt(1.0 - ab);
  vec3 x0 = textureLod(uTex, uv, uLod * nz).rgb * 2.0 - 1.0;
  x0 *= uHasTex;
  vec3 xt = sqrt(ab) * x0 + nz * eps * 0.62;
  o = vec4(clamp(xt * 0.5 + 0.5, 0.0, 1.0), 1.0);
}`;

export function initVitex() {
  const root = document.documentElement;
  const fig = document.querySelector<HTMLElement>('[data-vitex]');
  const video = document.querySelector<HTMLVideoElement>('[data-vitex-video]');
  const canvas = document.querySelector<HTMLCanvasElement>('[data-vitex-canvas]');
  const out = document.querySelector<HTMLElement>('[data-vitex-t]');
  const buttons = Array.from(document.querySelectorAll<HTMLButtonElement>('[data-scene]'));
  if (!fig || !video || !canvas) return;

  const still = isStill();
  const setScene = (id: string) => {
    video.src = `/media/vitex/${id}.mp4`;
    video.poster = `/media/vitex/${id}.jpg`;
    const b = buttons.find((x) => x.dataset.scene === id);
    if (b) video.setAttribute('aria-label', `ViTeX-Edit-14B replacing “${b.dataset.from}” with “${b.dataset.to}”: source on the left, edited video on the right.`);
    buttons.forEach((x) => x.setAttribute('aria-pressed', String(x.dataset.scene === id)));
    video.play().catch(() => {});
  };

  const gl = getGL(canvas, { alpha: true });
  let prog: ReturnType<typeof compile> | null = null;
  try {
    if (gl && !still) prog = compile(gl, QUAD_VS, FS);
  } catch {
    prog = null;
  }

  let visible = false;
  watchVisible(fig, (on) => {
    visible = on;
    if (on) {
      video.preload = 'auto';
      video.play().catch(() => {});
    } else {
      video.pause();
    }
  }, '240px');

  // Without the sampler (reduced motion or no WebGL) the buttons simply swap the clip.
  if (!gl || !prog) {
    buttons.forEach((b) => b.addEventListener('click', () => setScene(b.dataset.scene!)));
    if (out) out.closest('p')?.setAttribute('hidden', '');
    return;
  }

  root.classList.add('vitex-gl');
  video.removeAttribute('poster');
  const { program, u } = prog;
  const draw = fullscreenQuad(gl, program);
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  const stackMq = window.matchMedia('(max-width: 640px)');
  let dpr = 1;
  let hasTex = false;
  let t = 1;
  let introDone = false;
  let holdSwap = false;
  // A queue of t targets, one per frame, so every transition lands on the shared clock.
  let path: number[] = [];

  function size() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = canvas!.getBoundingClientRect();
    canvas!.width = Math.max(1, Math.round(r.width * dpr));
    canvas!.height = Math.max(1, Math.round(r.height * dpr));
  }
  size();
  new ResizeObserver(size).observe(canvas);

  function upload() {
    if (holdSwap || video!.readyState < 2) return;
    gl!.bindTexture(gl!.TEXTURE_2D, tex);
    gl!.pixelStorei(gl!.UNPACK_FLIP_Y_WEBGL, false);
    gl!.texImage2D(gl!.TEXTURE_2D, 0, gl!.RGBA, gl!.RGBA, gl!.UNSIGNED_BYTE, video!);
    gl!.generateMipmap(gl!.TEXTURE_2D);
    hasTex = true;
  }

  const ramp = (from: number, to: number, frames: number) =>
    Array.from({ length: frames }, (_, k) => from + (to - from) * ((k + 1) / frames));

  function tick(frame: number) {
    if (!visible && !path.length) return;
    if (!introDone && hasTex && visible && !path.length) {
      introDone = true;
      path = ramp(1, 0, 30);
    }
    // A -1 marks the peak of the forward process: hold there, grain alive, until the next clip has a frame.
    if (path.length && path[0] !== -1) t = path.shift()!;
    if (out) {
      out.textContent = fmtT(t);
      out.classList.toggle('live', t > 0.0005);
    }
    upload();
    const W = canvas!.width;
    const H = canvas!.height;
    gl!.viewport(0, 0, W, H);
    gl!.clearColor(0, 0, 0, 0);
    gl!.clear(gl!.COLOR_BUFFER_BIT);
    gl!.useProgram(program);
    gl!.activeTexture(gl!.TEXTURE0);
    gl!.bindTexture(gl!.TEXTURE_2D, tex);
    gl!.uniform1i(u('uTex'), 0);
    gl!.uniform1f(u('uHasTex'), hasTex ? 1 : 0);
    gl!.uniform1f(u('uT'), t);
    gl!.uniform1ui(u('uFrame'), frame >>> 0);
    gl!.uniform1f(u('uGrain'), Math.max(1, Math.round(dpr)));
    gl!.uniform1f(u('uLod'), 4.2);
    gl!.uniform2f(u('uSize'), W, H);
    gl!.uniform1f(u('uGap'), Math.round(8 * dpr));
    gl!.uniform1f(u('uStack'), stackMq.matches ? 1 : 0);
    draw();
  }
  onFrame(tick);

  let seq = 0;
  buttons.forEach((b) =>
    b.addEventListener('click', () => {
      const id = b.dataset.scene!;
      if (b.getAttribute('aria-pressed') === 'true') return;
      const my = ++seq;
      buttons.forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
      introDone = true;
      holdSwap = false;
      const peak = 0.72;
      path = [...ramp(t, peak, 9), -1];
      const swapAt = () => {
        holdSwap = true;
        setScene(id);
        const ready = () => {
          video.removeEventListener('loadeddata', ready);
          if (my !== seq) return;
          holdSwap = false;
          path = ramp(peak, 0, 15);
        };
        video.addEventListener('loadeddata', ready);
      };
      // Swap once the forward ramp has reached its peak.
      const wait = () => {
        if (my !== seq) return;
        if (path.length === 1 && path[0] === -1) swapAt();
        else requestAnimationFrame(wait);
      };
      wait();
    }),
  );
}
