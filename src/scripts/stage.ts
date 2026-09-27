import { QUAD_VS, NOISE_GLSL, compile, fullscreenQuad, getGL, isStill, onFrame, watchVisible, tReadout } from './gl';

// Each publication's before/after figure plays through the same sampler. It samples in once it is on screen;
// switching scenes runs the forward process up to t = 720, swaps the media, then samples back to t = 0.
// Media are 16:9 pairs laid side by side in one clip or still (source left, result right); a stage taller than it
// is wide (compact papers, phones) stacks them instead.

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
  // Cover-fit each 16:9 half of the media into its pane.
  float ta = 16.0 / 9.0;
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

function initStage(fig: HTMLElement) {
  const media = fig.querySelector<HTMLVideoElement | HTMLImageElement>('[data-stage-media]');
  const canvas = fig.querySelector<HTMLCanvasElement>('[data-stage-canvas]');
  const setT = tReadout(fig.querySelector<HTMLElement>('[data-stage-t]'));
  const buttons = Array.from(fig.querySelectorAll<HTMLButtonElement>('[data-scene]'));
  if (!media || !canvas) return;
  const base = fig.dataset.base!;
  const video = media instanceof HTMLVideoElement ? media : null;
  const ext = video ? 'mp4' : 'jpg';
  const ready = () => (video ? video.readyState >= 2 : (media as HTMLImageElement).complete && (media as HTMLImageElement).naturalWidth > 0);

  const still = isStill();
  const setScene = (id: string) => {
    const b = buttons.find((x) => x.dataset.scene === id);
    if (video) {
      video.poster = `${base}/${id}.jpg`;
      video.src = `${base}/${id}.mp4`;
      video.play().catch(() => {});
    } else {
      media.setAttribute('src', `${base}/${id}.${ext}`);
    }
    if (b?.dataset.alt) media.setAttribute('aria-label', b.dataset.alt);
    if (!video && b?.dataset.alt) media.setAttribute('alt', b.dataset.alt);
    buttons.forEach((x) => x.setAttribute('aria-pressed', String(x.dataset.scene === id)));
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
    if (!video) return;
    if (on) {
      video.preload = 'auto';
      video.play().catch(() => {});
    } else {
      video.pause();
    }
  }, '240px');

  // Without the sampler (reduced motion or no WebGL) the buttons simply swap the media.
  if (!gl || !prog) {
    buttons.forEach((b) => b.addEventListener('click', () => setScene(b.dataset.scene!)));
    fig.querySelector('.t-read')?.setAttribute('hidden', '');
    return;
  }

  fig.classList.add('is-gl');
  if (video) video.removeAttribute('poster');
  const { program, u } = prog;
  const draw = fullscreenQuad(gl, program);
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  let dpr = 1;
  let hasTex = false;
  let t = 1;
  let introDone = false;
  let holdSwap = false;
  // A still only needs a new texture when it changes, and a new frame when t or the canvas does.
  let stale = true;
  let dirty = true;
  // A queue of t targets, one per frame, so every transition lands on the shared clock.
  let path: number[] = [];

  function size() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = canvas!.getBoundingClientRect();
    canvas!.width = Math.max(1, Math.round(r.width * dpr));
    canvas!.height = Math.max(1, Math.round(r.height * dpr));
    dirty = true;
  }
  size();
  new ResizeObserver(size).observe(canvas);
  if (!video) media.addEventListener('load', () => (stale = true));

  function upload() {
    if (holdSwap || !ready() || (!video && !stale)) return;
    gl!.bindTexture(gl!.TEXTURE_2D, tex);
    gl!.pixelStorei(gl!.UNPACK_FLIP_Y_WEBGL, false);
    gl!.texImage2D(gl!.TEXTURE_2D, 0, gl!.RGBA, gl!.RGBA, gl!.UNSIGNED_BYTE, media!);
    gl!.generateMipmap(gl!.TEXTURE_2D);
    hasTex = true;
    stale = false;
    dirty = true;
  }

  const ramp = (from: number, to: number, frames: number) =>
    Array.from({ length: frames }, (_, k) => from + (to - from) * ((k + 1) / frames));

  function tick(frame: number) {
    if (!visible && !path.length) return;
    upload();
    if (!introDone && hasTex && visible && !path.length) {
      introDone = true;
      path = ramp(1, 0, 30);
    }
    // A -1 marks the peak of the forward process: hold there, grain alive, until the next media has loaded.
    const moving = path.length > 0;
    if (moving && path[0] !== -1) t = path.shift()!;
    setT(t);
    if (!video && !moving && !dirty) return;
    dirty = false;
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
    gl!.uniform1f(u('uStack'), H > W * 0.8 ? 1 : 0);
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
        const loaded = () => {
          media.removeEventListener(video ? 'loadeddata' : 'load', loaded);
          if (my !== seq) return;
          holdSwap = false;
          stale = true;
          path = ramp(peak, 0, 15);
        };
        if (!video && ready()) loaded();
        else media.addEventListener(video ? 'loadeddata' : 'load', loaded);
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

export function initStages() {
  document.querySelectorAll<HTMLElement>('[data-stage]').forEach(initStage);
}
