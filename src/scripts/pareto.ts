import { isStill, watchVisible } from './gl';

// The ViTeX-Bench Pareto front as a rotatable cube, drawn the way the benchmark's own leaderboard draws it:
// x = correctness (SeqAcc), y = temporal quality (Warp_c), z = locality (DreamSim_loc). Every axis runs
// from worst (-1) to best (+1), so the ideal method sits at the corner (1, 1, 1). Rows are read live from
// the leaderboard; the paper's numbers stand in if that fails.

const DATA = 'https://vitex-bench.github.io/ViTeX-Bench-Leaderboard/data/submissions.jsonl';

type Row = { method: string; kind: string; temporal_comparable: boolean; SeqAcc: number; Warp_crop: number; DreamSim_loc: number; layer?: number | null };
type Key = 'SeqAcc' | 'Warp_crop' | 'DreamSim_loc';

const FALLBACK: Row[] = [
  { method: 'TextCtrl', kind: 'editor', temporal_comparable: true, SeqAcc: 0.47475, Warp_crop: 2.08761, DreamSim_loc: 0.00429 },
  { method: 'ViTeX-Edit-14B (Composite)', kind: 'postprocessed', temporal_comparable: true, SeqAcc: 0.3449, Warp_crop: 1.55914, DreamSim_loc: 0.00233 },
  { method: 'ViTeX-Edit-14B', kind: 'editor', temporal_comparable: true, SeqAcc: 0.34121, Warp_crop: 1.53042, DreamSim_loc: 0.02352 },
  { method: 'VideoPainter', kind: 'editor', temporal_comparable: false, SeqAcc: 0.3645, Warp_crop: 3.34526, DreamSim_loc: 0.02391 },
  { method: 'FLUX-Text', kind: 'editor', temporal_comparable: true, SeqAcc: 0.52837, Warp_crop: 13.00985, DreamSim_loc: 0.01204 },
  { method: 'RS-STE', kind: 'editor', temporal_comparable: true, SeqAcc: 0.35397, Warp_crop: 1.81479, DreamSim_loc: 0.00732 },
  { method: 'AnyText2', kind: 'editor', temporal_comparable: true, SeqAcc: 0.27973, Warp_crop: 3.95164, DreamSim_loc: 0.0431 },
  { method: 'TextCtrl + AnyV2V', kind: 'editor', temporal_comparable: true, SeqAcc: 0.05679, Warp_crop: 3.96745, DreamSim_loc: 0.07322 },
  { method: 'Source video', kind: 'reference', temporal_comparable: true, SeqAcc: 0, Warp_crop: 1.26902, DreamSim_loc: 0 },
  { method: 'Wan2.1-VACE-14B', kind: 'editor', temporal_comparable: true, SeqAcc: 0, Warp_crop: 1.56097, DreamSim_loc: 0.00706 },
  { method: 'Kling Video 3.0 Omni', kind: 'editor', temporal_comparable: true, SeqAcc: 0, Warp_crop: 2.90209, DreamSim_loc: 0.06078 },
];

const AX: { key: Key; name: string; up: boolean; log: boolean; floor?: number }[] = [
  { key: 'SeqAcc', name: 'Correctness', up: true, log: false },
  { key: 'Warp_crop', name: 'Temporal', up: false, log: true, floor: 0.001 },
  { key: 'DreamSim_loc', name: 'Locality', up: false, log: true, floor: 0.001 },
];

type V3 = [number, number, number];
type Star = { r: Row; p: V3 };
type Proj = { s: Star; x: number; y: number; z: number; f: number; a?: number };
type Box = { x: number; y: number; w: number; h: number };

const better = (k: Key, a: number, b: number) => (AX.find((x) => x.key === k)!.up ? a > b : a < b);
const ranked = (r: Row) => r.kind === 'editor';

function layer(rows: Row[]) {
  const keys = AX.map((a) => a.key);
  const dominates = (o: Row, r: Row) => keys.every((k) => !better(k, r[k], o[k])) && keys.some((k) => better(k, o[k], r[k]));
  rows.forEach((r) => (r.layer = null));
  let left = rows.filter((r) => ranked(r) && r.temporal_comparable);
  for (let n = 1; left.length; n++) {
    const front = left.filter((r) => !left.some((o) => o !== r && dominates(o, r)));
    front.forEach((r) => (r.layer = n));
    left = left.filter((r) => !front.includes(r));
  }
}

function scale(a: (typeof AX)[number], rows: Row[]) {
  const vals = rows.map((r) => r[a.key]).filter((v) => v != null && isFinite(v));
  const tf = a.log ? (v: number) => Math.log10(Math.max(v, a.floor!)) : (v: number) => v;
  let lo: number, hi: number;
  if (a.log) {
    lo = Math.min(...vals.map(tf)); hi = Math.max(...vals.map(tf));
    const pad = (hi - lo) * 0.06; lo -= pad; hi += pad;
  } else {
    lo = Math.min(0, ...vals); hi = Math.max(...vals);
    hi = Math.ceil(hi * 10 + 0.3) / 10; lo -= (hi - lo) * 0.04;
  }
  const f = (v: number) => { let t = (tf(v) - lo) / (hi - lo); t = a.up ? t : 1 - t; return t * 2 - 1; };
  const ticks = () => {
    if (a.log) {
      const c = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 1, 1.5, 2, 3, 5, 10, 20].filter((v) => { const t = tf(v); return t >= lo && t <= hi; });
      return c.filter((_, i) => c.length <= 4 || i % 2 === 0);
    }
    const out: number[] = []; for (let v = 0.2; v <= hi + 1e-9; v += 0.2) out.push(+v.toFixed(2)); return out; // 0 would crowd the origin corner
  };
  return { f, ticks };
}

export function initPareto() {
  const canvas = document.querySelector<HTMLCanvasElement>('[data-pareto]');
  if (!canvas) return;
  const ctx = canvas.getContext('2d')!;
  const status = document.querySelector<HTMLElement>('[data-pareto-status]');
  const still = isStill();

  let W = 0, H = 0;
  let stars: Star[] = [], scales: ReturnType<typeof scale>[] = [], tris: V3[][] = [];
  const cam = { yaw: 0.52, pitch: 0.3, persp: 0.18 };
  let drag: { x: number; y: number; yaw: number; pitch: number; moved: boolean; id: number } | null = null;
  let hover: Proj | null = null, focus: string | null = null, touched = false, visible = false, raf = 0;
  const t0 = performance.now();
  let proj: Proj[] = [];

  // Colours come from the page's tokens, normalised through the canvas parser.
  let C = { ink: [0, 0, 0], graphite: [0, 0, 0], plate: [255, 255, 255], acc: [0, 0, 255], dark: false };
  const probe = document.createElement('canvas').getContext('2d')!;
  function rgb(name: string) {
    probe.fillStyle = '#000';
    probe.fillStyle = getComputedStyle(canvas!).getPropertyValue(name).trim() || '#000';
    const s = String(probe.fillStyle);
    if (s[0] === '#') { const n = parseInt(s.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
    const m = s.match(/[\d.]+/g) || ['0', '0', '0'];
    return [+m[0], +m[1], +m[2]];
  }
  function readColors() {
    C = { ink: rgb('--ink'), graphite: rgb('--graphite'), plate: rgb('--plate-2'), acc: rgb('--accent'), dark: document.documentElement.dataset.theme === 'dark' };
  }
  const rgba = (c: number[], a: number) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  const SANS = "'Geologica Variable', Geologica, system-ui, sans-serif";

  function build(rows: Row[]) {
    layer(rows);
    const pts = rows.filter((r) => r.temporal_comparable && AX.every((a) => r[a.key] != null));
    scales = AX.map((a) => scale(a, pts));
    stars = pts.map((r) => ({ r, p: [scales[0].f(r.SeqAcc), scales[1].f(r.Warp_crop), scales[2].f(r.DreamSim_loc)] as V3 }));
    // Delaunay triangulation of the front in the plane facing the ideal diagonal, drawn back in 3-D.
    const fp = stars.filter((s) => s.r.layer === 1);
    const uv = fp.map(({ p }) => [(p[0] - p[1]) / Math.SQRT2, (p[0] + p[1] - 2 * p[2]) / Math.sqrt(6)]);
    tris = [];
    for (let i = 0; i < uv.length; i++) for (let j = i + 1; j < uv.length; j++) for (let k = j + 1; k < uv.length; k++) {
      const [A, B, D] = [uv[i], uv[j], uv[k]];
      const d = 2 * (A[0] * (B[1] - D[1]) + B[0] * (D[1] - A[1]) + D[0] * (A[1] - B[1]));
      if (Math.abs(d) < 1e-9) continue;
      const a2 = A[0] ** 2 + A[1] ** 2, b2 = B[0] ** 2 + B[1] ** 2, c2 = D[0] ** 2 + D[1] ** 2;
      const ux = (a2 * (B[1] - D[1]) + b2 * (D[1] - A[1]) + c2 * (A[1] - B[1])) / d;
      const uy = (a2 * (D[0] - B[0]) + b2 * (A[0] - D[0]) + c2 * (B[0] - A[0])) / d;
      const r2 = (A[0] - ux) ** 2 + (A[1] - uy) ** 2;
      if (uv.every((P, m) => m === i || m === j || m === k || (P[0] - ux) ** 2 + (P[1] - uy) ** 2 > r2 * (1 + 1e-9))) tris.push([fp[i].p, fp[j].p, fp[k].p]);
    }
    if (status) status.hidden = true;
    draw();
  }

  function size() {
    const b = canvas!.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = b.width; H = b.height;
    canvas!.width = Math.max(1, Math.round(W * dpr)); canvas!.height = Math.max(1, Math.round(H * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function projector() {
    const small = W < 420;
    const R = Math.min(W * (small ? 0.25 : 0.28), (H * 0.5 - (small ? 40 : 30)) / 1.42);
    const cx = W * 0.5, cy = H * 0.5 + 2;
    const cyw = Math.cos(cam.yaw), syw = Math.sin(cam.yaw), cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    return (p: V3) => {
      const x1 = p[0] * cyw + p[2] * syw, z1 = -p[0] * syw + p[2] * cyw;
      const q = [x1, p[1] * cp - z1 * sp, p[1] * sp + z1 * cp];
      const f = 1 / (1 - cam.persp * q[2] * 0.33);
      return { x: cx + q[0] * R * f, y: cy - q[1] * R * f, z: q[2], f, R };
    };
  }
  const line = (P: ReturnType<typeof projector>, a: V3, b: V3, style: string, w = 1, dash: number[] = []) => {
    const p = P(a), q = P(b);
    ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y);
    ctx.strokeStyle = style; ctx.lineWidth = w; ctx.setLineDash(dash); ctx.stroke(); ctx.setLineDash([]);
  };
  function text(s: string, x: number, y: number, font: string, color: string, align: CanvasTextAlign = 'left', halo = false) {
    ctx.font = font; ctx.textAlign = align; ctx.textBaseline = 'middle';
    if (halo) { ctx.lineWidth = 4; ctx.lineJoin = 'round'; ctx.strokeStyle = rgba(C.plate, 0.9); ctx.strokeText(s, x, y); }
    ctx.fillStyle = color; ctx.fillText(s, x, y);
  }
  const hit = (b: Box, list: Box[]) => list.some((o) => b.x < o.x + o.w && o.x < b.x + b.w && b.y < o.y + o.h && o.y < b.y + b.h);

  function draw() {
    if (!W || !stars.length) return;
    const P = projector(), ink = C.ink, acc = C.acc;
    ctx.clearRect(0, 0, W, H);
    const ticks = scales.map((s) => s.ticks().map((v) => ({ v, t: s.f(v) })));
    const wall = rgba(ink, 0.08), edge = rgba(ink, 0.24);
    ticks[0].forEach((k) => { line(P, [k.t, -1, -1], [k.t, 1, -1], wall); line(P, [k.t, -1, -1], [k.t, -1, 1], wall); });
    ticks[1].forEach((k) => { line(P, [-1, k.t, -1], [1, k.t, -1], wall); line(P, [-1, k.t, -1], [-1, k.t, 1], wall); });
    ticks[2].forEach((k) => { line(P, [-1, -1, k.t], [1, -1, k.t], wall); line(P, [-1, -1, k.t], [-1, 1, k.t], wall); });
    const box = rgba(ink, 0.13);
    ([[[-1, 1, -1], [1, 1, -1]], [[1, -1, -1], [1, 1, -1]], [[1, -1, -1], [1, -1, 1]], [[-1, 1, -1], [-1, 1, 1]], [[-1, -1, 1], [-1, 1, 1]], [[-1, -1, 1], [1, -1, 1]],
      [[1, 1, -1], [1, 1, 1]], [[-1, 1, 1], [1, 1, 1]], [[1, -1, 1], [1, 1, 1]]] as V3[][]).forEach((e) => line(P, e[0], e[1], box));

    // Axes along the outermost parallel edge. Each carries one short name, pushed straight out from the
    // cube's centre past its arrow so it never sits on the volume; names that would touch step further out.
    // Tick values are left off: the key under the figure names each metric and its direction.
    const small = W < 420, placed: Box[] = [], O = P([0, 0, 0]);
    const nf = `500 ${small ? 11.5 : 12.5}px ${SANS}`;
    AX.forEach((a, i) => {
      const j = (i + 1) % 3, k2 = (i + 2) % 3;
      const o: V3 = [0, 0, 0], e: V3 = [0, 0, 0]; o[i] = -1; e[i] = 1.12;
      const so = P(o), se = P(e), len = Math.hypot(se.x - so.x, se.y - so.y) || 1;
      const ux = (se.x - so.x) / len, uy = (se.y - so.y) / len;
      let nx = -uy, ny = ux; if (-nx + ny < 0) { nx = -nx; ny = -ny; }
      let best: number[] = [-1, -1], bs = -Infinity;
      [[-1, -1], [-1, 1], [1, -1], [1, 1]].forEach((c) => {
        const m: V3 = [0, 0, 0]; m[j] = c[0]; m[k2] = c[1];
        const pm = P(m), sc = (pm.x - O.x) * nx + (pm.y - O.y) * ny;
        if (sc > bs + 0.5) { bs = sc; best = c; }
      });
      const at = (t: number): V3 => { const q: V3 = [0, 0, 0]; q[i] = t; q[j] = best[0]; q[k2] = best[1]; return q; };
      line(P, at(-1), at(1.12), edge);
      const tip = P(at(1.12));
      ctx.beginPath(); ctx.moveTo(tip.x, tip.y);
      ctx.lineTo(tip.x - ux * 7 - uy * 3.5, tip.y - uy * 7 + ux * 3.5); ctx.lineTo(tip.x - ux * 7 + uy * 3.5, tip.y - uy * 7 - ux * 3.5);
      ctx.closePath(); ctx.fillStyle = edge; ctx.fill();
      ctx.font = nf;
      const tw = ctx.measureText(a.name).width, th = 16;
      let rx = tip.x - O.x, ry = tip.y - O.y; const rl = Math.hypot(rx, ry) || 1; rx /= rl; ry /= rl;
      let b: Box = { x: 0, y: 0, w: 0, h: 0 }, cx = 0, cy = 0;
      for (let d = 14; d <= 74; d += 10) {
        // the label's centre sits on the ray, offset by half its size along that ray
        cx = tip.x + rx * (d + Math.abs(rx) * tw / 2); cy = tip.y + ry * (d + Math.abs(ry) * th / 2);
        cx = Math.max(4 + tw / 2, Math.min(W - 4 - tw / 2, cx)); cy = Math.max(4 + th / 2, Math.min(H - 4 - th / 2, cy));
        b = { x: cx - tw / 2 - 4, y: cy - th / 2 - 2, w: tw + 8, h: th + 4 };
        if (!hit(b, placed)) break;
      }
      text(a.name, cx, cy, nf, rgba(ink, 0.88), 'center', true);
      placed.push(b);
    });

    // The front as a surface, in the accent.
    tris.map((t) => { const q = t.map(P); return { q, z: (q[0].z + q[1].z + q[2].z) / 3 }; })
      .sort((a, b) => a.z - b.z)
      .forEach(({ q }) => {
        ctx.beginPath(); ctx.moveTo(q[0].x, q[0].y); ctx.lineTo(q[1].x, q[1].y); ctx.lineTo(q[2].x, q[2].y); ctx.closePath();
        ctx.fillStyle = rgba(acc, C.dark ? 0.13 : 0.1); ctx.fill();
        ctx.strokeStyle = rgba(acc, C.dark ? 0.6 : 0.7); ctx.lineWidth = 1; ctx.stroke();
      });

    // The ideal corner.
    const I = P([1, 1, 1]), spike = small ? 20 : 26;
    if (C.dark) {
      const g = ctx.createRadialGradient(I.x, I.y, 0, I.x, I.y, 56);
      g.addColorStop(0, rgba(ink, 0.45)); g.addColorStop(0.15, rgba(ink, 0.2)); g.addColorStop(1, rgba(ink, 0));
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(I.x, I.y, 56, 0, 6.2832); ctx.fill();
    }
    [[1, 0, spike], [0, 1, spike], [0.7071, 0.7071, spike * 0.45], [0.7071, -0.7071, spike * 0.45]].forEach((d) => {
      const sg = ctx.createLinearGradient(I.x - d[0] * d[2], I.y - d[1] * d[2], I.x + d[0] * d[2], I.y + d[1] * d[2]);
      sg.addColorStop(0, rgba(ink, 0)); sg.addColorStop(0.5, rgba(ink, 0.85)); sg.addColorStop(1, rgba(ink, 0));
      ctx.strokeStyle = sg; ctx.lineWidth = d[2] === spike ? 1.3 : 1;
      ctx.beginPath(); ctx.moveTo(I.x - d[0] * d[2], I.y - d[1] * d[2]); ctx.lineTo(I.x + d[0] * d[2], I.y + d[1] * d[2]); ctx.stroke();
    });
    ctx.beginPath(); ctx.arc(I.x, I.y, 5, 0, 6.2832); ctx.fillStyle = rgba(ink, 1); ctx.fill();
    ctx.beginPath(); ctx.arc(I.x, I.y, 10, 0, 6.2832); ctx.strokeStyle = rgba(ink, 0.45); ctx.lineWidth = 1; ctx.stroke();
    const ifont = `500 ${small ? 11.5 : 12.5}px ${SANS}`; ctx.font = ifont;
    const iw = ctx.measureText('Ideal').width; let ix = I.x + 13; if (ix + iw > W - 4) ix = I.x - 13 - iw;
    text('Ideal', ix, I.y - 15, ifont, rgba(ink, 0.95), 'left', true);
    placed.push({ x: Math.min(ix, I.x - spike) - 2, y: I.y - spike, w: Math.max(ix + iw, I.x + spike) - Math.min(ix, I.x - spike) + 4, h: spike * 2 });

    // Stars, far to near.
    proj = stars.map((s) => { const q = P(s.p); return { s, x: q.x, y: q.y, z: q.z, f: q.f }; }).sort((a, b) => a.z - b.z);
    const depth = (z: number) => 0.55 + 0.45 * (z + 1.8) / 3.6;
    const on = focus || (hover && hover.s.r.method);
    proj.forEach((q) => {
      const r = q.s.r, a = depth(q.z), k = q.f;
      if (!ranked(r)) {
        ctx.beginPath(); ctx.arc(q.x, q.y, 4 * k, 0, 6.2832); ctx.strokeStyle = rgba(ink, 0.7 * a); ctx.lineWidth = 1.1; ctx.stroke();
      } else if (r.layer === 1) {
        if (C.dark) {
          const g = ctx.createRadialGradient(q.x, q.y, 0, q.x, q.y, 18 * k);
          g.addColorStop(0, rgba(acc, 0.45 * a)); g.addColorStop(1, rgba(acc, 0));
          ctx.fillStyle = g; ctx.beginPath(); ctx.arc(q.x, q.y, 18 * k, 0, 6.2832); ctx.fill();
        }
        ctx.beginPath(); ctx.arc(q.x, q.y, 4.2 * k, 0, 6.2832); ctx.fillStyle = rgba(acc, Math.min(1, a + 0.15)); ctx.fill();
        ctx.beginPath(); ctx.arc(q.x, q.y, 8 * k, 0, 6.2832); ctx.strokeStyle = rgba(acc, 0.6 * a); ctx.lineWidth = 1; ctx.stroke();
      } else {
        ctx.beginPath(); ctx.arc(q.x, q.y, 2.6 * k * (0.7 + 0.3 * a), 0, 6.2832); ctx.fillStyle = rgba(ink, 0.6 * a); ctx.fill();
      }
      if (r.method === on) { ctx.beginPath(); ctx.arc(q.x, q.y, 12 * k, 0, 6.2832); ctx.strokeStyle = rgba(ink, 0.9); ctx.lineWidth = 1.2; ctx.stroke(); }
      q.a = a;
    });

    // Names: the front always, any other star while focused.
    const font = `500 ${small ? 11 : 12}px ${SANS}`;
    proj.slice().sort((a, b) => (b.s.r.method === on ? 1 : 0) - (a.s.r.method === on ? 1 : 0) || (a.s.r.layer === 1 ? -1 : 1))
      .forEach((q) => {
        const r = q.s.r, isOn = r.method === on;
        if (!isOn && r.layer !== 1) return;
        ctx.font = font;
        const w = ctx.measureText(r.method).width, h = 14, g = (r.layer === 1 ? 12 : 9) * q.f, d = g * 0.75;
        const cands: [number, number, CanvasTextAlign][] = [[g, 0, 'left'], [-g, 0, 'right'], [d, -d - 4, 'left'], [d, d + 4, 'left'], [-d, -d - 4, 'right'], [-d, d + 4, 'right'], [0, -g - 4, 'center'], [0, g + 6, 'center']];
        const boxFor = (c: [number, number, CanvasTextAlign]) => {
          const x0 = c[2] === 'left' ? q.x + c[0] : c[2] === 'right' ? q.x + c[0] - w : q.x - w / 2;
          return { x: x0 - 2, y: q.y + c[1] - h / 2, w: w + 4, h };
        };
        const free = (b: Box) => b.x >= 4 && b.x + b.w <= W - 4 && b.y >= 4 && b.y + b.h <= H - 4 && !hit(b, placed) &&
          !proj.some((o) => o !== q && o.x > b.x - 5 && o.x < b.x + b.w + 5 && o.y > b.y - 5 && o.y < b.y + b.h + 5);
        let pick: { c: [number, number, CanvasTextAlign]; b: Box; lead?: boolean } | null = null;
        for (const c of cands) { const b = boxFor(c); if (free(b)) { pick = { c, b }; break; } }
        if (!pick) {
          for (let ring = 28; ring <= 84 && !pick; ring += 14) for (let s = 0; s < 8 && !pick; s++) {
            const t = s * Math.PI / 4 - Math.PI / 8, dx = Math.cos(t) * ring, dy = Math.sin(t) * ring * 0.7;
            const c: [number, number, CanvasTextAlign] = [dx, dy, dx >= 0 ? 'left' : 'right'], b = boxFor(c);
            if (free(b)) pick = { c, b, lead: true };
          }
          if (!pick) { const c: [number, number, CanvasTextAlign] = [g, 0, 'left']; pick = { c, b: boxFor(c) }; }
        }
        placed.push(pick.b);
        if (pick.lead) {
          const ex = pick.c[2] === 'left' ? pick.b.x : pick.b.x + pick.b.w, ang = Math.atan2(pick.c[1], pick.c[0]), r0 = 9 * q.f;
          ctx.beginPath(); ctx.moveTo(q.x + Math.cos(ang) * r0, q.y + Math.sin(ang) * r0); ctx.lineTo(ex, q.y + pick.c[1]);
          ctx.strokeStyle = rgba(C.graphite, 0.8); ctx.lineWidth = 1; ctx.stroke();
        }
        text(r.method, pick.c[2] === 'center' ? q.x : q.x + pick.c[0], q.y + pick.c[1], font, isOn ? rgba(ink, 1) : rgba(ink, 0.85 * (q.a || 1)), pick.c[2], true);
      });
  }

  function loop() {
    if (raf) return;
    raf = requestAnimationFrame(function tick(now) {
      raf = 0;
      if (!visible || document.hidden) return;
      if (!touched && !drag && !still) cam.yaw = 0.52 + Math.sin((now - t0) / 9000) * 0.32;
      draw();
      if (!touched && !still) raf = requestAnimationFrame(tick);
    });
  }

  const nearest = (x: number, y: number) => {
    let best: Proj | null = null, bd = 20;
    proj.forEach((q) => { const d = Math.hypot(q.x - x, q.y - y); if (d < bd) { bd = d; best = q; } });
    return best as Proj | null;
  };
  canvas.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, yaw: cam.yaw, pitch: cam.pitch, moved: false, id: e.pointerId }; });
  canvas.addEventListener('pointermove', (e) => {
    const b = canvas.getBoundingClientRect();
    if (drag && drag.id === e.pointerId) {
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (!drag.moved && Math.hypot(dx, dy) > 4) {
        if (e.pointerType !== 'mouse' && Math.abs(dy) > Math.abs(dx)) { drag = null; return; } // let touch scroll the page
        drag.moved = true; touched = true;
        try { canvas.setPointerCapture(e.pointerId); } catch {}
        canvas.classList.add('is-dragging');
      }
      if (drag.moved) {
        cam.yaw = Math.max(-1.8, Math.min(1.8, drag.yaw + dx * 0.008));
        if (e.pointerType === 'mouse') cam.pitch = Math.max(-Math.PI / 2, Math.min(1.25, drag.pitch + dy * 0.008));
        draw(); return;
      }
    }
    const q = nearest(e.clientX - b.left, e.clientY - b.top);
    if ((q && q.s) !== (hover && hover.s)) { hover = q; canvas.classList.toggle('is-hover', !!q); if (!raf) draw(); }
  });
  const end = (e: PointerEvent) => {
    if (!drag || drag.id !== e.pointerId) return;
    const moved = drag.moved; drag = null; canvas.classList.remove('is-dragging');
    if (!moved && e.type === 'pointerup') {
      const b = canvas.getBoundingClientRect(), q = nearest(e.clientX - b.left, e.clientY - b.top);
      focus = q && focus !== q.s.r.method ? q.s.r.method : null;
      draw();
    }
  };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
  canvas.addEventListener('pointerleave', () => { if (hover) { hover = null; canvas.classList.remove('is-hover'); if (!raf) draw(); } });

  readColors(); size();
  new ResizeObserver(() => { size(); draw(); }).observe(canvas);
  document.addEventListener('themechange', () => requestAnimationFrame(() => { readColors(); draw(); }));
  watchVisible(canvas, (on) => { visible = on; if (on) loop(); }, '80px');
  if (document.fonts?.ready) document.fonts.ready.then(draw);

  build(FALLBACK.map((r) => ({ ...r })));
  fetch(DATA, { cache: 'no-cache' })
    .then((r) => { if (!r.ok) throw new Error(String(r.status)); return r.text(); })
    .then((t) => {
      const rows = t.split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l) as Row);
      if (rows.length) build(rows);
    })
    .catch(() => {});
}
