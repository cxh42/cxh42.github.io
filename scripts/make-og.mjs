// Renders the link-preview card (public/og.png, 1200×630) and the raster icons (public/apple-touch-icon.png,
// public/favicon.ico) with headless Chrome. The card is the hero in miniature: the name on its own ground,
// 陈星昊 on the surname's baseline, the particle portrait behind it, and the lede under a hairline.
//
//   npm i --no-save playwright-core && node scripts/make-og.mjs
//
// Needs Google Chrome at /usr/bin/google-chrome (or CHROME=/path/to/chrome).
import { readFile, writeFile } from 'node:fs/promises';
import { chromium } from 'playwright-core';
import { person } from '../src/data/site.ts';

const root = new URL('../', import.meta.url);
const read = (p) => readFile(new URL(p, root));
const dataUri = async (p, type) => `data:${type};base64,${(await read(p)).toString('base64')}`;

const sans = await dataUri('node_modules/@fontsource-variable/geologica/files/geologica-latin-wght-normal.woff2', 'font/woff2');
const kai = await dataUri('public/fonts/cjk/MaShanZheng.woff2', 'font/woff2');
const raw = await read('public/data/silhouette.bin');
const pts = Array.from(new Int16Array(raw.buffer, raw.byteOffset, raw.length / 2), (v) => v / 20000);

const card = /* html */ `<!doctype html>
<meta charset="utf-8">
<style>
  @font-face { font-family: G; src: url(${sans}) format('woff2'); font-weight: 100 900; }
  @font-face { font-family: K; src: url(${kai}) format('woff2'); }
  * { margin: 0; box-sizing: border-box; }
  body {
    width: 1200px; height: 630px; overflow: hidden; position: relative;
    background: radial-gradient(120% 95% at 68% 28%, #ffffff 0%, #f0f0ed 62%, #e9e9e5 100%);
    font-family: G, system-ui, sans-serif; color: #101113;
  }
  canvas { position: absolute; inset: 0; }
  .url { position: absolute; left: 64px; top: 52px; font-size: 21px; letter-spacing: 0.01em; color: #63676e; }
  .name {
    position: absolute; left: 58px; bottom: 186px;
    font-size: 116px; font-weight: 300; line-height: 0.9; letter-spacing: -0.04em; white-space: nowrap;
  }
  .cjk { font-family: K, serif; font-size: 0.25em; letter-spacing: 0.08em; color: #3a3c40; margin-left: 0.22em; }
  .foot { position: absolute; left: 64px; right: 64px; bottom: 56px; border-top: 1px solid rgb(16 17 19 / 0.18); padding-top: 22px; }
  .role { font-size: 31px; font-weight: 380; letter-spacing: -0.015em; line-height: 1.2; }
  .next { margin-top: 10px; font-size: 22px; color: #3a3c40; }
</style>
<canvas width="1200" height="630"></canvas>
<p class="url">cxh42.github.io</p>
<h1 class="name">${person.given} ${person.family}<span class="cjk">${person.cjk}</span></h1>
<div class="foot">
  <p class="role">${person.role}</p>
  <p class="next">Incoming Ph.D. student in Computer Science · Texas A&amp;M University · TACO Group</p>
</div>
<script>
  // The portrait, settled (t = 0): the bust stands on the hairline, right of centre.
  const p = ${JSON.stringify(pts)};
  const c = document.querySelector('canvas').getContext('2d');
  const floor = document.querySelector('.foot').getBoundingClientRect().top - 1;
  const scale = (floor - 64) / 0.9241, cx = 985;
  c.fillStyle = 'rgba(16, 17, 19, 0.44)';
  for (let i = 0; i < p.length; i += 2) {
    const x = cx + p[i] * scale, y = floor - (p[i + 1] + 0.4766) * scale;
    if (y > floor - 2) continue;
    const r = 0.8 + 0.5 * ((i * 0.309) % 1);
    c.beginPath(); c.arc(x, y, r, 0, Math.PI * 2); c.fill();
  }
</script>`;

const browser = await chromium.launch({ executablePath: process.env.CHROME || '/usr/bin/google-chrome' });
const page = await browser.newPage({ viewport: { width: 1200, height: 630 } });
await page.setContent(card);
await page.evaluate(() => document.fonts.ready);
await writeFile(new URL('public/og.png', root), await page.screenshot({ type: 'png' }));

// Icons: the favicon's grain-to-block mark on the plate, for home screens and for clients that want a raster.
const svg = (await read('public/favicon.svg')).toString();
const icon = async (size, pad) => {
  await page.setViewportSize({ width: size, height: size });
  await page.setContent(`<style>*{margin:0}body{width:${size}px;height:${size}px;background:#f7f7f5;display:grid;place-items:center}svg{width:${size - 2 * pad}px;height:${size - 2 * pad}px}</style>${svg}`);
  return page.screenshot({ type: 'png' });
};
await writeFile(new URL('public/apple-touch-icon.png', root), await icon(180, 26));
// favicon.ico holding one 48×48 PNG (the ICO container has allowed PNG payloads since Windows Vista).
const png = await icon(48, 4);
const head = Buffer.alloc(22);
head.writeUInt16LE(0, 0); head.writeUInt16LE(1, 2); head.writeUInt16LE(1, 4);
head.writeUInt8(48, 6); head.writeUInt8(48, 7); head.writeUInt8(0, 8); head.writeUInt8(0, 9);
head.writeUInt16LE(1, 10); head.writeUInt16LE(32, 12); head.writeUInt32LE(png.length, 14); head.writeUInt32LE(22, 18);
await writeFile(new URL('public/favicon.ico', root), Buffer.concat([head, png]));

await browser.close();
console.log('make-og: wrote public/og.png, public/apple-touch-icon.png, public/favicon.ico');
