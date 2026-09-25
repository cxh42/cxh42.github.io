// Pulls per-country visit counts from GoatCounter into src/data/visitors.json before each build.
// Needs GOATCOUNTER_TOKEN (a GitHub Actions secret). Without it, the committed file is left as is.
import { writeFile } from 'node:fs/promises';

const SITE = 'https://cxh42.goatcounter.com';
const token = process.env.GOATCOUNTER_TOKEN;
const out = new URL('../src/data/visitors.json', import.meta.url);

if (!token) {
  console.log('fetch-visitors: GOATCOUNTER_TOKEN not set, keeping existing data.');
  process.exit(0);
}

const get = async (path) => {
  const res = await fetch(`${SITE}/api/v0${path}`, {
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
  });
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
};

try {
  const start = '2026-01-01';
  const countries = [];
  let offset = 0;
  for (let page = 0; page < 10; page++) {
    const data = await get(`/stats/locations?start=${start}&limit=100&offset=${offset}`);
    for (const s of data.stats ?? []) {
      if (s.id && s.id.length === 2) countries.push({ code: s.id.toUpperCase(), name: s.name, count: s.count });
    }
    if (!data.more) break;
    offset += 100;
  }
  const total = (await get(`/stats/total?start=${start}`)).total ?? 0;
  const body = { updated: new Date().toISOString(), total, countries };
  await writeFile(out, JSON.stringify(body, null, 2) + '\n');
  console.log(`fetch-visitors: ${total} visits from ${countries.length} countries.`);
} catch (err) {
  // Stats are a nicety: never fail the deploy over them.
  console.warn('fetch-visitors: keeping existing data:', err.message);
}
