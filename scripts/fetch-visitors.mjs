// Pulls visit counts per country, and per region (state / province) inside each country, from GoatCounter
// into src/data/visitors.json before each build. GoatCounter has no city-level data; regions are recorded
// only for the countries listed under "collect regions" in the site settings (empty = all countries).
// Needs GOATCOUNTER_TOKEN (a GitHub Actions secret). Without it, the committed file is left as is.
import { writeFile } from 'node:fs/promises';

const SITE = 'https://cxh42.goatcounter.com';
const token = process.env.GOATCOUNTER_TOKEN;
const out = new URL('../src/data/visitors.json', import.meta.url);

if (!token) {
  console.log('fetch-visitors: GOATCOUNTER_TOKEN not set, keeping existing data.');
  process.exit(0);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const get = async (path, retry = 2) => {
  const res = await fetch(`${SITE}/api/v0${path}`, {
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
  });
  if (res.status === 429 && retry > 0) {
    await sleep(1000 * (Number(res.headers.get('X-Rate-Limit-Reset')) || 5));
    return get(path, retry - 1);
  }
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
};

// Regions of one country, e.g. US-TX "Texas". Visits with no known region are left out.
const regionsOf = async (code, start) => {
  try {
    const data = await get(`/stats/locations/${code}?start=${start}&limit=100`);
    return (data.stats ?? [])
      .filter((s) => typeof s.id === 'string' && /^[A-Za-z]{2}-[A-Za-z0-9]{1,4}$/.test(s.id) && s.count > 0)
      .map((s) => ({ code: s.id.toUpperCase(), name: s.name, count: s.count }));
  } catch (err) {
    console.warn(`fetch-visitors: no regions for ${code}:`, err.message);
    return [];
  }
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
  countries.sort((a, b) => b.count - a.count);
  // One call per country, gently, for the busiest 60.
  for (const c of countries.slice(0, 60)) {
    c.regions = await regionsOf(c.code, start);
    await sleep(350);
  }
  const total = (await get(`/stats/total?start=${start}`)).total ?? 0;
  const body = { updated: new Date().toISOString(), total, countries };
  await writeFile(out, JSON.stringify(body, null, 2) + '\n');
  const nr = countries.reduce((k, c) => k + (c.regions?.length ?? 0), 0);
  console.log(`fetch-visitors: ${total} visits from ${countries.length} countries, ${nr} regions.`);
} catch (err) {
  // Stats are a nicety: never fail the deploy over them.
  console.warn('fetch-visitors: keeping existing data:', err.message);
}
