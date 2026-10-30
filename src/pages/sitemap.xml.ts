import type { APIRoute } from 'astro';
import { lastModified } from '../data/build';

// A single-page site: one URL, dated by the last commit rather than the hourly rebuild.
export const GET: APIRoute = ({ site }) =>
  new Response(
    `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>${new URL('/', site)}</loc>
    <lastmod>${lastModified}</lastmod>
  </url>
</urlset>
`,
    { headers: { 'Content-Type': 'application/xml; charset=utf-8' } },
  );
