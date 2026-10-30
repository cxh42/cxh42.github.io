import type { APIRoute } from 'astro';

// Everything is open to crawlers; the sitemap points them at the one page that matters.
export const GET: APIRoute = ({ site }) =>
  new Response(`User-agent: *\nAllow: /\n\nSitemap: ${new URL('/sitemap.xml', site)}\n`, {
    headers: { 'Content-Type': 'text/plain; charset=utf-8' },
  });
