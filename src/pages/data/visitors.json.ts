import visitors from '../../data/visitors.json';

// The stats the page was built from, published so the next build can carry them forward
// when GoatCounter can't be reached (scripts/fetch-visitors.mjs).
export const GET = () => new Response(JSON.stringify(visitors));
