import { execSync } from 'node:child_process';

// When the page's content last changed: the date of the commit being built. The hourly rebuild only
// refreshes the visitor counts, so the build time would overstate it.
export const lastModified = (() => {
  try {
    return execSync('git log -1 --format=%cI', { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
  } catch {
    return new Date().toISOString();
  }
})();
