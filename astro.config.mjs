import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://cxh42.github.io',
  devToolbar: { enabled: false },
  // Keep inter-element whitespace: the copy relies on real spaces between inline elements.
  compressHTML: false,
  build: { inlineStylesheets: 'auto' },
});
