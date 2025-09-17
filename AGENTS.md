# Repository Guidelines

## Project Structure & Module Organization
The site arranges content through Jekyll collections: `_pages/` for static pages, `_posts/` for dated updates, plus `_portfolio/`, `_publications/`, `_talks/`, and `_teaching/` for resume-style sections. Shared Liquid snippets sit in `_includes/`, while `_layouts/` defines page shells and `_sass/` provides theme partials compiled into `assets/css/`. Client assets live in `assets/js/` and `images/`, with `main.min.js` generated from `_main.js`. Use `files/` for downloadable artifacts, `_data/` for structured YAML, and `markdown_generator/` or `scripts/` when bulk-importing content.

## Build, Test, and Development Commands
Run `bundle install` and `npm install` once per environment. Use `bundle exec jekyll serve -l -H localhost` for live previews, or `docker compose up` when you prefer a disposable container. Validate before committing with `bundle exec jekyll build --strict_front_matter` and `bundle exec jekyll doctor`. Regenerate JavaScript assets via `npm run build:js`, and use `npm run watch:js` while iterating.

## Coding Style & Naming Conventions
Front matter keys stay in `snake_case` and include `title`, `layout`, `permalink`, and `date` when applicable. Name markdown files in lowercase kebab-case, e.g., `_posts/2025-09-16-new-paper.md` or `_pages/about.md`. Follow two-space indentation for Liquid, HTML, and YAML; SCSS partials should extend shared variables instead of copying declarations. JavaScript follows the existing ES6 pattern in `_main.js`: arrow functions, semicolons, double quotes, and guarded access to globals.

## Testing Guidelines
Always run `bundle exec jekyll build --strict_front_matter` to surface missing front matter or assets, then spot-check the affected pages in the local server. When JS changes, rebuild with `npm run build:js` and verify dark/light theme toggles and navigation behaviors. Large TSV imports should be regenerated via `markdown_generator/` notebooks and reviewed for formatting drift.

## Commit & Pull Request Guidelines
Write short, imperative commit messages such as "Add keynote recording" or "Tweak hero spacing". PR descriptions should explain the user-facing impact, call out modified directories (e.g., `_sass/_theme.scss`), and link related issues. Attach before/after screenshots for visual edits and list the commands you executed locally.

## Content Authoring Tips
Prefer Markdown with Liquid includes over raw HTML, and place downloadable assets in `files/` referenced with `/files/...` URLs. Reuse data from `_data/` for repeated lists like people or sponsors to simplify updates. Always preview in the local server before merging to catch Liquid errors and layout regressions.
