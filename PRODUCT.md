# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Astro with static output, deployed to GitHub Pages (`cxh42.github.io`) through GitHub Actions. The user chose this over plain HTML/CSS/JS and a Next.js static export. The deciding reasons were content collections, so adding a publication or news item is a data edit, and built-in image optimization for large background photos.

## Users

This is the personal academic homepage of **Xinghao Chen**. It's a from-scratch rebuild that replaces an older Jekyll academic template, which the user rejected as generic ("like an NPC").

Visitors are the people who look up an early-career researcher. (This is inferred from the brief's framing as a personal academic homepage; the user didn't list audiences.) That includes faculty, fellow researchers and collaborators, conference attendees and reviewers arriving from the ViTeX-Bench paper, and anyone checking who he is and what he works on. They usually spend a short visit on it, on desktop or on a phone.

## Product Purpose

The site establishes who Xinghao Chen is: a **generative AI** researcher whose current focus is **video generation models**, first author of ViTeX-Bench (NeurIPS 2026), and an incoming PhD student at Texas A&M. It should present that identity with more craft and presence than a standard template. Success means a visitor remembers the name, understands the research focus, finds the publication, and can reach him by email.

It should include what a proper academic homepage has (identity, research focus, publications, education, news, contact) and nothing the user excluded (see Capabilities and Constraints).

## Positioning

The site belongs to a researcher who works on moving images (video generation and video scene-text editing), and it's meant to show that fluency itself instead of reading like a filled-in template. It is not a job-seeking page.

## Operating Context

- It's a personal site on GitHub Pages at `https://cxh42.github.io`, repo `cxh42/cxh42.github.io`, main branch `master`.
- The content is in English (inferred from the previous site). The Chinese name is shown as a secondary element.
- The site will grow slowly: more publications and news items over time, starting with exactly one paper.
- Visitor analytics use GoatCounter (site code `cxh42`, dashboard `https://cxh42.goatcounter.com`). Tracking is cookie-free.

## Capabilities and Constraints

**Identity**
- Name: **Xinghao Chen**. This is the most prominent element on the site.
- Chinese name: **陈星昊**, shown secondary.
- Research focus: **generative AI**, currently **video generation models** (updated 2026-09-26; the user asked not to be framed only as a video-generation researcher).
- Earlier research: **vision-language models (VLM)** and **3D Gaussian Splatting (3DGS)**.
- VR project (2025, UW course "Developing Immersive Experiences for AR/VR"): photographed UW landmark buildings and sculptures with a phone, reconstructed them as 3D Gaussian splats, and built a virtual campus tour for **Meta Quest**. The 3DGS reconstruction detail comes from the old site's news item; the user described the phone capture and the Quest tour.
- Status: will join **Texas A&M University** as a PhD student in **January 2027**, in the **TACO group** (`https://taco-group.github.io/`), advised by **Dr. Zhengzhong Tu** (`https://vztu.github.io/`).

**Publication (the only one to show for now)**
- Title: *ViTeX-Bench: Benchmarking High Fidelity Video Scene Text Editing*
- Authors, in order: **Xinghao Chen**, Xiangbo Gao, Jiongze Yu, Yuheng Wu, Zhengzhong Tu. Xinghao is first author, with no equal-contribution marking.
- Venue: **NeurIPS 2026, Evaluations & Datasets Track, accepted on 2026-09-24.** The user states the acceptance. The public project page still says "under review" and lists anonymous authors because it hasn't been deanonymized yet; the user's statement takes precedence.
- Links:
  - Project page: `https://vitex-bench.github.io/`
  - Dataset: `https://huggingface.co/datasets/ViTeX-Bench/ViTeX-Dataset`
  - Benchmark code: `https://huggingface.co/ViTeX-Bench/ViTeX-Bench`
  - Model and inference code: `https://huggingface.co/ViTeX-Bench/ViTeX-Edit-14B`
  - Leaderboard: `https://huggingface.co/spaces/ViTeX-Bench/ViTeX-Bench-Leaderboard`
- Facts from the project page: ViTeX-Dataset has 387 real-world 720p videos with masks and instructions. ViTeX-Bench is an evaluation protocol with 13 metrics covering text correctness, visual quality and edit locality. ViTeX-Edit-14B is the reference model, at 0.688 CharAcc.

**Education**
- Texas A&M University: Ph.D. in Computer Science (CSCE), starting Spring 2027 (January 2027), TACO group, advisor Zhengzhong Tu.
- University of Washington: M.S., Electrical & Computer Engineering (MSEE), September 2024 to December 2025.
- Henan University: B.E., Automation, September 2020 to June 2024.

**Contact and links (only these)**
- Email: `cxh4242@gmail.com`
- OpenReview: `https://openreview.net/profile?id=~Xinghao_Chen4`
- GitHub: `https://github.com/cxh42`
- LinkedIn: `https://www.linkedin.com/in/cxh42`

**Required features**
- Light/dark theme toggle.
- Global visitor map. This is a custom globe or map fed by GoatCounter location stats. A scheduled GitHub Action pulls the per-country data from the GoatCounter API and commits it as static data. The API token must be stored only as a GitHub Actions secret and never committed to the public repo or written into any file. The site had **zero recorded visits** at setup (2026-09-25), so the map needs a clear empty and low-data state.
- Good browsing experience on mobile.
- Ambitious, modern motion and interaction, per the user's explicit request.

**Excluded content (binding)**
- No personal photo for now. (The particle silhouette is allowed; see Evidence on Hand.)
- No CV PDF, and none of the old CV's past experiences.
- No Portfolio section or items.
- 3DGS may now appear, but only as earlier research and the VR project (the user reversed the earlier exclusion on 2026-09-26).
- No "looking for PhD positions" message.
- No Research Assistant role at TACO (the user chose not to show it).
- No Kaggle, Google Scholar or WeChat.
- No other emails (old `xhc42@outlook.com` and `cxh42@uw.edu` are retired from the site).
- No old news items (UW courses, the AR/VR course, the APL coastal-erosion project).

**Open decisions (don't invent these)**
- ViTeX paper link: the user will publish an arXiv version later. Leave the paper link out for now; don't show a dead or placeholder link.
- News items: the confirmed items are the ViTeX acceptance (2026-09-24), the UW graduation (12.2025) and the upcoming TAMU PhD start (Spring 2027). Add nothing else.
- Research statement wording beyond the facts above hasn't been provided; don't add claims about specific VLM work.

## Brand Commitments

These visual constraints came from the user and are binding. They're recorded without expansion.
- The overall look should be clean, minimal and deliberately designed, and very elegant. It must not look like a generic academic template.
- It should also have striking, state-of-the-art dynamic effects.
- The name is in the most prominent position; 陈星昊 is secondary.
- Education entries may have dark, high-resolution, representative imagery of each school as backgrounds. The user offered this as a suggestion, not a hard requirement ("不一定局限于此").
- Light/dark mode toggle.
- It must read well on phones.
- Added 2026-09-26:
  - The hero background must feel clean in both themes, with no "dirty" grain field.
  - 陈星昊 is set in a calligraphic but readable hand. A blinking cursor types it, erases it and retypes it in a different Chinese typeface each cycle.
  - A particle silhouette of the user sits in the hero and changes as the page scrolls.
  - The section navigator shows every section at all times and highlights the current one.
  - Each publication reads as its own distinct unit, and the list is marked as a selection.
  - The ViTeX "First → Last" example comes first.
  - Campus hand-overs must be smooth, with no grey or white gap, and cheap to render.

## Evidence on Hand

- ViTeX-Bench project page assets (the user's own work): `https://vitex-bench.github.io/static/images/teaser.png`, `pipeline.png`, `arch.png`.
- School logos from the old site are in git history at `HEAD~1:images/uwlogo.png` and `HEAD~1:images/henulogo.png`. No TAMU logo is on hand.
- Campus photographs (in `src/assets/campus/`):
  - Texas A&M: a photo the user chose from Texas A&M's own site (stories.tamu.edu). All rights stay with the university; it's credited in the footer.
  - UW: a photo the user chose from a UW site (cdn.uconnectlabs.com). All rights stay with the university; it's credited in the footer.
  - Henan University: "河南大学礼堂2020" by ScareCriterion12, CC BY-SA 4.0 via Wikimedia Commons.
- Particle portrait: the hero's particle silhouette is sampled from the outline of the old site's headshot (`HEAD~1:images/bio-photo.jpg`). Only the outline and heavily blurred light/dark masses are used, so no facial features can be recovered, and the photo itself is never shipped. The user asked for "粒子代表我的人物剪影" on 2026-09-26.
- There are no personal photos, testimonials, talks, teaching entries, awards or citation counts. Don't fabricate any of these.

## Product Principles

1. **The name comes first.** A visitor should leave knowing "Xinghao Chen, video generation, ViTeX-Bench, TAMU PhD." Everything else supports that.
2. **Show a curated truth, not everything.** Show only confirmed, current facts. The exclusion list is part of the product, not an oversight to fill in later.
3. **One paper should look intentional, and a dozen should still fit.** The structure has to feel complete with a single publication and absorb more without a redesign.
4. **Motion shows fluency, not decoration.** The research is about moving images, so the site's dynamism should feel native to that. It must never block reading, and all content must stay accessible with reduced motion or on a low-power phone.
5. **Keep it honest and private.** Use real data only, including the visitor map's empty state. Keep secrets out of the repo.

## Accessibility & Inclusion

No product-specific requirement was stated beyond the explicit ones: a theme toggle that respects the system preference, and full usability on mobile. Heavy motion needs a reduced-motion path because the brief calls for ambitious animation.
