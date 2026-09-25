---
version: 1
slug: "src-pages-index-astro"
primary_target: "src/pages/index.astro"
related_targets: []
---

# Home — single-page academic homepage

**Scope:** `src/pages/index.astro`, the only route. One long scroll: Hero → Research → Publication → Education → News → Visitors → Contact.
**Mode:** Experience. The work is who he is and what he builds, and it leads from the first viewport.
**Audience and job:** Faculty, peers and NeurIPS readers look him up, usually on a laptop or phone, for about a minute. They should leave knowing the name, "generative AI, video generation", ViTeX-Bench, and that he's joining TAMU for a PhD, and be able to email him.
**Constraints:** English only. Light and dark themes are both first-class and follow the system setting by default. No neon or cyber look, and no game-like or show-off motion (these are user vetoes). Reduced motion must be a complete, still page. The PRODUCT.md exclusion list binds.

## Direction contract

THESIS: The page is one diffusion sampling run. Each moment arrives from Gaussian noise to signal, coarse to fine, and each carries its own live t. Nothing grains at rest. The page closes on the clean sample x₀. It rejects the academic template (photo sidebar and static lists) and the creative-developer particle-blob hero.

OWN-WORLD: The light ground is a clean plate #F7F7F5 and the dark ground is a clean void #09090B. Neither carries a standing grain field (revised 2026-09-26 after the user found the grained grounds "dirty"). Ink is #101113 on light and #E9E8E4 on dark. Graphite is #63676E, and the plate tint #EEEEEB frames each project and paper. One cobalt accent (#2F4BD8, #8B98FF on dark) appears only on live t readouts and visitor marks. Grain exists only while something is being sampled: the name, the particle portrait, the ViTeX switch, the campus dissolves and the globe. Type is Geologica for Latin, KaTeX Computer Modern for notation, and six Chinese hands for 陈星昊 (楷书 Ma Shan Zheng by default, then 行书, 宋体, 草书, 楷体 and 行楷). Structure comes from hairline instruments and tabular numerals. No cards, shadows or pills; the plate tint is the only container.

STORY: The visitor watches the name and a particle silhouette of him sample out of noise together, then sees 陈星昊 written as a vertical inscription beside the silhouette, typed and retyped in a different Chinese hand each cycle. The role reads "generative AI, focused on video generation". Scrolling sends the silhouette back into noise, drifting upward and fading. Research lists four areas (video generation, VLMs, 3DGS, VR) and one framed project, the UW campus in VR. Selected Publications frames each paper as its own tinted unit and opens on the real First → Last edit. Campuses dissolve into one another in a dark band with no empty frame between them. Visitor points diffuse onto a globe, and the page closes on email and a colophon.

FIRST VIEWPORT: Clean ground. "Xinghao Chen" is set in Geologica Light at about 14vw, lower-left, breaking onto two lines on phones. A particle silhouette (about 94vh tall, centred near 65% of the width) stands behind the name's right half; on phones it is smaller and sits in the upper area. 陈星昊 is a vertical inscription beside the silhouette's head (about 47% from the left on desktop, the upper-right corner on phones), with a horizontal caret and a small graphite script label in the next column. Beneath the name is the role line and the incoming TAMU/TACO PhD line. Bottom-right is the KaTeX x_t equation with a t and ᾱ_t readout that is cobalt only while sampling. The right edge carries the schedule rail, with every section named, the current one in ink with a longer tick and the cobalt marker. The top bar has an Email link as the primary action and a Light/Dark switch. The small wordmark joins it only once the display name has left the viewport, because in the first viewport the name itself is the wordmark.

FORM: Denoising Trajectory, #1 of the ranked list (Impeccable's pick; the roll assigned #6, contact sheet), seed key 49fdd242. The signature is that every state change is a short noise ↔ signal trajectory on the shared 24 fps clock. That covers the name and the portrait sampled in together; the portrait diffusing out on scroll; the ViTeX scene switch; the campus dissolves (a coarse-to-fine threshold with a small noise bump, never through an empty frame); the globe points; and the theme toggle (a grain-threshold view transition). The inscription's typing, caret blink and hand changes run on the same clock. Nothing else moves: no hover lifts, parallax or cursor followers.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

## Memorable moment
The name condensing out of living grain, coarse shape first and then crisp edges, with the equation's t ticking to 0 beside it.

## Unresolved
- The ViTeX paper link waits for the arXiv version, so no Paper button is shown until it exists.
- The visitor globe shows the empty state until GoatCounter data exists and the Actions secret is set.
