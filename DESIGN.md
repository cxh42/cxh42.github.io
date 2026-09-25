---
name: Xinghao Chen
description: A personal academic homepage sampled like a diffusion run, from Gaussian grain to a clean x₀.
colors:
  plate: "#f7f7f5"
  plate-2: "#eeeeeb"
  ink: "#101113"
  ink-2: "#3a3c40"
  graphite: "#63676e"
  rule: "rgb(16 17 19 / 0.14)"
  rule-strong: "rgb(16 17 19 / 0.32)"
  accent: "#2f4bd8"
  select: "rgb(47 75 216 / 0.22)"
  void: "#09090b"
  void-2: "#131316"
  dark-ink: "#e9e8e4"
  dark-ink-2: "#b9b8b2"
  dark-graphite: "#8c9097"
  dark-rule: "rgb(231 230 225 / 0.13)"
  dark-rule-strong: "rgb(231 230 225 / 0.3)"
  dark-accent: "#8b98ff"
  dark-select: "rgb(139 152 255 / 0.28)"
  screen: "#0b0c0e"
  screen-dark: "#060708"
  on-screen: "#e7e6e1"
  on-screen-2: "#b5b4ae"
  on-screen-accent: "#9aa5ff"
  on-screen-rule: "rgb(231 230 225 / 0.18)"
  on-screen-rule-strong: "rgb(231 230 225 / 0.28)"
typography:
  display:
    fontFamily: "Geologica Variable, Geologica, system-ui, sans-serif"
    fontSize: "clamp(3.5rem, (min(100vw, 1360px + rail + 2 * gutter) - 2 * gutter - rail) / 6.2, 15rem)"
    fontWeight: 300
    lineHeight: 0.9
    letterSpacing: "-0.04em"
  display-mobile:
    fontFamily: "Geologica Variable, Geologica, system-ui, sans-serif"
    fontSize: "calc((100vw - 2 * gutter) / 3.75)"
    fontWeight: 300
    lineHeight: 0.92
    letterSpacing: "-0.04em"
  inscription:
    fontFamily: "CJK Kai, Kaiti SC, STKaiti, KaiTi, serif"
    fontSize: "clamp(2.6rem, 1.4rem + 3.2vw, 4.75rem)"
    lineHeight: 1.12
    letterSpacing: "0.06em"
  inscription-mobile:
    fontFamily: "CJK Kai, Kaiti SC, STKaiti, KaiTi, serif"
    fontSize: "clamp(2.4rem, 11vw, 3.4rem)"
    lineHeight: 1.12
    letterSpacing: "0.06em"
  inscription-label:
    fontFamily: "CJK Kai, Kaiti SC, STKaiti, KaiTi, serif"
    fontSize: "0.8125rem"
    lineHeight: 1.6
    letterSpacing: "0.3em"
  band-headline:
    fontFamily: "Geologica Variable, Geologica, system-ui, sans-serif"
    fontSize: "clamp(2.1rem, 1.3rem + 3.4vw, 4.75rem)"
    fontWeight: 300
    lineHeight: 1.02
    letterSpacing: "-0.035em"
  closing:
    fontFamily: "Geologica Variable, Geologica, system-ui, sans-serif"
    fontSize: "clamp(1.75rem, 1rem + 3.6vw, 4.25rem)"
    fontWeight: 320
    lineHeight: 1.05
    letterSpacing: "-0.035em"
  headline:
    fontFamily: "Geologica Variable, Geologica, system-ui, sans-serif"
    fontSize: "clamp(2rem, 2.9vw, 2.9rem)"
    fontWeight: 330
    lineHeight: 1.02
    letterSpacing: "-0.03em"
  statement:
    fontFamily: "Geologica Variable, Geologica, system-ui, sans-serif"
    fontSize: "clamp(2rem, 1.3rem + 2.8vw, 3.5rem)"
    fontWeight: 320
    lineHeight: 1.12
    letterSpacing: "-0.028em"
  title:
    fontFamily: "Geologica Variable, Geologica, system-ui, sans-serif"
    fontSize: "clamp(1.45rem, 1.1rem + 1.2vw, 1.875rem)"
    fontWeight: 380
    lineHeight: 1.25
    letterSpacing: "-0.015em"
  lead:
    fontFamily: "Geologica Variable, Geologica, system-ui, sans-serif"
    fontSize: "1.3125rem"
    fontWeight: 350
    lineHeight: 1.4
    letterSpacing: "-0.008em"
  body:
    fontFamily: "Geologica Variable, Geologica, system-ui, sans-serif"
    fontSize: "1.0625rem"
    fontWeight: 350
    lineHeight: 1.6
    letterSpacing: "normal"
    fontFeature: "\"tnum\" 1"
  small:
    fontFamily: "Geologica Variable, Geologica, system-ui, sans-serif"
    fontSize: "0.9375rem"
    fontWeight: 350
    lineHeight: 1.6
    letterSpacing: "0.005em"
    fontFeature: "\"tnum\" 1"
  label:
    fontFamily: "Geologica Variable, Geologica, system-ui, sans-serif"
    fontSize: "0.8125rem"
    fontWeight: 350
    lineHeight: 1.6
    letterSpacing: "0.01em"
    fontFeature: "\"tnum\" 1"
  rail-label:
    fontFamily: "Geologica Variable, Geologica, system-ui, sans-serif"
    fontSize: "0.8125rem"
    fontWeight: 350
    lineHeight: 1
    letterSpacing: "0.01em"
  readout:
    fontFamily: "KaTeX_Main, Times New Roman, serif"
    fontSize: "0.95rem"
    fontWeight: 400
    lineHeight: 1.6
  notation:
    fontFamily: "KaTeX_Main, Times New Roman, serif"
    fontSize: "1.08em"
    fontWeight: 400
    lineHeight: 1.2
  colophon-mark:
    fontFamily: "KaTeX_Main, Times New Roman, serif"
    fontSize: "1.6rem"
    fontWeight: 400
    lineHeight: 1
rounded:
  none: "0px"
  focus: "2px"
spacing:
  gutter: "clamp(20px, 5vw, 72px)"
  header: "64px"
  rail-space: "150px"
  container: "1360px"
  measure: "66ch"
  column-gap: "clamp(16px, 2vw, 32px)"
  row-gap: "32px"
  section-top: "clamp(88px, 13vh, 168px)"
  section-bottom: "clamp(56px, 8vh, 104px)"
  panel-pad: "clamp(16px, 2.6vw, 36px)"
  panel-gap: "clamp(16px, 2vw, 24px)"
  row-pad: "22px"
  area-pad: "16px"
  list-pad: "9px"
components:
  link-inline:
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
  link-arrow:
    textColor: "{colors.ink}"
    typography: "{typography.lead}"
  toggle-text:
    backgroundColor: "transparent"
    textColor: "{colors.graphite}"
    typography: "{typography.label}"
    padding: "6px 0"
  toggle-text-pressed:
    textColor: "{colors.ink}"
  scene-toggle:
    backgroundColor: "transparent"
    textColor: "{colors.graphite}"
    typography: "{typography.small}"
    padding: "6px 0"
  top-bar:
    backgroundColor: "{colors.plate}"
    textColor: "{colors.ink}"
    typography: "{typography.label}"
    height: "{spacing.header}"
  menu-sheet:
    backgroundColor: "{colors.plate}"
    textColor: "{colors.ink}"
    typography: "{typography.lead}"
    padding: "8px {spacing.gutter} 20px"
  schedule-rail:
    textColor: "{colors.graphite}"
    typography: "{typography.rail-label}"
    width: "120px"
  schedule-rail-current:
    textColor: "{colors.ink}"
  schedule-rail-marker:
    backgroundColor: "{colors.accent}"
    width: "3px"
    height: "16px"
  inscription:
    textColor: "{colors.ink}"
    typography: "{typography.inscription}"
  inscription-label:
    textColor: "{colors.graphite}"
    typography: "{typography.inscription-label}"
  plate-panel:
    backgroundColor: "{colors.plate-2}"
    rounded: "{rounded.none}"
    padding: "{spacing.panel-pad}"
  t-readout:
    textColor: "{colors.graphite}"
    typography: "{typography.readout}"
  t-readout-live:
    textColor: "{colors.accent}"
  ledger-row:
    textColor: "{colors.ink}"
    typography: "{typography.lead}"
    padding: "{spacing.row-pad} 0"
  area-row:
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    padding: "{spacing.area-pad} 0"
  screen-band:
    backgroundColor: "{colors.screen}"
    textColor: "{colors.on-screen}"
    typography: "{typography.band-headline}"
  colophon-mark:
    textColor: "{colors.ink}"
    typography: "{typography.colophon-mark}"
  skip-link:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.plate}"
    padding: "8px 14px"
---

# Design System: Xinghao Chen

## Overview

**Creative North Star: "The Denoising Trajectory"**

The page behaves like one diffusion sampling run. Every state that changes arrives as a short noise-to-signal trajectory: per-pixel Gaussian grain (PCG hash into Box-Muller, WebGL2) resolving coarse shape first and crisp edges last, on the cosine ᾱ schedule, while a KaTeX readout of the live timestep counts down to t = 0. The name and a particle portrait are sampled out of noise together in the first viewport; the portrait diffuses back into noise as the reader scrolls. ViTeX scene switches, the campus hand-overs, the visitor globe and the Light/Dark switch use the same grammar. When a trajectory ends, its grain is gone: the settled page is clean ground.

The material is an off-white plate in light and a near-black void in dark, printed in near-black or bone ink. Structure is carried by hairlines, tabular numerals and notation, with one container: a flat, square plate-2 tint panel that holds each publication and each project. Density is editorial and generous: long section air, a 12-column grid with margin headings at wide widths, and one full-bleed dark screen band where the campuses resolve. The name also stands in Chinese as a vertical inscription, typed and erased in six hands on the shared clock. The single cobalt accent is instrumentation: it lights when something is being sampled, and rests to graphite when t reaches 0.

Confirmed rejections: bordered, rounded or shadowed cards, pills, decorative shadows, eyebrows or kickers above headings, glow, gradients on the accent, neon or cyber styling, standing grain or texture at rest, and any hover motion beyond colour and underline changes.

**Key Characteristics:**
- Gaussian noise is the only transition vocabulary; it exists only while something is being sampled, and the ground is clean at rest.
- One shared 24 fps clock; every sampler, the inscription and every stepped CSS transition land on its frames.
- Hairline instruments (1px rules, 9px ticks, 1px underlines) and one flat plate-2 tint panel instead of boxes.
- Geologica at light weights with negative tracking; KaTeX Computer Modern for every number that means t; six calligraphic and print hands for 陈星昊.
- Light and dark are both first-class and follow the system until the visitor chooses.
- Reduced motion and no-WebGL are complete, still pages, never degraded ones.

## Colors

A near-monochrome instrument palette, off-white plate and cool ink, with one cobalt signal that exists only while something is live.

### Primary
- **Sampling Cobalt** (accent; dark-accent on void; on-screen-accent in the dark band): the live timestep. It colours a t or ᾱ_t readout only while t > 0, the schedule-rail marker (the reader's own position in the run), and visitor marks on the globe and the visit figure. At t = 0 readouts return to graphite. Functional affordances borrow it too: the focus ring, the text caret and the selection tint (select / dark-select). No glow, no gradient, no fills.

### Neutral
- **Plate** (plate): the light ground. Flat at rest; samplers draw over it only while they run.
- **Tint Plate** (plate-2): the one tonal step and the only container fill: the publication and project panels. Media inside a panel (the ViTeX stage) sits back on plate.
- **Ink** (ink): names, headings, primary text, pressed and current states, the inscription and its caret, the portrait's particles, the colophon x₀.
- **Soft Ink** (ink-2): body prose, lede, authors, the equation, area notes.
- **Graphite** (graphite): instrumentation at rest: meta lines, dates, counts, rail labels, idle readouts, unpressed toggles, the inscription's script label.
- **Hairline** (rule) and **Strong Hairline** (rule-strong): dividers and underline colour at rest (rule); rail ticks, theme separator, scrollbar thumb and resting link underline (rule-strong).
- **Void** set (void, void-2, dark-ink, dark-ink-2, dark-graphite, dark-rule, dark-rule-strong): the dark theme, same roles one-to-one; void-2 is the dark panel tint.
- **Screen** (screen; screen-dark in dark theme), **On-Screen** (on-screen, on-screen-2, on-screen-rule, on-screen-rule-strong): the always-dark Education band. It stays dark in both themes; any rail label, tick or marker that sits over the band repaints in on-screen values, element by element.

### Named Rules
**The Live-Only Accent Rule.** Cobalt means "t > 0 right now", "you are here" on the rail, or "a visitor was here". Anything at rest, including the colophon x₀, is ink or graphite.

**The Paired Theme Rule.** Every neutral has a void counterpart in the same role; never introduce a light-only or dark-only colour outside the screen band.

## Typography

**Display Font:** Geologica Variable (with Geologica, system-ui)
**Body Font:** Geologica Variable at weight 350
**Notation Font:** KaTeX Computer Modern (KaTeX_Main, Times New Roman) for equations, every numeric t readout and the colophon x₀
**CJK:** six self-hosted faces subset to 陈星昊 only (U+9648, U+661F, U+660A): CJK Kai (Ma Shan Zheng, 楷书), CJK Xing (Zhi Mang Xing, 行书), CJK Song (Noto Serif SC, 宋体), CJK Cao (Liu Jian Mao Cao, 草书), CJK WenKai (LXGW WenKai TC, 楷体), CJK XingKai (Long Cang, 行楷). --font-cjk defaults to CJK Kai, falling back to Kaiti SC, STKaiti, KaiTi, serif. Each is a single regular weight; font-display: block so a fallback hand never flashes.

**Character:** A clean, slightly technical grotesque kept thin and tightly tracked at scale, set against textbook Computer Modern so the math reads as math. The brush-written inscription is the page's one hand-made voice, a counterweight to the grotesque name.

### Hierarchy
- **Display** (300, container-width / 6.2 capped 3.5–15rem, line-height 0.9, -0.04em): the hero name only. On phones (≤760px) each word takes its own line at (100vw − 2 gutters) / 3.75. Never wraps within a word line.
- **Inscription** (CJK, clamp 2.6–4.75rem; 2.4–3.4rem at ≤760px; line-height 1.12, 0.06em): 陈星昊 set vertically (vertical-rl), holding a 3.55em column so erasing never shifts it.
- **Band Headline** (300, clamp 2.1–4.75rem, 1.02, -0.035em): degree titles inside the screen band.
- **Closing** (320, clamp 1.75–4.25rem, 1.05, -0.035em): the email address that ends the page, underlined at 1px with 0.16em offset.
- **Headline** (330, clamp 2–2.9rem, 1.02, -0.03em): section titles, set in the margin column at ≥1280px.
- **Statement** (320, step-3, 1.12, -0.028em, max 24ch): the research thesis; emphasis is weight 450, never italic.
- **Title** (380–400, step-2, 1.18–1.25, -0.015 to -0.018em): the role line, paper and project titles, school name, visitor lede.
- **Lead** (350, 1.3125rem, 1.4): news entries, menu links, profile links.
- **Body** (350, 1.0625rem, 1.6, tabular numerals): prose, capped at 66ch (measure).
- **Small** (350, 0.9375rem, 0.005em): secondary UI and list text: ViTeX scene buttons, area notes, country rows, the wordmark (there at 450, -0.01em).
- **Label** (350, 0.8125rem, 0.01–0.02em, graphite): meta, dates, venue, credits, bar controls, rail labels (current at 500), the CJK script label (0.3em). Sentence case, never uppercase.
- **Readout** (KaTeX_Main, 0.95rem): every t / ᾱ_t line; its numerals are Notation (1.08em of it).
- **Colophon Mark** (KaTeX_Main, 1.6rem, line-height 1, ink): the x₀ that signs the footer.

### Named Rules
**The Light Hand Rule.** Nothing on the page is heavier than 550 (author name and venue); headings sit between 300 and 400. Weight is not how hierarchy is made; size and tracking are.

**The Notation Rule.** Any number that is a timestep or schedule value is KaTeX_Main at 1.08em of a 0.95rem readout, with a 2.9ch minimum width, so it never reflows while it counts.

**The Three Glyph Rule.** CJK faces ship subset to the three glyphs of 陈星昊 and nothing else; a new CJK string means a new subset, never a full font.

## Layout

A centred container of 1360px plus a 150px rail allowance (rail-space, ≥1100px only), inset by a fluid gutter (clamp 20–72px). Sections breathe with clamp(88px, 13vh, 168px) above and clamp(56px, 8vh, 104px) below. Content sits on a 12-column grid (column gap clamp 16–32px, row gap 32px): section title in columns 1–3, body in 4–12 at ≥1280px; below that the title stacks above the body. Secondary splits inside the body are two-column (project head and text, publication text, research areas, visitors) and collapse at 900px (areas rows to one column at 520px).

The hero is a full-viewport field (min 100svh, 560px) with the name bottom-left and a hairline-topped foot: lede left, equation and readout right (stacked left-aligned on phones). The inscription hangs from below the bar at 42% of the width on desktop and against the right gutter on phones; the portrait is centred at 60% of the width on desktop (43% on phones, upper half). Panels stack with clamp(16px, 2vw, 24px) between them and pad clamp(16px, 2.6vw, 36px). Lists are ledgers: a top rule, then 1px bottom rules per row (22px padding for news, 16px for areas, 9px for country counts), dates in a fixed first column. The Education band is full-bleed; with WebGL it becomes a sticky 100svh screen inside a 340svh scroll track.

Breakpoints observed: 420, 520, 640, 760, 900, 1100 (rail appears, menu disappears), 1280 (margin headings).

## Elevation & Depth

Flat. Depth comes from sampling itself, the one tint step and the one dark screen band, not from lifting surfaces. Panels are tonal, never raised. The portrait canvas sits fixed behind the hero content and is clipped at the hero rule with a 56px soft edge, so nothing it draws reaches the foot or the sections below. The top bar gains a plate fill and a 1px hairline (`box-shadow: 0 1px 0 var(--rule)`) once scrolled; that is a rule drawn with box-shadow, not elevation.

### Shadow Vocabulary
- **Menu sheet** (`box-shadow: 0 1px 0 var(--rule), 0 18px 32px -24px rgb(0 0 0 / 0.35)`): the phone/tablet section menu only, because it overlays content.

### Named Rules
**The One Shadow Rule.** The menu sheet is the only surface allowed a cast shadow. Anything else that needs separation gets a hairline or the plate-2 tint.

**The Clean Ground Rule.** At rest the ground is flat plate or void. Grain and particles exist only while a trajectory runs; the hero canvas clears and stops once the name settles, and the portrait never draws below the hero rule.

## Shapes

Square everywhere: no radius on media, panels, rows, buttons or the menu. The only curve is the 2px radius of the focus outline (1.5px accent, 3px offset). Form is made of lines and one tint: 1px rules, rail ticks 9px at rest and 18px when current, 28px band ticks (2px when active), 1px underlines under pressed toggles, the 3×16px rail marker, the horizontal inscription caret (0.9em × 0.07em), and the square plate-2 panel. Portrait particles are soft round points (1.7px × dpr, ±25%). Icons are inline SVG line drawings at 1.2–1.5px stroke (arrows, menu bars).

## Components

### Links
Quiet and editorial. Inherit colour; 1px underline at 0.22em offset in rule-strong, turning to currentColor on hover over 160ms. Outbound profile and paper links carry a 0.72em inline SVG arrow. Bar links (Email) show no underline until hover.

### Text Toggles (theme switch, ViTeX scene list)
- **Shape:** bare text, no box, 6px vertical padding.
- **Rest:** graphite. **Hover:** ink (200ms colour). **Pressed:** ink with a 1px currentColor underline drawn under the label.
- Theme toggles are Label size; scene buttons are Small and read "source → target" with a 14×8 SVG arrow.

### Plate Panels
- **Corner Style:** square (0).
- **Background:** plate-2 (void-2 in dark). No border, no shadow.
- **Internal Padding:** clamp(16px, 2.6vw, 36px).
- **Use:** one per publication in Selected Publications (the ViTeX stage, scene row and paper text inside it) and one per project in Research (head and meta left, text right, tags under the text). It is the only container in the system; everything else is hairlines.

### Navigation
- **Top bar:** fixed, 64px, transparent over the hero, plate plus hairline when scrolled. Wordmark (Small, 450) is hidden until the hero name leaves the viewport. Right side: Email, Light | Dark (1px × 12px separator), Menu below 1100px.
- **Menu sheet:** plate, Lead-size links with 1px rule dividers, the only shadowed surface.
- **Schedule rail (≥1100px):** fixed right, vertically centred, 120px wide, 18px between rows. Every section is always labelled (Label size, graphite). The current section is ink at weight 500 and its tick grows from 9px to 18px in ink, done with transform: scaleX (0.5 at rest) from the right edge over 200ms. The cobalt 3×16px marker glides on the 1px line between ticks with section progress. Each label, tick and the marker switch to on-screen values individually whenever they sit over the Education band.

### Inscription (signature)
陈星昊 set vertical-rl in ink, followed by a horizontal ink caret and, in a second column, the Label-size graphite name of the current hand (楷书, 行书, 宋体, 草书, 楷体, 行楷). On the 24 fps clock, once the name has settled and all six faces are loaded: hold 84 frames (3.5s) with the caret blinking at 1Hz (12 frames on, 12 off), erase one character every 3 frames, a 14-frame gap in which the label steps out (250ms, steps(6)) and the hand changes, then type one character every 6 frames. It appears with the hero foot and pauses when off-screen. Reduced motion shows the first hand, still, with no caret.

### Live Readouts (signature)
A KaTeX label (t=, ᾱ_t=) followed by a numeral in Notation type, Readout size. Graphite at rest; accent (on-screen-accent in the band) while t > 0, with the colour change stepped (200ms, 5 steps). t is shown ×1000 as an integer; ᾱ_t to three decimals. Every sampler exposes one except the portrait, which shares the hero's.

### Grain Samplers (signature)
WebGL2 canvases sharing gl.ts noise and schedule, ticking on the one 24 fps clock:
- **Hero name:** 40 reverse steps (about 1.7s) from t = 1 to 0. The inscription and foot start settling below t = 0.34 (625ms in 15 steps, foot delayed 125ms). At t = 0 the DOM text takes over and two frames later the canvas clears and stops. A 3.2s CSS safety reveals text if the sampler never runs.
- **Particle portrait:** a fixed canvas of ink points (alpha 0.5 light, 0.62 dark) sampled over 44 steps alongside the name. On scroll (s = scrollY / 0.9 viewport heights) it is re-noised to t = 0.9·s^1.2, rises by 0.14 of its height and fades out between s = 0.12 and 0.55 (smoothstep), clipped at the hero rule. It stops drawing once past the hero. Reduced motion draws it once at rest.
- **ViTeX stage:** first view samples in over 30 frames; a scene switch noises forward over 9 frames, holds at the peak until the next clip has a frame, then samples back over 15.
- **Campus band:** scroll-driven, one pass with two textures. Each hand-over is a threshold dissolve over a field of two octaves of value noise plus pixel grain (0.6 / 0.25 / 0.15), with a 0.2 bump of forward noise at its midpoint; an incoming school holds at residual t = 0.14. The band opens on the first photo's own coarsest mip and resolves coarse to fine, never through black. Text swaps over 500ms in 12 steps.
- **Visitor globe:** points diffuse onto the sphere over 36 frames; the globe yaws slowly on the clock and can be dragged.

### Screen Band
Education, always dark (screen / on-screen). Static path: each school is its own full-bleed plate with a toned photograph (grayscale 0.35, brightness 0.52) under a bottom scrim for legibility. Live path: one sticky screen, the photograph sampled in WebGL (desaturated, darkened, scrimmed under the degree, the heading and the rail), a hairline index row of 28px ticks and a readout.

### Ledger Rows
News, research areas and country counts: a top rule, 1px bottom rule per row. News and counts carry a graphite date or count column in tabular numerals with ink text. Research areas are a two-column definition list of four rows: term at 450 in ink, note in Small soft ink.

### Colophon
A hairline-topped footer: the x₀ Colophon Mark in ink at left, Label-size meta and credits at right.

### Motion Grammar
- **Clock:** 24 fps (one frame = 41.667ms). Samplers and the inscription only act on shared frame boundaries.
- **Stepped CSS:** state transitions that belong to the world use `steps(n)` sized to whole frames (200ms/5, 250ms/6, 500ms/12, 625ms/15). Chrome transitions (link, toggle colour, rail label weight and tick, bar fill, wordmark) use 160–400ms on cubic-bezier(0.16, 1, 0.3, 1).
- **Theme switch:** a view transition revealed through a 12-frame grain threshold mask (500ms, step-end), a 128px field of 40% low-frequency and 60% white noise tiled at 256px. Falls back to an instant swap without View Transitions.
- **Reduced motion (`html.still`):** samplers render once at t = 0, no clock loop, the inscription stands still, instant theme swap, no smooth scroll; the page is complete and still.
- **No WebGL (`html.no-gl`):** text renders immediately, no portrait, the ViTeX video plays in place, the band shows the static plates, the globe is hidden.

## Do's and Don'ts

### Do:
- **Do** make every new state change a noise-to-signal trajectory on the shared 24 fps clock; content samplers also show a live t readout.
- **Do** clear and stop a sampler when its trajectory ends; the ground at rest is flat plate or void.
- **Do** keep the accent to live readouts (t > 0), the rail marker and visitor marks, plus focus ring, caret and selection; return readouts to graphite at t = 0.
- **Do** separate content with 1px hairlines (rule / rule-strong) and ledger rows; when a unit needs to stand apart, give it a square plate-2 panel.
- **Do** set numerals tabular and timesteps in KaTeX_Main at 1.08em with a 2.9ch minimum.
- **Do** keep headings at weight 300–400 with negative tracking (-0.015em to -0.04em).
- **Do** size stepped CSS transitions to whole frames (41.667ms multiples) with `steps(n)`.
- **Do** ship the still path and the no-WebGL path as finished pages.
- **Do** define every new neutral for plate, void and, if it can appear there, the screen band.
- **Do** subset any CJK face to the glyphs it sets and self-host it.

### Don't:
- **Don't** use bordered, rounded or shadowed cards, pills or badges; the only container is the square plate-2 panel, and radius is 0 except the 2px focus outline.
- **Don't** add shadows beyond the menu sheet.
- **Don't** place eyebrows or kickers above headings; supporting lines (venue, dates, notes) go below the title.
- **Don't** animate hover with lifts, scales, parallax or cursor followers; hover changes colour or underline only.
- **Don't** give the accent a glow, gradient, fill or rest state.
- **Don't** crossfade or slide where the world samples; no crossfades for theme or scene changes.
- **Don't** leave grain, particles or texture standing on the page at rest, or let the portrait draw below the hero rule.
- **Don't** use uppercase labels or letter-spaced small caps.
- **Don't** run a second animation clock or free-running requestAnimationFrame outside gl.ts.
