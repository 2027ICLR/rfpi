# RFPI anonymous project page

Open `index.html` directly in Chrome/Safari to preview. All assets are local: math is
pre-rendered with KaTeX (fonts in `static/katex/`), so there are no CDN dependencies.

## Editing
Edit `src/index.template.html` (write math as `\( ... \)` or `$$ ... $$`), then rebuild:

    npm i katex && node tools/build.js

The build renders all formulas and inlines `static/data/q_values.json` into `index.html`.
The interactive manifold demo lives in `static/js/playground.js`.

## Before publishing
1. Replace the Paper / Code `href="#"` placeholders with an OpenReview PDF link and an
   anonymized repository (e.g. anonymous.4open.science).
2. Host anonymously (fresh GitHub Pages account or a Netlify drop), not on a lab/personal domain.
3. Test-time real-robot numbers come from the draft's Table (marked provisional); the online
   curve uses the verified results. Update both if numbers change.

## Contents
- `static/videos/third_*.mp4`  third-person rollouts (0-2 RFPI success, 3-4 BC failure); only the chest logo is blurred
- `static/videos/head_*.mp4`   head camera, frame-aligned with the Q-value CSV (drives the synced chart)
- `static/images/`             paper figures rendered from PDF + video posters
- `static/js/playground.js`    exact posterior denoiser on a ring / wave manifold (R = E[A|y], J_R = Cov[A|y]/sigma^2)

Wrist-camera views are excluded: they show lab members' faces and the robot vendor logo.
