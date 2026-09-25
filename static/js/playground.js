/* RFPI manifold playground.
 * Everything here is computed exactly: behavior data are a dense set of points on a
 * 1-D curve in R^2 blurred by isotropic Gaussian noise of std sN (a Gaussian mixture).
 * For y = A + sigma Z the ideal flow's denoising map is the posterior mean
 *   R(y) = E[A | y],   J_R(y) = Cov[A | y] / sigma^2        (Tweedie)
 * with sigma = (1 - t_g) / t_g.
 */
(function () {
  const $ = id => document.getElementById(id);
  const cv = $('pg-canvas'); if (!cv) return;
  const ctx = cv.getContext('2d');
  const ev = $('pg-eig'), ectx = ev.getContext('2d');

  const COL = { ink: '#22262A', blue: '#2B6CB0', green: '#1F8A70', orange: '#D9822B', red: '#C8453A', purple: '#7B5EA7', gray: '#9AA3AD' };
  const METHODS = [
    { key: 'raw',    name: 'Raw ascent',  col: COL.red,    filt: false, retr: false },
    { key: 'filt',   name: 'Filter-only', col: COL.orange, filt: true,  retr: false },
    { key: 'rawR',   name: 'Raw + R',     col: COL.purple, filt: false, retr: true  },
    { key: 'rfpi',   name: 'RFPI',        col: COL.green,  filt: true,  retr: true  },
  ];

  const S = {
    shape: 'ring', tg: 0.9, sN: 0.01, beta: 2.5, L: 0.08, K: 40,
    a: null, pts: [], paths: null, pathsAlt: null, showAlt: false, anim: 0,
  };

  /* ---------------- manifolds ---------------- */
  const SH = {
    ring: {
      view: 1.7,
      pts(n) { const p = []; for (let i = 0; i < n; i++) { const t = 2 * Math.PI * i / n; p.push([Math.cos(t), Math.sin(t)]); } return p; },
      star: [Math.cos(-0.35 * Math.PI), Math.sin(-0.35 * Math.PI)],
      f(a) { const phi = Math.atan2(a[1], a[0]), ps = -0.35 * Math.PI; return Math.cos(phi - ps); },           // true value along M
      h(a) { return Math.hypot(a[0], a[1]) - 1; },                                                           // signed normal offset
      start: [Math.cos(0.72 * Math.PI) * 1.0, Math.sin(0.72 * Math.PI) * 1.0],
    },
    wave: {
      view: 1.7,
      c: x => 0.42 * Math.sin(2.1 * x),
      pts(n) { const p = []; for (let i = 0; i < n; i++) { const x = -1.9 + 3.8 * i / (n - 1); p.push([x, this.c(x)]); } return p; },
      get star() { return [1.05, this.c(1.05)]; },
      f(a) { return Math.cos(1.25 * (a[0] - 1.05)); },
      h(a) { return a[1] - this.c(a[0]); },
      start: [-1.15, 0.42 * Math.sin(2.1 * -1.15)],
    },
  };
  const M = () => SH[S.shape];

  /* critic: correct on M, extrapolated normal slope beta off M */
  const Qhat = (a, beta) => M().f(a) + beta * M().h(a);
  function gradQ(a, beta) {
    const e = 1e-4;
    return [(Qhat([a[0] + e, a[1]], beta) - Qhat([a[0] - e, a[1]], beta)) / (2 * e),
            (Qhat([a[0], a[1] + e], beta) - Qhat([a[0], a[1] - e], beta)) / (2 * e)];
  }

  /* ---------------- exact posterior denoiser ---------------- */
  function denoise(y) {
    const sig = (1 - S.tg) / S.tg, s2 = S.sN * S.sN;
    if (sig < 1e-5) return { R: y.slice(), J: [[1, 0], [0, 1]] };
    const g2 = sig * sig, v = s2 + g2, k = s2 / v, c = s2 * g2 / v;
    const P = S.pts, n = P.length;
    let maxl = -Infinity; const lw = new Float64Array(n);
    for (let i = 0; i < n; i++) { const dx = y[0] - P[i][0], dy = y[1] - P[i][1]; lw[i] = -(dx * dx + dy * dy) / (2 * v); if (lw[i] > maxl) maxl = lw[i]; }
    let W = 0, mx = 0, my = 0, sxx = 0, sxy = 0, syy = 0;
    for (let i = 0; i < n; i++) {
      const w = Math.exp(lw[i] - maxl); if (w < 1e-12) continue;
      const ux = P[i][0] + k * (y[0] - P[i][0]), uy = P[i][1] + k * (y[1] - P[i][1]);
      W += w; mx += w * ux; my += w * uy; sxx += w * ux * ux; sxy += w * ux * uy; syy += w * uy * uy;
    }
    mx /= W; my /= W; sxx /= W; sxy /= W; syy /= W;
    const Cxx = c + sxx - mx * mx, Cxy = sxy - mx * my, Cyy = c + syy - my * my;
    return { R: [mx, my], J: [[Cxx / g2, Cxy / g2], [Cxy / g2, Cyy / g2]] };
  }
  const mv = (J, g) => [J[0][0] * g[0] + J[0][1] * g[1], J[1][0] * g[0] + J[1][1] * g[1]];
  const norm = v => Math.hypot(v[0], v[1]);
  function eig2(J) { const a = J[0][0], b = J[0][1], d = J[1][1], tr = a + d, det = a * d - b * b, disc = Math.sqrt(Math.max(0, tr * tr / 4 - det)); return [tr / 2 + disc, tr / 2 - disc]; }

  /* nearest point on M and local frame */
  function nearest(a) {
    let bi = 0, bd = Infinity; const P = S.pts;
    for (let i = 0; i < P.length; i++) { const d = (a[0] - P[i][0]) ** 2 + (a[1] - P[i][1]) ** 2; if (d < bd) { bd = d; bi = i; } }
    const n = P.length, p0 = P[(bi - 1 + n) % n], p1 = P[(bi + 1) % n];
    let tx = p1[0] - p0[0], ty = p1[1] - p0[1];
    if (S.shape === 'wave' && (bi === 0 || bi === n - 1)) { const q = P[bi === 0 ? 1 : n - 2]; tx = q[0] - P[bi][0]; ty = q[1] - P[bi][1]; if (bi === n - 1) { tx = -tx; ty = -ty; } }
    const tn = Math.hypot(tx, ty); tx /= tn; ty /= tn;
    return { p: P[bi], d: Math.sqrt(bd), t: [tx, ty], n: [-ty, tx] };
  }

  /* one update of a given method */
  function step(a, m, beta) {
    const g = gradQ(a, beta);
    let d = g;
    if (m.filt) d = mv(denoise(a).J, g);
    const dn = norm(d) || 1e-12;
    let b = [a[0] + S.L * d[0] / dn, a[1] + S.L * d[1] / dn];
    if (m.retr) b = denoise(b).R;
    return b;
  }
  function runPaths(beta) {
    const out = {};
    for (const m of METHODS) { const p = [S.a.slice()]; let x = S.a.slice(); for (let k = 0; k < S.K; k++) { x = step(x, m, beta); if (!isFinite(x[0])) break; x[0] = Math.max(-3, Math.min(3, x[0])); x[1] = Math.max(-3, Math.min(3, x[1])); p.push(x); } out[m.key] = p; }
    return out;
  }


  /* ---- math-style canvas labels (KaTeX fonts are already loaded by the page) ---- */
  const MF = { rm: 'KaTeX_Main, "STIX Two Math", "Cambria Math", "Times New Roman", serif', it: 'KaTeX_Math, "STIX Two Math", "Cambria Math", "Times New Roman", serif', cal: 'KaTeX_Caligraphic, "Apple Chancery", "Times New Roman", serif' };
  // segs: [text, kind] with kind in rm | it | cal | sub | sup | hat(it with a hat)
  function mtext(c, segs, x, y, col, size = 15) {
    c.save(); c.fillStyle = col; c.strokeStyle = col; c.textAlign = 'left'; c.textBaseline = 'alphabetic';
    let cx = x;
    for (const [t, k] of segs) {
      const small = k === 'sub' || k === 'sup', fs = small ? Math.round(size * 0.7) : size;
      const fam = k === 'rm' ? MF.rm : k === 'cal' ? MF.cal : small ? MF.it : MF.it;
      c.font = `${k === 'rm' || k === 'sup' ? '' : 'italic '}${fs}px ${fam}`;
      if (k === 'sup' && /[+\-]/.test(t)) c.font = `${fs}px ${MF.rm}`;
      const dy = k === 'sub' ? size * 0.28 : k === 'sup' ? -size * 0.42 : 0;
      c.fillText(t, cx, y + dy);
      const w = c.measureText(t).width;
      if (k === 'hat') { const hx = cx + w * 0.58, hy = y - size * 0.86; c.lineWidth = 1.3; c.beginPath(); c.moveTo(hx - size * 0.22, hy + size * 0.13); c.lineTo(hx, hy - size * 0.02); c.lineTo(hx + size * 0.22, hy + size * 0.13); c.stroke(); }
      cx += w + (k === 'rm' ? 1 : 0.5);
    }
    c.restore();
  }

  /* ---------------- drawing ---------------- */
  let W = 0, H = 0, dpr = 1, heat = null, heatKey = '';
  function size() {
    dpr = window.devicePixelRatio || 1;
    const r = cv.getBoundingClientRect(); W = r.width; H = r.width; cv.style.height = H + 'px';
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    const re = ev.getBoundingClientRect(); ev.width = Math.round(re.width * dpr); ev.height = Math.round(re.height * dpr);
    heatKey = '';
  }
  const view = () => M().view;
  const X = x => (x + view()) / (2 * view()) * W, Y = y => (view() - y) / (2 * view()) * H;
  const invX = px => px / W * 2 * view() - view(), invY = py => view() - py / H * 2 * view();

  function buildHeat() {
    const key = S.shape + S.beta + W;
    if (key === heatKey) return; heatKey = key;
    const n = 110, img = document.createElement('canvas'); img.width = n; img.height = n;
    const ic = img.getContext('2d'), id = ic.createImageData(n, n);
    let lo = Infinity, hi = -Infinity; const vals = new Float32Array(n * n);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const x = -view() + 2 * view() * (i + .5) / n, y = view() - 2 * view() * (j + .5) / n;
      const q = Qhat([x, y], S.beta); vals[j * n + i] = q; lo = Math.min(lo, q); hi = Math.max(hi, q);
    }
    for (let k = 0; k < n * n; k++) {
      const t = (vals[k] - lo) / (hi - lo || 1), q = Math.round(t * 7) / 7;   // banded like a contour map
      id.data[4 * k] = 255 - q * 18; id.data[4 * k + 1] = 250 - q * 95; id.data[4 * k + 2] = 243 - q * 190; id.data[4 * k + 3] = 255;
    }
    ic.putImageData(id, 0, 0); heat = img;
  }
  function arrow(x0, y0, x1, y1, col, w = 2.4, dash = null, head = 9) {
    ctx.save(); ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = w; if (dash) ctx.setLineDash(dash);
    ctx.beginPath(); ctx.moveTo(X(x0), Y(y0)); ctx.lineTo(X(x1), Y(y1)); ctx.stroke(); ctx.setLineDash([]);
    const ang = Math.atan2(Y(y1) - Y(y0), X(x1) - X(x0)), L = Math.hypot(Y(y1) - Y(y0), X(x1) - X(x0));
    if (L > 4) { ctx.beginPath(); ctx.moveTo(X(x1), Y(y1)); ctx.lineTo(X(x1) - head * Math.cos(ang - .4), Y(y1) - head * Math.sin(ang - .4)); ctx.lineTo(X(x1) - head * Math.cos(ang + .4), Y(y1) - head * Math.sin(ang + .4)); ctx.closePath(); ctx.fill(); }
    ctx.restore();
  }
  function dot(p, r, fill, stroke) { ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), r, 0, 7); if (fill) { ctx.fillStyle = fill; ctx.fill(); } if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 2; ctx.stroke(); } }
  function starShape(p) {
    const cx = X(p[0]), cy = Y(p[1]); ctx.beginPath();
    for (let i = 0; i < 10; i++) { const r = i % 2 ? 4 : 10, t = -Math.PI / 2 + i * Math.PI / 5; ctx.lineTo(cx + r * Math.cos(t), cy + r * Math.sin(t)); }
    ctx.closePath(); ctx.fillStyle = COL.ink; ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();
  }

  function draw() {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    buildHeat(); ctx.imageSmoothingEnabled = true; ctx.drawImage(heat, 0, 0, W, H);
    // support band (±2 sN, drawn at least 3px) and curve
    const P = S.pts;
    ctx.save(); ctx.strokeStyle = 'rgba(34,38,42,0.12)'; ctx.lineWidth = Math.max(6, 4 * S.sN / (2 * view()) * W); ctx.lineJoin = 'round';
    ctx.beginPath(); P.forEach((p, i) => i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1]))); if (S.shape === 'ring') ctx.closePath(); ctx.stroke();
    ctx.strokeStyle = COL.ink; ctx.lineWidth = 2; ctx.stroke(); ctx.restore();
    const lp = S.shape === 'ring' ? [-0.95, -0.95] : [-1.6, -0.25]; mtext(ctx, [['M', 'cal'], ['s', 'sub']], X(lp[0]), Y(lp[1]), COL.ink, 18);
    starShape(M().star);

    // paths
    const drawPaths = (paths, dashed) => {
      if (!paths) return;
      const upto = Math.min(S.K, Math.floor(S.anim));
      for (const m of METHODS) {
        const p = paths[m.key]; ctx.save(); ctx.strokeStyle = m.col; ctx.fillStyle = m.col; ctx.lineWidth = dashed ? 1.6 : 2.2; ctx.globalAlpha = dashed ? .55 : 1; if (dashed) ctx.setLineDash([5, 4]);
        ctx.beginPath(); for (let k = 0; k <= Math.min(upto, p.length - 1); k++) { k ? ctx.lineTo(X(p[k][0]), Y(p[k][1])) : ctx.moveTo(X(p[k][0]), Y(p[k][1])); } ctx.stroke(); ctx.setLineDash([]);
        for (let k = 1; k <= Math.min(upto, p.length - 1); k++) { ctx.beginPath(); ctx.arc(X(p[k][0]), Y(p[k][1]), dashed ? 2 : 2.8, 0, 7); ctx.fill(); }
        ctx.restore();
      }
    };
    if (S.showAlt) drawPaths(S.pathsAlt, true);
    drawPaths(S.paths, false);

    // single-step anatomy at a
    const a = S.a, g = gradQ(a, S.beta), gn = norm(g) || 1, D = denoise(a), Jg = mv(D.J, g), nb = nearest(a);
    const sc = 0.5 / gn;                     // display scale: |g| -> 0.5 units
    const gt = (g[0] * nb.t[0] + g[1] * nb.t[1]), gnrm = (g[0] * nb.n[0] + g[1] * nb.n[1]);
    if (!S.paths || S.anim >= S.K) {
      // decomposition of g into tangent + normal
      arrow(a[0], a[1], a[0] + sc * gt * nb.t[0], a[1] + sc * gt * nb.t[1], COL.gray, 1.4, [4, 4], 7);
      arrow(a[0] + sc * gt * nb.t[0], a[1] + sc * gt * nb.t[1], a[0] + sc * g[0], a[1] + sc * g[1], COL.orange, 1.8, [4, 4], 7);
      arrow(a[0], a[1], a[0] + sc * g[0], a[1] + sc * g[1], COL.red, 2.6);
      arrow(a[0], a[1], a[0] + sc * Jg[0], a[1] + sc * Jg[1], COL.blue, 3);
      const jn = norm(Jg) || 1e-12, b = [a[0] + S.L * Jg[0] / jn, a[1] + S.L * Jg[1] / jn], ap = denoise(b).R;
      ctx.save(); ctx.strokeStyle = COL.green; ctx.setLineDash([3, 3]); ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(X(b[0]), Y(b[1])); ctx.lineTo(X(ap[0]), Y(ap[1])); ctx.stroke(); ctx.restore();
      dot(b, 5, '#fff', COL.blue); dot(ap, 5.5, COL.green, '#fff');
      ctx.font = '600 13px -apple-system,Segoe UI,sans-serif';
      const lab = (t, p, c, dx = 8, dy = -8) => { ctx.fillStyle = c; ctx.fillText(t, X(p[0]) + dx, Y(p[1]) + dy); };
      { const q = [a[0] + sc * g[0], a[1] + sc * g[1]]; mtext(ctx, [['∇', 'rm'], ['Q', 'hat']], X(q[0]) + 8, Y(q[1]) - 6, COL.red, 16); }
      { const q = [a[0] + sc * Jg[0], a[1] + sc * Jg[1]]; mtext(ctx, [['J', 'it'], ['R', 'sub'], [' ∇', 'rm'], ['Q', 'hat']], X(q[0]) + 14, Y(q[1]) + 24, COL.blue, 16); }
      mtext(ctx, [['a', 'it'], ['+', 'sup']], X(ap[0]) - 26, Y(ap[1]) + 24, COL.green, 16);
    }
    dot(a, 7, COL.ink, '#fff');
    if (demoOn && !userTookOver) { ctx.save(); ctx.font = '600 12px -apple-system,Segoe UI,sans-serif'; const t = 'Auto demo · drag the point or change a slider to take control'; const w = ctx.measureText(t).width; ctx.fillStyle = 'rgba(255,255,255,0.88)'; ctx.fillRect(10, H - 30, w + 16, 22); ctx.fillStyle = '#6B7480'; ctx.fillText(t, 18, H - 14); ctx.restore(); }

    // readouts
    const [l1, l2] = eig2(D.J);
    // tangent / normal gains measured in the true local frame
    const Jt = mv(D.J, nb.t), Jn = mv(D.J, nb.n);
    $('pg-lt').textContent = (Jt[0] * nb.t[0] + Jt[1] * nb.t[1]).toFixed(3);
    $('pg-ln').textContent = norm(Jn).toFixed(3);
    $('pg-norm').textContent = `${Math.abs(gnrm).toFixed(2)} → ${Math.abs(Jg[0] * nb.n[0] + Jg[1] * nb.n[1]).toFixed(3)}`;
    $('pg-sig').textContent = ((1 - S.tg) / S.tg).toFixed(3);
    drawEig(nb.p);
    board();
  }

  /* eigenvalues of J_R along t_g at the nearest on-manifold point */
  let eigCache = { key: '', data: null };
  function drawEig(p) {
    const w = ev.width / dpr, h = ev.height / dpr; ectx.setTransform(dpr, 0, 0, dpr, 0, 0); ectx.clearRect(0, 0, w, h);
    const key = S.shape + S.sN + p[0].toFixed(3) + p[1].toFixed(3);
    if (eigCache.key !== key) {
      const keep = S.tg, nb = nearest(p), out = [];
      for (let i = 0; i <= 50; i++) { const tg = 0.6 + 0.4 * i / 50; S.tg = Math.min(tg, 0.99999); const J = denoise(p).J, Jt = mv(J, nb.t), Jn = mv(J, nb.n); out.push([tg, Jt[0] * nb.t[0] + Jt[1] * nb.t[1], Jn[0] * nb.n[0] + Jn[1] * nb.n[1]]); }
      S.tg = keep; eigCache = { key, data: out };
    }
    const L = 34, R = 8, T = 10, B = 26, x = t => L + (t - 0.6) / 0.4 * (w - L - R), y = v => T + (1 - Math.max(-0.05, Math.min(1.1, v)) / 1.1) * (h - T - B);
    ectx.strokeStyle = '#E3E7EB'; ectx.fillStyle = '#6B7480'; ectx.font = '11px -apple-system,Segoe UI,sans-serif'; ectx.lineWidth = 1;
    [0, 0.5, 1].forEach(v => { ectx.beginPath(); ectx.moveTo(L, y(v)); ectx.lineTo(w - R, y(v)); ectx.stroke(); ectx.textAlign = 'right'; ectx.fillText(v.toFixed(1), L - 5, y(v) + 4); });
    ectx.textAlign = 'center'; [0.6, 0.7, 0.8, 0.9, 1.0].forEach(t => { ectx.textAlign = t === 1 ? 'right' : 'center'; ectx.fillText(t.toFixed(1), x(t) + (t === 1 ? 4 : 0), h - B + 14); }); ectx.textAlign = 'center';
    ectx.fillText('guidance level', (L + w - R) / 2 - 10, h - 2); { const tw = ectx.measureText('guidance level').width; mtext(ectx, [['t', 'it'], ['g', 'sub']], (L + w - R) / 2 - 10 + tw / 2 + 4, h - 3, '#6B7480', 12); }
    const line = (idx, col) => { ectx.strokeStyle = col; ectx.lineWidth = 2.2; ectx.beginPath(); eigCache.data.forEach((d, i) => i ? ectx.lineTo(x(d[0]), y(d[idx])) : ectx.moveTo(x(d[0]), y(d[idx]))); ectx.stroke(); };
    line(1, COL.blue); line(2, COL.orange);
    ectx.strokeStyle = COL.ink; ectx.setLineDash([4, 3]); ectx.beginPath(); ectx.moveTo(x(S.tg), T); ectx.lineTo(x(S.tg), h - B); ectx.stroke(); ectx.setLineDash([]);
  }

  function board() {
    const tb = $('pg-board'); if (!S.paths) { tb.innerHTML = '<tr><td colspan="3" class="muted">Press “Run updates” to compare the four updates.</td></tr>'; return; }
    const k = Math.min(S.K, Math.floor(S.anim));
    tb.innerHTML = METHODS.map(m => {
      const p = S.paths[m.key][Math.min(k, S.paths[m.key].length - 1)], nb = nearest(p), f = M().f(nb.p), off = nb.d > Math.max(0.05, 4 * S.sN);
      return `<tr${m.key === 'rfpi' ? ' class="ours"' : ''}><td><i style="background:${m.col}"></i>${m.name}</td><td>${f.toFixed(2)}</td><td style="color:${off ? COL.red : 'inherit'}">${nb.d.toFixed(3)}${off ? ' ✗' : ''}</td></tr>`;
    }).join('');
  }

  /* ---------------- interaction ---------------- */
  function reset(keepA) { S.pts = M().pts(S.shape === 'ring' ? 1600 : 1400); if (!keepA) S.a = M().start.slice(); S.paths = null; S.pathsAlt = null; S.anim = 0; eigCache.key = ''; heatKey = ''; draw(); }
  /* animation + auto demo: loops while the canvas is on screen, until the user interacts */
  const STEP_MS = 90, DEMO_PAUSE = 2600;
  let animId = 0, demoTimer = null, visible = false, userTookOver = false, demoOn = false;
  function run(fromDemo) {
    if (!fromDemo) { userTookOver = true; clearTimeout(demoTimer); }
    demoOn = !!fromDemo;
    S.paths = runPaths(S.beta); S.pathsAlt = runPaths(-S.beta); S.anim = 0;
    const id = ++animId; let t0 = null;
    const tick = t => {
      if (id !== animId) return;                       // a newer run replaced this one
      if (t0 === null) t0 = t;
      S.anim = Math.max(0, Math.min(S.K, (t - t0) / STEP_MS)); draw();
      if (S.anim < S.K) requestAnimationFrame(tick);
      else if (fromDemo) scheduleDemo(DEMO_PAUSE);
    };
    requestAnimationFrame(tick);
  }
  function scheduleDemo(delay) {
    clearTimeout(demoTimer);
    if (userTookOver || !visible) { demoOn = false; return; }
    demoTimer = setTimeout(() => { if (!userTookOver && visible) run(true); }, delay);
  }
  function takeOver() { userTookOver = true; demoOn = false; clearTimeout(demoTimer); }
  let dragging = false;
  const pos = e => { const r = cv.getBoundingClientRect(), p = e.touches ? e.touches[0] : e; return [invX(p.clientX - r.left), invY(p.clientY - r.top)]; };
  const down = e => { takeOver(); dragging = true; S.a = pos(e); S.paths = null; draw(); e.preventDefault(); };
  const move = e => { if (!dragging) return; S.a = pos(e); draw(); e.preventDefault(); };
  cv.addEventListener('mousedown', down); window.addEventListener('mousemove', move); window.addEventListener('mouseup', () => dragging = false);
  cv.addEventListener('touchstart', down, { passive: false }); cv.addEventListener('touchmove', move, { passive: false }); cv.addEventListener('touchend', () => dragging = false);

  const bind = (id, key, fmt, after) => { const el = $(id), out = $(id + '-v'); const f = () => { takeOver(); S[key] = +el.value; if (out) out.textContent = fmt(S[key]); if (after) after(); else { S.paths = null; } eigCache.key = ''; heatKey = ''; draw(); }; el.addEventListener('input', f); out && (out.textContent = fmt(+el.value)); };
  bind('pg-tg', 'tg', v => v.toFixed(3));
  bind('pg-sn', 'sN', v => v.toFixed(3));
  bind('pg-beta', 'beta', v => v.toFixed(1));
  bind('pg-L', 'L', v => v.toFixed(2));
  $('pg-run').onclick = () => run(false);
  $('pg-alt').onchange = e => { takeOver(); S.showAlt = e.target.checked; draw(); };
  $('pg-flip').onclick = () => { S.beta = -S.beta; $('pg-beta').value = S.beta; $('pg-beta-v').textContent = S.beta.toFixed(1); heatKey = ''; if (S.paths) run(false); else draw(); };
  document.querySelectorAll('[data-shape]').forEach(b => b.onclick = () => { takeOver(); S.shape = b.dataset.shape; document.querySelectorAll('[data-shape]').forEach(x => x.classList.toggle('on', x === b)); reset(false); });
  document.querySelectorAll('[data-preset]').forEach(b => b.onclick = () => {
    const p = b.dataset.preset;
    if (p === 'id') { S.tg = 1; } else if (p === 'paper') { S.tg = 0.9; S.sN = 0.01; } else if (p === 'weak') { S.tg = 0.99; }
    $('pg-tg').value = S.tg; $('pg-tg-v').textContent = S.tg.toFixed(3); $('pg-sn').value = S.sN; $('pg-sn-v').textContent = S.sN.toFixed(3);
    eigCache.key = ''; run(false);
  });
  window.addEventListener('resize', () => { size(); draw(); });

  size(); reset(false);
  if (document.fonts && document.fonts.load) Promise.all(['16px KaTeX_Main', 'italic 16px KaTeX_Math', '18px KaTeX_Caligraphic'].map(f => document.fonts.load(f))).catch(() => {}).then(() => { eigCache.key = ''; draw(); });
  // run once when first visible
  // auto demo: start once most of the canvas is on screen; pause when it scrolls away
  new IntersectionObserver(e => {
    visible = e[0].isIntersecting;
    if (visible) { if (!userTookOver && !demoOn) scheduleDemo(400); }
    else clearTimeout(demoTimer);
  }, { threshold: 0.55 }).observe(cv);
  document.addEventListener('visibilitychange', () => { if (document.hidden) clearTimeout(demoTimer); else if (visible && !demoOn) scheduleDemo(400); });
  window.__rfpiPlayground = { S, denoise, step, runPaths, nearest, METHODS, gradQ };
})();
