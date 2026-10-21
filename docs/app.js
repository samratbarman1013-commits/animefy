/* animefy — client-side video -> anime / cartoon / sketch / oil.
   Pure Canvas2D + typed arrays. Runs entirely in the browser. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const srcC = $("src"), outC = $("out");
  const sctx = srcC.getContext("2d", { willReadFrequently: true });
  const octx = outC.getContext("2d", { willReadFrequently: true });
  const video = $("video");

  let running = false, raf = null, recorder = null, chunks = [], maxW = 640;
  let lastT = 0, fpsEMA = 0;

  /* ------------------------------------------------------------------ styles
     Each style is just a bundle of pipeline parameters:
       smooth   bilateral radius (edge-preserving smoothing)
       range    bilateral colour-sigma (how different a pixel may be and still blend)
       colors   posterisation levels
       line     edge threshold (higher = fewer lines)
       thick    line weight (dilation radius)
       sat      saturation multiplier
       lineAlpha how dark the ink gets
       flatten  how far toward fully flat colour
       gray     1 = pencil-sketch base
       soft     extra ink softening (0..1) */
  const STYLES = {
    cartoon: { smooth: 3, range: 32, colors: 8,  line: 55, thick: 1, sat: 1.10, lineAlpha: 1.00, flatten: 1.00, gray: 0, soft: 0.0 },
    anime:   { smooth: 5, range: 45, colors: 6,  line: 50, thick: 1, sat: 1.35, lineAlpha: 0.85, flatten: 0.95, gray: 0, soft: 0.6 },
    comic:   { smooth: 3, range: 28, colors: 4,  line: 38, thick: 2, sat: 1.15, lineAlpha: 1.00, flatten: 1.00, gray: 0, soft: 0.0 },
    sketch:  { smooth: 3, range: 30, colors: 6,  line: 34, thick: 1, sat: 0.00, lineAlpha: 1.00, flatten: 1.00, gray: 1, soft: 0.3 },
    oil:     { smooth: 6, range: 60, colors: 10, line: 160, thick: 0, sat: 1.30, lineAlpha: 0.00, flatten: 1.00, gray: 0, soft: 0.0 },
  };

  function params() {
    const st = $("style").value;
    return {
      smooth: +$("smooth").value,
      range: STYLES[st].range,
      colors: +$("colors").value,
      line: +$("line").value,
      thick: +$("thick").value,
      sat: +$("sat").value / 10,
      lineAlpha: STYLES[st].lineAlpha,
      flatten: STYLES[st].flatten,
      gray: STYLES[st].gray,
      soft: STYLES[st].soft,
    };
  }

  function applyPreset(st) {
    const p = STYLES[st];
    $("smooth").value = p.smooth; $("colors").value = p.colors;
    $("line").value = p.line; $("thick").value = p.thick;
    $("sat").value = Math.round(p.sat * 10);
    syncLabels();
  }
  function syncLabels() {
    $("v-smooth").textContent = $("smooth").value;
    $("v-colors").textContent = $("colors").value;
    $("v-line").textContent = $("line").value;
    $("v-thick").textContent = $("thick").value;
    $("v-sat").textContent = (+$("sat").value / 10).toFixed(1);
  }

  /* ------------------------------------------------------------ image ops */
  const MAX_D2 = 3 * 255 * 255;
  const _lutCache = new Map();
  function rangeLUT(sigma) {
    let lut = _lutCache.get(sigma);
    if (lut) return lut;
    lut = new Float32Array(MAX_D2 + 1);
    const inv = 1 / (2 * sigma * sigma);
    for (let i = 0; i <= MAX_D2; i++) lut[i] = Math.exp(-i * inv);
    _lutCache.set(sigma, lut);
    return lut;
  }

  // Separable bilateral filter: horizontal pass.
  function bilatH(src, dst, w, h, r, lut) {
    const sw = new Float32Array(2 * r + 1);
    const ss2 = 2 * (r * 0.65) * (r * 0.65) + 1e-6;
    for (let k = -r; k <= r; k++) sw[k + r] = Math.exp(-(k * k) / ss2);
    for (let y = 0; y < h; y++) {
      const row = y * w * 4;
      for (let x = 0; x < w; x++) {
        const ci = row + x * 4, cr = src[ci], cg = src[ci + 1], cb = src[ci + 2];
        let sr = 0, sg = 0, sb = 0, ws = 0;
        const x0 = Math.max(0, x - r), x1 = Math.min(w - 1, x + r);
        for (let xx = x0; xx <= x1; xx++) {
          const ii = row + xx * 4;
          const dr = src[ii] - cr, dg = src[ii + 1] - cg, db = src[ii + 2] - cb;
          let d2 = dr * dr + dg * dg + db * db; if (d2 > MAX_D2) d2 = MAX_D2;
          const wgt = lut[d2] * sw[xx - x + r];
          sr += src[ii] * wgt; sg += src[ii + 1] * wgt; sb += src[ii + 2] * wgt; ws += wgt;
        }
        const o = ci; dst[o] = sr / ws; dst[o + 1] = sg / ws; dst[o + 2] = sb / ws; dst[o + 3] = 255;
      }
    }
  }

  function bilatV(src, dst, w, h, r, lut) {
    const sw = new Float32Array(2 * r + 1);
    const ss2 = 2 * (r * 0.65) * (r * 0.65) + 1e-6;
    for (let k = -r; k <= r; k++) sw[k + r] = Math.exp(-(k * k) / ss2);
    for (let x = 0; x < w; x++) {
      for (let y = 0; y < h; y++) {
        const ci = (y * w + x) * 4, cr = src[ci], cg = src[ci + 1], cb = src[ci + 2];
        let sr = 0, sg = 0, sb = 0, ws = 0;
        const y0 = Math.max(0, y - r), y1 = Math.min(h - 1, y + r);
        for (let yy = y0; yy <= y1; yy++) {
          const ii = (yy * w + x) * 4;
          const dr = src[ii] - cr, dg = src[ii + 1] - cg, db = src[ii + 2] - cb;
          let d2 = dr * dr + dg * dg + db * db; if (d2 > MAX_D2) d2 = MAX_D2;
          const wgt = lut[d2] * sw[yy - y + r];
          sr += src[ii] * wgt; sg += src[ii + 1] * wgt; sb += src[ii + 2] * wgt; ws += wgt;
        }
        const o = ci; dst[o] = sr / ws; dst[o + 1] = sg / ws; dst[o + 2] = sb / ws; dst[o + 3] = 255;
      }
    }
  }

  function bilateral(src, w, h, r, sigma) {
    if (r < 1) return src;
    const lut = rangeLUT(sigma);
    const dst = new Uint8ClampedArray(src.length);
    const tmp = new Uint8ClampedArray(src.length);
    bilatH(src, tmp, w, h, r, lut);
    bilatV(tmp, dst, w, h, r, lut);
    return dst;
  }

  function boxBlurGray(g, w, h, r) {
    if (r < 1) return;
    const win = 2 * r + 1, t = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      const row = y * w; let s = 0;
      for (let k = -r; k <= r; k++) s += g[row + Math.min(w - 1, Math.max(0, k))];
      for (let x = 0; x < w; x++) {
        t[row + x] = s / win;
        s += g[row + Math.min(w - 1, x + r + 1)] - g[row + Math.max(0, x - r)];
      }
    }
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let k = -r; k <= r; k++) s += t[Math.min(h - 1, Math.max(0, k)) * w + x];
      for (let y = 0; y < h; y++) {
        g[y * w + x] = s / win;
        s += t[Math.min(h - 1, y + r + 1) * w + x] - t[Math.max(0, y - r) * w + x];
      }
    }
  }

  // Separable max filter -> thickens the line mask.
  function maxFilter(a, w, h, r) {
    if (r < 1) return a;
    const t = new Float32Array(w * h), o = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) {
        let m = 0;
        for (let k = -r; k <= r; k++) { const xx = x + k; if (xx >= 0 && xx < w) { const v = a[row + xx]; if (v > m) m = v; } }
        t[row + x] = m;
      }
    }
    for (let x = 0; x < w; x++) {
      for (let y = 0; y < h; y++) {
        let m = 0;
        for (let k = -r; k <= r; k++) { const yy = y + k; if (yy >= 0 && yy < h) { const v = t[yy * w + x]; if (v > m) m = v; } }
        o[y * w + x] = m;
      }
    }
    return o;
  }

  const smoothstep = (e0, e1, x) => {
    let t = (x - e0) / (e1 - e0); t = t < 0 ? 0 : t > 1 ? 1 : t;
    return t * t * (3 - 2 * t);
  };

  /* ---------------------------------------------------------- the effect */
  function stylize(srcData, w, h, p) {
    const n = w * h;

    // 1. edge-preserving smoothing
    let work = new Uint8ClampedArray(srcData.data);
    work = bilateral(work, w, h, p.smooth, p.range);

    // 2. line strength from the smoothed image (anti-aliased, not binary)
    const gray = new Float32Array(n);
    for (let i = 0, j = 0; i < work.length; i += 4, j++) {
      gray[j] = work[i] * 0.299 + work[i + 1] * 0.587 + work[i + 2] * 0.114;
    }
    // Blur the edge channel well beyond the colour smoothing: the bilateral
    // filter keeps real edges *and* texture, and we only want the former to
    // become ink. A wider blur on the gradient source drops fur/foliage
    // texture and leaves the silhouette.
    boxBlurGray(gray, w, h, p.smooth + 1);

    const mag = new Float32Array(n), thr = p.line;
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x;
        const gx = -gray[i - w - 1] - 2 * gray[i - 1] - gray[i + w - 1]
                 + gray[i - w + 1] + 2 * gray[i + 1] + gray[i + w + 1];
        const gy = -gray[i - w - 1] - 2 * gray[i - w] - gray[i - w + 1]
                 + gray[i + w - 1] + 2 * gray[i + w] + gray[i + w + 1];
        mag[i] = Math.abs(gx) + Math.abs(gy);
      }
    }
    let strength = new Float32Array(n);
    const e1 = thr * 1.7;
    for (let j = 0; j < n; j++) strength[j] = smoothstep(thr, e1, mag[j]);
    if (p.thick > 0) strength = maxFilter(strength, w, h, p.thick);

    // 3. posterise with ordered dithering
    const L = Math.max(2, p.colors | 0), stepv = 255 / (L - 1);
    const bayer = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];
    const luts = [];
    for (let b = 0; b < 16; b++) {
      const lut = new Uint8ClampedArray(256), off = (bayer[b] / 16 - 0.5) * stepv * 0.45;
      for (let i = 0; i < 256; i++) {
        const v = Math.min(255, Math.max(0, i + off));
        lut[i] = Math.round(Math.round(v / 255 * (L - 1)) / (L - 1) * 255);
      }
      luts.push(lut);
    }
    const flat = new Uint8ClampedArray(work.length);
    const fl = Math.max(0, Math.min(1, p.flatten));
    for (let y = 0; y < h; y++) {
      const by = (y % 4) * 4;
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4, lut = luts[by + (x % 4)];
        flat[i]     = lut[work[i]] * fl + work[i] * (1 - fl);
        flat[i + 1] = lut[work[i + 1]] * fl + work[i + 1] * (1 - fl);
        flat[i + 2] = lut[work[i + 2]] * fl + work[i + 2] * (1 - fl);
        flat[i + 3] = 255;
      }
    }

    // 4. colour, then lay the ink over it
    const out = new Uint8ClampedArray(work.length);
    const la = p.lineAlpha * (1 - 0.35 * p.soft);
    const sat = p.sat;
    const INK = [18, 18, 26];
    for (let j = 0, i = 0; j < n; j++, i += 4) {
      let r = flat[i], g = flat[i + 1], b = flat[i + 2];
      if (p.gray) {
        const lum = r * 0.299 + g * 0.587 + b * 0.114;
        const paper = 255 - (255 - lum) * 0.45;      // light graphite shading
        r = g = b = paper;
      } else if (sat !== 1) {
        const gr = r * 0.299 + g * 0.587 + b * 0.114;
        r = gr + (r - gr) * sat; g = gr + (g - gr) * sat; b = gr + (b - gr) * sat;
      }
      const a = strength[j] * la;
      if (a > 0.001) {
        r = r * (1 - a) + INK[0] * a; g = g * (1 - a) + INK[1] * a; b = b * (1 - a) + INK[2] * a;
      }
      out[i] = r; out[i + 1] = g; out[i + 2] = b; out[i + 3] = 255;
    }
    return new ImageData(out, w, h);
  }

  /* -------------------------------------------------------- render loop */
  function render(ts) {
    if (!running) return;
    const vw = video.videoWidth, vh = video.videoHeight;
    if (vw && vh) {
      const scale = Math.min(1, maxW / vw);
      const w = Math.max(2, Math.round(vw * scale)), h = Math.max(2, Math.round(vh * scale));
      if (srcC.width !== w || srcC.height !== h) { srcC.width = outC.width = w; srcC.height = outC.height = h; }
      sctx.drawImage(video, 0, 0, w, h);
      const data = sctx.getImageData(0, 0, w, h);
      octx.putImageData(stylize(data, w, h, params()), 0, 0);
      if (ts) {
        if (lastT) { const dt = ts - lastT; if (dt > 0) { fpsEMA = fpsEMA ? fpsEMA * 0.9 + (1000 / dt) * 0.1 : 1000 / dt; $("fps").textContent = fpsEMA.toFixed(0) + " fps · " + w + "×" + h; } }
        lastT = ts;
      }
    }
    raf = requestAnimationFrame(render);
  }

  function start() {
    if (running) return;
    running = true;
    $("empty").style.display = "none";
    $("record").disabled = false; $("snap").disabled = false;
    lastT = 0;
    render();
  }
  function stop() { running = false; if (raf) cancelAnimationFrame(raf); raf = null; }

  /* ------------------------------------------------------------ sources */
  function stopWebcam() {
    const s = video.srcObject;
    if (s) { s.getTracks().forEach((t) => t.stop()); video.srcObject = null; }
  }

  $("file").addEventListener("change", (e) => {
    const f = e.target.files[0]; if (!f) return;
    stopWebcam();
    video.src = URL.createObjectURL(f); video.loop = true; video.muted = true;
    video.play().then(start).catch((err) => alert("Could not play that video: " + err.message));
  });

  $("webcam").addEventListener("click", async () => {
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: true });
      video.srcObject = s; video.loop = false; video.muted = true;
      await video.play(); start();
    } catch (err) {
      alert("Could not access the webcam: " + err.message);
    }
  });

  /* ---------------------------------------------------- compare slider */
  $("split").addEventListener("input", (e) => {
    const v = +e.target.value;
    srcC.style.clipPath = "inset(0 " + (100 - v) + "% 0 0)";
    $("divider").style.left = v + "%";
  });

  /* ----------------------------------------------------------- snapshot */
  $("snap").addEventListener("click", () => {
    if (!running) return;
    outC.toBlob((b) => {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(b); a.download = "animefy.png"; a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    }, "image/png");
  });

  /* ----------------------------------------------------------- recording */
  function sourceAudio() {
    try { if (video.captureStream) return video.captureStream().getAudioTracks(); } catch (e) {}
    return [];
  }

  $("record").addEventListener("click", () => {
    if (recorder && recorder.state === "recording") { recorder.stop(); return; }
    if (!running) return;
    if (typeof MediaRecorder === "undefined") { alert("Recording is not supported in this browser."); return; }
    const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9") ? "video/webm;codecs=vp9" : "video/webm";
    const cstream = outC.captureStream(25);
    const tracks = cstream.getVideoTracks().concat(sourceAudio());
    let stream;
    try { stream = new MediaStream(tracks); } catch (e) { stream = cstream; }
    chunks = [];
    recorder = new MediaRecorder(stream, { mimeType: mime });
    recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    recorder.onstop = () => {
      const url = URL.createObjectURL(new Blob(chunks, { type: "video/webm" }));
      const a = $("dl"); a.href = url; a.hidden = false;
      $("record").textContent = "● Record";
    };
    recorder.start();
    $("record").textContent = "■ Stop";
  });

  /* --------------------------------------------------------------- wiring */
  ["smooth", "colors", "line", "thick", "sat"].forEach((id) => $(id).addEventListener("input", syncLabels));
  $("style").addEventListener("change", (e) => applyPreset(e.target.value));
  $("res").addEventListener("change", (e) => { maxW = +e.target.value; });
  applyPreset("cartoon");

  /* -------------------------------------------- headless self-test hook */
  if (location.search.includes("selftest")) {
    const q = new URLSearchParams(location.search);
    const st = q.get("style") || "cartoon";
    const pre = STYLES[st];
    const sp = { smooth: pre.smooth, range: pre.range, colors: pre.colors, line: pre.line,
                 thick: pre.thick, sat: pre.sat, lineAlpha: pre.lineAlpha,
                 flatten: pre.flatten, gray: pre.gray, soft: pre.soft };
    const sizeCanvases = (w, h) => {
      if (srcC.width !== w || srcC.height !== h) { srcC.width = outC.width = w; srcC.height = outC.height = h; }
    };
    const run = (w, h) => {
      const d = sctx.getImageData(0, 0, w, h);
      octx.putImageData(stylize(d, w, h, sp), 0, 0);
      srcC.style.display = "none";          // show the stylised canvas only
      const e = $("empty"); if (e) e.style.display = "none";
      document.title = "selftest-done";
    };
    const img = q.get("img") || window.__SAMPLE__ || null;
    if (img) {
      const im = new Image();
      im.onload = () => {
        const w = 640, h = Math.round(640 * im.height / im.width);
        sizeCanvases(w, h);
        sctx.fillStyle = "#000"; sctx.fillRect(0, 0, w, h);
        sctx.drawImage(im, 0, 0, w, h);
        run(w, h);
      };
      im.onerror = () => { document.title = "selftest-error"; };
      im.src = img;
    } else {
      const w = 640, h = 360;
      sizeCanvases(w, h);
      const g = sctx.createLinearGradient(0, 0, w, h);
      g.addColorStop(0, "#9cc7ff"); g.addColorStop(1, "#20304a");
      sctx.fillStyle = g; sctx.fillRect(0, 0, w, h);
      sctx.fillStyle = "#2f7d4f"; sctx.fillRect(0, h * 0.7, w, h * 0.3);
      sctx.fillStyle = "#e8b23a"; sctx.beginPath();
      sctx.arc(w * 0.33, h * 0.48, h * 0.2, 0, Math.PI * 2); sctx.fill();
      sctx.fillStyle = "#c0392b"; sctx.fillRect(w * 0.6, h * 0.28, w * 0.18, h * 0.32);
      run(w, h);
    }
  }
})();
