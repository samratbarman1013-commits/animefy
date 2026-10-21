/* animefy — client-side video -> anime/cartoon.
   Pure Canvas2D + typed arrays; runs entirely in the browser. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const srcC = $("src"), outC = $("out");
  const sctx = srcC.getContext("2d", { willReadFrequently: true });
  const octx = outC.getContext("2d");
  const video = $("video");
  const MAXW = 640;

  let running = false, raf = null, recorder = null, chunks = [];

  /* ---------- style presets ---------- */
  const PRESETS = {
    cartoon: { smooth: 3, colors: 8, line: 55, sat: 11 },
    anime:   { smooth: 5, colors: 5, line: 45, sat: 15 },
  };

  function params() {
    const style = $("style").value;
    return {
      style,
      blur: +$("smooth").value,
      levels: +$("colors").value,
      thr: +$("line").value,
      sat: +$("sat").value / 10,
      lineAlpha: style === "anime" ? 0.7 : 1.0,
      extraSat: style === "anime" ? 1.12 : 1.0,
    };
  }

  function syncLabels() {
    $("v-smooth").textContent = $("smooth").value;
    $("v-colors").textContent = $("colors").value;
    $("v-line").textContent = $("line").value;
    $("v-sat").textContent = (+$("sat").value / 10).toFixed(1);
  }

  function applyPreset(style) {
    const p = PRESETS[style];
    $("smooth").value = p.smooth; $("colors").value = p.colors;
    $("line").value = p.line; $("sat").value = p.sat;
    syncLabels();
  }

  /* ---------- image ops ---------- */
  function blurH(src, dst, w, h, r) {
    const win = 2 * r + 1;
    for (let y = 0; y < h; y++) {
      const row = y * w * 4;
      let sr = 0, sg = 0, sb = 0;
      for (let k = -r; k <= r; k++) {
        const xx = Math.min(w - 1, Math.max(0, k));
        const i = row + xx * 4; sr += src[i]; sg += src[i + 1]; sb += src[i + 2];
      }
      for (let x = 0; x < w; x++) {
        const o = row + x * 4;
        dst[o] = sr / win; dst[o + 1] = sg / win; dst[o + 2] = sb / win; dst[o + 3] = 255;
        const ia = row + Math.min(w - 1, x + r + 1) * 4, is = row + Math.max(0, x - r) * 4;
        sr += src[ia] - src[is]; sg += src[ia + 1] - src[is + 1]; sb += src[ia + 2] - src[is + 2];
      }
    }
  }

  function blurV(src, dst, w, h, r) {
    const win = 2 * r + 1;
    for (let x = 0; x < w; x++) {
      let sr = 0, sg = 0, sb = 0;
      for (let k = -r; k <= r; k++) {
        const yy = Math.min(h - 1, Math.max(0, k));
        const i = (yy * w + x) * 4; sr += src[i]; sg += src[i + 1]; sb += src[i + 2];
      }
      for (let y = 0; y < h; y++) {
        const o = (y * w + x) * 4;
        dst[o] = sr / win; dst[o + 1] = sg / win; dst[o + 2] = sb / win; dst[o + 3] = 255;
        const ia = (Math.min(h - 1, y + r + 1) * w + x) * 4, is = (Math.max(0, y - r) * w + x) * 4;
        sr += src[ia] - src[is]; sg += src[ia + 1] - src[is + 1]; sb += src[ia + 2] - src[is + 2];
      }
    }
  }

  function boxBlur(data, w, h, r, passes) {
    if (r < 1) return;
    const tmp = new Uint8ClampedArray(data.length);
    for (let p = 0; p < passes; p++) { blurH(data, tmp, w, h, r); blurV(tmp, data, w, h, r); }
  }

  function boxBlurGray(g, w, h, r) {
    if (r < 1) return;
    const win = 2 * r + 1;
    const t = new Float32Array(w * h);
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

  /* ---------- the effect ---------- */
  function stylize(srcData, w, h, p) {
    const n = w * h;
    const work = new Uint8ClampedArray(srcData.data);

    // 1. smooth (two box-blur passes approximate a gaussian)
    boxBlur(work, w, h, p.blur, 2);

    // 2. detect edges on the *smoothed* image, BEFORE quantising - otherwise
    //    we would ink the boundaries of the posterisation bands themselves.
    const gray = new Uint8Array(n);
    for (let i = 0, j = 0; i < work.length; i += 4, j++) {
      gray[j] = (work[i] * 0.299 + work[i + 1] * 0.587 + work[i + 2] * 0.114) | 0;
    }

    // Blur the edge channel harder than the colour channel, so fine texture
    // (fur, foliage) does not flood the frame with ink - only real contours
    // survive to become lines.
    boxBlurGray(gray, w, h, p.blur + 2);

    const edge = new Uint8Array(n), thr = p.thr;
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x;
        const gx = -gray[i - w - 1] - 2 * gray[i - 1] - gray[i + w - 1]
                 + gray[i - w + 1] + 2 * gray[i + 1] + gray[i + w + 1];
        const gy = -gray[i - w - 1] - 2 * gray[i - w] - gray[i - w + 1]
                 + gray[i + w - 1] + 2 * gray[i + w] + gray[i + w + 1];
        if (Math.abs(gx) + Math.abs(gy) > thr) edge[i] = 1;
      }
    }

    // 3. posterise, with a little ordered dithering so smooth gradients
    //    break into a fine pattern instead of hard bands.
    const L = Math.max(2, p.levels | 0), stepv = 255 / (L - 1);
    const bayer = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];
    const luts = [];
    for (let b = 0; b < 16; b++) {
      const lut = new Uint8ClampedArray(256), off = (bayer[b] / 16 - 0.5) * stepv;
      for (let i = 0; i < 256; i++) {
        const v = Math.min(255, Math.max(0, i + off));
        lut[i] = Math.round(Math.round(v / 255 * (L - 1)) / (L - 1) * 255);
      }
      luts.push(lut);
    }
    for (let y = 0; y < h; y++) {
      const by = (y % 4) * 4;
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4, lut = luts[by + (x % 4)];
        work[i] = lut[work[i]]; work[i + 1] = lut[work[i + 1]]; work[i + 2] = lut[work[i + 2]];
      }
    }

    // 4. lay the ink over the flat colour, then push saturation.
    const out = new Uint8ClampedArray(work.length);
    const la = p.lineAlpha, sat = p.sat * p.extraSat;
    for (let j = 0, i = 0; j < n; j++, i += 4) {
      let r = work[i], g = work[i + 1], b = work[i + 2];
      if (sat !== 1) {
        const gr = r * 0.299 + g * 0.587 + b * 0.114;
        r = gr + (r - gr) * sat; g = gr + (g - gr) * sat; b = gr + (b - gr) * sat;
      }
      if (edge[j]) {
        r = r * (1 - la) + 16 * la; g = g * (1 - la) + 16 * la; b = b * (1 - la) + 22 * la;
      }
      out[i] = r; out[i + 1] = g; out[i + 2] = b; out[i + 3] = 255;
    }
    return new ImageData(out, w, h);
  }

  /* ---------- render loop ---------- */
  function render() {
    if (!running) return;
    const vw = video.videoWidth, vh = video.videoHeight;
    if (vw && vh) {
      const scale = Math.min(1, MAXW / vw);
      const w = Math.max(2, Math.round(vw * scale)), h = Math.max(2, Math.round(vh * scale));
      if (srcC.width !== w || srcC.height !== h) { srcC.width = outC.width = w; srcC.height = outC.height = h; }
      sctx.drawImage(video, 0, 0, w, h);
      const data = sctx.getImageData(0, 0, w, h);
      octx.putImageData(stylize(data, w, h, params()), 0, 0);
    }
    raf = requestAnimationFrame(render);
  }

  function start() {
    if (running) return;
    running = true;
    $("record").disabled = false;
    render();
  }
  function stop() {
    running = false;
    if (raf) cancelAnimationFrame(raf);
    raf = null;
  }

  /* ---------- sources ---------- */
  function stopWebcam() {
    const s = video.srcObject;
    if (s) { s.getTracks().forEach((t) => t.stop()); video.srcObject = null; }
  }

  $("file").addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (!f) return;
    stopWebcam();
    video.src = URL.createObjectURL(f);
    video.loop = true; video.muted = true;
    video.play().then(start).catch((err) => alert("Could not play that video: " + err.message));
  });

  $("webcam").addEventListener("click", async () => {
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
      video.srcObject = s; video.loop = false; video.muted = true;
      await video.play();
      start();
    } catch (err) {
      alert("Could not access the webcam: " + err.message);
    }
  });

  /* ---------- recording ---------- */
  $("record").addEventListener("click", () => {
    if (recorder && recorder.state === "recording") { recorder.stop(); return; }
    if (!running) return;
    const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
      ? "video/webm;codecs=vp9" : "video/webm";
    chunks = [];
    recorder = new MediaRecorder(outC.captureStream(25), { mimeType: mime });
    recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    recorder.onstop = () => {
      const url = URL.createObjectURL(new Blob(chunks, { type: "video/webm" }));
      const a = $("download"); a.href = url; a.hidden = false;
      $("record").textContent = "● Record";
    };
    recorder.start();
    $("record").textContent = "■ Stop";
  });

  /* ---------- UI wiring ---------- */
  ["smooth", "colors", "line", "sat"].forEach((id) =>
    $(id).addEventListener("input", syncLabels));
  $("style").addEventListener("change", (e) => applyPreset(e.target.value));

  applyPreset("cartoon");

  /* ---------- headless self-test hook (?selftest=1) ---------- */
  if (location.search.includes("selftest")) {
    const q = new URLSearchParams(location.search);
    const st = q.get("style") || "cartoon";
    const pre = PRESETS[st];
    const sp = { style: st, blur: pre.smooth, levels: pre.colors, thr: pre.line, sat: pre.sat / 10,
                 lineAlpha: st === "anime" ? 0.7 : 1, extraSat: st === "anime" ? 1.12 : 1 };
    const sizeCanvases = (w, h) => {
      if (srcC.width !== w || srcC.height !== h) { srcC.width = outC.width = w; srcC.height = outC.height = h; }
    };
    const run = (w, h) => {
      const d = sctx.getImageData(0, 0, w, h);
      octx.putImageData(stylize(d, w, h, sp), 0, 0);
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
