// ── Config ────────────────────────────────────────────────────────────────────

// Two simple layout choices — each generates BOTH output formats below
const LAYOUTS = [
  { id: "1-way", name: "Single Photo", photoCount: 1 },
  { id: "2-way", name: "2-Way Split", photoCount: 2 },
  { id: "3-way", name: "3-Way Split", photoCount: 3 },
];

// Both formats are always generated
const OUTPUT_FORMATS = [
  { label: "2×1", width: 2000, height: 1000, suffix: "2x1" },
  { label: "4×3", width: 2000, height: 1500, suffix: "4x3" },
];

const DIVIDER = { width: 12, color: "#ffffff" };

// ── State ─────────────────────────────────────────────────────────────────────

const state = {
  selectedLayout: null,
  images: [],       // [{ file, dataURL }]
  focalPoints: [],  // [{ x, y, faceFound }]
  imageEls: [],     // [HTMLImageElement] — loaded once, reused for re-renders
  adjustments: [],  // [formatIndex][panelIndex] = { panX, panY, scale } — independent per format
  targetFocalY: 0.5,
  composited: false,
  canvasEls: [],    // one HTMLCanvasElement per OUTPUT_FORMAT
  drag: null,       // { formatIndex, panelIndex, startMouseX, startMouseY, startPanX, startPanY }
  showGrid: false,
};

// ── DOM refs ──────────────────────────────────────────────────────────────────

const presetGrid        = document.getElementById("preset-grid");
const uploadZonesEl     = document.getElementById("upload-zones");
const uploadHint        = document.getElementById("upload-hint");
const btnCompose        = document.getElementById("btn-compose");
const btnDownload       = document.getElementById("btn-download");
const btnReset          = document.getElementById("btn-reset");
const canvasArea        = document.getElementById("canvas-area");
const canvasPlaceholder = document.getElementById("canvas-placeholder");
const canvasLoading     = document.getElementById("canvas-loading");
const creditReminder    = document.getElementById("credit-reminder");
const creditPattern     = document.getElementById("credit-reminder-pattern");
// (per-canvas adjustment controls are built and injected by buildAdjControls())

// ── Image quality checks ───────────────────────────────────────────────────────

function getImageSize(dataURL) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload  = () => resolve({ width: img.naturalWidth, height: img.naturalHeight });
    img.onerror = () => resolve({ width: 0, height: 0 });
    img.src = dataURL;
  });
}

// Laplacian variance — measures how much edge/detail is present.
// Sharp photos score high; blurry or heavily compressed ones score low.
function measureSharpness(dataURL) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const SIZE = 150;
      const c = document.createElement("canvas");
      c.width = c.height = SIZE;
      const ctx = c.getContext("2d", { willReadFrequently: true });
      ctx.drawImage(img, 0, 0, SIZE, SIZE);
      const { data } = ctx.getImageData(0, 0, SIZE, SIZE);

      // Convert to grayscale
      const gray = new Float32Array(SIZE * SIZE);
      for (let i = 0; i < SIZE * SIZE; i++) {
        const j = i * 4;
        gray[i] = 0.299 * data[j] + 0.587 * data[j + 1] + 0.114 * data[j + 2];
      }

      // Laplacian variance
      let sum = 0, sumSq = 0, n = 0;
      for (let y = 1; y < SIZE - 1; y++) {
        for (let x = 1; x < SIZE - 1; x++) {
          const i = y * SIZE + x;
          const lap = 4 * gray[i] - gray[i - 1] - gray[i + 1] - gray[i - SIZE] - gray[i + SIZE];
          sum += lap; sumSq += lap * lap; n++;
        }
      }
      resolve(sumSq / n - (sum / n) ** 2);
    };
    img.onerror = () => resolve(Infinity); // can't measure — assume ok
    img.src = dataURL;
  });
}

// ── Face Detection ────────────────────────────────────────────────────────────

let faceApiReady = false;

async function initFaceApi() {
  try {
    await faceapi.nets.tinyFaceDetector.loadFromUri("./weights");
    faceApiReady = true;
  } catch (e) {
    console.warn("face-api.js model failed to load:", e);
  }
}

async function detectFace(dataURL) {
  if (!faceApiReady) return { x: 0.5, y: 0.5, faceFound: false };
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = async () => {
      try {
        // Pre-scale to max 1200px on the longest dimension before detection.
        // TinyFaceDetector struggles with full-body shots where the face is a
        // small fraction of a high-res image. Pre-scaling normalises face size
        // for the detector without changing the normalised (0–1) coordinates.
        const MAX_DIM = 1200;
        const longestDim = Math.max(img.naturalWidth, img.naturalHeight);
        const prescale = longestDim > MAX_DIM ? MAX_DIM / longestDim : 1;

        let detectionEl = img;
        if (prescale < 1) {
          const c = document.createElement("canvas");
          c.width  = Math.round(img.naturalWidth  * prescale);
          c.height = Math.round(img.naturalHeight * prescale);
          c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
          detectionEl = c;
        }

        const detW = detectionEl instanceof HTMLCanvasElement ? detectionEl.width  : img.naturalWidth;
        const detH = detectionEl instanceof HTMLCanvasElement ? detectionEl.height : img.naturalHeight;

        // Try multiple inputSizes: 608 catches small faces in large images;
        // smaller values (416, 224) catch large/close-up faces that fill the frame.
        let detection = null;
        for (const inputSize of [608, 416, 224]) {
          detection = await faceapi.detectSingleFace(
            detectionEl,
            new faceapi.TinyFaceDetectorOptions({ inputSize, scoreThreshold: 0.35 })
          );
          if (detection) break;
        }

        if (!detection) return resolve({ x: 0.5, y: 0.5, faceFound: false });
        const box = detection.box;
        // Normalised coords are scale-invariant — no conversion needed.
        // Use estimated eye level (38% from top of bounding box) rather than
        // face centre (50%) — eyes are what the brain uses to judge "same plane".
        resolve({
          x:     (box.x + box.width  / 2)    / detW,
          y:     (box.y + box.height * 0.38) / detH,
          faceH: box.height / detH,
          faceFound: true,
        });
      } catch {
        resolve({ x: 0.5, y: 0.5, faceFound: false });
      }
    };
    img.onerror = () => resolve({ x: 0.5, y: 0.5, faceFound: false });
    img.src = dataURL;
  });
}

// ── Layouts UI ────────────────────────────────────────────────────────────────

function renderPresets() {
  presetGrid.innerHTML = "";
  LAYOUTS.forEach((layout) => {
    const card = document.createElement("div");
    card.className = "preset-card";
    card.dataset.id = layout.id;

    // Diagram — always show 2×1 aspect ratio
    const diagram = document.createElement("div");
    diagram.className = "preset-diagram";
    diagram.style.height = "36px";
    for (let i = 0; i < layout.photoCount; i++) {
      if (i > 0) {
        const div = document.createElement("div");
        div.className = "divider";
        diagram.appendChild(div);
      }
      diagram.appendChild(Object.assign(document.createElement("div"), { className: "panel" }));
    }

    const name = document.createElement("div");
    name.className = "preset-card-name";
    name.textContent = layout.name;

    const sub = document.createElement("div");
    sub.className = "preset-card-sub";
    sub.textContent = `${layout.photoCount} photo${layout.photoCount > 1 ? "s" : ""}`;

    const formats = document.createElement("div");
    formats.className = "preset-formats";
    OUTPUT_FORMATS.forEach((f) => {
      const badge = document.createElement("span");
      badge.className = "format-badge";
      badge.textContent = `${f.label} · ${f.width}×${f.height}`;
      formats.appendChild(badge);
    });

    card.appendChild(diagram);
    card.appendChild(name);
    card.appendChild(sub);
    card.appendChild(formats);
    card.addEventListener("click", () => selectLayout(layout));
    presetGrid.appendChild(card);
  });
}

function selectLayout(layout) {
  if (state.selectedLayout && state.selectedLayout.photoCount !== layout.photoCount) {
    state.images = [];
    state.focalPoints = [];
  }
  state.selectedLayout = layout;
  document.querySelectorAll(".preset-card").forEach((c) =>
    c.classList.toggle("selected", c.dataset.id === layout.id)
  );
  renderUploadZones(layout.photoCount);
  updateComposeButton();
  resetPreview();
}

// ── Upload zones ──────────────────────────────────────────────────────────────

function renderUploadZones(count) {
  uploadHint.textContent = `Upload ${count} photo${count > 1 ? "s" : ""} — or paste from clipboard (Ctrl+V / ⌘+V)`;
  uploadZonesEl.innerHTML = "";
  for (let i = 0; i < count; i++) {
    const zone = createUploadZone(i);
    uploadZonesEl.appendChild(zone);
    if (state.images[i]) applyImageToZone(zone, state.images[i].dataURL, state.focalPoints[i], state.images[i].qualityWarning);
  }
}

function createUploadZone(index) {
  const zone = document.createElement("div");
  zone.className = "upload-zone";
  zone.dataset.index = index;

  zone.appendChild(Object.assign(document.createElement("span"), {
    className: "zone-number", textContent: `Photo ${index + 1}`,
  }));

  const label = document.createElement("div");
  label.className = "zone-label";
  label.innerHTML = `<span class="zone-icon"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg></span><span>Drop or click to browse</span>`;
  zone.appendChild(label);

  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/*";
  input.addEventListener("change", (e) => { if (e.target.files[0]) loadImageFile(e.target.files[0], index); });
  zone.appendChild(input);

  const removeBtn = document.createElement("button");
  removeBtn.className = "remove-btn";
  removeBtn.textContent = "✕";
  removeBtn.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); removeImage(index); });
  zone.appendChild(removeBtn);

  zone.addEventListener("dragover",  (e) => { e.preventDefault(); zone.classList.add("dragover"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file?.type.startsWith("image/")) loadImageFile(file, index);
  });
  return zone;
}

async function loadImageFile(file, index) {
  const reader = new FileReader();
  reader.onload = async (e) => {
    const dataURL = e.target.result;
    const [focal, dims, sharpness] = await Promise.all([
      detectFace(dataURL),
      getImageSize(dataURL),
      measureSharpness(dataURL),
    ]);
    const isTooSmall = dims.width > 0 && Math.max(dims.width, dims.height) < 1500;
    const isBlurry   = sharpness < 100;
    const qualityWarning = isTooSmall ? "low resolution"
                         : isBlurry  ? "may look soft"
                         : null;
    state.images[index] = { file, dataURL, qualityWarning };
    state.focalPoints[index] = focal;
    const zone = uploadZonesEl.querySelector(`[data-index="${index}"]`);
    if (zone) applyImageToZone(zone, dataURL, focal, qualityWarning);
    updateComposeButton();
    resetPreview();
  };
  reader.readAsDataURL(file);
}

function applyImageToZone(zone, dataURL, focal, qualityWarning = null) {
  zone.querySelector(".preview-img")?.remove();
  zone.querySelector(".zone-face-badge")?.remove();
  zone.querySelector(".zone-warn-badge")?.remove();
  const img = Object.assign(document.createElement("img"), { className: "preview-img", src: dataURL });
  zone.appendChild(img);
  zone.classList.add("has-image");
  if (focal?.faceFound) {
    zone.appendChild(Object.assign(document.createElement("span"), {
      className: "zone-face-badge", textContent: "face detected",
    }));
  }
  if (qualityWarning) {
    zone.appendChild(Object.assign(document.createElement("span"), {
      className: "zone-warn-badge", textContent: `⚠ ${qualityWarning}`,
    }));
  }
}

function removeImage(index) {
  delete state.images[index];
  delete state.focalPoints[index];
  const zone = uploadZonesEl.querySelector(`[data-index="${index}"]`);
  if (zone) {
    zone.querySelector(".preview-img")?.remove();
    zone.querySelector(".zone-face-badge")?.remove();
    zone.querySelector(".zone-warn-badge")?.remove();
    zone.classList.remove("has-image");
  }
  updateComposeButton();
  resetPreview();
}

function swapImages(i, j) {
  [state.images[i],      state.images[j]]      = [state.images[j],      state.images[i]];
  [state.focalPoints[i], state.focalPoints[j]]  = [state.focalPoints[j], state.focalPoints[i]];
  renderUploadZones(state.selectedLayout.photoCount);
  if (state.composited) {
    // Swap loaded image elements and per-format adjustments so zoom/position
    // is preserved and there's no full recompose (which causes a page jump).
    [state.imageEls[i], state.imageEls[j]] = [state.imageEls[j], state.imageEls[i]];
    state.adjustments.forEach(fmtAdjs => {
      [fmtAdjs[i], fmtAdjs[j]] = [fmtAdjs[j], fmtAdjs[i]];
    });
    buildCanvasEntries();
    renderAllCanvases();
  }
}

function updateComposeButton() {
  if (!state.selectedLayout) { btnCompose.disabled = true; return; }
  let loaded = 0;
  for (let i = 0; i < state.selectedLayout.photoCount; i++) { if (state.images[i]) loaded++; }
  btnCompose.disabled = loaded < state.selectedLayout.photoCount;
}

// ── Compositing ───────────────────────────────────────────────────────────────

async function compose() {
  const layout = state.selectedLayout;
  if (!layout) return;

  btnCompose.disabled = true;
  resetPreview(false);
  canvasLoading.hidden = false;
  canvasPlaceholder.classList.add("hidden");

  const count = layout.photoCount;

  // Shared target focal Y
  const avgY = Array.from({ length: count }, (_, i) => state.focalPoints[i]?.y ?? 0.5)
    .reduce((s, v) => s + v, 0) / count;
  state.targetFocalY = Math.max(0.25, Math.min(0.65, avgY));

  // Load image elements once
  state.imageEls = await Promise.all(
    Array.from({ length: count }, (_, i) =>
      new Promise((resolve) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.src = state.images[i].dataURL;
      })
    )
  );

  // Init per-format, per-panel adjustments — align face Y position (no size equalization)
  state.adjustments = OUTPUT_FORMATS.map((fmt) => {
    const { width, height } = fmt;
    const { width: divW } = DIVIDER;
    const slotW = Math.floor((width - divW * (count - 1)) / count);

    // 1. Base scale for each panel (just enough to cover the slot)
    const baseScales = Array.from({ length: count }, (_, i) => {
      const img = state.imageEls[i];
      return Math.max(slotW / img.width, height / img.height);
    });

    // 2. Minimum scale so Y-alignment has room to shift the image.
    //    Ensure drawH >= 150% of slot height. Always clamp to >= 1.0 so cover is maintained.
    const minScales = Array.from({ length: count }, (_, i) =>
      Math.max(1.0, (height * 1.5) / (state.imageEls[i].height * baseScales[i]))
    );

    // 3. Rendered face heights at base scale
    const renderedFaceHeights = Array.from({ length: count }, (_, i) => {
      const focal = state.focalPoints[i];
      if (!focal?.faceFound || !focal.faceH) return null;
      return focal.faceH * state.imageEls[i].height * baseScales[i];
    });

    // 4. Target face height: equalize sizes, bounded by minScale floor and 3× cap.
    const floorFaceHeights = renderedFaceHeights.map((rfh, i) =>
      rfh != null ? rfh * minScales[i] : null
    );
    const validFloorFaceH = floorFaceHeights.filter(v => v != null);
    const validRenderedFaceH = renderedFaceHeights.filter(v => v != null);
    const targetFaceHeight = validFloorFaceH.length > 1
      ? Math.min(
          Math.max(...validFloorFaceH),
          Math.min(...validRenderedFaceH) * 3.0
        )
      : null;

    const scales = Array.from({ length: count }, (_, i) => {
      const hasFace = state.focalPoints[i]?.faceFound;
      const eqScale = (targetFaceHeight != null && renderedFaceHeights[i] != null)
        ? Math.min(3.0, targetFaceHeight / renderedFaceHeights[i])
        : 1.0;
      const floor = hasFace ? minScales[i] : 1.0;
      // Always >= 1.0 so adj.scale never zooms below cover, preventing white gaps
      return Math.min(3.0, Math.max(eqScale, floor));
    });

    // 3. Find the Y range each panel can place its eye level without image-boundary clamping.
    //    Valid eye Y for panel i: [height - (1-focal.y)*drawH,  focal.y*drawH]
    //    Intersect all ranges to get a shared achievable Y.
    const faceYRanges = Array.from({ length: count }, (_, i) => {
      const focal = state.focalPoints[i];
      if (!focal?.faceFound) return null;
      const drawH = state.imageEls[i].height * baseScales[i] * scales[i];
      return { lower: height - (1 - focal.y) * drawH, upper: focal.y * drawH };
    });
    const validRanges = faceYRanges.filter(r => r != null);

    let panY = 0;
    if (validRanges.length > 1) {
      const preferred = 0.38 * height;
      const commonLower = Math.max(...validRanges.map(r => r.lower));
      const commonUpper = Math.min(...validRanges.map(r => r.upper));
      // Use intersection if it exists; otherwise aim for preferred Y anyway —
      // each panel will clamp independently, which is still better than panY=0.
      const targetFaceY = (commonLower <= commonUpper)
        ? Math.min(commonUpper, Math.max(commonLower, preferred))
        : preferred;
      panY = targetFaceY - state.targetFocalY * height;
    }

    return Array.from({ length: count }, (_, i) => ({
      panX: 0,
      panY: state.focalPoints[i]?.faceFound ? panY : 0,
      scale: scales[i],
    }));
  });

  // Build dual canvas UI
  canvasLoading.hidden = true;
  buildCanvasEntries();

  state.composited = true;
  renderAllCanvases();

  btnCompose.disabled = false;
  btnDownload.disabled = false;
  btnDownload.hidden = false;

  // Show credit line reminder
  const n = state.selectedLayout.photoCount;
  creditPattern.textContent = Array.from({ length: n }, (_, i) => `Photo ${i + 1} Credit`).join("; ");
  creditReminder.hidden = false;
}

// ── Build canvas entry elements ───────────────────────────────────────────────

function buildCanvasEntries() {
  // Remove any existing canvas entries
  canvasArea.querySelectorAll(".canvas-entry").forEach((el) => el.remove());
  state.canvasEls = [];

  OUTPUT_FORMATS.forEach((fmt, fi) => {
    const entry = document.createElement("div");
    entry.className = "canvas-entry";

    // Header row: label + download button
    const header = document.createElement("div");
    header.className = "canvas-entry-header";

    const labelEl = document.createElement("span");
    labelEl.className = "canvas-entry-label";
    labelEl.textContent = `${fmt.label} — ${fmt.width} × ${fmt.height} px`;

    const dlBtn = document.createElement("button");
    dlBtn.className = "btn-download-single";
    dlBtn.textContent = `↓ ${fmt.label}`;
    dlBtn.addEventListener("click", () => downloadOne(fi));

    header.appendChild(labelEl);
    header.appendChild(dlBtn);

    // Canvas
    const canvas = document.createElement("canvas");
    canvas.width  = fmt.width;
    canvas.height = fmt.height;
    attachCanvasDrag(canvas, fi);

    // Swap arrow overlay (pointer-events: none so drag still works;
    // only the arrow buttons themselves are pointer-events: auto)
    const wrapper = document.createElement("div");
    wrapper.className = "canvas-wrapper";
    wrapper.appendChild(canvas);

    const count = state.selectedLayout.photoCount;
    if (count > 1) {
      const overlay = document.createElement("div");
      overlay.className = "canvas-overlay";

      for (let i = 0; i < count; i++) {
        if (i > 0) {
          const divider = document.createElement("div");
          divider.className = "canvas-overlay-divider";
          overlay.appendChild(divider);
        }
        const panel = document.createElement("div");
        panel.className = "canvas-overlay-panel";

        const svgArrow = (dir) => `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${
          dir === 'left'
            ? '<line x1="19" y1="12" x2="5" y2="12"/><polyline points="12,19 5,12 12,5"/>'
            : '<line x1="5" y1="12" x2="19" y2="12"/><polyline points="12,5 19,12 12,19"/>'
        }</svg>`;

        if (i > 0) {
          const btn = document.createElement("button");
          btn.className = "canvas-swap-arrow";
          btn.innerHTML = svgArrow('left');
          btn.title = `Swap photos ${i} & ${i + 1}`;
          btn.addEventListener("click", () => swapImages(i, i - 1));
          panel.appendChild(btn);
        }
        if (i < count - 1) {
          const btn = document.createElement("button");
          btn.className = "canvas-swap-arrow";
          btn.innerHTML = svgArrow('right');
          btn.title = `Swap photos ${i + 1} & ${i + 2}`;
          btn.addEventListener("click", () => swapImages(i, i + 1));
          panel.appendChild(btn);
        }
        overlay.appendChild(panel);
      }
      wrapper.appendChild(overlay);
    }

    // Per-canvas adjustment controls
    const adjsEl = buildAdjControls(fi);

    entry.appendChild(header);
    entry.appendChild(wrapper);
    entry.appendChild(adjsEl);
    canvasArea.appendChild(entry);
    state.canvasEls.push(canvas);
  });
}

// ── Render all canvases ───────────────────────────────────────────────────────

function renderAllCanvases() {
  OUTPUT_FORMATS.forEach((fmt, fi) => renderForFormat(fi, fmt));
}

function renderForFormat(fi, fmt) {
  const canvas = state.canvasEls[fi];
  if (!canvas) return;

  const { width, height } = fmt;
  const { width: divW, color: divColor } = DIVIDER;
  const count = state.selectedLayout.photoCount;
  const slotW = Math.floor((width - divW * (count - 1)) / count);

  const ctx = canvas.getContext("2d");

  // Fill background with divider color — eliminates any sub-pixel gaps
  ctx.fillStyle = divColor;
  ctx.fillRect(0, 0, width, height);

  let x = 0;
  for (let i = 0; i < count; i++) {
    if (i > 0) x += divW;
    // Last panel takes any remaining pixels to avoid sub-pixel gaps on the right edge
    const panelW = i === count - 1 ? width - x : slotW;
    drawPanel(ctx, state.imageEls[i], x, 0, panelW, height, i, state.adjustments[fi][i]);
    x += panelW;
  }

  if (state.showGrid) drawGrid(ctx, width, height, count, slotW, divW);
}

function drawGrid(ctx, width, height, count, slotW, divW) {
  ctx.save();
  ctx.strokeStyle = "rgba(255,255,255,0.85)";
  ctx.lineWidth = 4;
  ctx.setLineDash([]);

  // Horizontal lines span full canvas at 1/3 and 2/3 height
  [1/3, 2/3].forEach(t => {
    ctx.beginPath(); ctx.moveTo(0, height * t); ctx.lineTo(width, height * t); ctx.stroke();
  });

  // Vertical lines at 1/3 and 2/3 within each panel individually
  let x = 0;
  for (let i = 0; i < count; i++) {
    if (i > 0) x += divW;
    [1/3, 2/3].forEach(t => {
      ctx.beginPath(); ctx.moveTo(x + slotW * t, 0); ctx.lineTo(x + slotW * t, height); ctx.stroke();
    });
    x += slotW;
  }

  ctx.restore();
}

// ── Draw one panel ────────────────────────────────────────────────────────────

function drawPanel(ctx, img, x, y, w, h, index, adj) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, w, h);
  ctx.clip();

  const focal = state.focalPoints[index] ?? { x: 0.5, y: 0.5 };

  const baseScale   = Math.max(w / img.width, h / img.height);
  const effectScale = baseScale * adj.scale;
  const drawW       = img.width  * effectScale;
  const drawH       = img.height * effectScale;

  // Auto-position from focal point
  let offsetX = 0.5 * w            - focal.x * drawW;
  let offsetY = state.targetFocalY * h - focal.y * drawH;

  // Apply user pan — clamp BEFORE adding so pan is measured from valid range
  const minOffX = w - drawW; // ≤ 0 (image must cover right edge)
  const minOffY = h - drawH; // ≤ 0
  offsetX = Math.min(0, Math.max(minOffX, offsetX + adj.panX));
  offsetY = Math.min(0, Math.max(minOffY, offsetY + adj.panY));

  ctx.drawImage(img, x + offsetX, y + offsetY, drawW, drawH);
  ctx.restore();
}

// ── Canvas drag interaction ───────────────────────────────────────────────────

function getPanelIndex(cssX, canvas, formatWidth) {
  const rect    = canvas.getBoundingClientRect();
  const scale   = formatWidth / rect.width;
  const cx      = cssX * scale;
  const count   = state.selectedLayout.photoCount;
  const slotW   = Math.floor((formatWidth - DIVIDER.width * (count - 1)) / count);

  let cursor = 0;
  for (let i = 0; i < count; i++) {
    if (i > 0) cursor += DIVIDER.width;
    if (cx >= cursor && cx < cursor + slotW) return i;
    cursor += slotW;
  }
  return -1;
}

function attachCanvasDrag(canvas, formatIndex) {
  const fmt = OUTPUT_FORMATS[formatIndex];

  canvas.addEventListener("mousedown", (e) => {
    if (!state.composited) return;
    const rect = canvas.getBoundingClientRect();
    const panelIndex = getPanelIndex(e.clientX - rect.left, canvas, fmt.width);
    if (panelIndex < 0) return;

    const adj = state.adjustments[formatIndex][panelIndex];
    state.drag = {
      canvas,
      formatIndex,
      formatWidth: fmt.width,
      panelIndex,
      startMouseX: e.clientX,
      startMouseY: e.clientY,
      startPanX: adj.panX,
      startPanY: adj.panY,
    };
    canvas.classList.add("dragging");
    e.preventDefault();
  });
}

window.addEventListener("mousemove", (e) => {
  if (!state.drag) return;
  const { canvas, formatIndex, formatWidth, panelIndex, startMouseX, startMouseY, startPanX, startPanY } = state.drag;
  const rect        = canvas.getBoundingClientRect();
  const cssToCanvas = formatWidth / rect.width;

  const adj = state.adjustments[formatIndex][panelIndex];
  adj.panX = startPanX + (e.clientX - startMouseX) * cssToCanvas;
  adj.panY = startPanY + (e.clientY - startMouseY) * cssToCanvas;

  // Only re-render the format being dragged
  renderForFormat(formatIndex, OUTPUT_FORMATS[formatIndex]);
});

window.addEventListener("mouseup", () => {
  if (state.drag) {
    state.drag.canvas.classList.remove("dragging");
    state.drag = null;
  }
});

// ── Per-canvas adjustment controls ───────────────────────────────────────────

function buildAdjControls(fi) {
  const count = state.selectedLayout.photoCount;
  const fmt   = OUTPUT_FORMATS[fi];

  const wrapper = document.createElement("div");
  wrapper.className = "canvas-adjs";

  const hintRow = document.createElement("div");
  hintRow.className = "canvas-adjs-hint";

  const hintText = document.createElement("span");
  hintText.textContent = "Drag image to reposition · Use sliders to zoom";

  const gridLabel = document.createElement("label");
  gridLabel.className = "grid-toggle";
  const gridCheck = document.createElement("input");
  gridCheck.type = "checkbox";
  gridCheck.checked = state.showGrid;
  gridCheck.addEventListener("change", () => {
    state.showGrid = gridCheck.checked;
    // Sync all other grid checkboxes
    document.querySelectorAll(".grid-toggle input").forEach(cb => cb.checked = state.showGrid);
    renderAllCanvases();
  });
  gridLabel.append(gridCheck, document.createTextNode(" Show grid lines"));

  hintRow.append(hintText, gridLabel);
  wrapper.appendChild(hintRow);

  for (let i = 0; i < count; i++) {
    const focal = state.focalPoints[i];
    const adj   = state.adjustments[fi][i];

    // Single compact row: [Photo N  face aligned?]  [Zoom --slider-- 1.0×]  [Reset]
    const card = document.createElement("div");
    card.className = "adj-card";

    const labelEl = document.createElement("div");
    labelEl.className = "adj-card-label";
    labelEl.textContent = `Photo ${i + 1}`;

    const zoomRow = document.createElement("div");
    zoomRow.className = "adj-row";

    const zoomLabel = Object.assign(document.createElement("label"), { textContent: "Zoom" });
    const slider = Object.assign(document.createElement("input"), {
      type: "range", min: "1", max: "3", step: "0.05", value: String(adj.scale.toFixed(2)),
    });
    const valueEl = Object.assign(document.createElement("span"), {
      className: "adj-value", textContent: adj.scale.toFixed(1) + "×",
    });

    slider.addEventListener("input", () => {
      adj.scale = parseFloat(slider.value);
      valueEl.textContent = adj.scale.toFixed(1) + "×";
      renderForFormat(fi, fmt);
    });

    zoomRow.append(zoomLabel, slider, valueEl);

    const resetBtn = Object.assign(document.createElement("button"), {
      className: "secondary adj-reset", textContent: "Reset",
    });
    resetBtn.addEventListener("click", () => {
      adj.panX = 0; adj.panY = 0; adj.scale = 1.0;
      slider.value = "1";
      valueEl.textContent = "1.0×";
      renderForFormat(fi, fmt);
    });

    // All inline in one row
    card.append(labelEl, zoomRow, resetBtn);
    wrapper.appendChild(card);
  }

  return wrapper;
}

// ── Download ──────────────────────────────────────────────────────────────────

function downloadOne(fi) {
  const fmt  = OUTPUT_FORMATS[fi];
  const name = (state.selectedLayout.name + "-" + fmt.suffix).toLowerCase().replace(/\s+/g, "-");
  const link = Object.assign(document.createElement("a"), {
    download: `composite-${name}.jpg`,
    href: state.canvasEls[fi].toDataURL("image/jpeg", 0.92),
  });
  link.click();
}

function downloadAll() {
  OUTPUT_FORMATS.forEach((fmt, fi) => {
    // Slight delay between downloads so browsers don't block multiple at once
    setTimeout(() => {
      const name = (state.selectedLayout.name + "-" + fmt.suffix).toLowerCase().replace(/\s+/g, "-");
      const link = Object.assign(document.createElement("a"), {
        download: `composite-${name}.jpg`,
        href: state.canvasEls[fi].toDataURL("image/jpeg", 0.92),
      });
      link.click();
    }, fi * 300);
  });
}

// ── Reset ─────────────────────────────────────────────────────────────────────

function resetPreview(clearComposited = true) {
  if (clearComposited) {
    state.composited  = false;
    state.imageEls    = [];
    state.adjustments = [];
    state.canvasEls   = [];
    canvasArea.querySelectorAll(".canvas-entry").forEach((el) => el.remove());
  }
  canvasPlaceholder.classList.remove("hidden");
  canvasLoading.hidden = true;
  btnDownload.disabled = true;
  btnDownload.hidden = true;
  creditReminder.hidden = true;
  state.showGrid = false;
}

function resetAll() {
  state.selectedLayout = null;
  state.images         = [];
  state.focalPoints    = [];
  document.querySelectorAll(".preset-card").forEach((c) => c.classList.remove("selected"));
  uploadHint.textContent = "Select a split first";
  uploadZonesEl.innerHTML = "";
  resetPreview();
  updateComposeButton();
}

// ── Events ────────────────────────────────────────────────────────────────────

btnCompose.addEventListener("click", compose);
btnDownload.addEventListener("click", downloadAll);
btnReset.addEventListener("click", resetAll);

// Clipboard paste — loads image into the next empty slot
document.addEventListener("paste", (e) => {
  if (!state.selectedLayout) return;
  const items = Array.from(e.clipboardData?.items ?? []);
  const imageItem = items.find((item) => item.type.startsWith("image/"));
  if (!imageItem) return;

  const count = state.selectedLayout.photoCount;
  // Find the first empty slot
  let targetIndex = -1;
  for (let i = 0; i < count; i++) {
    if (!state.images[i]) { targetIndex = i; break; }
  }
  // If all slots filled, overwrite the last one
  if (targetIndex === -1) targetIndex = count - 1;

  const file = imageItem.getAsFile();
  if (file) loadImageFile(file, targetIndex);
});

// ── Init ──────────────────────────────────────────────────────────────────────

initFaceApi();
renderPresets();
updateComposeButton();
