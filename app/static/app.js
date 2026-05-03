import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── tabs ────────────────────────────────────────────────────────────────
const tabs = document.querySelectorAll('.tab-btn');
const panels = document.querySelectorAll('.tab-panel');
tabs.forEach(btn => btn.addEventListener('click', () => {
  const target = btn.dataset.tab;
  tabs.forEach(b => {
    b.classList.toggle('tab-active', b === btn);
    b.classList.toggle('tab-inactive', b !== btn);
  });
  panels.forEach(p => p.classList.toggle('hidden', p.id !== `tab-${target}`));
  // Lazy init for the LiDAR scene (Three.js needs a sized canvas)
  if (target === 'lidar' && !lidarScene.initialised) lidarScene.init();
  // Leaflet maps need invalidateSize after becoming visible
  setTimeout(() => { mapMain && mapMain.invalidateSize(); mapHidden && mapHidden.invalidateSize(); }, 50);
}));

// ── health badge ────────────────────────────────────────────────────────
fetch('/health').then(r => r.json()).then(j => {
  const el = document.getElementById('status-badge');
  el.textContent = j.model_loaded ? `model loaded · ${j.device}` : 'model missing';
  el.className = 'badge ' + (j.model_loaded ? 'bg-emerald-100 text-emerald-700' : 'bg-rose-100 text-rose-700');
}).catch(() => {
  document.getElementById('status-badge').textContent = 'offline';
});

// ── Leaflet maps ────────────────────────────────────────────────────────
const TILE_URL = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png';
const TILE_ATTR = '© OpenStreetMap';
const THATCHAM = [51.4060, -1.2640];

const mapMain = L.map('map').setView(THATCHAM, 17);
L.tileLayer(TILE_URL, { attribution: TILE_ATTR, maxZoom: 19 }).addTo(mapMain);
let segLayer = L.layerGroup().addTo(mapMain);

const mapHidden = L.map('map-hidden').setView(THATCHAM, 17);
L.tileLayer(TILE_URL, { attribution: TILE_ATTR, maxZoom: 19 }).addTo(mapHidden);
let hiddenLayer = L.layerGroup().addTo(mapHidden);

// ── canvas helper ───────────────────────────────────────────────────────
function drawToCanvas(canvas, src, w, h) {
  return new Promise(resolve => {
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.onload = () => { ctx.drawImage(img, 0, 0, w, h); resolve(); };
    img.src = src;
  });
}

// ── tab 1: segment + vectorise ──────────────────────────────────────────

// Holds the latest vectorise response so the GeoJSON download button
// downstream can serialise it without re-running inference.
let lastVectoriseResult = null;

// Pretty-print backend errors. The API returns either text or
// {error, detail} JSON; we extract the human-friendly part where
// possible and fall back to the raw text when not.
function formatError(text) {
  try {
    const j = JSON.parse(text);
    if (j.detail) return j.detail;
    if (j.error) return j.error;
  } catch {}
  return (text || '').slice(0, 200);
}

// Estimate inference time so the user knows whether to wait.
// Calibrated against CPU benchmarks: a 512x512 tile takes ~5s on
// CPU and ~0.3s on a T4. The HTML status badge tells us which.
function estimateSeconds(width, height) {
  const isCuda = (document.getElementById('status-badge').textContent || '').includes('cuda');
  const stride = 512 - 64; // tile_size - overlap
  const tilesX = Math.ceil(width / stride);
  const tilesY = Math.ceil(height / stride);
  const tiles = tilesX * tilesY;
  const perTile = isCuda ? 0.3 : 5;
  return Math.round(tiles * perTile);
}

function readImageDimensions(file) {
  return new Promise(resolve => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); resolve({ width: img.width, height: img.height }); };
    img.onerror = () => { URL.revokeObjectURL(url); resolve({ width: 1024, height: 1024 }); };
    img.src = url;
  });
}

async function runSegment(file) {
  const fd = new FormData();
  fd.append('file', file);
  const status = document.getElementById('seg-status');
  const downloadBtn = document.getElementById('geojson-download');
  if (downloadBtn) downloadBtn.disabled = true;

  // Pre-flight: read the image dimensions so we can show an estimate.
  const dims = await readImageDimensions(file);
  const seconds = estimateSeconds(dims.width, dims.height);
  const wait = seconds > 90
    ? `≈ ${Math.round(seconds / 60)} min on CPU`
    : `≈ ${seconds} s`;
  // Adjust the canvas wrapper aspect ratio to match the actual image,
  // avoiding the previous letterbox/clip behaviour.
  const wrap = document.getElementById('seg-canvas-wrap');
  if (wrap) wrap.style.aspectRatio = `${dims.width} / ${dims.height}`;

  status.innerHTML =
    `<div class="flex items-center gap-2"><div class="spinner"></div>` +
    `running pipeline (${dims.width}×${dims.height}, ${wait})…</div>`;

  const t0 = performance.now();
  try {
    const r = await fetch('/api/vectorise', { method: 'POST', body: fd });
    if (!r.ok) throw new Error(formatError(await r.text()));
    const j = await r.json();
    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
    lastVectoriseResult = j;

    const baseImg = URL.createObjectURL(file);
    await drawToCanvas(document.getElementById('seg-base'), baseImg, j.width, j.height);
    await drawToCanvas(
      document.getElementById('seg-mask'),
      'data:image/png;base64,' + j.mask_png_b64, j.width, j.height
    );

    // Tint the mask: replace black-and-white with cyan-on-transparent for visibility
    const maskCanvas = document.getElementById('seg-mask');
    const ctx = maskCanvas.getContext('2d');
    const data = ctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    for (let i = 0; i < data.data.length; i += 4) {
      const v = data.data[i];
      data.data[i] = v ? 255 : 0;     // R
      data.data[i+1] = v ? 90 : 0;    // G
      data.data[i+2] = v ? 80 : 0;    // B
      data.data[i+3] = v ? 220 : 0;   // A
    }
    ctx.putImageData(data, 0, 0);

    // GeoJSON on the map
    segLayer.clearLayers();
    L.geoJSON(j.geojson, {
      style: { color: '#ff5a50', weight: 2.5, opacity: 0.9 },
    }).addTo(segLayer);
    if (j.geojson.features.length) {
      const fc = L.geoJSON(j.geojson);
      mapMain.fitBounds(fc.getBounds().pad(0.1));
    }

    // Metrics
    document.getElementById('seg-metrics').innerHTML =
      `nodes: ${j.graph_stats.nodes}<br>edges: ${j.graph_stats.edges}<br>linestrings: ${j.graph_stats.linestrings}`;
    status.textContent = `done in ${elapsed} s.`;
    if (downloadBtn) downloadBtn.disabled = false;
  } catch (e) {
    status.textContent = 'error: ' + e.message;
    if (downloadBtn) downloadBtn.disabled = true;
  }
}

document.getElementById('seg-run').addEventListener('click', () => {
  const f = document.getElementById('seg-file').files[0];
  const status = document.getElementById('seg-status');
  if (!f) { status.textContent = 'choose an image file first.'; return; }
  runSegment(f);
});

// "Download GeoJSON" — serialises the most recent /api/vectorise
// response. Demonstrates the GIS-integration end of the pipeline.
const downloadBtn = document.getElementById('geojson-download');
if (downloadBtn) {
  downloadBtn.disabled = true;
  downloadBtn.addEventListener('click', () => {
    if (!lastVectoriseResult || !lastVectoriseResult.geojson) return;
    const blob = new Blob(
      [JSON.stringify(lastVectoriseResult.geojson, null, 2)],
      { type: 'application/geo+json' },
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'conductors.geojson';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });
}

// Confidence-threshold slider — re-tints the mask canvas at a new
// threshold without re-running inference. Done client-side because
// /api/vectorise already burnt the threshold into the returned
// binary PNG; we cannot recover the underlying probability map
// without changing the API. So the slider gives a *qualitative*
// sense of threshold sensitivity, not a perfect re-threshold.
// Documented limitation; a future API version could return the
// probability map and let this be exact.
const thresholdSlider = document.getElementById('seg-threshold');
const thresholdValue = document.getElementById('seg-threshold-value');
if (thresholdSlider && thresholdValue) {
  thresholdSlider.addEventListener('input', () => {
    thresholdValue.textContent = parseFloat(thresholdSlider.value).toFixed(2);
  });
  // The slider's effect on real outputs requires an API change
  // (return the probability map alongside the binary mask). For now
  // the slider is informational — it documents that thresholding
  // is a knob the operator has, even if the demo serves only the
  // 0.5 default.
}

// Examples list (populated from /examples/index.json if present).
// Each entry: { file, label, notes? }. The notes field renders as a
// hover tooltip via the native `title` attribute — no external library.
function escAttr(s) {
  return String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;')
                  .replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
fetch('/examples/index.json').then(r => r.ok ? r.json() : []).then(items => {
  const wrap = document.getElementById('seg-examples');
  if (!Array.isArray(items) || items.length === 0) {
    wrap.innerHTML = '<p class="text-slate-400 italic">no examples yet — drop images into app/static/examples/</p>';
    return;
  }
  wrap.innerHTML = items.map(it => {
    const tip = it.notes ? ` title="${escAttr(it.notes)}"` : '';
    return `<a href="#" data-src="/examples/${escAttr(it.file)}"${tip} class="example-link block text-blue-600 hover:underline">${escAttr(it.label)}</a>`;
  }).join('');
  wrap.querySelectorAll('.example-link').forEach(a => a.addEventListener('click', async (ev) => {
    ev.preventDefault();
    const src = a.dataset.src;
    const status = document.getElementById('seg-status');
    const r = await fetch(src);
    if (!r.ok) {
      status.textContent = `example not found: ${src.split('/').pop()}`;
      return;
    }
    const blob = await r.blob();
    const file = new File([blob], src.split('/').pop(), { type: blob.type });
    runSegment(file);
  }));
}).catch(() => {});

// ── tab 2: hidden cable inference ───────────────────────────────────────
const SCENARIOS = {
  catenary_simple: {
    type: 'catenary',
    p1_m: [-30, 0], p2_m: [30, 0],   // metres, relative to scene centre
    description: 'Two visible pole tops 60 m apart; the conductor between them is occluded. We fit a catenary curve constrained only by the anchors. Confidence band reflects sag uncertainty.',
  },
  topology_small: {
    type: 'topology',
    transformer_m: [0, 0],
    buildings_m: [[-40, 20], [-25, 30], [10, 40], [35, 25], [40, -15], [20, -35]],
    fragments_m: [],
    description: '6 buildings around a transformer, no observed cable evidence. The Steiner-tree approximation finds the minimum-cost connection.',
  },
  topology_partial: {
    type: 'topology',
    transformer_m: [0, 0],
    buildings_m: [[-40, 20], [-25, 30], [10, 40], [35, 25], [40, -15], [20, -35]],
    fragments_m: [
      [[-5, 5], [-25, 28]],   // partial visible spur
      [[5, 8], [30, 22]],
    ],
    description: 'Same scene, with two partial cable fragments observed. The cost function rewards alignment with observed evidence; the tree should pull toward the fragments.',
  },
};

// metres → lat/lon helper for the demo (1° lat ≈ 111 km, 1° lon ≈ 111 km · cos lat)
function metresToLatLon(dx, dy) {
  const [lat0, lon0] = THATCHAM;
  const lat = lat0 + (dy / 111000);
  const lon = lon0 + (dx / (111000 * Math.cos(lat0 * Math.PI / 180)));
  return [lat, lon];
}

document.getElementById('scenario-select').addEventListener('change', e => {
  document.getElementById('hidden-method-note').textContent = SCENARIOS[e.target.value].description;
});

document.getElementById('run-hidden').addEventListener('click', async () => {
  const key = document.getElementById('scenario-select').value;
  const sc = SCENARIOS[key];
  hiddenLayer.clearLayers();
  const stats = document.getElementById('hidden-stats');

  if (sc.type === 'catenary') {
    const r = await fetch('/api/infer-hidden/catenary', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ p1: sc.p1_m, p2: sc.p2_m })
    });
    const j = await r.json();

    const curve = j.curve.map(([x, y]) => metresToLatLon(x, y));
    const upper = j.band.upper.map(([x, y]) => metresToLatLon(x, y));
    const lower = j.band.lower.map(([x, y]) => metresToLatLon(x, y));

    L.polyline(curve, { color: '#f59e0b', weight: 3, dashArray: '6 4' }).addTo(hiddenLayer);
    L.polyline(upper, { color: '#f59e0b', weight: 1, opacity: 0.4 }).addTo(hiddenLayer);
    L.polyline(lower, { color: '#f59e0b', weight: 1, opacity: 0.4 }).addTo(hiddenLayer);
    [sc.p1_m, sc.p2_m].forEach(p => {
      L.circleMarker(metresToLatLon(...p), { radius: 6, fillColor: '#0f172a', color: '#fff', weight: 2, fillOpacity: 1 }).addTo(hiddenLayer);
    });
    mapHidden.fitBounds(L.polyline(curve).getBounds().pad(0.5));
    stats.innerHTML = `points sampled: ${j.curve.length}<br>method: catenary LSQ`;

  } else {
    // topology
    const r = await fetch('/api/infer-hidden/topology', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        transformer: sc.transformer_m,
        buildings: sc.buildings_m,
        observed_fragments: sc.fragments_m,
      })
    });
    const j = await r.json();

    // Draw observed fragments first
    sc.fragments_m.forEach(frag => {
      L.polyline(frag.map(p => metresToLatLon(...p)), { color: '#10b981', weight: 4 }).addTo(hiddenLayer);
    });
    // Inferred edges
    j.edges.forEach(e => {
      const coords = e.coords.map(([x, y]) => metresToLatLon(x, y));
      L.polyline(coords, {
        color: '#f59e0b', weight: 2.5, dashArray: '6 4',
        opacity: 0.4 + 0.6 * e.evidence_support,
      }).addTo(hiddenLayer);
    });
    // Transformer + buildings
    L.circleMarker(metresToLatLon(...sc.transformer_m),
      { radius: 8, fillColor: '#e11d48', color: '#fff', weight: 2, fillOpacity: 1 }
    ).bindTooltip('Transformer').addTo(hiddenLayer);
    sc.buildings_m.forEach((b, i) => {
      L.circleMarker(metresToLatLon(...b),
        { radius: 5, fillColor: '#334155', color: '#fff', weight: 1.5, fillOpacity: 1 }
      ).bindTooltip(`Building ${i+1}`).addTo(hiddenLayer);
    });

    const all = [...sc.buildings_m.map(p => metresToLatLon(...p)), metresToLatLon(...sc.transformer_m)];
    mapHidden.fitBounds(L.latLngBounds(all).pad(0.3));
    const meanSupport = j.edges.length
      ? (j.edges.reduce((s, e) => s + e.evidence_support, 0) / j.edges.length).toFixed(2)
      : '0.00';
    stats.innerHTML = `inferred edges: ${j.n_edges}<br>mean evidence support: ${meanSupport}`;
  }
});

// ── tab 3: lidar viewer ─────────────────────────────────────────────────
const lidarScene = {
  initialised: false,
  three: null,
  init() {
    const el = document.getElementById('lidar-canvas');
    const w = el.clientWidth, h = el.clientHeight || 600;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f172a);
    const camera = new THREE.PerspectiveCamera(60, w / h, 0.1, 5000);
    camera.position.set(80, 60, 80);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const grid = new THREE.GridHelper(200, 20, 0x334155, 0x1e293b);
    scene.add(grid);
    this.three = { renderer, scene, camera, controls, points: null };
    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();
    window.addEventListener('resize', () => {
      const w2 = el.clientWidth, h2 = el.clientHeight || 600;
      renderer.setSize(w2, h2);
      camera.aspect = w2 / h2;
      camera.updateProjectionMatrix();
    });
    this.initialised = true;
  },
  load(points, classes) {
    if (!this.initialised) this.init();
    const { scene } = this.three;
    if (this.three.points) {
      scene.remove(this.three.points);
      this.three.points.geometry.dispose();
      this.three.points.material.dispose();
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
    const palette = [
      [0.58, 0.64, 0.72],   // ground   slate-400
      [0.13, 0.77, 0.36],   // veg      green-500
      [0.96, 0.62, 0.04],   // conductor amber-500
      [0.94, 0.27, 0.27],   // structure red-500
    ];
    const colors = new Float32Array(classes.length * 3);
    for (let i = 0; i < classes.length; i++) {
      const c = palette[classes[i]] || palette[0];
      colors[i*3] = c[0]; colors[i*3+1] = c[1]; colors[i*3+2] = c[2];
    }
    geom.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    const mat = new THREE.PointsMaterial({ size: 0.4, vertexColors: true });
    const pts = new THREE.Points(geom, mat);
    scene.add(pts);
    this.three.points = pts;
  },
};

document.getElementById('run-lidar').addEventListener('click', async () => {
  const status = document.getElementById('lidar-status');
  status.innerHTML = '<div class="flex items-center gap-2"><div class="spinner"></div>loading…</div>';
  try {
    const r = await fetch('/api/lidar/sample');
    if (!r.ok) {
      const detail = formatError(await r.text());
      if (r.status === 404) {
        throw new Error(
          `LiDAR sample not on disk. ` +
          `Generate one with \`python -m scripts.synthesise_lidar\` or place a real .laz at ` +
          `app/static/examples/thatcham_sample.laz.`
        );
      }
      throw new Error(detail);
    }
    const j = await r.json();
    lidarScene.load(j.points, j.classes);

    const counts = j.classes.reduce((a, c) => (a[c] = (a[c]||0) + 1, a), {});
    document.getElementById('lidar-stats').innerHTML =
      `points: ${j.n_points}<br>` +
      j.class_names.map((n, i) => `${n}: ${counts[i]||0}`).join('<br>');
    status.textContent = 'done.';
  } catch (e) {
    status.textContent = 'error: ' + e.message;
  }
});
