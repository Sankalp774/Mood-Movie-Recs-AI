/**
 * Mood Movie Recs frontend
 * - OpenCV emotion via /api/emotion (webcam frame)
 * - TF-IDF recs (API or static)
 * - Match loading bar + 4:3 Netflix glass stack (linked-list nav)
 */

const MOODS = [
  { id: "cozy", label: "Cozy", emoji: "☕" },
  { id: "tense", label: "Tense", emoji: "😰" },
  { id: "romantic", label: "Romantic", emoji: "💕" },
  { id: "mind-bending", label: "Mind-bending", emoji: "🧠" },
  { id: "adrenaline", label: "Adrenaline", emoji: "⚡" },
  { id: "nostalgic", label: "Nostalgic", emoji: "📼" },
  { id: "inspirational", label: "Inspirational", emoji: "✨" },
  { id: "dark", label: "Dark", emoji: "🌑" },
  { id: "whimsical", label: "Whimsical", emoji: "🎩" },
  { id: "emotional", label: "Emotional", emoji: "💧" },
  { id: "curious", label: "Curious", emoji: "🔍" },
  { id: "epic", label: "Epic", emoji: "🏔️" },
  { id: "intense", label: "Intense", emoji: "🔥" },
  { id: "reflective", label: "Reflective", emoji: "🪞" },
];

const state = {
  movies: [],
  mode: "static",
  selectedMood: "cozy",
  liked: loadSet("mmr_liked"),
  watchlist: loadSet("mmr_watchlist"),
  ratings: loadObj("mmr_ratings"),
  index: null,
  lastRecs: [],
  stack: { head: null, nodes: [], index: 0 }, // linked-list style
  camStream: null,
  apiBase: null,
};

/* ---------- storage ---------- */
function loadSet(key) {
  try { return new Set(JSON.parse(localStorage.getItem(key) || "[]")); }
  catch { return new Set(); }
}
function saveSet(key, set) { localStorage.setItem(key, JSON.stringify([...set])); }
function loadObj(key) {
  try { return JSON.parse(localStorage.getItem(key) || "{}"); }
  catch { return {}; }
}
function saveObj(key, obj) { localStorage.setItem(key, JSON.stringify(obj)); }

function toast(msg) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 2400);
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

/* ---------- TF-IDF static engine ---------- */
function tokenize(text) {
  return (text || "").toLowerCase().replace(/[^a-z0-9\s-]/g, " ").split(/\s+/).filter(Boolean);
}
function buildIndex(movies) {
  const docs = movies.map((m) =>
    tokenize([m.title, ...(m.genres || []), ...(m.moods || []), m.overview].join(" "))
  );
  const df = new Map();
  docs.forEach((toks) => new Set(toks).forEach((t) => df.set(t, (df.get(t) || 0) + 1)));
  const vocab = [...df.keys()];
  const vIndex = new Map(vocab.map((t, i) => [t, i]));
  const N = docs.length;
  const matrix = docs.map((toks) => {
    const tf = new Map();
    toks.forEach((t) => tf.set(t, (tf.get(t) || 0) + 1));
    const vec = new Float32Array(vocab.length);
    let norm = 0;
    for (const [t, c] of tf) {
      const i = vIndex.get(t);
      if (i === undefined) continue;
      const idf = Math.log((N + 1) / ((df.get(t) || 0) + 1)) + 1;
      const val = (c / toks.length) * idf;
      vec[i] = val;
      norm += val * val;
    }
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < vec.length; i++) vec[i] /= norm;
    return vec;
  });
  return { vocab, vIndex, matrix, df, N };
}
function vectorizeQuery(text, index) {
  const toks = tokenize(text);
  const tf = new Map();
  toks.forEach((t) => tf.set(t, (tf.get(t) || 0) + 1));
  const vec = new Float32Array(index.vocab.length);
  let norm = 0;
  for (const [t, c] of tf) {
    const i = index.vIndex.get(t);
    if (i === undefined) continue;
    const idf = Math.log((index.N + 1) / ((index.df.get(t) || 0) + 1)) + 1;
    const val = (c / Math.max(toks.length, 1)) * idf;
    vec[i] = val;
    norm += val * val;
  }
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < vec.length; i++) vec[i] /= norm;
  return vec;
}
function cosine(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
function localRecommend({ mood, query, topK = 8 }) {
  const parts = [];
  if (mood) parts.push(mood.replace(/-/g, " "));
  if (query) parts.push(query);
  for (const id of state.liked) {
    const m = state.movies.find((x) => x.id === id);
    if (m) parts.push([...(m.moods || []), ...(m.genres || []), m.overview].join(" "));
  }
  const qVec = vectorizeQuery(parts.join(" ") || "cozy warm light", state.index);
  const scored = state.movies.map((m, i) => {
    const sim = cosine(qVec, state.index.matrix[i]);
    const score = 0.85 * sim + 0.15 * ((m.rating || 7) / 10);
    return { ...m, score, match_pct: Math.round(score * 100) };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored.filter((m) => !state.liked.has(m.id)).slice(0, topK);
}

/* ---------- Linked-list stack model ---------- */
function buildLinkedList(recs) {
  // nodes: { movie, next, prev }
  const nodes = recs.map((movie) => ({ movie, next: null, prev: null }));
  for (let i = 0; i < nodes.length; i++) {
    if (i > 0) nodes[i].prev = nodes[i - 1];
    if (i < nodes.length - 1) nodes[i].next = nodes[i + 1];
  }
  state.stack = { head: nodes[0] || null, nodes, index: 0 };
}

/* ---------- Match bar ---------- */
function showMatchOverlay(sub) {
  const ov = document.getElementById("match-overlay");
  document.getElementById("match-sub").textContent = sub || "Scoring catalog against your vibe";
  document.getElementById("match-bar-fill").style.width = "0%";
  document.getElementById("match-pct").textContent = "0%";
  ov.classList.add("open");
  ov.setAttribute("aria-hidden", "false");
}
function setMatchProgress(p) {
  const pct = Math.max(0, Math.min(100, Math.round(p)));
  document.getElementById("match-bar-fill").style.width = pct + "%";
  document.getElementById("match-pct").textContent = pct + "%";
}
function hideMatchOverlay() {
  const ov = document.getElementById("match-overlay");
  ov.classList.remove("open");
  ov.setAttribute("aria-hidden", "true");
}
async function animateMatchBar(durationMs = 1200) {
  showMatchOverlay();
  const start = performance.now();
  return new Promise((resolve) => {
    function tick(now) {
      const t = Math.min(1, (now - start) / durationMs);
      // ease-out curve with slight overshoot feel
      const eased = 1 - Math.pow(1 - t, 3);
      setMatchProgress(eased * 100);
      if (t < 1) requestAnimationFrame(tick);
      else resolve();
    }
    requestAnimationFrame(tick);
  });
}

/* ---------- Stack UI ---------- */
function openStack(startIndex = 0) {
  if (!state.lastRecs.length) {
    toast("Run recommendations first");
    return;
  }
  buildLinkedList(state.lastRecs);
  state.stack.index = Math.min(startIndex, state.stack.nodes.length - 1);
  document.getElementById("stack-overlay").classList.add("open");
  document.getElementById("stack-overlay").setAttribute("aria-hidden", "false");
  renderStack();
}
function closeStack() {
  document.getElementById("stack-overlay").classList.remove("open");
  document.getElementById("stack-overlay").setAttribute("aria-hidden", "true");
}
function stackGo(dir) {
  const n = state.stack.nodes.length;
  if (!n) return;
  const next = state.stack.index + dir;
  if (next < 0 || next >= n) return;
  state.stack.index = next;
  renderStack(dir);
}
function renderStack(dir = 0) {
  const deck = document.getElementById("stack-deck");
  const nodes = state.stack.nodes;
  const i = state.stack.index;
  deck.innerHTML = "";

  // paint current + 2 behind (linked-list depth illusion)
  const order = [i + 2, i + 1, i].filter((x) => x >= 0 && x < nodes.length);
  order.forEach((idx) => {
    const m = nodes[idx].movie;
    const card = document.createElement("article");
    card.className = "stack-card glass";
    if (idx === i) card.classList.add("is-active");
    else if (idx === i + 1) card.classList.add("is-behind-1");
    else card.classList.add("is-behind-2");

    const match = m.match_pct ?? Math.round((m.score || 0) * 100);
    const poster = document.createElement("img");
    poster.className = "sc-poster";
    poster.alt = m.title;
    poster.src = m.poster;
    poster.onerror = () => {
      poster.style.background = `linear-gradient(160deg, ${m.color || "#243049"}, #0b1020)`;
      poster.removeAttribute("src");
    };

    const body = document.createElement("div");
    body.className = "sc-body";
    body.innerHTML = `
      <div class="stack-match">${match}% match</div>
      <div class="stack-match-bar"><i style="width:${match}%"></i></div>
      <h2>${escapeHtml(m.title)}</h2>
      <div class="meta">
        <span class="rating">★ ${Number(m.rating || 0).toFixed(1)}</span>
        <span>${m.year || ""}</span>
        <span>${m.runtime || "?"} min</span>
      </div>
      <div class="tags">
        ${(m.moods || []).slice(0, 4).map((g) => `<span class="tag mood-tag">${escapeHtml(g)}</span>`).join("")}
        ${(m.genres || []).slice(0, 3).map((g) => `<span class="tag">${escapeHtml(g)}</span>`).join("")}
      </div>
      <p style="color:var(--muted);font-size:0.92rem;line-height:1.45;margin:0">${escapeHtml(m.overview || "")}</p>
      <div class="sc-actions">
        <button class="btn" type="button" data-act="like">${state.liked.has(m.id) ? "♥ Liked" : "♡ Like"}</button>
        <button class="btn secondary" type="button" data-act="wl">${state.watchlist.has(m.id) ? "✓ Watchlist" : "+ Watchlist"}</button>
        <button class="btn secondary" type="button" data-act="detail">Details</button>
      </div>`;
    body.querySelector('[data-act="like"]').onclick = (e) => {
      e.stopPropagation();
      toggleLike(m.id);
      renderStack();
      renderLiked();
    };
    body.querySelector('[data-act="wl"]').onclick = (e) => {
      e.stopPropagation();
      toggleWatchlist(m.id);
      renderStack();
      renderWatchlist();
    };
    body.querySelector('[data-act="detail"]').onclick = (e) => {
      e.stopPropagation();
      openModal(m.id);
    };

    card.appendChild(poster);
    card.appendChild(body);
    deck.appendChild(card);
  });

  document.getElementById("stack-counter").textContent = `${i + 1} / ${nodes.length}`;
  document.getElementById("stack-prev").disabled = i <= 0;
  document.getElementById("stack-next").disabled = i >= nodes.length - 1;
}

function toggleLike(id) {
  if (state.liked.has(id)) state.liked.delete(id);
  else state.liked.add(id);
  saveSet("mmr_liked", state.liked);
  toast(state.liked.has(id) ? "Liked — recs will adapt" : "Removed like");
}
function toggleWatchlist(id) {
  if (state.watchlist.has(id)) state.watchlist.delete(id);
  else state.watchlist.add(id);
  saveSet("mmr_watchlist", state.watchlist);
  toast(state.watchlist.has(id) ? "Added to watchlist" : "Removed from watchlist");
}

/* ---------- cards grid ---------- */
function posterEl(m) {
  const img = document.createElement("img");
  img.className = "poster";
  img.alt = m.title;
  img.loading = "lazy";
  img.src = m.poster;
  img.onerror = () => {
    const d = document.createElement("div");
    d.className = "poster-fallback";
    d.style.background = `linear-gradient(160deg, ${m.color || "#243049"}, #0b1020)`;
    d.textContent = (m.title || "?").slice(0, 1);
    img.replaceWith(d);
  };
  return img;
}
function cardHTML(m, opts = {}) {
  const el = document.createElement("article");
  el.className = "card";
  el.onclick = () => {
    const idx = state.lastRecs.findIndex((x) => x.id === m.id);
    if (idx >= 0) openStack(idx);
    else openModal(m.id);
  };
  if (opts.match != null) {
    const b = document.createElement("div");
    b.className = "badge";
    b.textContent = `${opts.match}% match`;
    el.appendChild(b);
  }
  el.appendChild(posterEl(m));
  const body = document.createElement("div");
  body.className = "card-body";
  body.innerHTML = `
    <h3>${escapeHtml(m.title)}</h3>
    <div class="meta">
      <span>${m.year || ""}</span>
      <span class="rating">★ ${(m.rating || 0).toFixed(1)}</span>
    </div>`;
  el.appendChild(body);
  return el;
}
function renderGrid(id, movies, withMatch = false) {
  const root = document.getElementById(id);
  root.innerHTML = "";
  if (!movies.length) {
    root.innerHTML = `<div class="empty">Nothing here yet.</div>`;
    return;
  }
  movies.forEach((m) => {
    root.appendChild(
      cardHTML(m, withMatch ? { match: m.match_pct ?? Math.round((m.score || 0) * 100) } : {})
    );
  });
}
function renderMoods() {
  const root = document.getElementById("moods");
  root.innerHTML = "";
  MOODS.forEach((mood) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "mood" + (state.selectedMood === mood.id ? " selected" : "");
    b.textContent = `${mood.emoji} ${mood.label}`;
    b.onclick = () => {
      state.selectedMood = mood.id;
      renderMoods();
    };
    root.appendChild(b);
  });
}
function discoveryFeed() {
  const feed = [...state.movies].sort(
    (a, b) => (b.rating || 0) - (a.rating || 0) || (b.year || 0) - (a.year || 0)
  );
  renderGrid("feed", feed.slice(0, 12));
}
function renderWatchlist() {
  renderGrid("watchlist", state.movies.filter((m) => state.watchlist.has(m.id)));
}
function renderLiked() {
  renderGrid("liked", state.movies.filter((m) => state.liked.has(m.id)));
}

/* ---------- Modal ---------- */
function openModal(id) {
  const m = state.movies.find((x) => x.id === id);
  if (!m) return;
  const backdrop = document.getElementById("modal");
  const body = document.getElementById("modal-body");
  const liked = state.liked.has(m.id);
  const wl = state.watchlist.has(m.id);
  const rating = state.ratings[m.id] || 0;
  body.innerHTML = `
    <button class="close-x" id="close-modal" aria-label="Close">×</button>
    <div class="modal-top">
      <div id="modal-poster"></div>
      <div>
        <h2>${escapeHtml(m.title)} <span class="meta">(${m.year})</span></h2>
        <div class="meta">
          <span class="rating">★ ${Number(m.rating).toFixed(1)}</span>
          <span>${m.runtime || "?"} min</span>
        </div>
        <div class="tags">
          ${(m.genres || []).map((g) => `<span class="tag">${escapeHtml(g)}</span>`).join("")}
          ${(m.moods || []).map((g) => `<span class="tag mood-tag">${escapeHtml(g)}</span>`).join("")}
        </div>
        <p>${escapeHtml(m.overview || "")}</p>
        <div class="modal-actions">
          <button class="btn" id="btn-like">${liked ? "♥ Liked" : "♡ Like"}</button>
          <button class="btn secondary" id="btn-wl">${wl ? "✓ Watchlist" : "+ Watchlist"}</button>
          <button class="btn secondary" id="btn-rate">Rate: ${rating || "—"}/5</button>
        </div>
      </div>
    </div>`;
  body.querySelector("#modal-poster").appendChild(posterEl(m));
  backdrop.classList.add("open");
  body.querySelector("#close-modal").onclick = closeModal;
  body.querySelector("#btn-like").onclick = () => { toggleLike(m.id); openModal(m.id); renderLiked(); };
  body.querySelector("#btn-wl").onclick = () => { toggleWatchlist(m.id); openModal(m.id); renderWatchlist(); };
  body.querySelector("#btn-rate").onclick = () => {
    const next = (rating % 5) + 1;
    state.ratings[m.id] = next;
    saveObj("mmr_ratings", state.ratings);
    if (next >= 4) { state.liked.add(m.id); saveSet("mmr_liked", state.liked); }
    openModal(m.id);
    renderLiked();
    toast(`Rated ${next}/5`);
  };
}
function closeModal() {
  document.getElementById("modal").classList.remove("open");
}

/* ---------- Camera + OpenCV emotion API ---------- */
function apiBases() {
  // Try same-origin first, then common local dev ports
  const origins = [
    window.location.origin,
    "http://127.0.0.1:8012",
    "http://localhost:8012",
  ];
  // de-dupe
  return [...new Set(origins.filter(Boolean))];
}

async function apiFetch(path, options = {}) {
  let lastErr;
  for (const base of apiBases()) {
    try {
      const url = base.replace(/\/$/, "") + path;
      const ctrl =
        options.signal ||
        (typeof AbortSignal !== "undefined" && AbortSignal.timeout
          ? AbortSignal.timeout(12000)
          : undefined);
      const r = await fetch(url, { ...options, signal: ctrl });
      if (r.ok || r.status < 500) {
        // prefer first reachable host
        state.apiBase = base.replace(/\/$/, "");
        state.mode = "api";
        return r;
      }
      lastErr = new Error(`HTTP ${r.status} from ${base}`);
    } catch (e) {
      lastErr = e;
    }
  }
  throw lastErr || new Error("API unreachable");
}

function stopCamera() {
  if (state.camStream) {
    state.camStream.getTracks().forEach((t) => t.stop());
    state.camStream = null;
  }
  const video = document.getElementById("cam");
  if (video) video.srcObject = null;
  document.getElementById("cam-wrap")?.classList.remove("live");
  const box = document.getElementById("cam-face-box");
  if (box) box.classList.remove("show");
}

async function enableCamera() {
  const status = document.getElementById("cam-status");
  const btn = document.getElementById("btn-cam");
  const scanBtn = document.getElementById("btn-scan");

  if (!window.isSecureContext && location.hostname !== "localhost" && location.hostname !== "127.0.0.1") {
    status.textContent = "Camera needs HTTPS or localhost (not file://)";
    toast("Open via http://localhost:8012 — file:// blocks the camera");
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    status.textContent = "Browser has no camera API";
    toast("Use Chrome/Edge/Firefox with camera permission");
    return;
  }

  btn.disabled = true;
  btn.textContent = "Starting…";
  status.textContent = "Requesting camera permission…";
  stopCamera();

  const attempts = [
    { video: { facingMode: { ideal: "user" }, width: { ideal: 640 }, height: { ideal: 480 } }, audio: false },
    { video: { facingMode: "user" }, audio: false },
    { video: true, audio: false },
  ];

  let stream = null;
  let lastErr = null;
  for (const constraints of attempts) {
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
      break;
    } catch (e) {
      lastErr = e;
      console.warn("getUserMedia attempt failed", constraints, e);
    }
  }

  if (!stream) {
    btn.disabled = false;
    btn.textContent = "Enable camera";
    const name = lastErr?.name || "Error";
    const msg =
      name === "NotAllowedError"
        ? "Permission denied — allow camera in browser settings"
        : name === "NotFoundError"
          ? "No camera found on this device"
          : name === "NotReadableError"
            ? "Camera is busy (close Zoom/FaceTime and retry)"
            : `Camera failed: ${lastErr?.message || name}`;
    status.textContent = msg;
    toast(msg);
    return;
  }

  try {
    state.camStream = stream;
    const video = document.getElementById("cam");
    video.setAttribute("playsinline", "true");
    video.setAttribute("webkit-playsinline", "true");
    video.muted = true;
    video.autoplay = true;
    video.srcObject = stream;

    // Critical: some browsers need an explicit play()
    await video.play().catch(() => {});

    await new Promise((resolve, reject) => {
      if (video.readyState >= 2 && video.videoWidth > 0) {
        resolve();
        return;
      }
      const onReady = () => {
        cleanup();
        resolve();
      };
      const onErr = () => {
        cleanup();
        reject(new Error("Video failed to load"));
      };
      const t = setTimeout(() => {
        cleanup();
        if (video.videoWidth > 0) resolve();
        else reject(new Error("Camera timed out — try again"));
      }, 10000);
      function cleanup() {
        clearTimeout(t);
        video.removeEventListener("loadedmetadata", onReady);
        video.removeEventListener("loadeddata", onReady);
        video.removeEventListener("error", onErr);
      }
      video.addEventListener("loadedmetadata", onReady);
      video.addEventListener("loadeddata", onReady);
      video.addEventListener("error", onErr);
    });

    document.getElementById("cam-wrap")?.classList.add("live");
    status.textContent = `Camera live (${video.videoWidth}×${video.videoHeight}) — face the lens`;
    scanBtn.disabled = false;
    btn.textContent = "Restart camera";
    btn.disabled = false;
    toast("Camera enabled");
  } catch (e) {
    console.error(e);
    stopCamera();
    status.textContent = e.message || "Could not start video";
    toast(status.textContent);
    btn.disabled = false;
    btn.textContent = "Enable camera";
  }
}

function captureFrameDataUrl() {
  const video = document.getElementById("cam");
  const canvas = document.getElementById("cam-canvas");
  if (!video.srcObject) throw new Error("Camera not started — click Enable camera");
  if (!video.videoWidth || !video.videoHeight) {
    throw new Error("Camera not ready yet — wait a second and retry");
  }
  // Downscale for faster API upload
  const maxW = 480;
  const scale = Math.min(1, maxW / video.videoWidth);
  const w = Math.round(video.videoWidth * scale);
  const h = Math.round(video.videoHeight * scale);
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  // Capture un-mirrored (video element is only mirrored via CSS)
  ctx.drawImage(video, 0, 0, w, h);
  return canvas.toDataURL("image/jpeg", 0.82);
}

function showFaceBox(box, videoW, videoH) {
  const el = document.getElementById("cam-face-box");
  const wrap = document.getElementById("cam-wrap");
  if (!el || !wrap || !box) {
    el?.classList.remove("show");
    return;
  }
  // Map image coords → preview (mirrored CSS)
  const rect = wrap.getBoundingClientRect();
  const scaleX = rect.width / videoW;
  const scaleY = rect.height / videoH;
  // because preview is mirrored, flip x
  const x = videoW - box.x - box.w;
  el.style.left = `${x * scaleX}px`;
  el.style.top = `${box.y * scaleY}px`;
  el.style.width = `${box.w * scaleX}px`;
  el.style.height = `${box.h * scaleY}px`;
  el.classList.add("show");
}

function applyEmotionResult(data) {
  document.getElementById("face-expr").textContent = data.expression;
  document.getElementById("face-mood").textContent = `→ mood: ${data.mood}`;
  document.getElementById("face-conf").textContent = data.confidence != null
    ? `confidence ${(data.confidence * 100).toFixed(1)}%`
    : "";
  document.getElementById("face-engine").textContent = data.engine || "opencv";
  document.getElementById("cam-status").textContent =
    `Detected ${data.expression} → ${data.mood}`;

  if (data.mood) {
    state.selectedMood = data.mood;
    renderMoods();
  }
  if (data.query_hint) {
    document.getElementById("query").value = data.query_hint;
  }
  toast(`Expression: ${data.expression} → ${data.mood}`);
}

async function scanExpression() {
  const btn = document.getElementById("btn-scan");
  const status = document.getElementById("cam-status");
  btn.disabled = true;
  btn.textContent = "Scanning…";
  status.textContent = "Capturing frame…";

  try {
    const image = captureFrameDataUrl();
    status.textContent = "OpenCV analyzing face…";

    let data;
    try {
      const r = await apiFetch("/api/emotion", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image }),
      });
      data = await r.json();
      if (!r.ok) {
        throw new Error(data.detail || data.message || `HTTP ${r.status}`);
      }
    } catch (apiErr) {
      console.warn("OpenCV API failed, trying browser FaceDetector", apiErr);
      // Browser fallback so camera still produces a mood without backend
      data = await browserFaceFallback(image);
      if (!data) {
        throw apiErr;
      }
    }

    if (data.ok === false) {
      document.getElementById("face-expr").textContent = data.error || "failed";
      document.getElementById("face-mood").textContent = data.message || "Try better lighting";
      document.getElementById("face-conf").textContent = "";
      document.getElementById("face-engine").textContent = data.engine || "";
      status.textContent = data.message || "No face";
      document.getElementById("cam-face-box")?.classList.remove("show");
      toast(data.message || "Expression failed");
      return;
    }

    const video = document.getElementById("cam");
    if (data.face_box) {
      showFaceBox(data.face_box, video.videoWidth, video.videoHeight);
    }
    applyEmotionResult(data);
  } catch (e) {
    console.error(e);
    const hint =
      "Start server: uvicorn backend.app.main:app --port 8012  · then open http://localhost:8012";
    document.getElementById("face-expr").textContent = "scan failed";
    document.getElementById("face-mood").textContent = hint;
    status.textContent = e.message || "Scan failed";
    toast(e.message || "Expression scan failed");
  } finally {
    btn.disabled = false;
    btn.textContent = "Read expression";
  }
}

/** Fallback when FastAPI/OpenCV is offline — browser FaceDetector if available */
async function browserFaceFallback(dataUrl) {
  if (!("FaceDetector" in window)) return null;
  try {
    const img = new Image();
    await new Promise((res, rej) => {
      img.onload = res;
      img.onerror = rej;
      img.src = dataUrl;
    });
    const detector = new FaceDetector({ fastMode: true, maxDetectedFaces: 1 });
    const faces = await detector.detect(img);
    if (!faces.length) {
      return {
        ok: false,
        error: "no_face",
        message: "No face detected (browser). Face the camera with light on your face.",
        engine: "browser-FaceDetector",
      };
    }
    const b = faces[0].boundingBox;
    // Without OpenCV expression model, use neutral/reflective and still unlock flow
    return {
      ok: true,
      expression: "neutral",
      confidence: 0.5,
      mood: "reflective",
      query_hint: "reflective calm drama thoughtful",
      face_box: {
        x: Math.round(b.x),
        y: Math.round(b.y),
        w: Math.round(b.width),
        h: Math.round(b.height),
      },
      engine: "browser-FaceDetector (start API for full OpenCV FER)",
    };
  } catch (e) {
    console.warn(e);
    return null;
  }
}

/* ---------- Recommend with match bar + stack ---------- */
async function fetchRecs() {
  const query = document.getElementById("query").value.trim();
  const mood = state.selectedMood;
  if (state.mode === "api") {
    try {
      const r = await apiFetch("/api/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mood,
          query: query || null,
          top_k: 8,
          liked_ids: [...state.liked],
        }),
      });
      const data = await r.json();
      return data.recommendations;
    } catch (e) {
      console.warn("API recommend failed, using local TF-IDF", e);
    }
  }
  return localRecommend({ mood, query, topK: 8 });
}

async function runRecommend() {
  const btn = document.getElementById("btn-rec");
  btn.disabled = true;
  try {
    // parallel: animate bar + fetch recs
    const [_, recs] = await Promise.all([
      animateMatchBar(1300),
      fetchRecs(),
    ]);
    // hold at 100% briefly
    setMatchProgress(100);
    await new Promise((r) => setTimeout(r, 280));
    hideMatchOverlay();

    state.lastRecs = recs;
    renderGrid("recs", recs, true);
    document.getElementById("recs-sub").textContent =
      `Tuned for “${state.selectedMood}” · ${state.mode === "api" ? "API + OpenCV path" : "on-device TF-IDF"}`;

    // open Netflix-style stack automatically
    openStack(0);
  } catch (e) {
    hideMatchOverlay();
    toast("Could not load recommendations");
    console.error(e);
  } finally {
    btn.disabled = false;
  }
}

/* ---------- boot ---------- */
async function detectMode() {
  for (const base of apiBases()) {
    try {
      const r = await fetch(`${base.replace(/\/$/, "")}/api/health`, {
        signal: AbortSignal.timeout?.(2000),
      });
      if (r.ok) {
        state.mode = "api";
        state.apiBase = base.replace(/\/$/, "");
        return;
      }
    } catch {}
  }
  state.mode = "static";
  state.apiBase = null;
}

async function loadMovies() {
  if (state.mode === "api") {
    try {
      const r = await apiFetch("/api/movies");
      const data = await r.json();
      state.movies = data.movies;
      state.index = buildIndex(state.movies);
      return;
    } catch (e) {
      console.warn(e);
      state.mode = "static";
    }
  }
  const base = document.querySelector("script[data-base]")?.dataset.base || "./";
  const r = await fetch(`${base}assets/movies.json`);
  state.movies = await r.json();
  state.index = buildIndex(state.movies);
}

async function init() {
  renderMoods();
  await detectMode();
  const pill = document.getElementById("api-pill");
  pill.textContent = state.mode === "api" ? "API + OpenCV live" : "Static host mode";
  pill.classList.toggle("live", state.mode === "api");

  await loadMovies();
  document.getElementById("stat-movies").textContent = String(state.movies.length);
  document.getElementById("stat-moods").textContent = String(MOODS.length);

  discoveryFeed();
  renderWatchlist();
  renderLiked();
  // initial recs without auto-stack spam
  state.lastRecs = await fetchRecs();
  renderGrid("recs", state.lastRecs, true);

  document.getElementById("btn-rec").onclick = runRecommend;
  document.getElementById("btn-clear").onclick = () => {
    document.getElementById("query").value = "";
  };
  document.getElementById("btn-open-stack").onclick = () => openStack(0);
  document.getElementById("btn-cam").onclick = enableCamera;
  document.getElementById("btn-scan").onclick = scanExpression;

  // Netflix-style horizontal row scroll buttons
  document.querySelectorAll(".row-scroll-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = btn.getAttribute("data-scroll");
      const dir = Number(btn.getAttribute("data-dir") || 1);
      const row = document.getElementById(id);
      if (!row) return;
      const step = Math.max(row.clientWidth * 0.75, 240);
      row.scrollBy({ left: dir * step, behavior: "smooth" });
    });
  });
  document.getElementById("stack-prev").onclick = () => stackGo(-1);
  document.getElementById("stack-next").onclick = () => stackGo(1);
  document.getElementById("stack-close").onclick = closeStack;
  document.getElementById("query").addEventListener("keydown", (e) => {
    if (e.key === "Enter") runRecommend();
  });
  document.getElementById("modal").addEventListener("click", (e) => {
    if (e.target.id === "modal") closeModal();
  });
  window.addEventListener("keydown", (e) => {
    if (!document.getElementById("stack-overlay").classList.contains("open")) return;
    if (e.key === "ArrowLeft") stackGo(-1);
    if (e.key === "ArrowRight") stackGo(1);
    if (e.key === "Escape") closeStack();
  });
}

init().catch((e) => {
  console.error(e);
  toast("Failed to boot app");
});
