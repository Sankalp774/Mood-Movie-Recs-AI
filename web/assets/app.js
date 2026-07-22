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
async function enableCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    state.camStream = stream;
    const video = document.getElementById("cam");
    video.srcObject = stream;
    document.getElementById("cam-status").textContent = "Camera live — face the lens";
    document.getElementById("btn-scan").disabled = false;
    toast("Camera enabled");
  } catch (e) {
    document.getElementById("cam-status").textContent = "Camera permission denied";
    toast("Could not open camera");
    console.error(e);
  }
}

function captureFrameDataUrl() {
  const video = document.getElementById("cam");
  const canvas = document.getElementById("cam-canvas");
  if (!video.videoWidth) throw new Error("Camera not ready");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  // un-mirror for model
  ctx.drawImage(video, 0, 0);
  return canvas.toDataURL("image/jpeg", 0.85);
}

async function scanExpression() {
  if (state.mode !== "api") {
    toast("Start the FastAPI server for OpenCV emotion (uvicorn …)");
    document.getElementById("face-expr").textContent = "API required";
    document.getElementById("face-mood").textContent =
      "OpenCV FER runs on the backend. Run: uvicorn backend.app.main:app --port 8000";
    return;
  }
  const btn = document.getElementById("btn-scan");
  btn.disabled = true;
  btn.textContent = "Scanning…";
  document.getElementById("cam-status").textContent = "OpenCV analyzing face…";
  try {
    const image = captureFrameDataUrl();
    const r = await fetch("/api/emotion", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image }),
    });
    const data = await r.json();
    if (!r.ok || data.ok === false) {
      document.getElementById("face-expr").textContent = data.error || "failed";
      document.getElementById("face-mood").textContent = data.message || "Try better lighting";
      document.getElementById("face-conf").textContent = "";
      document.getElementById("cam-status").textContent = data.message || "No face";
      toast(data.message || "Expression failed");
      return;
    }
    document.getElementById("face-expr").textContent = data.expression;
    document.getElementById("face-mood").textContent = `→ mood: ${data.mood}`;
    document.getElementById("face-conf").textContent =
      `confidence ${(data.confidence * 100).toFixed(1)}%`;
    document.getElementById("face-engine").textContent = data.engine || "opencv";
    document.getElementById("cam-status").textContent =
      `Detected ${data.expression} → ${data.mood}`;

    // auto-select mood + optional query hint
    state.selectedMood = data.mood;
    renderMoods();
    if (data.query_hint) {
      document.getElementById("query").value = data.query_hint;
    }
    toast(`Expression: ${data.expression} → ${data.mood}`);
  } catch (e) {
    console.error(e);
    toast("Expression scan failed — is the API running?");
  } finally {
    btn.disabled = false;
    btn.textContent = "Read expression";
  }
}

/* ---------- Recommend with match bar + stack ---------- */
async function fetchRecs() {
  const query = document.getElementById("query").value.trim();
  const mood = state.selectedMood;
  if (state.mode === "api") {
    const r = await fetch("/api/recommend", {
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
  try {
    const r = await fetch("/api/health", { signal: AbortSignal.timeout(1200) });
    if (r.ok) {
      state.mode = "api";
      return;
    }
  } catch {}
  state.mode = "static";
}

async function loadMovies() {
  if (state.mode === "api") {
    const r = await fetch("/api/movies");
    const data = await r.json();
    state.movies = data.movies;
  } else {
    const base = document.querySelector("script[data-base]")?.dataset.base || "./";
    const r = await fetch(`${base}assets/movies.json`);
    state.movies = await r.json();
  }
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
