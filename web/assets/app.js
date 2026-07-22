/**
 * Mood Movie Recs — frontend
 * Uses FastAPI when available; otherwise pure client-side TF-IDF recommender
 * so the site works fully hosted on GitHub Pages.
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
];

const state = {
  movies: [],
  mode: "static", // static | api
  selectedMood: "cozy",
  liked: loadSet("mmr_liked"),
  watchlist: loadSet("mmr_watchlist"),
  ratings: loadObj("mmr_ratings"),
  vocab: null,
  matrix: null,
};

function loadSet(key) {
  try { return new Set(JSON.parse(localStorage.getItem(key) || "[]")); }
  catch { return new Set(); }
}
function saveSet(key, set) {
  localStorage.setItem(key, JSON.stringify([...set]));
}
function loadObj(key) {
  try { return JSON.parse(localStorage.getItem(key) || "{}"); }
  catch { return {}; }
}
function saveObj(key, obj) {
  localStorage.setItem(key, JSON.stringify(obj));
}

function toast(msg) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 2200);
}

// --- Minimal TF-IDF (client-side) ---
function tokenize(text) {
  return (text || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, " ")
    .split(/\s+/)
    .filter(Boolean);
}

function buildIndex(movies) {
  const docs = movies.map((m) =>
    tokenize(
      [m.title, ...(m.genres || []), ...(m.moods || []), m.overview].join(" ")
    )
  );
  const df = new Map();
  docs.forEach((toks) => {
    new Set(toks).forEach((t) => df.set(t, (df.get(t) || 0) + 1));
  });
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
  const qText = parts.join(" ") || "cozy warm light";
  const qVec = vectorizeQuery(qText, state.index);
  const scored = state.movies.map((m, i) => {
    const sim = cosine(qVec, state.index.matrix[i]);
    const score = 0.85 * sim + 0.15 * ((m.rating || 7) / 10);
    return { ...m, score, match_pct: Math.round(score * 100) };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored.filter((m) => !state.liked.has(m.id)).slice(0, topK);
}

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
  el.onclick = () => openModal(m.id);
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

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
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
      runRecommend();
    };
    root.appendChild(b);
  });
}

function renderGrid(id, movies, withMatch = false) {
  const root = document.getElementById(id);
  root.innerHTML = "";
  if (!movies.length) {
    root.innerHTML = `<div class="empty">Nothing here yet — rate or heart a few films.</div>`;
    return;
  }
  movies.forEach((m) => {
    root.appendChild(
      cardHTML(m, withMatch ? { match: m.match_pct ?? Math.round((m.score || 0) * 100) } : {})
    );
  });
}

async function runRecommend() {
  const btn = document.getElementById("btn-rec");
  btn.disabled = true;
  btn.textContent = "Finding…";
  try {
    const recs = await fetchRecs();
    renderGrid("recs", recs, true);
    document.getElementById("recs-sub").textContent = state.selectedMood
      ? `Tuned for “${state.selectedMood}” · ${state.mode === "api" ? "API engine" : "on-device TF-IDF"}`
      : "";
  } catch (e) {
    toast("Could not load recommendations");
    console.error(e);
  } finally {
    btn.disabled = false;
    btn.textContent = "Get recommendations";
  }
}

function discoveryFeed() {
  const feed = [...state.movies].sort(
    (a, b) => (b.rating || 0) - (a.rating || 0) || (b.year || 0) - (a.year || 0)
  );
  renderGrid("feed", feed.slice(0, 12));
}

function renderWatchlist() {
  const items = state.movies.filter((m) => state.watchlist.has(m.id));
  renderGrid("watchlist", items);
}

function renderLiked() {
  const items = state.movies.filter((m) => state.liked.has(m.id));
  renderGrid("liked", items);
}

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
          <span>${(m.genres || []).join(" · ")}</span>
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
  const holder = body.querySelector("#modal-poster");
  holder.appendChild(posterEl(m));
  backdrop.classList.add("open");
  body.querySelector("#close-modal").onclick = closeModal;
  body.querySelector("#btn-like").onclick = () => {
    if (state.liked.has(m.id)) state.liked.delete(m.id);
    else state.liked.add(m.id);
    saveSet("mmr_liked", state.liked);
    renderLiked();
    openModal(m.id);
    runRecommend();
    toast(state.liked.has(m.id) ? "Saved to likes — recs will adapt" : "Removed like");
  };
  body.querySelector("#btn-wl").onclick = () => {
    if (state.watchlist.has(m.id)) state.watchlist.delete(m.id);
    else state.watchlist.add(m.id);
    saveSet("mmr_watchlist", state.watchlist);
    renderWatchlist();
    openModal(m.id);
    toast(state.watchlist.has(m.id) ? "Added to watchlist" : "Removed from watchlist");
  };
  body.querySelector("#btn-rate").onclick = () => {
    const next = ((rating % 5) + 1);
    state.ratings[m.id] = next;
    saveObj("mmr_ratings", state.ratings);
    if (next >= 4) {
      state.liked.add(m.id);
      saveSet("mmr_liked", state.liked);
    }
    openModal(m.id);
    renderLiked();
    runRecommend();
    toast(`Rated ${next}/5`);
  };
}

function closeModal() {
  document.getElementById("modal").classList.remove("open");
}

async function init() {
  renderMoods();
  await detectMode();
  const pill = document.getElementById("api-pill");
  pill.textContent = state.mode === "api" ? "API live" : "Static host mode";
  pill.classList.toggle("live", state.mode === "api");

  await loadMovies();
  document.getElementById("stat-movies").textContent = String(state.movies.length);
  document.getElementById("stat-moods").textContent = String(MOODS.length);

  discoveryFeed();
  renderWatchlist();
  renderLiked();
  await runRecommend();

  document.getElementById("btn-rec").onclick = runRecommend;
  document.getElementById("btn-clear").onclick = () => {
    document.getElementById("query").value = "";
    runRecommend();
  };
  document.getElementById("query").addEventListener("keydown", (e) => {
    if (e.key === "Enter") runRecommend();
  });
  document.getElementById("modal").addEventListener("click", (e) => {
    if (e.target.id === "modal") closeModal();
  });
}

init().catch((e) => {
  console.error(e);
  toast("Failed to boot app — check console");
});
