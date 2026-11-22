/* Green Vision – Dashboard front-end logic */
const POLL_MS = 2000;
let online = false;

// ── Clock ───────────────────────────────────────────────────────────────────
function tickClock() {
  document.getElementById('clock').textContent = new Date().toLocaleTimeString();
}
setInterval(tickClock, 1000);
tickClock();

// ── Badge ───────────────────────────────────────────────────────────────────
function setBadge(on) {
  const el = document.getElementById('status-badge');
  if (on !== online) {
    online = on;
    el.textContent = on ? 'ONLINE' : 'OFFLINE';
    el.className = 'badge ' + (on ? 'online' : 'offline');
  }
}

// ── Fetch helpers ───────────────────────────────────────────────────────────
async function getJSON(url) {
  try {
    const r = await fetch(url, { cache: 'no-store' });
    if (!r.ok) throw new Error(r.status);
    return await r.json();
  } catch (e) {
    console.warn(`GET ${url} failed:`, e.message);
    return null;
  }
}

// ── Renderers ───────────────────────────────────────────────────────────────
function renderState(d) {
  if (!d || d.error) { setBadge(false); return; }
  setBadge(true);
  setText('scenario', d.scenario || '—');
  setText('decisions', String(d.total_decisions ?? 0));
  setText('episode', d.episode_active ? 'yes' : 'no');
  const streak = d.no_op_streak ?? 0;
  const stEl = document.getElementById('noop-streak');
  stEl.textContent = String(streak);
  stEl.className = 'stat-value' + (streak > 5 ? ' warn' : '');
  setText('console', d.console_enabled ? 'ON' : 'off');
}

function renderHealth(h) {
  if (!h || h.gpu?.error) return;
  const g = h.gpu;
  if (g.vram_used_mib != null) {
    setText('vram-used', String(g.vram_used_mib));
    setText('vram-total', String(g.vram_total_mib));
    setText('clock-mhz', String(g.clock_mhz));
    setText('power-w', g.power_w.toFixed(1));
  }
}

function renderGoals(data) {
  const ul = document.getElementById('goals-list');
  if (!data || !data.active || data.active.length === 0) {
    ul.innerHTML = '<li class="muted">No active goals</li>';
    return;
  }
  ul.innerHTML = data.active.slice(0, 8).map(g => {
    const prog = Math.round((g.progress ?? 0) * 100);
    const pri = (g.priority || 'MEDIUM').toUpperCase();
    const desc = (g.desc || g.description || '?').slice(0, 40);
    return `<li>
      <span class="goal-pri pri-${pri}">${pri}</span>${desc}
      <div class="goals-progress"><div class="goals-progress-fill" style="width:${prog}%"></div></div>
    </li>`;
  }).join('');
}

function renderThoughts(thoughts) {
  const feed = document.getElementById('thoughts-feed');
  if (!thoughts || thoughts.length === 0) {
    feed.innerHTML = '<p class="muted">No thoughts yet</p>';
    return;
  }
  // newest first
  const items = [...thoughts].reverse().slice(0, 20).map(t => {
    const ts = t.ts ? new Date(t.ts * 1000).toLocaleTimeString() : '';
    const conf = t.confidence ?? 0;
    const decision = t.decision || '?';
    const reasoning = (t.reasoning || '').slice(0, 120);
    return `<div class="thought-item">
      <span class="ts">${ts}</span> → <span class="decision">${decision}</span>
      <span class="conf-bar"><span class="conf-fill" style="width:${Math.round(conf*100)}%"></span></span>
      <div class="reasoning">${reasoning}</div>
    </div>`;
  }).join('');
  feed.innerHTML = items;
}

function renderActions(actions) {
  const tbody = document.querySelector('#actions-table tbody');
  if (!actions || actions.length === 0) {
    tbody.innerHTML = '<tr><td colspan="3" class="muted">No actions recorded</td></tr>';
    return;
  }
  tbody.innerHTML = [...actions].reverse().map(a => {
    const ts = a.ts ? new Date(a.ts * 1000).toLocaleTimeString() : '';
    const action = a.action || '?';
    const detail = (a.detail || a.description || '').slice(0, 60);
    return `<tr><td>${ts}</td><td>${action}</td><td>${detail}</td></tr>`;
  }).join('');
}

// ── Utility ─────────────────────────────────────────────────────────────────
function setText(id, val) { const el = document.getElementById(id); if (el) el.textContent = val; }

// ── Poll loop ───────────────────────────────────────────────────────────────
async function poll() {
  const [state, health, goals, thoughts, actions] = await Promise.all([
    getJSON('/api/state'),
    getJSON('/api/health'),
    getJSON('/api/goals'),
    getJSON('/api/thoughts'),
    getJSON('/api/actions'),
  ]);
  renderState(state);
  renderHealth(health);
  renderGoals(goals);
  renderThoughts(thoughts);
  renderActions(actions);
}

poll();
setInterval(poll, POLL_MS);
