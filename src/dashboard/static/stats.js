/* Green Vision – Stats page logic v2
 * Computes Learning Maturity composite from raw /api/stats data.
 */
const POLL_MS = 5000;
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

async function postJSON(url, body) {
  try {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    return await r.json();
  } catch (e) {
    return { error: e.message };
  }
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function setBarWidth(id, pct) {
  const el = document.getElementById(id);
  if (el) el.style.width = Math.max(0, Math.min(100, pct)) + '%';
}

// ── Learning Maturity computation ───────────────────────────────────────────
function computeMaturity(d) {
  const m = { trend: 0, consistency: 0, breadth: 0, success: 0, explore: 0, overall: 0, stage: 'Initialising…' };

  // 1. Reward Trend — first ¼ vs last ¼ of reward window
  const rh = d.reward_history || [];
  if (rh.length >= 8) {
    const q = Math.floor(rh.length / 4);
    const firstQ = rh.slice(0, q).reduce((a, b) => a + b, 0) / q;
    const lastQ = rh.slice(-q).reduce((a, b) => a + b, 0) / q;
    const range = Math.abs(lastQ - firstQ);
    // Normalise: if improvement > 50% of |firstQ| or absolute 0.5, cap at 100
    const denom = Math.max(Math.abs(firstQ), 0.5);
    m.trend = Math.round(Math.min(100, Math.max(0, ((lastQ - firstQ) / denom) * 100)));
    if (m.trend < 0) m.trend = 0;
  } else {
    m.trend = 0;
  }

  // 2. Consistency — low variance in recent rewards = stable policy
  if (rh.length >= 10) {
    const mean = rh.reduce((a, b) => a + b, 0) / rh.length;
    const variance = rh.reduce((s, v) => s + (v - mean) ** 2, 0) / rh.length;
    const stdDev = Math.sqrt(variance);
    // Score: 100 when stdDev ≈ 0, drops to 0 when stdDev > 2×mean magnitude
    const scale = Math.max(Math.abs(mean), 0.3);
    m.consistency = Math.round(Math.max(0, 100 * (1 - stdDev / (scale * 2))));
  } else {
    m.consistency = 0;
  }

  // 3. Action Breadth — unique actions used out of 31 total
  const topActions = d.planner?.top_actions || [];
  const totalActions = d.planner?.actions_total || 0;
  if (totalActions > 0) {
    // We only get top-10; estimate breadth from that + total count
    const uniqueKnown = new Set(topActions.map(a => a.name)).size;
    // If total >> top-10 sum, there are more unique actions beyond the top 10
    const topSum = topActions.reduce((s, a) => s + a.count, 0);
    const estUnique = uniqueKnown + Math.min(totalActions - topSum, 21); // cap at remaining 21
    m.breadth = Math.round(Math.min(100, (estUnique / 31) * 100));
  } else {
    m.breadth = 0;
  }

  // 4. Episode Success rate
  const ep = d.episodes || {};
  const totalEp = ep.total || 0;
  if (totalEp > 0) {
    m.success = Math.round(((ep.success || 0) / totalEp) * 100);
  } else {
    m.success = 0;
  }

  // 5. Exploration Decay — intrinsic reward should decrease over time
  const cur = d.curiosity || {};
  const avgIntr = cur.avg_intrinsic ?? 0;
  const numSteps = cur.num_steps ?? 0;
  if (numSteps >= 100) {
    // If avg intrinsic is very low (< 0.001), the agent has mapped most of the world → high score
    // If still high (> 0.05), it's still exploring a lot → lower maturity on this axis
    m.explore = Math.round(Math.max(0, 100 * (1 - avgIntr / 0.05)));
  } else if (numSteps > 0) {
    m.explore = Math.round(numSteps / 100 * 30); // partial credit for early steps
  } else {
    m.explore = 0;
  }

  // Composite: weighted average (trend 30%, consistency 20%, breadth 15%, success 20%, explore 15%)
  m.overall = Math.round(
    m.trend * 0.30 +
    m.consistency * 0.20 +
    m.breadth * 0.15 +
    m.success * 0.20 +
    m.explore * 0.15
  );

  // Stage label
  if (m.overall >= 80) m.stage = '🏆 Expert — policy is stable and broad';
  else if (m.overall >= 60) m.stage = '💪 Competent — solid decisions with room to grow';
  else if (m.overall >= 40) m.stage = '📈 Developing — learning new strategies';
  else if (m.overall >= 20) m.stage = '🌱 Emerging — building basic competence';
  else m.stage = '🔰 Initialising — early exploration phase';

  return m;
}

// ── Reward chart (canvas line graph) ────────────────────────────────────────
function drawRewardChart(data, bestVal) {
  const canvas = document.getElementById('reward-chart');
  if (!canvas || !data || data.length === 0) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const pad = { top: 12, right: 20, bottom: 30, left: 55 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  ctx.clearRect(0, 0, W, H);

  let minV = Math.min(...data);
  let maxV = Math.max(...data);
  if (minV === maxV) { minV -= 0.5; maxV += 0.5; }
  const range = maxV - minV;

  // Grid lines + Y labels
  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (plotH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(W - pad.right, y);
    ctx.stroke();
    const val = maxV - (range / 4) * i;
    ctx.fillStyle = '#8b949e';
    ctx.font = '11px Segoe UI';
    ctx.textAlign = 'right';
    ctx.fillText(val.toFixed(2), pad.left - 6, y + 4);
  }

  // X labels
  ctx.textAlign = 'center';
  ctx.fillStyle = '#8b949e';
  ctx.fillText('0', pad.left, H - 8);
  ctx.fillText(String(Math.floor(data.length / 2)), pad.left + plotW / 2, H - 8);
  ctx.fillText(String(data.length - 1), W - pad.right, H - 8);

  // Line + fill
  if (data.length > 1) {
    ctx.beginPath();
    ctx.strokeStyle = '#3fb950';
    ctx.lineWidth = 2;
    for (let i = 0; i < data.length; i++) {
      const x = pad.left + (plotW / (data.length - 1)) * i;
      const y = pad.top + plotH - ((data[i] - minV) / range) * plotH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.lineTo(pad.left + plotW, pad.top + plotH);
    ctx.lineTo(pad.left, pad.top + plotH);
    ctx.closePath();
    ctx.fillStyle = 'rgba(63, 185, 80, 0.07)';
    ctx.fill();
  }

  // Best reward dashed line
  if (bestVal != null && bestVal > minV && bestVal < maxV) {
    const by = pad.top + plotH - ((bestVal - minV) / range) * plotH;
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = '#d29922';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, by);
    ctx.lineTo(W - pad.right, by);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#d29922';
    ctx.font = '10px Segoe UI';
    ctx.textAlign = 'left';
    ctx.fillText(`best: ${bestVal.toFixed(3)}`, W - pad.right - 80, by - 5);
  }
}

// ── Render all stats ────────────────────────────────────────────────────────
function renderStats(d) {
  if (!d || d.error) { setBadge(false); return; }
  setBadge(true);

  // ── Learning Maturity ──
  const m = computeMaturity(d);
  setText('mat-overall', m.overall + '%');
  setText('mat-stage-label', m.stage);
  setBarWidth('mat-overall-bar', m.overall);

  setText('mat-trend-score', (m.trend >= 50 ? '▲' : m.trend > 0 ? '↗' : '—') + ' ' + m.trend + '%');
  document.getElementById('mat-trend-score').className = 'maturity-sub-score ' + (m.trend >= 50 ? 'trend-up' : m.trend > 0 ? 'trend-flat' : 'trend-down');
  setBarWidth('mat-trend-bar', m.trend);

  setText('mat-consist-score', m.consistency + '%');
  document.getElementById('mat-consist-score').className = 'maturity-sub-score ' + (m.consistency >= 60 ? 'trend-up' : 'trend-flat');
  setBarWidth('mat-consist-bar', m.consistency);

  setText('mat-breadth-score', m.breadth + '%');
  document.getElementById('mat-breadth-score').className = 'maturity-sub-score ' + (m.breadth >= 50 ? 'trend-up' : 'trend-flat');
  setBarWidth('mat-breadth-bar', m.breadth);

  setText('mat-success-score', m.success + '%');
  document.getElementById('mat-success-score').className = 'maturity-sub-score ' + (m.success >= 50 ? 'trend-up' : 'trend-flat');
  setBarWidth('mat-success-bar', m.success);

  setText('mat-explore-score', m.explore + '%');
  document.getElementById('mat-explore-score').className = 'maturity-sub-score ' + (m.explore >= 60 ? 'trend-up' : 'trend-flat');
  setBarWidth('mat-explore-bar', m.explore);

  // ── Summary cards ──
  setText('stat-decisions', String(d.planner?.total_decisions ?? 0));
  setText('stat-best-reward', (d.best_reward ?? 0).toFixed(4));
  const rh = d.reward_history || [];
  const curAvg = rh.length > 0 ? (rh.reduce((a, b) => a + b, 0) / rh.length).toFixed(4) : '—';
  setText('stat-cur-avg-reward', curAvg);
  setText('stat-episodes', String(d.episodes?.total ?? 0));
  setText('stat-goals-done', `${d.planner?.goals_completed ?? 0}/${d.planner?.goals_abandoned ?? 0}`);
  setText('stat-checkpoints', String(d.checkpoints?.length ?? 0));

  // ── Reward chart ──
  drawRewardChart(rh, d.best_reward);
  const capEl = document.getElementById('reward-caption');
  if (capEl && rh.length > 0) {
    capEl.textContent = `Window: ${rh.length} steps · Current avg: ${curAvg} · Best ever: ${(d.best_reward ?? 0).toFixed(4)} · Trend: ${m.trend >= 50 ? 'improving ▲' : m.trend > 0 ? 'slight gain ↗' : 'flat/declining ▼'}`;
  }

  // ── Episode outcome bars ──
  const ep = d.episodes || {};
  const totalEp = Math.max(1, ep.total || 1);
  setBarWidth('bar-success', ((ep.success || 0) / totalEp * 100));
  setBarWidth('bar-stall', ((ep.stall || 0) / totalEp * 100));
  setBarWidth('bar-failure', ((ep.failure || 0) / totalEp * 100));
  setText('count-success', String(ep.success ?? 0));
  setText('count-stall', String(ep.stall ?? 0));
  setText('count-failure', String(ep.failure ?? 0));

  // Episode avg reward table (3 columns now)
  const tbody = document.querySelector('#episode-avg-table tbody');
  if (tbody) {
    tbody.innerHTML = [
      ['Success', ep.avg_reward_success, ep.success],
      ['Stall', ep.avg_reward_stall, ep.stall],
      ['Failure', ep.avg_reward_failure, ep.failure],
    ].map(([label, val, cnt]) => `<tr><td>${label}</td><td>${val != null ? Number(val).toFixed(4) : '—'}</td><td>${cnt ?? 0}</td></tr>`).join('');
  }

  // ── Curiosity ──
  const cur = d.curiosity || {};
  setText('cur-total', (cur.total_intrinsic ?? 0).toFixed(4));
  setText('cur-steps', String(cur.num_steps ?? 0));
  setText('cur-avg', (cur.avg_intrinsic ?? 0).toExponential(3));
  const phase = (cur.avg_intrinsic ?? 0) < 0.005 ? '🗺️ Mapped' : (cur.avg_intrinsic ?? 0) < 0.02 ? '🔍 Refining' : '🚀 Exploring';
  setText('cur-phase', phase);

  // ── PPO ──
  const ppo = d.ppo || {};
  setText('ppo-device', ppo.device || '—');
  setText('ppo-rollout', `${ppo.rollout_buffer ?? 0} / ${ppo.batch_size * 8 || '?'}`);
  setText('ppo-replay', String(ppo.replay_buffer ?? 0));
  setText('ppo-update-every', String(ppo.update_every_steps ?? '?') + ' steps');
  setText('ppo-batch', String(ppo.batch_size ?? '?'));
  setText('ppo-policy-loss', ppo.policy_loss != null ? Number(ppo.policy_loss).toExponential(3) : '—');
  setText('ppo-value-loss', ppo.value_loss != null ? Number(ppo.value_loss).toExponential(3) : '—');
  setText('ppo-entropy', ppo.entropy != null ? Number(ppo.entropy).toFixed(5) : '—');
  setText('ppo-kl', ppo.kl_divergence != null ? Number(ppo.kl_divergence).toExponential(3) : (ppo.approx_kl != null ? Number(ppo.approx_kl).toExponential(3) : '—'));

  // ── Action distribution ──
  const distList = document.getElementById('action-dist-list');
  if (distList && d.planner?.top_actions?.length > 0) {
    const maxCount = Math.max(...d.planner.top_actions.map(a => a.count), 1);
    distList.innerHTML = d.planner.top_actions.map(a => `
      <li class="dist-item">
        <span class="dist-name">${a.name}</span>
        <div class="bar-track"><div class="bar-fill action-bar" style="width:${(a.count / maxCount * 100).toFixed(0)}%"></div></div>
        <span class="bar-count">${a.count}</span>
      </li>
    `).join('');
  }

  // ── State vector ──
  setText('state-dim', String(d.state?.dim ?? 0));
  setText('state-norm-val', (d.state?.norm ?? 0).toFixed(4));
  const norm = d.state?.norm ?? 0;
  setText('state-quality', norm === 0 ? '⚠️ No data' : norm < 5 ? '🟡 Sparse' : norm < 20 ? '🟢 Good' : '💪 Rich');

  // ── Health config ──
  const hw = d.health_config || {};
  setText('hw-ocr-thresh', String(hw.ocr_empty_threshold ?? '—'));
  setText('hw-mem-timeout', String(hw.mem_fail_timeout_s ?? '—'));
  setText('hw-min-fps', String(hw.capture_min_fps ?? '—'));

  // ── Checkpoints list ──
  const ckptList = document.getElementById('ckpt-list');
  if (ckptList && d.checkpoints?.length > 0) {
    ckptList.innerHTML = d.checkpoints.slice(-10).reverse().map(c => `<li class="dist-item"><span class="dist-name" style="width:auto;flex:1">${c}</span></li>`).join('');
  }
}

// ── Header controls ─────────────────────────────────────────────────────────
document.getElementById('btn-pause')?.addEventListener('click', () => {
  postJSON('/api/control', { cmd: 'pause' });
});
document.getElementById('btn-resume')?.addEventListener('click', () => {
  postJSON('/api/control', { cmd: 'resume' });
});

// ── Poll loop ───────────────────────────────────────────────────────────────
async function poll() {
  const stats = await getJSON('/api/stats');
  renderStats(stats);
}
poll();
setInterval(poll, POLL_MS);
