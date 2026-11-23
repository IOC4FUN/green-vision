/* Green Vision – Unified Dashboard + Stats (SPA v5.0)
   Simultaneous GPU-composited slide — both views animate together.
   No page reload, no flash, no blocking fetches.
*/
const POLL_MS = 2000;
let online = false;
let currentTab = 'dashboard';
let transitioning = false;
const visitedTabs = new Set(['dashboard']);



// ═══════════════════════════════════════════════════════════
// ANIMATED COUNTERS
// ═══════════════════════════════════════════════════════════
const animValues = {};
function animateValue(id, target, decimals = 0) {
  const el = document.getElementById(id);
  if (!el) return;
  const prev = animValues[id] ?? 0;
  if (prev === target) return;
  animValues[id] = target;
  el.classList.remove('pulse-up', 'pulse-down');
  void el.offsetWidth;
  el.classList.add(target > prev ? 'pulse-up' : 'pulse-down');
  setTimeout(() => el.classList.remove('pulse-up', 'pulse-down'), 500);
  const arrowEl = document.querySelector(`.trend-arrow[data-id="${id}"]`);
  if (arrowEl) {
    arrowEl.className = 'trend-arrow ' + (target > prev ? 'up' : target < prev ? 'down' : '');
    arrowEl.textContent = target > prev ? '\u25b2' : target < prev ? '\u25bc' : '';
  }
  const start = performance.now();
  const dur = 380;
  function step(now) {
    const t = Math.min(1, (now - start) / dur);
    const ease = 1 - Math.pow(1 - t, 3);
    const val = prev + (target - prev) * ease;
    el.textContent = decimals > 0 ? val.toFixed(decimals) : String(Math.round(val));
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ═══════════════════════════════════════════════════════════
// TOAST NOTIFICATIONS
// ═══════════════════════════════════════════════════════════
function toast(msg, type = 'info') {
  const c = document.getElementById('toast-container');
  if (!c) return;
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  c.appendChild(el);
  setTimeout(() => el.remove(), 2600);
}

// ═══════════════════════════════════════════════════════════
// TAB SWITCHING — simultaneous GPU keyframe slide
// Both views animate in the same frame: old exits, new enters.
// Data fetches run in background; UI never blocks on network.
// ═══════════════════════════════════════════════════════════
const ANIM_CLASSES = ['exit-left','exit-right','enter-from-right','enter-from-left'];

function moveTabIndicator(activeLink) {
  const nav = document.querySelector('.tab-nav');
  const ind = document.querySelector('.tab-indicator');
  if (!nav || !ind || !activeLink) return;
  const navRect = nav.getBoundingClientRect();
  const linkRect = activeLink.getBoundingClientRect();
  ind.style.width = linkRect.width + 'px';
  ind.style.left = (linkRect.left - navRect.left) + 'px';
  ind.classList.remove('indicator-pop');
  void ind.offsetWidth;
  ind.classList.add('indicator-pop');
}

function setAmbientTint(tab) {
  const glow = document.getElementById('ambient-glow');
  if (!glow) return;
  glow.classList.toggle('tint-stats', tab === 'stats');
}

async function switchTab(tab) {
  if (tab === currentTab || transitioning) return;
  transitioning = true;

  // Update tab links + indicator immediately
  const link = document.querySelector(`.tab-link[data-tab="${tab}"]`);
  document.querySelectorAll('.tab-link').forEach(l => l.classList.remove('active'));
  if (link) { link.classList.add('active'); moveTabIndicator(link); }

  const oldView = document.getElementById('view-' + currentTab);
  const newView = document.getElementById('view-' + tab);
  const goingRight = tab === 'stats';

  const vc = document.querySelector('.views-container');

  // Kick off data fetch in background — does NOT block the animation
  fireBackgroundFetch(tab);

  // Clean stale animation classes on both views
  [oldView, newView].forEach(v => v.classList.remove(...ANIM_CLASSES));

  // Reset scroll to top for clean entry
  if (vc) vc.scrollTop = 0;

  history.replaceState(null, '', '#' + tab);
  setAmbientTint(tab);

  // ── Simultaneous transition: both views animate in the same frame ──
  // Force a style recalc so the browser registers the "from" state
  void oldView.offsetWidth;

  oldView.classList.add(goingRight ? 'exit-left' : 'exit-right');
  newView.classList.add('active', goingRight ? 'enter-from-right' : 'enter-from-left');

  // Stagger cards on first visit only
  if (!visitedTabs.has(tab)) {
    visitedTabs.add(tab);
    newView.classList.add('stagger-in');
    setTimeout(() => newView.classList.remove('stagger-in'), 650);
  }

  // Wait for enter animation to finish (0.3s × 1.15 ≈ 345ms + buffer)
  await new Promise(r => setTimeout(r, 380));

  // Cleanup
  oldView.classList.remove('active', ...ANIM_CLASSES);
  newView.classList.remove(...ANIM_CLASSES);
  transitioning = false;
  currentTab = tab;
}

// Background data fetch — renders when ready, never blocks UI
function fireBackgroundFetch(tab) {
  if (tab === 'stats') {
    getJSON('/api/stats').then(d => { if (d && !d.error) renderStats(d); });
  } else {
    Promise.all([
      getJSON('/api/state'), getJSON('/api/health'),
      getJSON('/api/goals'), getJSON('/api/thoughts'), getJSON('/api/actions')
    ]).then(([st, h, g, th, ac]) => {
      if (st && !st.error) renderState(st);
      if (h) renderHealth(h);
      if (g) renderGoals(g);
      if (th) renderThoughts(th);
      if (ac) renderActions(ac);
    });
  }
}

document.querySelectorAll('.tab-link').forEach(link => {
  link.addEventListener('click', e => { e.preventDefault(); switchTab(link.dataset.tab); });
});

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === '1') switchTab('dashboard');
  else if (e.key === '2') switchTab('stats');
  else if (e.key === 'ArrowLeft') switchTab('dashboard');
  else if (e.key === 'ArrowRight') switchTab('stats');
});

(function initFromHash() {
  const h = location.hash.replace('#','');
  if (h === 'stats' || h === 'dashboard') {
    if (h !== currentTab) {
      document.getElementById('view-' + currentTab).classList.remove('active');
      document.getElementById('view-' + h).classList.add('active');
      document.querySelectorAll('.tab-link').forEach(l => l.classList.toggle('active', l.dataset.tab === h));
      currentTab = h;
      visitedTabs.add(h);
      setAmbientTint(h);
    }
  }
})();

window.addEventListener('load', () => {
  const active = document.querySelector('.tab-link.active');
  if (active) moveTabIndicator(active);
});
window.addEventListener('resize', () => {
  const active = document.querySelector('.tab-link.active');
  if (active) moveTabIndicator(active);
});

// ═══════════════════════════════════════════════════════════
// CLOCK
// ═══════════════════════════════════════════════════════════
function tickClock() { document.getElementById('clock').textContent = new Date().toLocaleTimeString(); }
setInterval(tickClock, 1000); tickClock();

// ═══════════════════════════════════════════════════════════
// BADGE
// ═══════════════════════════════════════════════════════════
let wasOnline = false;
function setBadge(on) {
  const el = document.getElementById('status-badge');
  if (on !== online) {
    online = on;
    el.textContent = on ? 'ONLINE' : 'OFFLINE';
    el.className = 'badge ' + (on ? 'online' : 'offline');
    if (wasOnline && !on) toast('Connection lost', 'error');
    else if (!wasOnline && on) toast('Connected to agent', 'success');
    wasOnline = on;
  }
}

// ═══════════════════════════════════════════════════════════
// FETCH HELPERS
// ═══════════════════════════════════════════════════════════
async function getJSON(url) {
  try { const r = await fetch(url, { cache: 'no-store' }); if (!r.ok) throw new Error(r.status); return await r.json(); } catch(e) { console.warn(`GET ${url}:`, e.message); return null; }
}
async function postJSON(url, body) {
  try { const r = await fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body) }); return await r.json(); } catch(e) { return { error:e.message }; }
}
function setText(id, val) { const el = document.getElementById(id); if (el) el.textContent = val; }
function setBarWidth(id, pct) { const el = document.getElementById(id); if (el) el.style.width = Math.max(0, Math.min(100, pct)) + '%'; }

// ═══════════════════════════════════════════════════════════
// DASHBOARD RENDERERS
// ═══════════════════════════════════════════════════════════
function renderState(d) {
  if (!d || d.error) { setBadge(false); return; }
  setBadge(true);
  setText('scenario', d.scenario || '\u2014');
  animateValue('decisions', d.total_decisions ?? 0);
  setText('episode', d.episode_active ? 'yes' : 'no');
  const streak = d.no_op_streak ?? 0;
  const stEl = document.getElementById('noop-streak');
  if (stEl) { stEl.textContent = String(streak); stEl.className = 'stat-value' + (streak > 5 ? ' warn' : ''); }
  animateValue('frames', d.frame_count ?? 0);
  setText('state-norm', (d.state_norm ?? 0).toFixed(2));
  setText('console', d.console_enabled ? 'ON' : 'off');
  setText('decision-interval', (d.decision_interval_s ?? 0.5) + 's');
  const modeEl = document.getElementById('ai-mode');
  if (modeEl) { modeEl.textContent = d.autonomous_paused ? '\u23f8 PAUSED' : 'autonomous'; modeEl.style.color = d.autonomous_paused ? '#d29922' : '#3fb950'; }
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
  if (!ul) return;
  if (!data || !data.active || data.active.length === 0) { ul.innerHTML = '<li class="muted">No active goals</li>'; return; }
  ul.innerHTML = data.active.slice(0,8).map(g => {
    const prog = Math.round((g.progress ?? 0)*100);
    const pri = (g.priority||'MEDIUM').toUpperCase();
    const desc = (g.desc||g.description||'?').slice(0,45);
    return `<li><span class="goal-pri pri-${pri}">${pri}</span>${desc}<div class="goals-progress"><div class="goals-progress-fill" style="width:${prog}%"></div></div></li>`;
  }).join('');
}
let lastThoughtCount = 0;
function renderThoughts(thoughts) {
  const feed = document.getElementById('thoughts-feed');
  if (!feed) return;
  if (!thoughts || thoughts.length === 0) { feed.innerHTML = '<p class="muted">No thoughts yet \u2014 start the game to see AI reasoning.</p>'; return; }
  lastThoughtCount = thoughts.length;
  feed.innerHTML = [...thoughts].reverse().slice(0,20).map(t => {
    const ts = t.ts ? new Date(t.ts*1000).toLocaleTimeString() : '';
    const conf = t.confidence ?? 0;
    return `<div class="thought-item"><span class="ts">${ts}</span> \u2192 <span class="decision">${t.decision||'?'}</span><span class="conf-bar"><span class="conf-fill" style="width:${Math.round(conf*100)}%"></span></span>${(t.reasoning||'').slice(0,140)?`<div class="reasoning">${(t.reasoning||'').slice(0,140)}</div>`:''}</div>`;
  }).join('');
}
function renderActions(actions) {
  const tbody = document.querySelector('#actions-table tbody');
  if (!tbody) return;
  if (!actions || actions.length === 0) { tbody.innerHTML = '<tr><td colspan="3" class="muted">No actions recorded yet</td></tr>'; return; }
  tbody.innerHTML = [...actions].reverse().map(a => {
    const ts = a.ts ? new Date(a.ts*1000).toLocaleTimeString() : '';
    return `<tr><td>${ts}</td><td>${a.action||'?'}</td><td>${(a.detail||a.description||'').slice(0,60)}</td></tr>`;
  }).join('');
}

// ═══════════════════════════════════════════════════════════
// MANUAL CONTROLS
// ═══════════════════════════════════════════════════════════
async function loadActionButtons() {
  const groups = await getJSON('/api/actions-available');
  const container = document.getElementById('action-groups');
  if (!container) return;
  if (!groups || !Array.isArray(groups)) { container.innerHTML = '<p class="muted">Could not load actions.</p>'; return; }
  container.innerHTML = groups.map(g => `<div class="action-group"><h4>${g.group}</h4><div class="action-btns">${g.actions.map(a=>`<button class="action-btn" data-action="${a.name}" title="${a.name}">${a.label}</button>`).join('')}</div></div>`).join('');
  container.querySelectorAll('.action-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const res = await postJSON('/api/action', { action: btn.dataset.action });
      if (res && res.ok) {
        btn.classList.remove('flash'); void btn.offsetWidth; btn.classList.add('flash');
        toast(`\u2713 ${btn.dataset.action} sent`, 'success');
      } else {
        toast(`\u2717 Action failed: ${res?.error||'unknown'}`, 'error');
      }
    });
  });
}

// ═══════════════════════════════════════════════════════════
// STATS RENDERERS + MATURITY COMPUTATION
// ═══════════════════════════════════════════════════════════
let lastStatsData = null;
function computeMaturity(d) {
  const m = { trend:0, consistency:0, breadth:0, success:0, explore:0, overall:0, stage:'Initialising\u2026' };
  const rh = d.reward_history || [];
  if (rh.length >= 8) {
    const q = Math.floor(rh.length/4);
    const firstQ = rh.slice(0,q).reduce((a,b)=>a+b,0)/q;
    const lastQ = rh.slice(-q).reduce((a,b)=>a+b,0)/q;
    const denom = Math.max(Math.abs(firstQ), 0.5);
    m.trend = Math.round(Math.min(100, Math.max(0, ((lastQ-firstQ)/denom)*100)));
  }
  if (rh.length >= 10) {
    const mean = rh.reduce((a,b)=>a+b,0)/rh.length;
    const variance = rh.reduce((s,v)=>s+(v-mean)**2,0)/rh.length;
    const stdDev = Math.sqrt(variance);
    const scale = Math.max(Math.abs(mean), 0.3);
    m.consistency = Math.round(Math.max(0, 100*(1-stdDev/(scale*2))));
  }
  const topActions = d.planner?.top_actions || [];
  const totalActions = d.planner?.actions_total || 0;
  if (totalActions > 0) {
    const uniqueKnown = new Set(topActions.map(a=>a.name)).size;
    const topSum = topActions.reduce((s,a)=>s+a.count,0);
    const estUnique = uniqueKnown + Math.min(totalActions-topSum, 21);
    m.breadth = Math.round(Math.min(100,(estUnique/31)*100));
  }
  const ep = d.episodes || {};
  if ((ep.total||0) > 0) m.success = Math.round(((ep.success||0)/ep.total)*100);
  const cur = d.curiosity || {};
  const avgIntr = cur.avg_intrinsic ?? 0;
  const numSteps = cur.num_steps ?? 0;
  if (numSteps >= 100) m.explore = Math.round(Math.max(0, 100*(1-avgIntr/0.05)));
  else if (numSteps > 0) m.explore = Math.round(numSteps/100*30);
  m.overall = Math.round(m.trend*0.30+m.consistency*0.20+m.breadth*0.15+m.success*0.20+m.explore*0.15);
  if (m.overall>=80) m.stage='\ud83c\udfc6 Expert \u2014 policy is stable and broad';
  else if (m.overall>=60) m.stage='\ud83d\udcaa Competent \u2014 solid decisions with room to grow';
  else if (m.overall>=40) m.stage='\ud83d\udcc8 Developing \u2014 learning new strategies';
  else if (m.overall>=20) m.stage='\ud83c\udf31 Emerging \u2014 building basic competence';
  else m.stage='\ud83d\udd30 Initialising \u2014 early exploration phase';
  return m;
}
function drawRewardChart(data, bestVal) {
  const canvas = document.getElementById('reward-chart');
  if (!canvas || !data || data.length === 0) return;
  const ctx = canvas.getContext('2d');
  const W=canvas.width, H=canvas.height;
  const pad={top:12,right:20,bottom:30,left:55};
  const plotW=W-pad.left-pad.right, plotH=H-pad.top-pad.bottom;
  ctx.clearRect(0,0,W,H);
  let minV=Math.min(...data), maxV=Math.max(...data);
  if(minV===maxV){minV-=0.5;maxV+=0.5;}
  const range=maxV-minV;
  ctx.strokeStyle='rgba(48,54,61,0.6)';ctx.lineWidth=1;
  for(let i=0;i<=4;i++){
    const y=pad.top+(plotH/4)*i;
    ctx.beginPath();ctx.moveTo(pad.left,y);ctx.lineTo(W-pad.right,y);ctx.stroke();
    const val=maxV-(range/4)*i;
    ctx.fillStyle='#8b949e';ctx.font='11px Segoe UI';ctx.textAlign='right';
    ctx.fillText(val.toFixed(2),pad.left-6,y+4);
  }
  ctx.textAlign='center';ctx.fillStyle='#8b949e';
  ctx.fillText('0',pad.left,H-8);
  ctx.fillText(String(Math.floor(data.length/2)),pad.left+plotW/2,H-8);
  ctx.fillText(String(data.length-1),W-pad.right,H-8);
  if(data.length>1){
    ctx.beginPath();ctx.strokeStyle='#3fb950';ctx.lineWidth=2;ctx.lineJoin='round';
    for(let i=0;i<data.length;i++){
      const x=pad.left+(plotW/(data.length-1))*i;
      const y=pad.top+plotH-((data[i]-minV)/range)*plotH;
      if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
    }
    ctx.stroke();
    const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top+plotH);
    grad.addColorStop(0, 'rgba(63,185,80,0.2)');
    grad.addColorStop(1, 'rgba(63,185,80,0.0)');
    ctx.lineTo(pad.left+plotW,pad.top+plotH);ctx.lineTo(pad.left,pad.top+plotH);ctx.closePath();
    ctx.fillStyle=grad;ctx.fill();
  }
  if(bestVal!=null&&bestVal>minV&&bestVal<maxV){
    const by=pad.top+plotH-((bestVal-minV)/range)*plotH;
    ctx.setLineDash([4,4]);ctx.strokeStyle='#d29922';ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(pad.left,by);ctx.lineTo(W-pad.right,by);ctx.stroke();
    ctx.setLineDash([]);ctx.fillStyle='#d29922';ctx.font='10px Segoe UI';ctx.textAlign='left';
    ctx.fillText(`best: ${bestVal.toFixed(3)}`,W-pad.right-80,by-5);
  }
}
function renderStats(d) {
  lastStatsData = d;
  if (!d || d.error) return;
  const m = computeMaturity(d);
  setText('mat-overall', m.overall+'%');
  setText('mat-stage-label', m.stage);
  setBarWidth('mat-overall-bar', m.overall);
  setText('mat-trend-score',(m.trend>=50?'\u25b2':m.trend>0?'\u2197':'\u2014')+' '+m.trend+'%');
  document.getElementById('mat-trend-score').className='maturity-sub-score '+(m.trend>=50?'trend-up':m.trend>0?'trend-flat':'trend-down');
  setBarWidth('mat-trend-bar',m.trend);
  setText('mat-consist-score',m.consistency+'%');
  document.getElementById('mat-consist-score').className='maturity-sub-score '+(m.consistency>=60?'trend-up':'trend-flat');
  setBarWidth('mat-consist-bar',m.consistency);
  setText('mat-breadth-score',m.breadth+'%');
  document.getElementById('mat-breadth-score').className='maturity-sub-score '+(m.breadth>=50?'trend-up':'trend-flat');
  setBarWidth('mat-breadth-bar',m.breadth);
  setText('mat-success-score',m.success+'%');
  document.getElementById('mat-success-score').className='maturity-sub-score '+(m.success>=50?'trend-up':'trend-flat');
  setBarWidth('mat-success-bar',m.success);
  setText('mat-explore-score',m.explore+'%');
  document.getElementById('mat-explore-score').className='maturity-sub-score '+(m.explore>=60?'trend-up':'trend-flat');
  setBarWidth('mat-explore-bar',m.explore);
  animateValue('stat-decisions', d.planner?.total_decisions??0);
  setText('stat-best-reward',(d.best_reward??0).toFixed(4));
  const rh=d.reward_history||[];
  const curAvg=rh.length>0?(rh.reduce((a,b)=>a+b,0)/rh.length).toFixed(4):'\u2014';
  setText('stat-cur-avg-reward',curAvg);
  animateValue('stat-episodes', d.episodes?.total??0);
  setText('stat-goals-done',`${d.planner?.goals_completed??0}/${d.planner?.goals_abandoned??0}`);
  animateValue('stat-checkpoints', d.checkpoints?.length??0);
  drawRewardChart(rh,d.best_reward);
  const capEl=document.getElementById('reward-caption');
  if(capEl&&rh.length>0)capEl.textContent=`Window: ${rh.length} steps \u00b7 Avg: ${curAvg} \u00b7 Best: ${(d.best_reward??0).toFixed(4)} \u00b7 Trend: ${m.trend>=50?'improving \u25b2':m.trend>0?'slight gain \u2197':'flat/declining \u25bc'}`;
  const ep=d.episodes||{};const totalEp=Math.max(1,ep.total||1);
  setBarWidth('bar-success',((ep.success||0)/totalEp*100));setBarWidth('bar-stall',((ep.stall||0)/totalEp*100));setBarWidth('bar-failure',((ep.failure||0)/totalEp*100));
  setText('count-success',String(ep.success??0));setText('count-stall',String(ep.stall??0));setText('count-failure',String(ep.failure??0));
  const tbody=document.querySelector('#episode-avg-table tbody');
  if(tbody)tbody.innerHTML=[['Success',ep.avg_reward_success,ep.success],['Stall',ep.avg_reward_stall,ep.stall],['Failure',ep.avg_reward_failure,ep.failure]].map(([l,v,c])=>`<tr><td>${l}</td><td>${v!=null?Number(v).toFixed(4):'\u2014'}</td><td>${c??0}</td></tr>`).join('');
  const cur=d.curiosity||{};
  setText('cur-total',(cur.total_intrinsic??0).toFixed(4));setText('cur-steps',String(cur.num_steps??0));setText('cur-avg',(cur.avg_intrinsic??0).toExponential(3));
  setText('cur-phase',(cur.avg_intrinsic??0)<0.005?'\ud83d\uddfa\ufe0f Mapped':(cur.avg_intrinsic??0)<0.02?'\ud83d\udd0d Refining':'\ud83d\ude80 Exploring');
  const ppo=d.ppo||{};
  setText('ppo-device',ppo.device||'\u2014');setText('ppo-rollout',`${ppo.rollout_buffer??0} / ${ppo.batch_size*8||'?'}`);setText('ppo-replay',String(ppo.replay_buffer??0));
  setText('ppo-update-every',String(ppo.update_every_steps??'?')+' steps');setText('ppo-batch',String(ppo.batch_size??'?'));
  setText('ppo-policy-loss',ppo.policy_loss!=null?Number(ppo.policy_loss).toExponential(3):'\u2014');
  setText('ppo-value-loss',ppo.value_loss!=null?Number(ppo.value_loss).toExponential(3):'\u2014');
  setText('ppo-entropy',ppo.entropy!=null?Number(ppo.entropy).toFixed(5):'\u2014');
  setText('ppo-kl',ppo.kl_divergence!=null?Number(ppo.kl_divergence).toExponential(3):(ppo.approx_kl!=null?Number(ppo.approx_kl).toExponential(3):'\u2014'));
  const distList=document.getElementById('action-dist-list');
  if(distList&&d.planner?.top_actions?.length>0){const maxC=Math.max(...d.planner.top_actions.map(a=>a.count),1);distList.innerHTML=d.planner.top_actions.map(a=>`<li class="dist-item"><span class="dist-name">${a.name}</span><div class="bar-track"><div class="bar-fill action-bar" style="width:${(a.count/maxC*100).toFixed(0)}%"></div></div><span class="bar-count">${a.count}</span></li>`).join('');}
  setText('state-dim',String(d.state?.dim??0));setText('state-norm-val',(d.state?.norm??0).toFixed(4));
  const norm=d.state?.norm??0;setText('state-quality',norm===0?'\u26a0\ufe0f No data':norm<5?'\ud83d\udfe1 Sparse':norm<20?'\ud83d\udfe2 Good':'\ud83d\udcaa Rich');
  const hw=d.health_config||{};
  setText('hw-ocr-thresh',String(hw.ocr_empty_threshold??'\u2014'));setText('hw-mem-timeout',String(hw.mem_fail_timeout_s??'\u2014'));setText('hw-min-fps',String(hw.capture_min_fps??'\u2014'));
  const ckptList=document.getElementById('ckpt-list');
  if(ckptList&&d.checkpoints?.length>0)ckptList.innerHTML=d.checkpoints.slice(-10).reverse().map(c=>`<li class="dist-item"><span class="dist-name" style="width:auto;flex:1">${c}</span></li>`).join('');
}

// ═══════════════════════════════════════════════════════════
// HEADER CONTROLS
// ═══════════════════════════════════════════════════════════
document.getElementById('btn-pause').addEventListener('click', async () => {
  const r = await postJSON('/api/control',{cmd:'pause'});
  toast(r?.ok ? '\u23f8 AI paused' : 'Pause failed', r?.ok ? 'info' : 'error');
});
document.getElementById('btn-resume').addEventListener('click', async () => {
  const r = await postJSON('/api/control',{cmd:'resume'});
  toast(r?.ok ? '\u25b6 AI resumed' : 'Resume failed', r?.ok ? 'success' : 'error');
});
document.getElementById('btn-stop').addEventListener('click', async () => {
  if(!confirm('Stop the agent entirely? This cannot be undone without restarting.')) return;
  const r = await postJSON('/api/control',{cmd:'stop'});
  toast(r?.ok ? '\u23f9 Agent stopped' : 'Stop failed', r?.ok ? 'info' : 'error');
});

// ═══════════════════════════════════════════════════════════
// UNIFIED POLL LOOP — renders both views regardless of active tab
// so switching back always shows fresh data instantly.
// ═══════════════════════════════════════════════════════════
async function poll() {
  const [state,health,goals,thoughts,actions,stats] = await Promise.all([
    getJSON('/api/state'),getJSON('/api/health'),getJSON('/api/goals'),
    getJSON('/api/thoughts'),getJSON('/api/actions'),getJSON('/api/stats')
  ]);
  renderState(state);renderHealth(health);renderGoals(goals);renderThoughts(thoughts);renderActions(actions);
  if (stats) renderStats(stats);
}
poll();
setInterval(poll, POLL_MS);
loadActionButtons();
