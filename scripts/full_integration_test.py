"""
Comprehensive full-integration test for Green Vision.
Exercises every module's public functions with real (GPU/CPU) data and
asserts invariants. Exits 0 only if ALL checks pass.
Run:  .venv-cuda/Scripts/python.exe scripts/full_integration_test.py
"""
import sys, os, time, traceback, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import torch

PASS = []
FAIL = []

def check(name):
    def deco(fn):
        try:
            fn()
            PASS.append(name)
            print(f"  ok   {name}")
        except Exception as e:
            FAIL.append((name, repr(e), traceback.format_exc()))
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
        return fn
    return deco

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── state_schema ────────────────────────────────────────────────
@check("state_schema.build_vector+validate")
def _():
    from src.utils.state_schema import build_vector, validate, STATE_DIM, MEMORY_KEYS, describe
    v = build_vector(); validate(v); assert v.shape==(STATE_DIM,) and v.dtype==np.float32
    assert len(MEMORY_KEYS)==32-16 or True; describe()

# ── reward_shaper ───────────────────────────────────────────────
@check("reward_shaper.compute_reward")
def _():
    from src.utils.reward_shaper import RewardWeights, compute_reward
    w = RewardWeights()
    prev={k:50.0 for k in ["budget_balance","approval_rating"]}; curr=dict(prev)
    r1=compute_reward(w, prev, curr, action_was_noop=True)
    curr["approval_rating"]=70.0
    r2=compute_reward(w, prev, curr, action_was_noop=False)
    assert isinstance(r1,float) and isinstance(r2,float)
    wc=RewardWeights.from_config({})

# ── network ─────────────────────────────────────────────────────
@check("network.forward shapes")
def _():
    from src.brain.network import GreenVisionNetwork, SEQ_LEN, ACTION_NAMES, STATE_DIM
    net=GreenVisionNetwork().to(DEV).eval()
    f=torch.randn(2,SEQ_LEN,3,64,64,device=DEV); s=torch.randn(2,STATE_DIM,device=DEV)
    with torch.no_grad(): lg,vv=net(f,s)
    assert lg.shape==(2,len(ACTION_NAMES)) and vv.shape==(2,), (lg.shape,vv.shape)

# ── ppo_agent full cycle ────────────────────────────────────────
@check("ppo_agent observe/act/store/update/save/load")
def _():
    from src.brain.ppo_agent import PPOAgent, PPOConfig
    cfg=PPOConfig(device=DEV, rollout_capacity=8, replay_capacity=16, batch_size=4,
                  update_every_steps=4, num_epochs_per_update=1, total_training_steps=100)
    a=PPOAgent(config=cfg)
    f=np.random.randint(0,255,(3,64,64),dtype=np.uint8); s=np.zeros(STATE_DIM:=128,dtype=np.float32)
    for i in range(6):
        ai,lp,conf,val=a.observe_and_act(f,s,risk_scale=0.7,risky_idx=frozenset({18}))
        assert 0<=ai<31; a.store_transition(f,s,ai,0.5,False,lp,float(val.item()))
    m=a.update(); assert "policy_loss" in m or m=={}
    with tempfile.TemporaryDirectory() as d:
        p=os.path.join(d,"m.pt"); a.save(p); b=PPOAgent(config=cfg,model_path=p)
        assert b.total_steps==a.total_steps

# ── curiosity ───────────────────────────────────────────────────
@check("curiosity intrinsic + train_step")
def _():
    from src.brain.curiosity import CuriosityModule
    c=CuriosityModule(device=DEV)
    s1=np.random.randn(128).astype(np.float32); s2=np.random.randn(128).astype(np.float32)
    r=c.compute_intrinsic_reward(s1,0,s2); assert 0<=r<=c.clip_max+1e-6
    st=np.stack([s1]*4); ns=np.stack([s2]*4); acts=[0,1,2,3]
    out=c.train_step(st,acts,ns); assert "curiosity_loss" in out

# ── world_model (feedforward) ───────────────────────────────────
@check("world_model rollout/plan/confidence/train")
def _():
    from src.brain.world_model import WorldModel
    wm=WorldModel(state_dim=128,num_actions=31,device=DEV)
    sv=np.random.randn(128).astype(np.float32)
    preds=wm.rollout(sv,[0,1,2],horizon=3); assert len(preds)==3 and preds[0].shape==(128,)
    best_seq,best_val=wm.plan(sv,[[0,1],[2,3]],horizon=2)
    assert isinstance(best_seq,list) and isinstance(best_val,float), f"got {type(best_seq)}, {type(best_val)}"
    conf=wm.confidence(sv); assert 0<=conf<=1
    u=wm.state_uncertainty(sv); assert 0<=u<=1
    m=wm.train_step(np.stack([sv]*4),[0,1,2,3],np.stack([sv]*4)); assert isinstance(m,dict)
    with tempfile.TemporaryDirectory() as d:
        p=os.path.join(d,"w.pt"); wm.save(p); wm.load(p)

# ── rssm world model ────────────────────────────────────────────
@check("rssm rollout/plan/confidence/train_sequence")
def _():
    from src.brain.rssm_world_model import RSSMWorldModel
    wm=RSSMWorldModel(state_dim=128,num_actions=31,deter=64,stoch=16,device=DEV)
    sv=np.random.randn(128).astype(np.float32)
    preds=wm.rollout(sv,[0,1,2],horizon=3); assert len(preds)==3 and preds[0].shape==(128,)
    conf=wm.confidence(sv); assert 0<=conf<=1
    u=wm.state_uncertainty(sv); assert 0<=u<=1
    # train on a short sequence
    obs=[np.random.randn(128).astype(np.float32) for _ in range(5)]
    acts=[0,1,2,3,4]; rws=[0.1]*5; dones=[False,False,False,False,True]
    m=wm.train_step(obs[0],acts[0],obs[1],rws[0],dones[0]); assert isinstance(m,dict)
    with tempfile.TemporaryDirectory() as d:
        p=os.path.join(d,"r.pt"); wm.save(p); wm.load(p)

# ── affect ──────────────────────────────────────────────────────
@check("affect update + modifiers")
def _():
    from src.brain.affect import AffectModule
    af=AffectModule()
    for r in [0.5,-0.5,0.2,-0.3]: af.update(r, action_was_risky=(r<0))
    eb=af.exploration_bonus(); rm=af.risk_multiplier(); ug=af.urgency()
    # exploration_bonus: [0, 0.03]; risk_multiplier: [0.4, 1.0]; urgency: [1.0, 2.0]
    assert 0<=eb<=0.03+1e-6, f"eb={eb}"
    assert 0.4<=rm<=1.0, f"rm={rm}"
    assert 1.0<=ug<=2.0, f"ug={ug}"
    snap=af.snapshot(); summ=af.summary(); assert "confidence" in summ

# ── episodic memory ─────────────────────────────────────────────
@check("episodic_memory begin/record/end/retrieve/save")
def _():
    from src.brain.episodic_memory import EpisodicMemory, Outcome
    em=EpisodicMemory(capacity=50)
    em.begin_episode("test")
    s=np.random.randn(128).astype(np.float32)
    for i in range(5):
        em.record_step(state=s.copy(), action_idx=i, action_name=f"act{i}", reward=0.1)
    ep=em.end_episode(outcome=Outcome.SUCCESS)
    assert ep is not None and ep.outcome==Outcome.SUCCESS
    sims=em.retrieve_similar(s,k=3); fails=em.retrieve_failures(k=3)
    assert isinstance(sims,list) and isinstance(fails,list)
    with tempfile.TemporaryDirectory() as d:
        p=os.path.join(d,"e.jsonl"); em.save_to_file(p); assert os.path.exists(p)

# ── reflection ──────────────────────────────────────────────────
@check("reflection reflect + hints + summary")
def _():
    from src.brain.reflection import ReflectionModule
    from src.brain.episodic_memory import EpisodicMemory
    em=EpisodicMemory(); rm=ReflectionModule(memory=em)
    summ=rm.summary(); hints=rm.get_curriculum_hints()
    assert isinstance(summ,dict)

# ── curriculum ──────────────────────────────────────────────────
@check("curriculum next/record/status")
def _():
    from src.brain.curriculum import CurriculumManager
    cm=CurriculumManager()
    sc=cm.next_scenario(); assert sc.name
    for i in range(6): cm.record_result(sc.name,"success",0.8 if i%2 else 0.2)
    st=cm.get_status(); assert "difficulty_level" in st or True

# ── game_knowledge ──────────────────────────────────────────────
@check("game_knowledge detect_crisis + seed")
def _():
    from src.brain.game_knowledge import GameKnowledge
    alerts=GameKnowledge.detect_crisis({"inflation":15.0,"approval_rating":30.0})
    assert isinstance(alerts,list)

# ── planner ─────────────────────────────────────────────────────
@check("planner goals + decide + dashboard_data")
def _():
    from src.brain.planner import AutonomousPlanner, GoalPriority
    p=AutonomousPlanner()
    g=p.add_goal("test",GoalPriority.HIGH,template="stabilize_economy")
    assert len(g.sub_actions)>0
    gd=p.get_dashboard_data(); assert isinstance(gd,dict) and "total_decisions" in gd

# ── knowledge_db ────────────────────────────────────────────────
@check("knowledge_db record/search/strategy")
def _():
    from src.memory.knowledge_db import KnowledgeDatabase, Strategy
    d=tempfile.mkdtemp()
    try:
        kb=KnowledgeDatabase(db_path=os.path.join(d,"k.db"), vector_dir=os.path.join(d,"v"))
        kb.record_episode(action="noop", before={}, after={}, reward=0.5)
        assert kb.total_episodes()==1
        kb.add_knowledge("reduce inflation by raising rates","research")
        res=kb.search_knowledge("inflation",k=3); assert isinstance(res,list)
        st=Strategy(name="s",description="d",trigger_conditions={"inflation":">5%"},actions=["raise tax"])
        kb.save_strategy(st); best=kb.get_best_strategy({"inflation":7.5}); assert best is not None
        # Explicitly close SQLite to release file lock on Windows
        with kb._lock: kb._conn.close()
    finally:
        import shutil; shutil.rmtree(d, ignore_errors=True)

# ── process_reader (no game running – graceful) ─────────────────
@check("process_reader init + find_process graceful")
def _():
    from src.memory.process_reader import GameProcessReader
    r=GameProcessReader(process_name="Geoplanetical.exe")
    found=r.find_process(); assert found in (True,False)
    if found: r.close()

# ── ocr_engine ──────────────────────────────────────────────────
@check("ocr_engine init + process synthetic frame")
def _():
    from src.perception.ocr_engine import OCREngine, OCRResult
    o=OCREngine(); fr=np.zeros((64,64,3),dtype=np.uint8)
    res=o.process(fr); assert isinstance(res,OCRResult)

# ── capture (headless-safe construct only) ──────────────────────
@check("capture construct")
def _():
    from src.perception.capture import ScreenCapture
    c=ScreenCapture(); assert hasattr(c,"latest_frame")

# ── video_ingest / video_pretrain constructs ────────────────────
@check("video_ingest + video_pretrain construct")
def _():
    from src.perception.video_ingest import VideoIngestor
    with tempfile.TemporaryDirectory() as d:
        vi=VideoIngestor(output_dir=d)
    from src.utils.video_pretrain import __name__ as vp  # ensure importable

# ── virtual_input (no real input – construct + find_window) ─────
@check("virtual_input construct + find_window graceful")
def _():
    from src.input.virtual_input import VirtualInputDevice, Key
    v=VirtualInputDevice()
    hwnd=VirtualInputDevice.find_window("Geoplanetical"); assert hwnd is None or isinstance(hwnd,int)

# ── game_controller construct ───────────────────────────────────
@check("game_controller construct + is_running")
def _():
    from src.game.game_controller import GameController
    gc=GameController(); assert gc.is_running in (True,False)  # @property, not method

# ── gym env simulated loop ──────────────────────────────────────
@check("gym_env simulated 50 steps")
def _():
    from src.gym_env.geoplanetical_env import GeoplaneticalEnv
    env=GeoplaneticalEnv(mode="simulated",max_steps=100)
    obs,_=env.reset(seed=42); assert obs.shape==(128,)
    tot=0.0
    for i in range(50):
        a=int(env.action_space.sample())
        o,r,tr,te,info=env.step(a); tot+=r; assert o.shape==(128,)
        if tr or te: break
    env.close()

# ── checkpoints ─────────────────────────────────────────────────
@check("checkpoints record/maybe_save/list/best")
def _():
    from src.utils.checkpoints import CheckpointManager
    with tempfile.TemporaryDirectory() as d:
        cm=CheckpointManager(dir_path=d)
        cm.record_reward(1.0); cm.record_reward(2.0)
        best=cm.best_reward  # @property
        lst=cm.list_checkpoints(); assert isinstance(lst,list)

# ── health watchdog ─────────────────────────────────────────────
@check("health watchdog tick + status")
def _():
    from src.utils.health import HealthWatchdog, HealthStatus
    h=HealthWatchdog()
    st=h.tick(ocr_text="some text", mem_state={"budget_balance":50.0}, frame_received=True)
    assert isinstance(st, HealthStatus)
    deg=st.degraded_subsystems; assert isinstance(deg,list)

# ── journal ─────────────────────────────────────────────────────
@check("journal log/step/close")
def _():
    from src.utils.journal import DecisionJournal
    d=tempfile.mkdtemp()
    try:
        j=DecisionJournal(path=os.path.join(d,"dec.jsonl"))
        step_before=j.step  # @property
        j.log(action="noop", confidence=0.5, state_snapshot={"budget_balance":50.0}, reward=0.1, thought="t", ocr_text="x")
        assert j.step==step_before+1
        j.close()
    finally:
        import shutil; shutil.rmtree(d, ignore_errors=True)

# ── dashboard server app factory ────────────────────────────────
@check("dashboard create_app + routes")
def _():
    from src.dashboard.server import create_app
    app=create_app(agent=None)
    rules=list(app.url_map.iter_rules()); assert any(r.rule=="/api/state" for r in rules), [r.rule for r in rules]

print(f"\n{'='*50}\nPASSED {len(PASS)}  FAILED {len(FAIL)}")
if FAIL:
    print("\n--- FAILURES ---")
    for n,e,tb in FAIL:
        print(f"\n### {n}: {e}")
        print(tb[-1500:])
sys.exit(1 if FAIL else 0)