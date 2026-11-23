#!/usr/bin/env python3
"""Import-check every src module + tools. Exits 0 only if all pass."""
import sys, os, traceback
sys.path.insert(0, r'E:\Green Vision')
os.chdir(r'E:\Green Vision')

mods = [
    'src',
    'src.brain.network',
    'src.brain.ppo_agent',
    'src.brain.planner',
    'src.brain.game_knowledge',
    'src.brain.affect',
    'src.brain.curiosity',
    'src.brain.curriculum',
    'src.brain.episodic_memory',
    'src.brain.reflection',
    'src.brain.world_model',
    'src.brain.rssm_world_model',
    'src.memory.knowledge_db',
    'src.memory.process_reader',
    'src.perception.capture',
    'src.perception.ocr_engine',
    'src.perception.video_ingest',
    'src.input.virtual_input',
    'src.game.game_controller',
    'src.gym_env.geoplanetical_env',
    'src.dashboard.server',
    'src.utils.checkpoints',
    'src.utils.health',
    'src.utils.journal',
    'src.utils.reward_shaper',
    'src.utils.state_schema',
    'src.utils.video_pretrain',
]
ok = fail = 0
for m in mods:
    try:
        __import__(m)
        print(f'OK   {m}')
        ok += 1
    except Exception as e:
        print(f'FAIL {m}: {type(e).__name__}: {e}')
        traceback.print_exc()
        fail += 1
print(f'\n{ok} imported, {fail} failed')
sys.exit(1 if fail else 0)
