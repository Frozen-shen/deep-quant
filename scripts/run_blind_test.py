"""
盲测 — ★ 参数冻结, 只跑一次, 结果即为最终答案

用法: python scripts/run_blind_test.py

运行后:
  - config.yaml 的 trial_count 自动 +1
  - 结果不可用于调参
  - 报告即为最终评估依据
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.pipeline import load_config, QuantPipeline

config = load_config()
bt = config.get("blind_test", {})

if bt.get("locked", False):
    print("⚠️ 参数已锁定。若需重新盲测，请先重置 config.yaml 中的 blind_test.locked=false")

pipeline = QuantPipeline(config, mode="blind")
pipeline.run()
