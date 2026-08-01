"""
开发期 Walk-Forward 验证 — 可用此脚本调参

用法: python scripts/run_backtest.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.pipeline import load_config, QuantPipeline

config = load_config()
pipeline = QuantPipeline(config, mode="dev")
pipeline.run()
