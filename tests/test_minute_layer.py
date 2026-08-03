# tests/test_minute_layer.py
"""分钟因子独立叠加层 (方案B) 测试。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "active"))


def test_minute_layer_config():
    """config.yaml 的 minute_layer 段存在且含默认值。"""
    import yaml
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ml = cfg.get("minute_layer", {})
    assert ml.get("enabled") is True
    assert ml.get("lambda") == 0.3
    assert ml.get("min_icir") == 0.3
