"""
eastmoney_proxy.py — 东财接口代理初始化 (cheapproxy / akshare-proxy-patch)

背景: 东财对本地直连 IP 间歇性封禁 (push2/push2his 频繁 RemoteDisconnected, 2026-08 起)。
      通过 cheapproxy 网关转发请求可绕过本地 IP 封禁 (按积分计费)。

用法 (fetch 脚本顶部, 必须在 akshare/efinance 首次 import 之前):
    import eastmoney_proxy
    eastmoney_proxy.setup_from_config()

配置: config.yaml 的 eastmoney_proxy 段 (enabled/gateway/auth_token/hook_domains)
"""

import os

# 必须早于 akshare/efinance 的任何 import
_CONFIG_LOADED = False


def load_proxy_config(base_dir: str = None) -> dict:
    """从 config.yaml 读取 eastmoney_proxy 配置。"""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(base_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        return {}
    try:
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return (cfg.get("eastmoney_proxy") or {})
    except Exception:
        return {}


def setup_from_config(base_dir: str = None, force: bool = False) -> bool:
    """
    按 config.yaml 配置启用东财代理插件。

    Returns:
      True=代理已启用, False=未配置/已禁用
    """
    global _CONFIG_LOADED
    if _CONFIG_LOADED and not force:
        return True

    cfg = load_proxy_config(base_dir)
    if not cfg.get("enabled"):
        return False

    gateway = cfg.get("gateway", "101.201.173.125")
    token = cfg.get("auth_token", "")
    retry = int(cfg.get("retry", 30))
    hook_domains = cfg.get("hook_domains", [])

    if not token:
        return False

    # 清空环境代理变量, 避免与插件网关冲突
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(k, None)

    try:
        import akshare_proxy_patch
        akshare_proxy_patch.install_patch(
            gateway,
            auth_token=token,
            retry=retry,
            hook_domains=hook_domains,
            fast=True,
        )
        _CONFIG_LOADED = True
        return True
    except ImportError:
        # 未安装 akshare-proxy-patch 时静默降级为直连
        return False


def setup_from_env(base_dir: str = None) -> bool:
    """兼容入口: 优先走 config, 环境变量 BQ_EM_PROXY=1 时强制。"""
    if os.environ.get("BQ_EM_PROXY") == "0":
        return False
    return setup_from_config(base_dir)
