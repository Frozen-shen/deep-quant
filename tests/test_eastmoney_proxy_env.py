"""tests/test_eastmoney_proxy_env.py — 代理凭据环境变量化 (2026-09-02 安全整改)

凭据只从环境变量 CHEAPPROXY_AUTH_TOKEN 读取 (项目根 .env 文件或真实环境),
config.yaml 的 auth_token 字段不再作为凭据来源——防止硬编码凭据再次入库
(公开仓库曾泄露该网关 token)。
"""
import pytest

import eastmoney_proxy

_CFG_TMPL = """\
eastmoney_proxy:
  enabled: true
  gateway: "1.2.3.4"
  auth_token: "{cfg_token}"
  retry: 3
  hook_domains:
    - "push2.eastmoney.com"
"""


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("CHEAPPROXY_AUTH_TOKEN", raising=False)
    yield
    # _load_env_file 用 os.environ.setdefault 写入, 不受 monkeypatch 回滚
    import os
    os.environ.pop("CHEAPPROXY_AUTH_TOKEN", None)


def _setup(tmp_path, monkeypatch, cfg_token=""):
    """写临时 config/.env, 拦截 install_patch, 返回捕获字典。"""
    (tmp_path / "config.yaml").write_text(
        _CFG_TMPL.format(cfg_token=cfg_token), encoding="utf-8")
    captured = {}
    import akshare_proxy_patch
    monkeypatch.setattr(
        akshare_proxy_patch, "install_patch",
        lambda gw, auth_token, retry, hook_domains, fast:
            captured.update(token=auth_token, gateway=gw))
    return str(tmp_path), captured


def test_token_loaded_from_env_file(tmp_path, monkeypatch):
    base, captured = _setup(tmp_path, monkeypatch)
    (tmp_path / ".env").write_text(
        "CHEAPPROXY_AUTH_TOKEN=tok-from-env\n", encoding="utf-8")
    assert eastmoney_proxy.setup_from_config(base, force=True) is True
    assert captured["token"] == "tok-from-env"


def test_real_env_wins_over_env_file(tmp_path, monkeypatch):
    base, captured = _setup(tmp_path, monkeypatch)
    (tmp_path / ".env").write_text(
        "CHEAPPROXY_AUTH_TOKEN=file-token\n", encoding="utf-8")
    monkeypatch.setenv("CHEAPPROXY_AUTH_TOKEN", "real-env-token")
    assert eastmoney_proxy.setup_from_config(base, force=True) is True
    assert captured["token"] == "real-env-token"


def test_config_token_field_no_longer_used(tmp_path, monkeypatch):
    """config.yaml 里的 auth_token 值不再被读取 (回归防线)。"""
    base, captured = _setup(tmp_path, monkeypatch, cfg_token="config-secret")
    assert eastmoney_proxy.setup_from_config(base, force=True) is False
    assert "token" not in captured


def test_missing_token_disables_proxy(tmp_path, monkeypatch):
    base, captured = _setup(tmp_path, monkeypatch)
    assert eastmoney_proxy.setup_from_config(base, force=True) is False
    assert "token" not in captured
