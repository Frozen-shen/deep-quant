"""
A股行业分类 — 基于代码前缀的粗分类

用法:
  from sector_analyzer import build_a_share_sector_map
  sector_map = build_a_share_sector_map(["600519", "000858"])
  # → {"600519": "食品饮料", "000858": "食品饮料"}
"""

import os, json

# ================================================================
#  A股板块映射 — 基于代码前缀的粗分类 (Phase 2.3)
# ================================================================

def _classify_a_stock_by_prefix(symbol: str) -> str:
    """根据A股代码前缀推断申万一级行业 (粗略版)。"""
    code = str(symbol)
    # 银行
    if code.startswith(("601398","601939","601288","601988","601328","600016","600036","600000","601166","601818","002142","000001")):
        return "银行"
    # 保险
    if code.startswith(("601318","601628","601601","601336","000627")):
        return "非银金融"
    # 白酒/食品
    if code.startswith(("600519","000858","000568","002304","600809","000799","600702","603369")):
        return "食品饮料"
    # 医药
    if code.startswith(("300760","600276","000538","002007","300122","688180","600085","300015","300347","603259","002821")):
        return "医药生物"
    # 半导体
    if code.startswith(("688981","002371","002049","688012","688396","603986","300782","688008","002185","600584")):
        return "电子"
    # 新能源
    if code.startswith(("300750","002594","601012","688005","300274","300014","002812","688567","300763")):
        return "电力设备"
    # AI/TMT
    if code.startswith(("300033","002230","688111","300454","688561","002415","300059")):
        return "计算机"
    # 军工
    if code.startswith(("600760","601668","600893","002013","600118","000768")):
        return "国防军工"
    # 家电
    if code.startswith(("000651","600690","002050","603486","000333")):
        return "家用电器"
    # 地产/建材
    if code.startswith(("000002","001979","600048","600585","000786")):
        return "房地产"
    # 汽车
    if code.startswith(("600104","000625","002594","601238","601633","000800")):
        return "汽车"
    # 煤炭/石油
    if code.startswith(("601088","600188","601857","600028","600348","000983")):
        return "煤炭"
    # 有色
    if code.startswith(("601899","600547","603993","002466","000630","600489")):
        return "有色金属"
    # 交通/物流
    if code.startswith(("601919","601006","600009","600029","600115","002352")):
        return "交通运输"
    # 电信
    if code.startswith(("600050","600941","601728","002583")):
        return "通信"
    # 默认: 按代码段分类
    if code.startswith("600"): return "主板金融"
    if code.startswith("601"): return "主板工业"
    if code.startswith("603"): return "主板消费"
    if code.startswith("000"): return "深市主板"
    if code.startswith("002"): return "中小板"
    if code.startswith("300"): return "创业板"
    if code.startswith("688"): return "科创板"
    return "其他"


def build_a_share_sector_map(symbols: list) -> dict:
    """
    Phase 2.3: 构建A股行业映射 {symbol: sector_name}。

    优先从缓存文件读取；若不存在则根据代码前缀推断。

    返回: {symbol: sector_name}
    """
    cache_path = os.path.join(os.path.dirname(__file__), "data_cache", "a_sectors.json")

    # 尝试从缓存加载
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            # 返回已缓存的 (可能不包含所有symbols)
            return cached
        except:
            pass

    # 按代码前缀推断
    sector_map = {}
    for sym in symbols:
        sector_map[str(sym)] = _classify_a_stock_by_prefix(sym)

    # 保存缓存
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(sector_map, f, ensure_ascii=False)
    except:
        pass

    return sector_map
