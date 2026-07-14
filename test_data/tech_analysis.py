import os, json
for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy']: os.environ[k]=''
os.environ['NO_PROXY']='*'

data = [
  {'day':'2026-06-01','open':16.01,'high':16.55,'low':15.92,'close':16.22,'volume':37326779},
  {'day':'2026-06-02','open':16.35,'high':16.48,'low':15.52,'close':16.07,'volume':42586874},
  {'day':'2026-06-03','open':15.94,'high':16.49,'low':15.72,'close':16.12,'volume':43945073},
  {'day':'2026-06-04','open':15.95,'high':16.26,'low':15.71,'close':15.88,'volume':33954106},
  {'day':'2026-06-05','open':15.85,'high':16.07,'low':15.41,'close':15.69,'volume':34784589},
  {'day':'2026-06-08','open':15.12,'high':15.59,'low':14.78,'close':15.07,'volume':36227126},
  {'day':'2026-06-09','open':15.31,'high':15.43,'low':15.07,'close':15.34,'volume':28325825},
  {'day':'2026-06-10','open':15.67,'high':16.87,'low':15.67,'close':16.87,'volume':60681866},
  {'day':'2026-06-11','open':15.77,'high':16.46,'low':15.54,'close':15.66,'volume':76663699},
  {'day':'2026-06-12','open':15.89,'high':15.98,'low':15.18,'close':15.29,'volume':53288375},
  {'day':'2026-06-15','open':15.48,'high':16.04,'low':15.40,'close':16.01,'volume':53023466},
  {'day':'2026-06-16','open':16.06,'high':16.51,'low':15.70,'close':16.30,'volume':55144642},
  {'day':'2026-06-17','open':16.49,'high':17.48,'low':16.18,'close':17.28,'volume':95443132},
  {'day':'2026-06-18','open':16.89,'high':17.44,'low':16.68,'close':17.06,'volume':66950100},
  {'day':'2026-06-22','open':17.00,'high':18.27,'low':16.99,'close':17.75,'volume':77290112},
  {'day':'2026-06-23','open':17.29,'high':17.55,'low':16.45,'close':16.55,'volume':62509679},
  {'day':'2026-06-24','open':16.28,'high':17.28,'low':15.55,'close':16.99,'volume':63969112},
  {'day':'2026-06-25','open':16.67,'high':16.84,'low':15.80,'close':15.85,'volume':54561713},
  {'day':'2026-06-26','open':15.52,'high':15.80,'low':14.58,'close':14.60,'volume':51023622},
  {'day':'2026-06-29','open':14.51,'high':14.76,'low':14.02,'close':14.10,'volume':38975296},
  {'day':'2026-06-30','open':14.07,'high':14.77,'low':13.84,'close':14.55,'volume':44519132},
  {'day':'2026-07-01','open':14.48,'high':15.18,'low':14.18,'close':14.69,'volume':51080566},
  {'day':'2026-07-02','open':14.49,'high':14.85,'low':14.21,'close':14.24,'volume':36833133},
  {'day':'2026-07-03','open':14.22,'high':14.44,'low':13.85,'close':13.86,'volume':34022519},
  {'day':'2026-07-06','open':13.85,'high':13.95,'low':13.42,'close':13.50,'volume':27947980},
  {'day':'2026-07-07','open':13.51,'high':13.56,'low':12.93,'close':13.00,'volume':25829671},
  {'day':'2026-07-08','open':13.01,'high':13.92,'low':12.96,'close':13.52,'volume':59043754},
  {'day':'2026-07-09','open':13.36,'high':13.85,'low':13.05,'close':13.77,'volume':51509316},
  {'day':'2026-07-10','open':13.72,'high':14.40,'low':13.55,'close':14.07,'volume':53136191},
  {'day':'2026-07-13','open':13.95,'high':14.37,'low':13.18,'close':13.30,'volume':36382698},
]

n = len(data)
closes = [d['close'] for d in data]
highs = [d['high'] for d in data]
lows = [d['low'] for d in data]
opens = [d['open'] for d in data]
vols = [d['volume'] for d in data]

def ma(arr, period):
    if len(arr) < period: return sum(arr)/len(arr)
    return sum(arr[-period:])/period

ma5 = ma(closes,5); ma10 = ma(closes,10); ma20 = ma(closes,20)

print('=== 均线系统 ===')
print(f'MA5={ma5:.2f}  MA10={ma10:.2f}  MA20={ma20:.2f}')
print(f'收盘13.30 vs MA5: {(13.30/ma5-1)*100:+.1f}% | MA10: {(13.30/ma10-1)*100:+.1f}% | MA20: {(13.30/ma20-1)*100:+.1f}%')
print(f'排列: {"空头" if ma5<ma10<ma20 else "多头" if ma5>ma10>ma20 else "粘合"}')

print('\n=== 多周期涨跌幅 ===')
for p, name in [(5,'5日'),(10,'10日'),(20,'20日')]:
    if n > p:
        chg = (closes[-1]/closes[-p-1]-1)*100
        print(f'近{name}: {chg:+.1f}%')

# MACD
ema12 = closes[0]; ema26 = closes[0]
difs = []; deas = []; macds = []
for i, c in enumerate(closes):
    ema12 = c * 2/13 + ema12 * 11/13
    ema26 = c * 2/27 + ema26 * 25/27
    dif = ema12 - ema26
    difs.append(dif)
    if i == 0:
        deas.append(dif); macds.append(0)
    else:
        dea = deas[-1] * 8/10 + dif * 2/10
        deas.append(dea)
        macds.append((dif - dea) * 2)

print('\n=== MACD(12,26,9) ===')
print(f'DIF={difs[-1]:.4f}  DEA={deas[-1]:.4f}  MACD柱={macds[-1]:.4f}')
print(f'5日前DIF={difs[-6]:.4f} -> 今日={difs[-1]:.4f}')
is_golden = difs[-2] <= deas[-2] and difs[-1] > deas[-1]
is_dead = difs[-2] >= deas[-2] and difs[-1] < deas[-1]
print(f'信号: {"金叉!" if is_golden else "死叉!" if is_dead else "延续"} | {"多头" if difs[-1]>deas[-1] else "空头"}')

# KDJ
k_vals = [50]; d_vals = [50]; j_vals = [50]
for i in range(8, n):
    hh = max(highs[i-8:i+1]); ll = min(lows[i-8:i+1])
    rsv = (closes[i] - ll) / (hh - ll) * 100 if hh != ll else 50
    k = k_vals[-1] * 2/3 + rsv * 1/3
    d = d_vals[-1] * 2/3 + k * 1/3
    j = 3*k - 2*d
    k_vals.append(k); d_vals.append(d); j_vals.append(j)

print('\n=== KDJ(9,3,3) ===')
print(f'K={k_vals[-1]:.1f}  D={d_vals[-1]:.1f}  J={j_vals[-1]:.1f}')
j_val = j_vals[-1]
print(f'信号: {"超卖(J<0)→反弹机会" if j_val<0 else "超买(J>100)→回调风险" if j_val>100 else "中性偏弱" if j_val<50 else "中性偏强"}')

# RSI
gains = [max(closes[i]-closes[i-1],0) for i in range(1,n)]
losses = [max(closes[i-1]-closes[i],0) for i in range(1,n)]
avg_gain = sum(gains[-14:])/14; avg_loss = sum(losses[-14:])/14
rsi = 100 - 100/(1+avg_gain/avg_loss) if avg_loss>0 else 100
print(f'\n=== RSI(14) ===')
print(f'RSI={rsi:.1f}  {"超卖" if rsi<30 else "超买" if rsi>70 else "中性偏弱" if rsi<50 else "中性偏强"}')

# Bollinger
std20 = (sum((c-ma20)**2 for c in closes[-20:])/20) ** 0.5
bb_upper = ma20+2*std20; bb_lower = ma20-2*std20
bb_width = (4*std20/ma20)*100
print(f'\n=== 布林带(20,2) ===')
print(f'上轨={bb_upper:.2f}  中轨={ma20:.2f}  下轨={bb_lower:.2f}')
print(f'价格在布林: {"下轨附近→超卖" if closes[-1]<ma20-std20 else "上轨附近→超买" if closes[-1]>ma20+std20 else "中轨下方→偏弱"}')
print(f'带宽={bb_width:.1f}%  {"收窄→变盘信号" if bb_width<10 else "正常"}')

# Volume
avg_vol_5 = sum(vols[-6:-1])/5; avg_vol_20 = sum(vols[-21:-1])/20
print(f'\n=== 量价分析 ===')
print(f'5日均量: {avg_vol_5/1e4:.0f}万  20日均量: {avg_vol_20/1e4:.0f}万')
print(f'今日量: {vols[-1]/1e4:.0f}万  量比(5日): {vols[-1]/avg_vol_5:.2f}')
today_chg = closes[-1]-closes[-2]
vol_ratio = vols[-1]/avg_vol_5
if vol_ratio < 0.7 and today_chg < 0: vol_signal = '缩量下跌→抛压减弱'
elif vol_ratio < 0.7 and today_chg > 0: vol_signal = '缩量上涨→反弹乏力'
elif vol_ratio > 1.5 and today_chg < 0: vol_signal = '放量下跌→恐慌抛售'
elif vol_ratio > 1.5 and today_chg > 0: vol_signal = '放量上涨→多头入场'
else: vol_signal = '量价正常'
print(f'信号: {vol_signal}')

# High/low
print(f'\n=== 波段高低点 ===')
h30 = max(highs); l30 = min(lows)
h10 = max(highs[-10:]); l10 = min(lows[-10:])
print(f'30日: 高{h30:.2f}(6/22) | 低{l30:.2f}(7/7)')
print(f'10日: 高{h10:.2f} | 低{l10:.2f}')
print(f'当前位置: 距30日高{(13.30/h30-1)*100:.1f}% | 距30日低{(13.30/l30-1)*100:+.1f}%')

# Candle pattern
body = closes[-1]-opens[-1]
upper = highs[-1]-max(opens[-1],closes[-1])
lower = min(opens[-1],closes[-1])-lows[-1]
body_pct = abs(body)/opens[-1]*100
print(f'\n=== 今日K线形态 ===')
print(f'开{opens[-1]:.2f} 高{highs[-1]:.2f} 低{lows[-1]:.2f} 收{closes[-1]:.2f}')
print(f'实体={body:+.2f}({body_pct:.1f}%)  上影={upper:.2f}  下影={lower:.2f}')
if body < 0:
    if body_pct > 5: print('形态: 大阴线!')
    elif body_pct > 2: print('形态: 中阴线')
    else: print('形态: 小阴线')
if lower > abs(body)*2: print('→ 长下影线: 下方有承接盘')

# Lifecycle
print(f'\n=== 生命周期阶段 ===')
ma_arr = '空头排列' if ma5<ma10<ma20 else '多头排列' if ma5>ma10>ma20 else '粘合'
macp_pos = '零轴下方' if difs[-1]<0 else '零轴上方'
print(f'均线: {ma_arr} | MACD: {macp_pos} | 量价: 缩量下跌 | 高低点: 逐级降低(18.27->14.69->13.00)')
if ma5<ma10<ma20 and difs[-1]<0:
    stage = '下跌期'
elif difs[-2]>difs[-1] and macds[-1]<macds[-2] and ma5<ma10:
    stage = '出货期后期→下跌期'
elif ma5>ma10>ma20 and difs[-1]>deas[-1] and difs[-1]>0:
    stage = '主升浪'
else:
    stage = '洗盘/筑底期'
print(f'综合判断: {stage}')
