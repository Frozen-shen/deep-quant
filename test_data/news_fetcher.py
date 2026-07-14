#!/usr/bin/env python3
"""
消息面数据获取脚本
用法: py news_fetcher.py 000815 东数西算,数据中心

获取: 公司公告 + 行业关键词公告
输出: JSON
"""

import os, sys, json, requests
from datetime import datetime, timedelta

for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy']:
    os.environ[k] = ''
os.environ['NO_PROXY'] = '*'

def get_announcements(code, days=45):
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    url = 'https://np-anotice-stock.eastmoney.com/api/security/ann'
    params = {
        'page_size': 10, 'page_index': 1, 'ann_type': 'A',
        'stock_list': code, 'f_node': 0, 's_node': 0,
        'begin_time': start, 'end_time': end
    }
    
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if data.get('success') != 1:
            return []
        
        results = []
        bull_kw = ['预增','增持','回购','中标','合同','订单','获批','上市','战略合作','新品','突破']
        bear_kw = ['减持','亏损','退市','风险提示','问询','监管','诉讼','冻结','质押','预降','预亏','终止']
        
        for item in data['data']['list']:
            title = item.get('title', '')
            sent = '中性'
            for kw in bull_kw:
                if kw in title: sent = '利好'; break
            for kw in bear_kw:
                if kw in title: sent = '利空'; break
            
            results.append({
                'date': item.get('notice_date', '')[:10],
                'title': title,
                'type': item.get('columns', [{}])[0].get('column_name', '其他'),
                'sentiment': sent
            })
        return results
    except Exception as e:
        return [{'error': str(e)[:100]}]

def get_keyword_announcements(keyword, days=30):
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    url = 'https://np-anotice-stock.eastmoney.com/api/security/ann'
    params = {
        'page_size': 5, 'page_index': 1, 'ann_type': 'A',
        'key_word': keyword,
        'begin_time': start, 'end_time': end
    }
    
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200: return []
        data = r.json()
        if data.get('success') != 1: return []
        
        results = []
        for item in data['data']['list'][:5]:
            results.append({
                'date': item.get('notice_date', '')[:10],
                'title': item.get('title', ''),
                'company': [c.get('short_name', '') for c in item.get('codes', [])]
            })
        return results
    except:
        return []

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({'error': '用法: py news_fetcher.py <股票代码> [关键词1,关键词2]'}, ensure_ascii=False))
        sys.exit(1)
    
    code = sys.argv[1]
    keywords = sys.argv[2].split(',') if len(sys.argv) > 2 else []
    
    output = {
        'stock_code': code,
        'fetch_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'company_announcements': get_announcements(code),
        'industry_announcements': {}
    }
    
    for kw in keywords:
        output['industry_announcements'][kw] = get_keyword_announcements(kw)
    
    print(json.dumps(output, ensure_ascii=False, indent=2))
