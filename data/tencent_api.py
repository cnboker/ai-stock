import httpx
import asyncio
from typing import List, Dict

class StockProvider:
    def __init__(self):
        # 腾讯接口地址，s_ 前缀代表简版数据，速度更快
        self.base_url = "https://qt.gtimg.cn/q="
        self.headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64) AppleWebKit/537.36"
        }

    async def get_price_map(self, symbols: List[str]) -> Dict[str, float]:
        """
        返回精简的价格字典，例如: {'sz300961': 12.77, 'sh600519': 1650.0}
        """
        if not symbols:
            return {}

        # 统一转为小写处理
        symbols = [s.lower() for s in symbols]
        query_str = ",".join([f"s_{s}" for s in symbols])
        url = f"{self.base_url}{query_str}"
        print(f"url={url}")
        proxy_url = "http://127.0.0.1:7890"
        
        async with httpx.AsyncClient(proxy=proxy_url, trust_env=False) as client:
            try:
                response = await client.get(url, headers=self.headers, timeout=5.0)
                if response.status_code != 200:
                    return {}
                
                return self._parse_to_dict(response.text)
            except Exception as e:
                print(f"获取行情失败: {e}")
                return {}
            
    def _parse_to_dict(self, text: str) -> Dict[str, float]:
        price_map = {}
        lines = text.strip().split(';')
        for line in lines:
            if '~' not in line:
                continue
            
            # 腾讯返回格式示例: v_s_sz300961="51~深水海纳~300961~12.77~...";
            try:
                # 提取代码（从 v_s_ 开始到 = 结束）
                raw_code = line.split('=')[0].split('_')[-1]
                # 提取价格（~ 分割后的第 4 个元素）
                content = line.split('"')[1]
                current_price = float(content.split('~')[3])
                
                price_map[raw_code] = current_price
            except (IndexError, ValueError):
                continue
                
        return price_map

    async def get_realtime_quotes(self, symbols: List[str]) -> List[Dict]:
        """
        支持多只股票批量获取实时行情
        :param symbols: 股票代码列表，如 ['sz300961', 'sh600519']
        """
        if not symbols:
            return []

        # 构造请求 URL，多个代码用逗号隔开
        query_str = ",".join([f"s_{s.lower()}" for s in symbols])
        url = f"{self.base_url}{query_str}"

        async with httpx.AsyncClient(trust_env=False) as client:
            response = await client.get(url, headers=self.headers)
            if response.status_code != 200:
                return []

            # 解析返回的字符串内容
            return self._parse_response(response.text)

    def _parse_response(self, text: str) -> List[Dict]:
        results = []
        # 腾讯返回的数据以分号间隔每只股票
        lines = text.strip().split(';')
        for line in lines:
            if not line or '~' not in line:
                continue
            
            # 数据格式: v_s_sz300961="51~深水海纳~300961~12.77~-0.49~-3.70~64265~8302~";
            content = line.split('"')[1]
            parts = content.split('~')
            
            results.append({
                "name": parts[1],
                "code": parts[2],
                "current": float(parts[3]),  # 当前价
                "chg": float(parts[4]),      # 涨跌额
                "percent": float(parts[5]),  # 涨跌幅
                "volume": float(parts[6]),   # 成交量（手）
                "amount": float(parts[7]),   # 成交额（万元）
            })
        return results

# --- 使用示例 ---
# async def main():
#     provider = StockProvider()
#     # 模拟获取深水海纳和贵州茅台
#     stocks = await provider.get_realtime_quotes(['sz300961', 'sh600519'])
#     for s in stocks:
#         print(f"{s['name']} ({s['code']}): 价格 {s['current']} 涨跌幅 {s['percent']}%")

# if __name__ == "__main__":
#     asyncio.run(main())