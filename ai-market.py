from http import client

from google import genai
import json

# 1. 配置 API
client =genai.Client(api_key="AIzaSyC5Lp2o-V6kmE2QsvyDpPANPjJqOSdWwK4")


def get_ai_market_picks(market_context):
    # 使用目前最稳定的 Flash 模型名
    model_id = "gemini-3-flash-preview" # 或者根据官网最新的 gemini-3-flash

    prompt = f"""
    你是顶级量化分析师。基于以下市场上下文，筛选出10只最值得量化模型(Optuna)寻优的标的。
    市场上下文: {market_context}
    
    输出要求：
    1. 严格使用 JSON。
    2. 字段: symbol, name, sector, logic_score(0-1), vol_type(HIGH/LOW), reason。
    """

    try:
        # 新版 SDK 的调用方法
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config={
                'response_mime_type': 'application/json', # 强制要求输出 JSON
            }
        )
        
        # 新版解析方式：直接访问 text 属性
        return json.loads(response.text)
    except Exception as e:
        print(f"API 调用异常: {e}")
        return None

# 模拟测试
market_info = "A股通讯设备板块异动，低估值红利指数回调，北向资金净流入电子元件行业。"
picks = get_ai_market_picks(market_info)

if picks:
    print("=== AI 筛选结果 ===")
    print(json.dumps(picks, indent=2, ensure_ascii=False))