# Daily Review Script (Cron Task)
import requests
import json
from datetime import datetime

# 配置
API_BASE_URL = "http://localhost:8080/api/v1" # 你的 FastAPI 地址
HERMES_API_URL = "你的模型API地址" # 如果是本地模型或 OpenAI 格式接口
AUDIT_LOG_PATH = "logs/audit_records.md"

def fetch_daily_data():
    """从 FastAPI 获取今日交易表现"""
    try:
        response = requests.get(f"{API_BASE_URL}/analytics/daily_review")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ 获取数据失败: {e}")
        return None

def ask_hermes(data):
    """向 Hermes 发送审计请求"""
    # 构造 Prompt，这是 Week 1 的核心
    prompt = f"""
    # 角色
    你是一名资深的量化交易审计员 Hermes。
    
    # 任务
    分析今日 {data['date']} 的交易预测偏差，并生成复盘记忆。
    
    # 今日数据
    市场概况: {data['summary']}
    详细记录: {json.dumps(data['data'], ensure_ascii=False, indent=2)}
    
    # 分析要求
    1. 归因：找出 'error' 最大的标的，判断是模型过度拟合了某种特征，还是受大盘系统性影响。
    2. 模式识别：是否存在“缩量必涨错”或“高开必杀跌”等规律？
    3. 记忆：生成一条格式为 [MEMORY_ENTRY] 的短语，用于后续指导调优。
    """
    
    # 这里根据你的模型调用方式修改（例如 OpenAI SDK 或直接 requests）
    print("🤖 Hermes 正在分析数据...")
    # 示例返回
    return "### Hermes 每日复盘报告\n" + "分析结果..." # 实际应调用接口返回

def run_audit():
    # 1. 抓取数据
    data = fetch_daily_data()
    if not data or not data.get("data"):
        print("📭 今日无交易预测数据。")
        return

    # 2. 交互复盘
    analysis = ask_hermes(data)
    
    # 3. 记录日志
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n\n## {data['date']} 复盘\n")
        f.write(analysis)
    
    print(f"✅ 复盘完成，记忆已存入 {AUDIT_LOG_PATH}")

if __name__ == "__main__":
    run_audit()