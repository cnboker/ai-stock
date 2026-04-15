import requests

def run_hermes_audit():
    # 1. 获取今日原始数据
    raw_data = requests.get("http://localhost:8000/api/v1/daily_review").json()
    
    # 2. 调用 Hermes API (假设你通过 LangChain 或直接 API 调用)
    prompt = f"这是今日 ai-stock 的表现数据：{raw_data}。请分析 Chronos 预测失败的原因，并给出一条针对性的改进建议。"
    
    analysis = hermes.ask(prompt)
    
    # 3. 将分析结果存入你的本地 Markdown 复盘日志或向量库
    with open(f"logs/audit_{raw_data['date']}.md", "w") as f:
        f.write(analysis)
    
    print(f"✅ {raw_data['date']} 复盘记忆已生成。")

if __name__ == "__main__":
    run_hermes_audit()