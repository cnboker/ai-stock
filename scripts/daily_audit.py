import requests
import json
import subprocess # 用于调用本地 hermes cli
import os
from datetime import datetime

# 配置
API_BASE_URL = "http://localhost:8080/api/v1"
AUDIT_LOG_PATH = "logs/audit_records.md"
# 假设你的 cli 命令是 'hermes'，如果是路径则写全称如 '/usr/local/bin/hermes'
HERMES_CLI_CMD = "hermes" 

def fetch_daily_data():
    """从 FastAPI 获取今日交易表现"""
    try:
        response = requests.get(f"{API_BASE_URL}/analytics/daily_review")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        # 调试用：如果接口没开，返回一个模拟结构
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "summary": "本地接口连接失败，使用模拟数据调试",
            "data": [{"symbol": "NVDA", "expected": 0.05, "actual": -0.01, "error": 0.06}]
        }

def ask_hermes_cli(prompt_text):
    """通过配置了特定代理的环境变量调用本地 Hermes CLI"""
    
    # 1. 构造独立的环境变量字典
    # 拷贝当前系统的环境变量
    env = os.environ.copy()
    
    # 2. 先 unset（即从字典中删除旧的代理设置）
    proxy_keys = [
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", 
        "http_proxy", "https_proxy", "all_proxy"
    ]
    for key in proxy_keys:
        env.pop(key, None)
    
    # 3. 设置新的代理
    # 针对你本地的 Clash/代理端口 (7891)
    new_proxy = "socks5://127.0.0.1:7891"
    env["HTTP_PROXY"] = new_proxy
    env["HTTPS_PROXY"] = new_proxy
    env["ALL_PROXY"] = new_proxy

    try:
        process = subprocess.Popen(
        [HERMES_CLI_CMD, "chat", "-q", prompt_text],   # 关键改动在这里
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        encoding='utf-8'
        )

        stdout, stderr = process.communicate()
        clean_stdout = stdout.strip()
        if process.returncode != 0:
            print("错误:", stderr)
        else:
            print("Hermes 输出:", stdout)
        return clean_stdout
    except Exception as e:
        return f"❌ 无法启动 Hermes CLI: {e}"


"""
你完全可以只把结果存成一个本地的 .md 文件，但通过 gh 创建 Issue 有三个核心优势：

版本化审计记录：每一天的复盘都变成了一个带编号的 Issue（例如 #12, #13）。你可以清晰地看到模型预测能力的演变过程。

闭环调优：如果你根据 Hermes 的建议（比如 [MEMORY_ENTRY] 调低 RSI 权重）修改了代码，你可以在提交代码（Commit）时写上 Closes #12。这样，审计 -> 建议 -> 修复 -> 关闭 Issue 就形成了一个自动化的开发闭环。

多端同步：Issue 提交后，你在手机上的 GitHub App 或电脑网页端都能随时看到 Hermes 的审计结果，无需非得盯着这台服务器。
"""
def create_gh_issue(date, analysis):
    # --- 核心修复：强制检查 analysis 是否有效 ---
    if analysis is None:
        analysis = "Warning: Hermes CLI failed to return an analysis."
    else:
        analysis = str(analysis).strip()
        if not analysis: # 处理空字符串
            analysis = "Analysis returned empty content."
    title = f"Audit {date}"
    try:
        # 使用列表模式，无需担心 analysis 里的引号导致命令断裂
        subprocess.run([
            "gh", "issue", "create", 
            "--title", title, 
            "--label", "hermes-audit,quant-review",
            "--body", analysis
        ], check=True)
        print("✅ GitHub Issue 创建成功！")
    except subprocess.CalledProcessError as e:
        print(f"❌ GitHub CLI 执行失败: {e}")

import re

def clean_hermes_output(text):
    # 1. 移除 ANSI 颜色码
    text = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)
    
    # 2. 定位有效内容的起点
    # 我们找“深度诊断”或者“好的，Hermes”作为正文开始的标志
    start_markers = ["好的", "深度诊断", "Hermes 量化审计", "Audit"]
    start_index = 0
    for marker in start_markers:
        pos = text.find(marker)
        if pos != -1:
            start_index = pos
            break
            
    # 截取正文
    content = text[start_index:]
    
    # 3. 定位有效内容的终点
    # 丢弃后面关于 Session, Duration 等统计信息
    end_marker = "Resume this session with:"
    end_index = content.find(end_marker)
    if end_index != -1:
        content = content[:end_index]

    # 4. 最后清理掉残余的框线字符和点阵字符（那些盲文符号）
    # [\u2800-\u28FF] 是 Unicode 里的盲文点阵列，也就是那个大 Logo 的组成部分
    content = re.sub(r'[╭╮╰╯│─━═┐└┘┌┐┊░⚕\u2800-\u28FF]+', '', content)
    
    # 5. 将连续的空行压缩成一行
    content = re.sub(r'\n\s*\n', '\n\n', content)
    
    return content.strip()

def run_audit():
    # 1. 抓取数据
    data = fetch_daily_data()
    if not data or not data.get("data"):
        print("📭 今日无交易预测数据。")
        return
    print(f"data={data}")
    
    # 2. 构造初始 Prompt
    # 根据你提供的 JSON 结构解析数据
    date = data['date']
    summary = data['summary']
    details = data['data']

    initial_prompt = f"""
    # 角色
    你是一名深度量化审计专家 Hermes，专门负责识别“信号生成层”与“风险控制层”之间的逻辑断层。

    # 复盘数据概览 ({date})
    - 市场情绪: {summary['market_sentiment']}
    - 信号/误差: 共 {summary['total_signals']} 个信号，平均误差 {summary['avg_error']}

    # 原始数据集
    {json.dumps(details, ensure_ascii=False)}

    # 审计逻辑 (通用型)
    
    1. **信号质量与分布 (Signal Quality)**:
       - 识别数据中是否存在“低分过热”现象：即 ModelScore 接近 0.5 但频繁产生动作信号的情况。
       - 分析预测值(Pred_Change)相对于该标的 ATR 的倍数。如果倍数 > 1 且 Score < 0.6，诊断是否存在“过度拟合波动”的倾向。

    2. **风控拦截审计 (Risk Control Efficiency)**:
       - 统计因 "equity_slope_break" 或其他风控原因被拦截(no_trade)的比例。
       - **盈亏回溯**：对比被拦截信号的 Pred 与 Actual。如果 Actual 波动极小，判定风控为“精准防御”；如果 Actual 波动巨大且方向一致，判定风控为“过度抑制”。

    3. **流动性与摩擦分析 (Friction & Churn)**:
       - 针对“有预测无波动”（Actual_Change 接近 0）的案例，分析是否属于“无效成交量”导致的模型误导。
       - 区分“突破失败”与“流动性黑洞”：在 A 股 ETF 或中小盘股中，这种现象往往代表市场处于深度分歧或庄家吸筹。

    4. **系统级进化记忆**:
       - 请根据今日所有信号的统计特征，生成一条 [SYSTEM_OPTIMIZATION]：给出关于置信度阈值(Gate)或风控灵敏度的具体调优建议。
       - 请生成一条 [MODEL_BIAS_ALERT]：识别模型是否存在系统性的多头或空头偏向（例如在横盘期由于微小放量而习惯性看跌）。

    # 输出要求
    请保持冷峻、专业、数据驱动的语调，避免废话，直接指出系统逻辑中最薄弱的环节。
    """
    # 3. 第一次分析
    print("🤖 Hermes CLI 正在分析...")
    analysis = ask_hermes_cli(initial_prompt)
    analysis = clean_hermes_output(analysis)
    print(f"\n--- 初次审计结果 ---\n{analysis}")

    # 4. 进入交互模式
    while True:
        user_input = input("\n💬 追问 Hermes (输入 's' 保存并同步 GitHub, 'q' 退出): ").strip()
        
        if user_input.lower() == 'q':
            break
        elif user_input.lower() == 's':
            # 记录到本地
            with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"\n\n## {data['date']} 复盘\n{analysis}")
            
            # 使用 GitHub CLI 同步 (你的第二个 Skill)
            print("🚀 正在通过 GitHub CLI 创建 Issue...")
            create_gh_issue(data['date'],analysis)
            
            break
        else:
            # 将上下文带入追问
            full_prompt = f"背景数据: {json.dumps(data['data'])}\n之前分析: {analysis}\n用户追问: {user_input}"
            analysis = ask_hermes_cli(full_prompt)
            print(f"\n--- Hermes 回复 ---\n{analysis}")


if __name__ == "__main__":
    run_audit()
  