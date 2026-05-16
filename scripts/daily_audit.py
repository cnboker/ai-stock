import requests
import subprocess  # 用于调用本地 hermes cli
import os
from datetime import datetime

from scripts.hermes_prompt import prepare_hermes_prompt, process_hermes_smart_save

# 配置
API_BASE_URL = "http://localhost:8080/api/v1"
AUDIT_LOG_PATH = "logs/audit_records.md"
# 假设你的 cli 命令是 'hermes'，如果是路径则写全称如 '/usr/local/bin/hermes'
HERMES_CLI_CMD = "hermes"


def run_audit():
    # 1. 抓取数据
    daily_json = fetch_daily_data('2026-05-15')
    print(f"📥 从 API 获取的原始数据:\n{daily_json}")
    if not daily_json or not daily_json.get("details"):
        print("📭 今日无交易预测数据。")
        return

    # 2. 构造初始 Prompt

    final_prompt, current_date = prepare_hermes_prompt(daily_json)

    # 3. 第一次分析
    print("🤖 Hermes CLI 正在分析...")
    #print(f"🔍 初始 Prompt:\n{final_prompt}")  # 只打印前500字符，避免过长
    analysis = ask_hermes_cli(final_prompt)
    hermes_output = clean_hermes_output(analysis)
    print(f"\n--- 初次审计结果 ---\n{analysis}")
    # 实际操作中，你会把 Hermes 的输出传给这个函数

    # 4. 进入交互模式
    while True:
        user_input = input(
            "\n💬 追问 Hermes (输入 's' 保存并同步 GitHub, 'q' 退出): "
        ).strip()

        if user_input.lower() == "q":
            break
        elif user_input.lower() == "s":
            # 记录到本地
            process_hermes_smart_save(
                hermes_output=hermes_output, date_str=current_date
            )
            # 同步到 GitHub
            create_gh_issue(current_date, analysis)
            break


def fetch_daily_data(date=datetime.now().date()):
    """从 FastAPI 获取今日交易表现"""
    try:
        print(f"{API_BASE_URL}/analytics/daily_review/{date}")
        response = requests.get(f"{API_BASE_URL}/analytics/daily_review/{date}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        # 调试用：如果接口没开，返回一个模拟结构
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "summary": "本地接口连接失败，使用模拟数据调试",
            "data": [
                {"symbol": "NVDA", "expected": 0.05, "actual": -0.01, "error": 0.06}
            ],
        }


def ask_hermes_cli(prompt_text):
    """通过配置了特定代理的环境变量调用本地 Hermes CLI"""

    # 1. 构造独立的环境变量字典
    # 拷贝当前系统的环境变量
    env = os.environ.copy()

    # 2. 先 unset（即从字典中删除旧的代理设置）
    proxy_keys = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]
    for key in proxy_keys:
        env.pop(key, None)

    # 3. 设置新的代理
    # 针对你本地的 Clash/代理端口 (7891)
    new_proxy = "http://127.0.0.1:7890"
    env["HTTP_PROXY"] = new_proxy
    env["HTTPS_PROXY"] = new_proxy
    env["ALL_PROXY"] = new_proxy

    try:
        process = subprocess.Popen(
            [HERMES_CLI_CMD, "chat", "-q", prompt_text],  # 关键改动在这里
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            encoding="utf-8",
        )

        stdout, stderr = process.communicate()
        clean_stdout = stdout.strip()
        if process.returncode != 0:
            print("错误:", stderr)
        else:
            pass
            #print("Hermes 输出:", stdout)
        return clean_stdout
    except Exception as e:
        return f"❌ 无法启动 Hermes CLI: {e}"


def create_gh_issue(date, analysis):
    # --- 核心修复：强制检查 analysis 是否有效 ---
    if analysis is None:
        analysis = "Warning: Hermes CLI failed to return an analysis."
    else:
        analysis = str(analysis).strip()
        if not analysis:  # 处理空字符串
            analysis = "Analysis returned empty content."
    title = f"Audit {date}"
    try:
        # 使用列表模式，无需担心 analysis 里的引号导致命令断裂
        subprocess.run(
            [
                "gh",
                "issue",
                "create",
                "--title",
                title,
                "--label",
                "hermes-audit,quant-review",
                "--body",
                analysis,
            ],
            check=True,
        )
        print("✅ GitHub Issue 创建成功！")
    except subprocess.CalledProcessError as e:
        print(f"❌ GitHub CLI 执行失败: {e}")


import re


def clean_hermes_output(text):
    # 你的原始多行文本数据
    log_data = """
    ┊ 📖 read      /home/scott/code/ai-stock/agents/hermes/recent_lessons.md  0.8s
    ┊ 📖 read      /home/scott/code/ai-stock/agents/hermes/memory/short_term/recent_lessons.md  0.7s
    ┊ 🔎 find      core_lessons*  0.7s

    ╭─ ⚕ Hermes ───────────────────────────────────────────────────────────────────╮
        数据已获取。让我分析所有可用的记忆文件和信号数据，生成审计报告。
    ╰──────────────────────────────────────────────────────────────────────────────╯
    ┊ 📖 preparing read_file…
    ┊ 📖 read      /home/scott/code/ai-stock/agents/hermes  0.4s [error]
    ┊ 🔎 preparing search_files…
    ┊ 🔎 preparing search_files…
    ┊ 🔎 find      *lessons*  0.4s
    ┊ 🔎 find      *core*  0.4s
    ┊ 📖 preparing read_file…
    ┊ 📖 read      /home/scott/code/ai-stock/agents/hermes/memory/long_term/core_lessions.md  0.9s

    ╭─ ⚕ Hermes ───────────────────────────────────────────────────────────────────╮
        现在我已经掌握了所有上下文。内存文件（corelessions.md 和 recentlessons.md）目前都是空/占位状态，所以我会在报告中注明。以下是完整的审计报告：
        
        Hermes 深度审计报告
        日期: 2026-05-16 | 市场情绪: Low Volatility (仅3个预测,1条详细信号) | 主要标的: sh600584
        
        1. 信号质量与分布
        ... (此处省略部分重复的日志内容以便阅读) ...
        参考核心记忆：corelessions.md 存在但为空文件；recentlessons.md 内容为占位符。已基于模板中提到的历史教训（幅度低估、Regime缺陷、低分无效交易）进行分析。
    ╰──────────────────────────────────────────────────────────────────────────────╯
    """

    # 定义正则表达式
    # ╭─ ⚕ Hermes [^╮]*╮ : 匹配开头行
    # (.*?) : 惰性匹配中间的所有内容（包含换行）
    # ╰─[^╯]*╯ : 匹配结尾行
    pattern = r"╭─ ⚕ Hermes [^╮]*╮\n(.*?)\n╰─[^╯]*╯"
    matches = re.findall(pattern, text, re.DOTALL)

    # 将列表中的所有片段合并为一个字符串，并去除两端多余空格/换行
    result_string = "\n\n--- 报告分段 ---\n\n".join([m.strip() for m in matches])

    # 此时 result_string 就是纯字符串了
    #print(result_string)
    return result_string

if __name__ == "__main__":
    run_audit()
#    process_hermes_smart_save(
#                 hermes_output=output,
#                 date_str='2026-4-30'
#             )
