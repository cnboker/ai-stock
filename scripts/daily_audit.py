import requests
import subprocess # 用于调用本地 hermes cli
import os
from datetime import datetime

from scripts.hermes_prompt import prepare_hermes_prompt, process_hermes_smart_save

# 配置
API_BASE_URL = "http://localhost:8080/api/v1"
AUDIT_LOG_PATH = "logs/audit_records.md"
# 假设你的 cli 命令是 'hermes'，如果是路径则写全称如 '/usr/local/bin/hermes'
HERMES_CLI_CMD = "hermes" 

output = """
Hermes 
    现在我已获取 recent_lessons.md 中的核心教训（2026-04-29审计结果），core_lessons.md 不存在。开始深度分析原始数据。

    Hermes 深度审计报告
    日期: 2026-05-01 | 市场情绪: Low Volatility / 局部下行趋势 | 主要标的: sh515150（上证ETF）、sh601899（紫金矿业）、sz159908（创业板ETF）、sz002463（沪电股份）

    1. 信号质量与分布

    低分过热检测：
    - 4个信号的 model_score < 0.60，全部产生交易：
      - sz159908 10:30 (score=0.588) → pred=-0.0007, actual=-0.0026 ✅方向正确但预测幅度仅27%
      - sz159908 11:00 (score=0.295) → pred=-0.0104, actual=-0.0032 ❌幅度高估3.25x
      - sz002463 14:00 (score=0.417) → pred=-0.0149, actual=-0.0017 ❌幅度高估8.76x
      - sz002463 14:30 (score=0.500) → pred=-0.0076, actual=-0.0022 ❌幅度高估3.45x

      结论：score<0.60的信号中，仅1个方向合理但大幅低估幅度，其余3个全部严重偏离。低分信号的质量极度不可靠。

    Pred/ATR 分析：
    | 标的 | Pred/ATR范围 | 结论 |
    |------|-------------|------|
    | sh515150 | 0.478x ~ 0.644x | 持续低估 |
    | sh601899 | 0.057x ~ 0.096x | 严重低估（不到ATR的1/10） |
    | sz159908 | 0.029x ~ 0.491x | 严重低估为主 |
    | sz002463 | 0.006x ~ 0.011x | 在巨大ATR面前几乎零预测 |

    全量信号 Pred/ATR 均 ≤0.65x。核心教训"幅度严重低估"再次全面验证。sh601899和sz002463尤其离谱——预测幅度不到实际波动范围的1/10。

    关键异常信号：
    1. sh601899 13:30 — 偏差最大（error=0.039）：pred=+2.59%, actual=-1.31%, 方向完全相反且幅度偏差3.9%。score=1.0（最高分），model极度自信但全错。
    2. sz002463 14:00 — error=0.0132：pred=-1.49%, actual=-0.17%, 幅度高估8.76倍。score=0.417低分，ATR=1.31（极大）。

    2. 风控拦截效率

    拦截统计：共26个信号（summary所示），14个可审计，全部被拦截（REDUCE状态）。如可审计信号代表全量，拦截率100%。

    拦截回溯：
    - 所有信号都被REDUCE（gate_mult = 0.418~0.648，即仓位被削减至41%~65%）。
    - 拦截后的实际仓位仍然产生了-485.6的净亏损（仅sz159908 10:00盈利+68.9，其余全部亏损）。
    - 风控层虽然降低了每笔交易的规模，但方向性错误导致拦截后仍全面亏损。

    Gate逻辑检查：
    - Gate_mult 与 model_score 正相关（高分→高gate_mult），但方向性错误使得高分信号的gate_mult反而放大了亏损。sh601899 13:30 (score=1.0, gate=0.648) 产生了-88的亏损。
    - 存在Gate悖论：Gate仓位削减的逻辑是"降低不确定信号的风险"，但实际上低分信号的幅度偏差比高分信号的方向性错误危害更小。当前Gate以分数为权重依据，未区分"方向错误"与"幅度偏差"——这是重大缺陷。

    3. 流动性与方向性错误分析（核心部分）

    方向错误大爆发：
    本日最严重的问题是系统性方向错误，14个信号中仅6个方向正确（42.9%准确率），且多集中在sz159908。

    重点案例尸检：

    sh515150（上证ETF）— 全线方向错误
    | 时间 | Pred | Actual | 方向 |
    |-----|------|--------|:----:|
    | 10:30 | +0.34% | -0.36% | ❌ |
    | 11:00 | +0.36% | -0.48% | ❌ |
    | 13:00 | +0.30% | -0.42% | ❌ |
    | 13:30 | +0.23% | -0.42% | ❌ |

    5个信号全部预测上涨，实际全部下跌。模型对上证ETF全天呈现持续的看涨偏向，忽视了持续阴跌的现实。

    sh601899（紫金矿业）— 全线方向错误
    | 时间 | Pred | Actual | 方向 |
    |-----|------|--------|:----:|
    | 11:00 | +1.95% | -0.45% | ❌ |
    | 13:00 | +1.57% | -0.48% | ❌ |
    | 13:30 | +2.59% | -1.31% | ❌ |
    | 14:00 | +1.82% | -1.16% | ❌ |

    4个信号全部预测大幅上涨，实际持续下跌。score最高达1.0（13:30, 14:00），model极度自信但全天方向错误。

    Regime分类器表现：
    所有信号的regime均为"good"。这不是传统意义的"neutral误判"，而是更危险的"good误判"——模型认为市场处于"good"（适合交易/趋势明确）状态，但实际上sh515150和sh601899处于持续下行通道，模型却一直输出看涨预测。这暗示regime分类器不仅会在趋势中输出neutral，还可能在下跌趋势中错误标记为good并输出看涨方向。

    高波动 vs 低波动标的：
    - sh601899（高波动，ATR≈0.25~0.29）: 方向性错误最严重，损失最大（-311）
    - sh515150（低波动，ATR≈0.005~0.006）: 持续方向错误，损失可控但频率高（-145）
    - sz002463（极高ATR，ATR≈1.2~1.3）: pred/ATR几乎为零，预测完全无意义

    4. 系统性偏差诊断

    核心教训回溯：
    1. ✅ 幅度严重低估 — 再次出现。全部信号pred/ATR < 0.65x，sh601899低至0.06x。
    2. ✅ Regime分类器误判 — 本次不是输出neutral，而是在下跌趋势中错误输出good并持续做多，危害更严重。
    3. ✅ 低分无效交易 — sz002463 score=0.417和0.500的信号产生大量偏离。
    4. 新发现：系统性方向偏向 — 模型在sh515150和sh601899上呈现顽固的看涨偏向，持续预测上涨而实际下跌。

    本日最主要的逻辑断层：
    模型在局部下行周期中丧失了方向感知能力。全部出错信号都指向同一个问题：信号生成器在下跌环境中顽固地输出看涨预测。无论score多高、regime多"good"，方向错误使一切精确定量失去意义。这比之前发现的"幅度低估"严重得多——幅度低估可以靠Gate缩放，但方向错误直接导致亏损。

    5. 系统优化建议

    [SYSTEM_OPTIMIZATION]
    1. 引入方向合理性检测：当连续2个以上信号对同一标的在同方向全部错误时，强制触发方向怀疑机制，反转或暂停该标的的预测。sh515150在10:30第一次方向错误时，后续4个信号应该被标记为高风险。
    2. Gate逻辑升级：当前Gate以score为权重（高分→高仓位），但高分伴随方向错误时风险最大。建议引入"方向置信度"作为独立维度的Gate因子——当pred_sign与近期实际走势持续背离时大幅降低gate_mult，不论score多高。
    3. Regime分类器加强：增加"持续下行识别"维度——当标的连续N个K线收盘价低于前日时，regime不应输出"good"或至少应在方向输出上加入下行偏向。
    4. sz002463 Pred/ATR保护：ATR高达1.2-1.3时，pred变化仅0.7-1.5%，信号与波动无关，建议对pred/ATR<0.05的信号自动拦截（零仓位）。

    [MODEL_BIAS_ALERT]
    强劲的看涨偏向：2026-04-30的审计中发现了明确的看涨系统性偏向——sh515150（8次预测方向）和sh601899（5次预测方向）全线预测上涨，即使实际持续下跌。这不是偶然误差，而是信号生成器在局部下行环境中的顽固做多偏向。可能原因：
    - 训练数据中上涨样本过占主导
    - 趋势检测逻辑对"上涨延续"比"下跌延续"更敏感
    - 特征提取在上行窗口更有效，下行时信号衰减

    总结：本日最薄弱环节

    优先级排序的修复建议：
    1. 方向错误优先于幅度偏差：幅度可以靠Gate缩放，方向错误直接导致亏损。必须引入方向合理性检测机制——当预测方向与近期走势持续背离时，强制限制或暂停交易。这是本日最严重的单点故障。
    2. Regime分类器增加下降趋势检测：当前regime输出"good"在下跌环境中反而助长了错误交易。增加持续下行判断逻辑，在下跌趋势中自动降低看涨信号的置信度。
    3. 低分+低pred/ATR双重拦截：对 score<0.60 且 pred/ATR<0.05 的信号自动零仓位。sz002463的两个信号完全符合此特征，本不必要的交易被Gate削减后仍然造成了-59的损失。

    参考核心记忆：已参考 recent_lessons.md 中的幅度低估、Regime缺陷、低分无效交易等关键教训。本日新发现的核心问题是方向性系统性偏差——比幅度低估更致命的逻辑断层。

    现在更新 recent_lessons.md：

   ✍️ preparing write_file…
   ✍️  write     ./agents/hermes/recent_lessons.md  0.7s
   review diff
a/agents/hermes/recent_lessons.md → b/agents/hermes/recent_lessons.md
@@ -2,18 +2,20 @@
 description: 最近7-14天的教训摘要（会定期更新）

 # Hermes Recent Lessons - 近期教训摘要
-更新日期: 2026-04-29
+更新日期: 2026-05-01

-## 当前核心教训（基于2026-04-29审计）
+## 当前核心教训（基于2026-04-30审计）

-- **幅度低估偏差**：模型 pred_change 普遍只有实际波动的 1/10 左右，在强趋势中严重低估涨幅。需重点监控 Pred/ATR 比例。
-- **Regime 分类器缺陷**：在连续上涨趋势中容易错误输出 "neutral"，导致信号生成器做反向或过保守预测。这是当前最主要的逻辑断层。
+- **新：方向性系统性偏向** — 2026-04-30审计中sh515150(5/5)和sh601899(5/5)全线预测上涨但实际下跌。模型在局部下行周期中顽固输出看涨预测。这比幅度低估更致命——幅度可Gate缩放，方向错误直接导致亏损。
+- **幅度低估偏差**：模型 pred_change 普遍只有实际波动的 1/10 左右，在强趋势中严重低估涨幅。需重点监控 Pred/ATR 比例。本日 sh601899 Pred/ATR 低至 0.06x。
+- **Regime 分类器缺陷**：在连续上涨趋势中容易错误输出"neutral"（已确认）。2026-04-30新发现：在持续下跌环境中可能输出"good"并配合错误看涨方向，危害更大。需要增加下降趋势识别维度。
 - **低波动标的处理**：score 在 0.54~0.60 区间的信号容易产生无效微小交易（如 sh603871），建议对此类标的提升最低 score 阈值至 0.60。
-- **高波动标的建议**：Gate 偏保守，在 model_score > 0.7 时可考虑允许更大预测幅度。
+- **低分+低Pred/ATR双重拦截需求**：sz002463 score=0.417/0.500 且 pred/ATR=0.006x/0.011x 的信号本应零仓位。建议对 score<0.60 且 pred/ATR<0.05 的信号自动拦截。
+- **Gate悖论**：当前Gate以score为权重（高分→高仓位），但高分信号的方向错误放大亏损（sh601899 score=1.0但方向全错）。需引入"方向置信度"作为独立Gate因子。
 - **风控救场现象**：风控层有时能挽救信号层错误，但不能长期依赖，核心仍需提升信号层质量。
-- **开盘信号风险**：09:23 前后的竞价阶段信号需加强时间上下文过滤，避免误判挂单堆积为趋势启动。
+- **开盘信号风险**：09:23前后的竞价阶段信号需加强时间上下文过滤，避免误判挂单堆积为趋势启动。

 **更新规则**：
 - 每次审计完成后，Hermes需输出更新后的 recent_lessons 内容。
 - 保持总条数不超过 8 条，优先保留最新、最严重的系统性问题。
-- 每7天进行一次元复盘，决定是否合并或移入 core_lessons.md。+- 每7天进行一次元复盘，决定是否合并或移入 core_lessons.md。

  Hermes 
    审计完成。recent_lessons.md 已更新，新增了"方向性系统性偏向"和"Gate悖论"两条核⼼教训。

    报告要点速览：

    本日最大发现：方向性系统性错误比幅度低估更严重。sh515150和sh601899全线预测上涨但实际下跌，且高分信号（score=0.84~1.0）的错误方向配合高gate_mult放大了亏损。当前Gate按score分配仓位，但在方向错误时这逻辑正好反作用。

    最需优先修复的3件事：
    1. 引入方向合理性检测 — 连续方向错误时暂停该标的交易
    2. Gate增加方向置信度维度 — 脱离唯score论
    3. Regime分类器增加下降趋势识别 — 避免下跌中输出"good"配合看涨预测
"""

def run_audit():
    # 1. 抓取数据
    daily_json = fetch_daily_data()
    if not daily_json or not daily_json.get("data"):
        print("📭 今日无交易预测数据。")
        return
   
    # 2. 构造初始 Prompt

    final_prompt, current_date = prepare_hermes_prompt(daily_json)

    # 3. 第一次分析
    print("🤖 Hermes CLI 正在分析...")
    print(f"🔍 初始 Prompt:\n{final_prompt}")  # 只打印前500字符，避免过长
    analysis = ask_hermes_cli(final_prompt)
    hermes_output = clean_hermes_output(analysis)
    print(f"\n--- 初次审计结果 ---\n{analysis}")
    # 实际操作中，你会把 Hermes 的输出传给这个函数
    
    # 4. 进入交互模式
    while True:
        user_input = input("\n💬 追问 Hermes (输入 's' 保存并同步 GitHub, 'q' 退出): ").strip()
        
        if user_input.lower() == 'q':
            break
        elif user_input.lower() == 's':
            # 记录到本地
            process_hermes_smart_save(
                hermes_output=hermes_output, 
                date_str=current_date
            )
            # 同步到 GitHub
            create_gh_issue(current_date, analysis)
            break
            



def fetch_daily_data():
    """从 FastAPI 获取今日交易表现"""
    try:
        date = datetime.now().date()
        date = '2026-04-30'
        response = requests.get(f"{API_BASE_URL}/analytics/daily_review/{date}")
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
    new_proxy = "http://127.0.0.1:7890"
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


if __name__ == "__main__":
    #run_audit()
   process_hermes_smart_save(
                hermes_output=output, 
                date_str='2026-4-30'
            )
