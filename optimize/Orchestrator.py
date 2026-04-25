import os
from pathlib import Path
from typing import Any
from anyio import Path

from dotenv import load_dotenv
from optimize.config_factory import ConfigFactory
from optimize.diagnostic_scanner import DiagnosticScanner
from google import genai
from pydantic import BaseModel, Field
from typing import Dict, Any

from optimize.opt_study import run_optuna_study
from optimize.persist_manager import PersistManager
import os
# 如果你的网络可以直接访问，或者你希望 SDK 不走这个 socks 代理
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('all_proxy', None)


# 自动寻找当前文件所在目录的父目录中的 .env
# 确保路径正确（.parent 代表当前文件所在目录）
env_path = Path(__file__).parent.parent / '.env' 

# 修正参数名为 dotenv_path
result = load_dotenv(dotenv_path=env_path)

print(f"是否成功加载文件: {result}") # 如果返回 False，说明路径还是不对
gemini_api_key = os.getenv("gemini_api_key")
print(f'gemini_api_key: {gemini_api_key}')


class ParameterSpace(BaseModel):
    low: float
    high: float

class InitialTrialConfig(BaseModel):
    strength_alpha: float
    model_th: float
    atr_stop: float
    risk: float
    kelly: float
    slope: float
    max_stop: float
    predict_up: float
    tp1: float
    tp2: float
    init_pt: float

class SearchSpaceConfig(BaseModel):
    strength_alpha: ParameterSpace
    model_th: ParameterSpace
    atr_stop: ParameterSpace
    risk: ParameterSpace
    kelly: ParameterSpace
    slope: ParameterSpace
    max_stop: ParameterSpace
    predict_up: ParameterSpace
    tp1: ParameterSpace
    tp2: ParameterSpace
    init_pt: ParameterSpace

class GeminiOptimizationResponse(BaseModel):
    analysis: str
    action_taken: str
    suggest_search_space: SearchSpaceConfig
    recommended_initial_trial: InitialTrialConfig 
    give_up: bool

client = None

def ask_gemini_to_fix_config(report: Dict[str, Any]):
    if not client:
        client = genai.Client(api_key=gemini_api_key)

    ticker = report['ticker']
    status = report.get("status", "ANEMIC")
    
    # 基础上下文
    base_context = f"""
        你是一个量化交易专家。目前正在优化股票 {ticker} 的 30分钟线 交易策略。
        当前诊断拦截报告: {report['intercept_report']}
        当前参数配置: {report['current_config']}
        """

    # 根据不同状态定制核心任务指令
    if status == "OPTUNA_FAIL":
        task_instruction = f"""
        ### 紧急情况：OPTUNA_FAIL
        警告：Optuna 在 10 次尝试后，Trade Count 依然为 0。这意味着目前的搜索空间（Search Space）和初始参数完全无法触发入场逻辑。
        
        你的核心任务是【开闸放水】：
        1. **大幅调低入场门槛**：下调 model_th (模型置信度) 的 low 边界至 0.1-0.2，下调 slope (斜率) 至接近 0。
        2. **削弱强度抑制**：减小 strength_alpha，让小波动的信号也能产生 Strength。
        3. **缩减触发步长**：降低 init_pt。
        
        请重新定义搜索空间，确保至少能产生交易信号。如果这只股票在 30分钟周期下完全没有波动趋势，请将 give_up 设为 true。
        """
    else:
        # 构造 Prompt 内容
        task_instruction = f"""
        ### 任务：预检优化 (ANEMIC)
        目前预检入场次数仅为 {report['success_count']} 次。
        请根据拦截报告分析瓶颈，并微调参数空间以增加信号捕获率。
        """
    prompt = f"""
        {base_context}
        
        {task_instruction}
        ### 输出要求 (JSON ONLY):
    请严格按照定义的 Schema 返回 JSON，包含 analysis (瓶颈分析), action_taken (采取动作), suggested_search_space (修正空间), recommended_initial_trial (推荐初始值), give_up (是否放弃)。
    """

    model_id = "gemini-3-flash-preview" # 或者根据官网最新的 gemini-3-flash

    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": GeminiOptimizationResponse,
            },
        )

        # 直接获取解析后的 Pydantic 对象
        advice = response.parsed
        print(f"🤖 Gemini 诊断建议: {advice.analysis}")
        print(f"🛠️ 采取行动: {advice.action_taken}")

        return advice
    except Exception as e:
        print(f"❌ 调用 Gemini 失败: {e}")
        return None

def start_optimization_cycle(ticker: str, ticker_period: str, reset_study: bool = False):
    cfg = ConfigFactory.load_ticker_config(ticker)
    initial_trial = cfg["initial_trial"]

    # --- 1. 初次诊断 ---
    report = DiagnosticScanner.run_body_check(ticker, ticker_period, initial_trial)

    # --- 2. 如果诊断不合格，先让 Gemini 改一版参数 ---
    # if report["status"] == "ANEMIC":
    #     advice = ask_gemini_to_fix_config(report)
    #     if not advice or advice.give_up:
    #         print(f"🚫 Gemini 建议放弃 {ticker}")
    #         return
    #     initial_trial = advice.recommended_initial_trial
    #     # 更新配置以便下一步 Optuna 使用
    #     PersistManager.save_ticker_config(ticker, advice)

    # --- 3. 运行 Optuna ---
    study = run_optuna_study(ticker, ticker_period, n_trials=50, reset_study=reset_study, slope_stats=report["slope_stats"])
    
    # --- 4. 核心改动：检查 Optuna 运行后的“出勤率” ---
    # 假设你的 backtest 函数会将交易次数存在 study 的 user_attr 里，或者通过 study 的结果判断
    # 如果 10 次 Trial 后，最好的结果依然是“无交易”（Trade Count = 0）
    # best_value = study.best_value
    best_trial = study.best_trial
    print(f"best_trial={best_trial}")
    # 判断逻辑：如果最好的一次 Trial 收益为 0 且成交数为 0
    # if best_value < 0 or best_trial.user_attrs.get("train_trade_count", 0) == 0:
    #     print(f"⚠️ Optuna 运行 10 次后仍无成交。正在进行【深度反馈优化】...")
        
    #     # 构造一个更深度的数据包给 Gemini
    #     enhanced_report = {
    #         "ticker": ticker,
    #         "status": "OPTUNA_FAIL",
    #         "total_scans": 10,
    #         "success_count": 0,
    #         "intercept_report": report["intercept_report"], # 沿用之前的诊断
    #         "current_config": study.best_params, # 喂给它 Optuna 刚尝试失败的参数
    #         "failure_note": "Optuna tried 10 combinations but couldn't find a single trade. Filters might be too tight for the full backtest period."
    #     }
        
    #     # 再次调用 Gemini
    #     new_advice = ask_gemini_to_fix_config(enhanced_report)
        
    #     if new_advice and not new_advice.give_up:
    #         print(f"♻️ Gemini 重新调整了搜索空间。写入文件并尝试最后一次 Optuna...")
    #         PersistManager.save_ticker_config(ticker, new_advice)
            
    #         # 使用全新的空间再跑 30 次
    #         run_optuna_study(ticker, ticker_period, n_trials=30, reset_study=False, slope_stats=enhanced_report["current_config"])
    #     else:
    #         print(f"🛑 深度优化失败，彻底放弃 {ticker}。")

