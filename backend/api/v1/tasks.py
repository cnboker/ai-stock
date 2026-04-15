# Week 2: Optuna Task APIs
# tasks 将负责耗时的 Optuna 调优任务
from fastapi import APIRouter, BackgroundTasks, Depends
from sqlmodel import Session
from api.deps import get_session
from services.optuna_tuner import run_optimization_task

router = APIRouter()

@router.post("/run_optuna")
async def trigger_optuna(
    background_tasks: BackgroundTasks,
    focus: str = "balanced", # 可由 Hermes 决定：'volatility', 'drawdown', 'profit'
    session: Session = Depends(get_session)
):
    """
    由 Hermes 或人工触发的异步调优任务
    """
    # 异步执行，立即返回 202 状态码给调用者
    background_tasks.add_task(run_optimization_task, focus=focus)
    
    return {
        "status": "task_started",
        "message": f"Optuna 调优任务已启动，优化侧重点: {focus}",
        "timestamp": "2026-04-15 11:40:00" # 示例当前时间
    }

@router.get("/task_status")
async def get_task_status():
    # 未来这里可以从 Redis 或 数据库读取任务进度
    return {"status": "idle", "last_run": "2026-04-12"}