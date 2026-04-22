import numpy as np
from typing import List, Optional, Dict
from log import signal_log


class SignalArbiter:
    def __init__(self, max_slots: int = 1):
        """
        :param max_slots: 最终允许下单的名额数量（比如前 1 名）
        """
        self.bucket = []
        self.max_slots = max_slots

    def add_candidate(self, candidate):
        ctx = candidate["ctx"]
        std = candidate["std"]

        """
        将满足基础开仓条件的信号放入漏斗
        """
        if not ctx.gate_allow or ctx.raw_signal != "LONG":
            return

        # 计算风险调整后收益 (预测夏普比率)
        # 这里的 ctx.predicted_up 是预期涨幅，std.mean() 是预期波动
        avg_std = std.mean() if std is not None else 0.01  # 兜底值
        rank_score = ctx.predicted_up / (avg_std + 1e-6)

        self.bucket.append({**candidate, "rank_score": rank_score, "std_val": avg_std})

    def get_best_decisions(self) -> List[Dict]:
        """
        对漏斗进行排序，返回排名前 N 的决策
        """
        if not self.bucket:
            return []

        # 1. 按 rank_score 降序排列 (分数越高越好)
        self.bucket.sort(key=lambda x: x["rank_score"], reverse=True)

        # 2. 提取前 N 名
        selected = self.bucket[: self.max_slots]

        # 打印选秀结果日志
        for i, item in enumerate(selected):
            signal_log(
                f"🏆 漏斗排名 [{i+1}]: {item['ticker']} | "
                f"得分: {item['rank_score']:.4f} | "
                f"预涨: {item['ctx'].predicted_up:.4%}"
            )

        return selected

    def clear(self):
        """清空漏斗，准备下一个周期"""
        self.bucket = []
