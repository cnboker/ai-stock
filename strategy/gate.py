import numpy as np
from dataclasses import dataclass

from log import get_logger


@dataclass
class GateResult:
    allow: bool
    score: float
    reason: str
    meta: dict


class PredictionGate:
    def __init__(
        self,
        score_threshold: float = 0.6,
        min_width_ratio: float = 0.3,
        vol_window: int = 32,
    ):
        """
        score_threshold : 低于该值，直接拦信号
        min_width_ratio : 区间/历史波动 最小比例
        vol_window      : 计算历史波动窗口
        """
        self.score_threshold = score_threshold
        self.min_width_ratio = min_width_ratio
        self.vol_window = vol_window

    # -------- 核心评分 --------
    def _operational_score(
        self,
        lower: np.ndarray,
        mid: np.ndarray,
        upper: np.ndarray,
        y_proxy: np.ndarray,
        close_df: np.ndarray,
    ) -> tuple[float, dict]:

        # 1. 空间
        hist_vol = np.std(close_df[-self.vol_window :]) + 1e-6
        width = np.mean(upper - lower)
        space_score = np.tanh(width / (hist_vol * self.min_width_ratio))

        # 2. 覆盖（用 proxy）
        coverage = np.mean((y_proxy >= lower) & (y_proxy <= upper))
        coverage_score = np.clip(coverage / 0.8, 0, 1)

        # 3. 方向
        pred_trend = np.sign(mid[-1] - mid[0])
        true_trend = np.sign(y_proxy[-1] - close_df[-1])
        direction_score = 1.0 if pred_trend == true_trend else 0.0

        score = 0.4 * space_score + 0.4 * coverage_score + 0.2 * direction_score

        meta = {
            "space": space_score,
            "coverage": coverage_score,
            "direction": direction_score,
            "width": float(width),
            "hist_vol": float(hist_vol),
        }

        return score, meta

    # -------- 对外接口 --------
    def evaluate(
        self,
        lower: np.ndarray,
        mid: np.ndarray,
        upper: np.ndarray,
        close_df: np.ndarray,
        y_proxy: np.ndarray | None = None,
    ) -> GateResult:

        if y_proxy is None:
            # 实盘：用 mid 自身做 proxy（保守）
            y_proxy = mid

        score, meta = self._operational_score(
            lower=lower,
            mid=mid,
            upper=upper,
            y_proxy=y_proxy,
            close_df=close_df,
        )

        #print("score", score)

        allow = score >= self.score_threshold
        reason = "OK" if allow else "LOW_OPERATIONAL_SCORE"
     
        gate_result = GateResult(
            allow=allow,
            score=float(score),
            reason=reason,
            meta=meta,
        )
        '''
        score：市场可交易性评分（0.8 / 1.0 都是 允许交易）
        space=1.00：仓位空间充足
        coverage=1.00：信号覆盖完整
        '''
        get_logger("gate").info(
            f"[GATE] score={gate_result.score:.3f} "
            f"space={gate_result.meta['space']:.2f} "
            f"coverage={gate_result.meta['coverage']:.2f}"
        )
        return gate_result

gater = PredictionGate(score_threshold=0.6)
