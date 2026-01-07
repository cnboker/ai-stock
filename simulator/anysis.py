import csv
import math
from pathlib import Path

def dump_abnormal_predictions(
    results,
    csv_path="pre_result_abnormal.csv",
    model_score_threshold=0.35,
):
    """
    导出可能导致未下单的 PredictionResult：
    - ATR = 0 / NaN
    - model_score 低于阈值
    """
    csv_path = Path(csv_path)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # ===== CSV 表头 =====
        writer.writerow([
            "index",
            "atr",
            "atr_is_zero",
            "model_score",
            "model_score_low",
            "low_last",
            "median_last",
            "high_last",
            "predicted_up",
            #"gate_allow",
            "regime",
            #"gate_mult",
            "raw_score",            
            "action",
            "confidence"
        ])

        # ⚠️ 注意：只遍历 pre_result（list）
        for i, x in enumerate(results):
            # ⚠️ r 是 PredictionResult，不是 iterable
            r = x.prediction
            decision = x.decision
            atr = r.atr
            model_score = r.model_score

            atr_is_zero = (
                atr is None
                or (isinstance(atr, float) and (math.isnan(atr) or atr == 0.0))
            )

            model_score_low = model_score < model_score_threshold

            low_last = float(r.low[-1])
            median_last = float(r.median[-1])
            high_last = float(r.high[-1])

            # 是否预测向上（但可能被策略过滤）
            predicted_up = high_last > median_last > low_last
            regime = decision['regime']
            #gate_mult = decision['gate_mult']
            raw_score = decision['raw_score']
            action = decision['action']
            confidence = decision['confidence']
            # 只导出“可能导致不下单”的情况
            if atr_is_zero or model_score_low:
                writer.writerow([
                    i,
                    atr,
                    atr_is_zero,
                    model_score,
                    model_score_low,
                    low_last,
                    median_last,
                    high_last,
                    predicted_up,
                    regime,
                    #gate_mult,
                    raw_score,
                    action,
                    confidence
                ])

    print(f"[OK] abnormal predictions dumped to: {csv_path.resolve()}")
