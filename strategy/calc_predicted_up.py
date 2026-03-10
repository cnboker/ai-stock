import numpy as np


def calc(        
        low: np.ndarray,
        median: np.ndarray,
        high: np.ndarray,
        latest_price: float,
    ):
        if latest_price <= 0:
            return 0.0

        if len(low) == 0 or len(median) == 0:
            return 0.0

        pred = median[-1]
        low_pred = low[-1]

        up = (pred - latest_price) / latest_price
        risk = max(0.0, (latest_price - low_pred) / latest_price)

        return float(up - 0.1 * risk)