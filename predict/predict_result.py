from dataclasses import dataclass, field
from typing import Union

import pandas as pd

@dataclass
class PredictionResult:
    low: Union[float, pd.Series]
    median: Union[float, pd.Series]
    high: Union[float, pd.Series]
    model_score: float
    atr:float