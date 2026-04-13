from abc import ABC, abstractmethod
import pandas as pd

class BaseTSFMAdapter(ABC):
    @abstractmethod
    def predict(self, df: pd.DataFrame, prediction_length: int) -> pd.DataFrame:
        """统一返回包含: low, median, high 三列的 DataFrame"""
        pass