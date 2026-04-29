import sys
import os
import pandas as pd

from infra.utils.time_profile import timer_decorator
from .base import BaseTSFMAdapter

class KronosAdapter(BaseTSFMAdapter):
    def __init__(self, model_name="kronos-base", base_path: str = "."):
        self.model_name = model_name
      
        
        # 1. 这里的 base_path 可能是相对路径，将其转为绝对路径
        absolute_base_path = os.path.abspath(base_path)
        model_path = os.path.join(absolute_base_path, "models", model_name)
        tokenizer_path = os.path.join(absolute_base_path, "models", "kronos-tokenizer-base") 
        # 调试用：打印一下路径，确保它指向正确的文件夹
        print(f"DEBUG: Looking for model at {model_path}")

        # 2. 动态注入 Kronos_Source 路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 注意：这里的相对层级需根据你实际的目录结构确认
        # 假设：predict/adapters/kronos_adapter.py -> ../../models/Kronos_Source
        kronos_src = os.path.abspath(os.path.join(current_dir, "../../models/Kronos_Source"))
        
        if kronos_src not in sys.path:
            sys.path.insert(0, kronos_src)

        from model import Kronos, KronosTokenizer, KronosPredictor
        
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
        model = Kronos.from_pretrained(model_path)
        
        # 适配 V100 开启半精度
        self.predictor = KronosPredictor(model, tokenizer, max_context=512)
        #self.predictor.model.cuda().half() 
    
    def predict(self, df, prediction_length):
        # 1. 确保索引为 Datetime 类型
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        x_df = df[['open', 'high', 'low', 'close', 'volume']].tail(512)
        # 将 DataFrame 的索引（通常是时间戳）转换为 Pandas Series。
        # 这是因为 Kronos 的 predictor.predict 方法通常要求时间轴作为独立的序列输入，以便处理时间特征
        x_ts = x_df.index.to_series()
        
        # 2. 自动获取频率
        # 自动推断: 它查看最后 10 条数据的时间间隔。
        inferred_freq = pd.infer_freq(df.index[-10:]) or '5T'
        
        # 3. 构造 y_ts 并强制转换为 Series (修复 AttributeError 的核心)
        y_range = pd.date_range(start=x_ts.iloc[-1], periods=prediction_length+1, freq=inferred_freq)[1:]
        y_ts = pd.Series(y_range) # 关键：将 DatetimeIndex 转为 Series

        # 4. 执行预测
        # res = self.predictor.predict(x_df, x_ts, y_ts, prediction_length, sample_count=5,verbose=False)
        
        # return pd.DataFrame({
        #     "low": res["close"].values * 0.99,
        #     "median": res["close"].values,
        #     "high": res["close"].values * 1.01
        # })
        # 修改执行预测部分
        sample_count = 10 # 增加采样数以获得真实的分布
        res = self.predictor.predict(x_df, x_ts, y_ts, prediction_length, sample_count=sample_count, verbose=False)

        # res["close"] 通常是一个 shape 为 (sample_count, prediction_length) 的数组
        # 计算真实的预测区间
        mean_pred = res["close"].mean(axis=0)
        std_pred = res["close"].std(axis=0)

        return pd.DataFrame({
            "median": mean_pred,
            "std": std_pred,
            "upper_bound": mean_pred + 2 * std_pred, # 动态压力位
            "lower_bound": mean_pred - 2 * std_pred,  # 动态支撑位
            "low": res["close"].values * 0.99,
            "high": res["close"].values * 1.01
        })