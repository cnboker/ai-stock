import sys
import os
# 确保能引用到上一级目录的 strategy 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategy.your_strategy_file import YourStrategyClass # 引用你实际的策略类
import pandas as pd
import matplotlib.pyplot as plt

class BacktestEngine:
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
    def run(self, gate_val, threshold_val, mode='original'):
        # 初始化资产
        cash = 70000 
        hold_shares = 0
        equity_history = []
        
        for index, row in self.data.iterrows():
            # 1. 获取当前数据
            current_slope = row['slope']
            current_price = row['price']
            
            # 2. 调用你的核心逻辑 (此处手动模拟或引用你的函数)
            # 如果是 'ai_mode', 我们在这里加入你想要的动态逻辑
            if mode == 'ai_mode':
                # 示例：如果斜率方向连续3次一致，临时增强 gate
                gate = gate_val * 1.2 
            else:
                gate = gate_val
            
            strength = current_slope * gate
            
            # 3. 模拟买卖动作
            if strength > threshold_val and cash >= current_price * 100:
                # 买入逻辑
                hold_shares += 100
                cash -= current_price * 100
            elif strength < -threshold_val and hold_shares > 0:
                # 卖出逻辑
                cash += hold_shares * current_price
                hold_shares = 0
            
            # 4. 计算当前总资产
            total_equity = cash + (hold_shares * current_price)
            equity_history.append(total_equity)
            
        return equity_history

# --- 执行对比 ---
engine = BacktestEngine('../data/history_backtest.csv')

# 跑一遍你现在的参数 (假设 gate=0.5, threshold=1.0)
old_curve = engine.run(0.5, 1.0, mode='original')

# 跑一遍你设想的优化参数 (比如降低门槛)
new_curve = engine.run(0.3, 0.7, mode='ai_mode')

# 绘图输出到文件
plt.plot(old_curve, label='Old Strategy')
plt.plot(new_curve, label='New Strategy')
plt.legend()
plt.savefig('comparison_result.png')