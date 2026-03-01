import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from research.log_parser import parse_all_logs

class AIStockLab:
    def __init__(self, raw_df):        
        self.raw_df = raw_df
           

    def calculate_hurst(self, series):
        """核心算法：计算 Hurst 指数"""
        if len(series) < 20: return 0.5
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return reg[0] * 2.0

    def run_backtest(self, h_threshold):
        """根据给定的 Hurst 阈值执行模拟回测"""
        initial_cash = 70000.0
        equity_curve = []
        
        # 预计算所有点的 Hurst (为了加速)
        dd_values = self.raw_df['dd'].values
        
        for i in range(len(self.raw_df)):
            # 计算当前窗口的 Hurst
            window = dd_values[max(0, i-20):i+1]
            h_val = self.calculate_hurst(window)
            
            # --- 核心逻辑：Hurst 断路器 ---
            if h_val < h_threshold:
                # 低于阈值：模拟清仓/空仓状态
                # 资产保持在上一时刻的水平 (形成阶梯)
                if len(equity_curve) > 0:
                    current_equity = equity_curve[-1]
                else:
                    current_equity = initial_cash
            else:
                # 高于阈值：跟随原始市场波动
                current_equity = initial_cash * (1 + self.raw_df.iloc[i]['dd']/100)
            
            equity_curve.append(current_equity)
            
        return equity_curve

    def optimize(self):
        """寻找最佳阈值"""
        thresholds = np.arange(0.35, 0.56, 0.01) # 扫描 0.35 到 0.55
        final_profits = []
        
        print(f"正在对 {len(thresholds)} 个参数进行网格搜索...")
        
        for h in thresholds:
            curve = self.run_backtest(h)
            final_profits.append(curve[-1])
            
        # 寻找最大值
        best_idx = np.argmax(final_profits)
        best_h = thresholds[best_idx]
        max_val = final_profits[best_idx]
        
        print(f"\n[结果] 最佳 Hurst 阈值: {best_h:.2f}")
        print(f"[结果] 最终资产最大化: {max_val:.2f}")
        
        # 绘制寻优结果
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, final_profits, marker='o', color='royalblue')
        plt.axvline(x=best_h, color='red', linestyle='--', label=f'Best Threshold: {best_h:.2f}')
        plt.title('Hurst Threshold vs Final Equity Optimizer')
        plt.xlabel('Hurst Threshold (Stop-loss Trigger)')
        plt.ylabel('Final Equity Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig('optimization_result.png')
        plt.show()
        
        # 最后用最佳参数再跑一次，对比原始曲线
        best_curve = self.run_backtest(best_h)
        original_curve = [70000 * (1 + d/100) for d in self.raw_df['dd']]
        
        return best_h, best_curve, original_curve

   
# --- 执行入口 ---
if __name__ == "__main__":
    # 请确保目录下有日志文件，并在此修改文件名
    LOG_DIR = 'logs' 
    log_df = parse_all_logs(LOG_DIR)
    lab = AIStockLab(log_df)
    
    best_h, best_curve, original_curve = lab.optimize()
    
    # 对比绘图
    plt.figure(figsize=(12, 6))
    plt.plot(original_curve, label='Original (No Protection)', alpha=0.6)
    plt.plot(best_curve, label=f'Optimized Protection (H < {best_h:.2f})', linewidth=2)
    plt.title('Final Comparison: Original vs Optimized AI Logic')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()

   