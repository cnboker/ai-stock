
-- 2. 清理旧数据（可选，调试用）
-- TRUNCATE orders, predictions RESTART IDENTITY;

-- 3. 插入模拟预测数据 (Predictions)
INSERT INTO predictions (symbol, timestamp, expected_return, confidence_interval_low, confidence_interval_high, features_snapshot, model_version)
VALUES 
('NVDA', NOW() - INTERVAL '2 days', 0.035, 0.028, 0.042, '{"rsi": 65.4, "vol_ratio": 1.2, "ma_cross": "golden"}', 'chronos-bolt'),
('AAPL', NOW() - INTERVAL '1 day', -0.015, -0.020, -0.010, '{"rsi": 42.1, "vol_ratio": 0.8, "ma_cross": "death"}', 'chronos-v1'),
('TSLA', NOW() - INTERVAL '12 hours', 0.052, 0.040, 0.065, '{"rsi": 72.8, "vol_ratio": 2.5, "ma_cross": "golden"}', 'chronos-bolt'),
('BTCUSDT', NOW() - INTERVAL '2 hours', 0.012, 0.005, 0.018, '{"rsi": 55.0, "vol_ratio": 1.5, "ma_cross": "none"}', 'chronos-v1');

-- 4. 插入对应的订单数据 (Orders)
INSERT INTO orders (symbol, side, entry_price, exit_price, entry_time, exit_time, actual_return, pnl_amount, prediction_id, status)
VALUES 
-- 场景 1: 完美符合预测 (NVDA)
('NVDA', 'buy', 900.50, 932.00, NOW() - INTERVAL '47 hours', NOW() - INTERVAL '40 hours', 0.035, 3150.00, 1, 'closed'),

-- 场景 2: 预测失败 - 止损出场 (AAPL)
-- 预期跌(Sell)，结果反弹了
('AAPL', 'sell', 170.00, 175.00, NOW() - INTERVAL '23 hours', NOW() - INTERVAL '20 hours', -0.029, -500.00, 2, 'closed'),

-- 场景 3: 盈利但不及预期 (TSLA)
('TSLA', 'buy', 180.00, 182.50, NOW() - INTERVAL '11 hours', NOW() - INTERVAL '5 hours', 0.013, 250.00, 3, 'closed'),

-- 场景 4: 正在持仓中 (BTCUSDT)
('BTCUSDT', 'buy', 65000.00, NULL, NOW() - INTERVAL '1 hour', NULL, 0.0, 0.0, 4, 'open');

-- 5. 调试查询语句示例：复盘 Chronos 预测准确度
/*
SELECT 
    p.symbol, 
    p.expected_return, 
    o.actual_return, 
    (o.actual_return - p.expected_return) as alpha_error,
    p.features_snapshot
FROM predictions p
JOIN orders o ON p.id = o.prediction_id
WHERE o.status = 'closed';
*/