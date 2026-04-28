```txt
           ┌─────────────┐
           │ Market Data │
           └─────┬───────┘
                 │
                 ▼
         ┌─────────────────┐
         │ Determine Regime │
         │ (neutral/bull/bear) │
         └─────┬───────────┘
               │
       ┌───────┴─────────┐
       │                 │
       ▼                 ▼
   regime = neutral   regime ≠ neutral
       │                 │
       │                 ▼
       │          ┌────────────────────┐
       │          │ Calculate slope     │
       │          │ slope_threshold ?   │
       │          └─────┬──────────────┘
       │                │
       │        slope*gate > threshold?
       │                │
       │        ┌───────┴─────────┐
       │        │                 │
       ▼        ▼                 ▼
  action=None  action=LONG     action=SHORT
  strength=0   strength= f(slope, gate) 

  逻辑说明

    Regime 判断

    neutral → 系统倾向 不操作

    bull / bear → 根据趋势和阈值决定多空

    Slope + Gate

    斜率 slope 表示趋势方向和强度

    gate 是调节系数

    系统只有当 slope*gate 超过内部阈值才会生成动作

    Action & Strength

    action=None → 不买卖

    strength=0 → 持仓变动力度为 0

    当条件满足时，strength 会随趋势强度变化

```

# 周期 (Timeframe),每天 K 线数量,1000 条数据可覆盖时长,是否满足 6 个月？
# 日线 (Daily),1 条,~1000 个交易日 (约 4 年),是 (过度覆盖)
# 1 小时 (1H),4 条,~250 个交易日 (约 12 个月),是 (完美契合)
# 30 分钟 (30M),8 条,~125 个交易日 (约 6 个月),是 (刚好满足)
# 15 分钟 (15M),16 条,~62.5 个交易日 (约 3 个月),否
# 5 分钟 (5M),48 条,~21 个交易日 (约 1 个月),否

# ETF 优化参数
```js
 config = {
        "MODEL_TH": trial.suggest_float("model_th", 0.42, 0.48),
        "SLOPE": trial.suggest_float("slope", -0.001, 0.005),
        "PREDICT_UP": trial.suggest_float("predict_up", 0.0, 0.005),
        "INIT_PT": trial.suggest_float("init_pt", 0.02, 0.04),
        "TREND_STAGE": trial.suggest_float("trend_pt", 0.05, 0.25),
        "TP1": trial.suggest_float("tp1", 1.03, 1.06),
        "TP2": trial.suggest_float("tp2", 1.1, 1.25),
        "KELLY": trial.suggest_float("kelly", 0.3, 0.5),
        "RISK": trial.suggest_float("risk", 0.01, 0.015),
        "ATR_STOP": trial.suggest_float("atr_stop", 2.5, 3.5),
        "MAX_STOP": trial.suggest_float("max_stop", 0.03, 0.10),
        "MIN_STOP": trial.suggest_float("min_stop", 0.01, 0.02),
        
        "STRENGTH_ALPHA": trial.suggest_float("strength_alpha", 1.2, 1.5),
        "CONFIRM_N": trial.suggest_float("confirm_n", 2, 5),
    }
```

### 全自动参数优化
    ```bash
    python -m optimize.auto_tune
    ```
{
    "atr_stop": {"low": 3.0, "high": 7.0},       // 允许更大的波动
    "max_stop": {"low": 0.05, "high": 0.12},     // 接受 5%-12% 的波段回撤
    "min_stop": {"low": 0.02, "high": 0.03},     // 过滤掉 2% 以内的所有噪音
    "tp1": {"low": 1.10, "high": 1.20},          // 止盈设高点，不翻倍不走
    "strength_alpha": {"low": 2.0, "high": 15.0} // 降低止损随价格上涨的移动速度
}


### install hermes
# 执行官方安装脚本
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash

# 安装完成后刷新 shell
source ~/.bashrc

# 检查是否成功 (当前稳定版通常为 v0.7.0+)
hermes --version

# 启动qwen-coder
```bash
python -m vllm.entrypoints.openai.api_server \
    --model  /home/scott/models/Qwen2.5-Coder-14B-Instruct-GPTQ-Int4 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.7 \
    --trust-remote-code \
    --quantization gptq \
    --dtype float16
```


# 配置orgin-x


### 进入orginx l4t-pytorch 环境
```bash
  APP_DIR="/stock_v3"
  sudo ctr run --rm --privileged --tty \
  --net-host \
  --device /dev/nvmap \
  --device /dev/nvgpu/igpu0 \
  --device /dev/nvhost-ctrl \
  --device /dev/nvhost-vic \
  --device /dev/nvhost-power-gpu \
  --mount type=bind,src=/usr/lib,dst=/usr/lib/host,options=rbind:ro \
  --mount type=bind,src=/usr/lib/aarch64-linux-gnu/tegra,dst=/usr/lib/aarch64-linux-gnu/tegra,options=rbind:ro \
  --mount type=bind,src=$APP_DIR,dst=/app,options=rbind:rw \
  --env LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/host \
  nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3 test \
  /bin/bash
```

### set source url
```bash
pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple
```


### 创建运行环境 for l4t
* 在容器内重新创建一个正确的虚拟环境（只做这一次）

```bash
cd /app
#1. Upgrade numpy first (this is the most important one)
pip install "numpy>=1.24.0" --upgrade 
# 2. Upgrade typing-extensions
pip install "typing-extensions>=4.6.0" --upgrade 
# 3. Re-install the requirements (or just pandas if you prefer)
pip install -r requirements.txt
# 
pip install -r ./models/Kronos_Source/requirements.txt
```

### 一键安装运行
```bash
cat > /stock_v3/l4t_pytorch_start.sh << 'EOF'
#!/bin/bash

APP_DIR="/stock_v3"
CONTAINER_NAME="kronos"

echo "=== 启动 Jetson Kronos 容器 ==="

sudo ctr run --rm --privileged --tty \
  --net-host \
  --device /dev/nvmap \
  --device /dev/nvgpu/igpu0 \
  --device /dev/nvhost-ctrl \
  --device /dev/nvhost-vic \
  --device /dev/nvhost-power-gpu \
  --mount type=bind,src=/usr/lib,dst=/usr/lib/host,options=rbind:ro \
  --mount type=bind,src=/usr/lib/aarch64-linux-gnu/tegra,dst=/usr/lib/aarch64-linux-gnu/tegra,options=rbind:ro \
  --mount type=bind,src=$APP_DIR,dst=/app,options=rbind:rw \
  --env LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/host \
  nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3 "$CONTAINER_NAME" \
  /bin/bash -c '
    cd /app

    echo "=== 设置 PyPI 腾讯源 ==="
    pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple
    pip config set global.trusted-host mirrors.cloud.tencent.com
    # 升级关键基础包
    echo "=== 升级 numpy 和 typing-extensions ==="
    pip install "numpy>=1.24.0" "typing-extensions>=4.7.0" --upgrade

    # 安装依赖
    echo "=== 安装 requirements.txt ==="
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi

    if [ -f "./models/Kronos_Source/requirements.txt" ]; then
        echo "=== 安装 Kronos 依赖 ==="
        pip install -r ./models/Kronos_Source/requirements.txt
    fi

    echo "=== 环境准备完成 ==="
    echo "当前目录: $(pwd)"
    echo "Python: $(which python)"
    exec /bin/bash
  '
EOF
chmod +x /stock_v3/start_kronos.sh
python3 -m optimize.auto_tune
```

# cp 
rsync -av --exclude='models' ./stock-model/   root@192.168.10.22:/opt/hdm/stock-model/

