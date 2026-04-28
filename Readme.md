# Kronos (ai-stock v3) 项目手册

本项目是一个基于 **Hermes Agent** 与 **Qwen2.5-Coder** 驱动的 A 股量化调优系统，运行于 Jetson L4T 容器环境。

---

## 1. 项目运行逻辑

系统根据市场行情识别环境（Regime），通过斜率（Slope）决定操作方向和力度。

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
```

**逻辑说明：**
* **Regime 判断**：`neutral` 倾向观望；`bull/bear` 根据趋势和阈值决定多空。
* **Slope + Gate**：`slope` 表示趋势方向和强度，`gate` 是调节系数。
* **门槛机制**：仅当 `slope * gate` 超过内部阈值才会生成动作。
* **Action & Strength**：满足条件时，`strength` 会随趋势强度动态变化。

---

## 2. 回测周期建议 (Remark)

基于 1000 条 K 线数据可覆盖的时长分析：

| 周期 (Timeframe) | 每天 K 线数 | 1000 条数据覆盖时长 | 是否满足 6 个月？ |
| :--- | :--- | :--- | :--- |
| **日线 (Daily)** | 1 条 | ~1000 交易日 (约 4 年) | ✅ 是 (过度覆盖) |
| **1 小时 (1H)** | 4 条 | ~250 交易日 (约 12 个月) | ✅ 是 (完美契合) |
| **30 分钟 (30M)** | 8 条 | ~125 交易日 (约 6 个月) | ✅ 是 (刚好满足) |
| **15 分钟 (15M)** | 16 条 | ~62.5 交易日 (约 3 个月) | ❌ 否 |
| **5 分钟 (5M)** | 48 条 | ~21 交易日 (约 1 个月) | ❌ 否 |

---

## 3. 环境部署

### 3.1 安装 Hermes Agent
```bash
curl -fsSL [https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh](https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh) | bash
source ~/.bashrc
hermes --version
```

### 3.2 启动 Qwen-Coder 推理服务
```bash
python -m vllm.entrypoints.openai.api_server \\
    --model /home/scott/models/Qwen2.5-Coder-14B-Instruct-GPTQ-Int4 \\
    --max-model-len 4096 \\
    --gpu-memory-utilization 0.7 \\
    --trust-remote-code \\
    --quantization gptq \\
    --dtype float16
```

### 3.3 容器环境配置 (L4T PyTorch)
针对 Jetson 平台，使用 `ctr` 运行 `l4t-pytorch` 容器并挂载项目目录。

**一键启动与初始化脚本 (`start_kronos.sh`):**
```bash
#!/bin/bash

APP_DIR="/stock_v3"
CONTAINER_NAME="kronos"

echo "=== 启动 Jetson Kronos 容器 ==="

sudo ctr run --rm --privileged --tty \\
  --net-host \\
  --device /dev/nvmap \\
  --device /dev/nvgpu/igpu0 \\
  --device /dev/nvhost-ctrl \\
  --device /dev/nvhost-vic \\
  --device /dev/nvhost-power-gpu \\
  --mount type=bind,src=/usr/lib,dst=/usr/lib/host,options=rbind:ro \\
  --mount type=bind,src=/usr/lib/aarch64-linux-gnu/tegra,dst=/usr/lib/aarch64-linux-gnu/tegra,options=rbind:ro \\
  --mount type=bind,src=$APP_DIR,dst=/app,options=rbind:rw \\
  --env LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/host \\
  nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3 "$CONTAINER_NAME" \\
  /bin/bash -c '
    cd /app
    echo "=== 设置 PyPI 腾讯源 ==="
    pip config set global.index-url [https://mirrors.cloud.tencent.com/pypi/simple](https://mirrors.cloud.tencent.com/pypi/simple)
    pip config set global.trusted-host mirrors.cloud.tencent.com
    
    echo "=== 升级关键基础包 ==="
    pip install "numpy>=1.24.0" "typing-extensions>=4.7.0" --upgrade

    echo "=== 安装依赖库 ==="
    [ -f "requirements.txt" ] && pip install -r requirements.txt
    [ -f "./models/Kronos_Source/requirements.txt" ] && pip install -r ./models/Kronos_Source/requirements.txt

    echo "=== 环境准备完成 ==="
    exec /bin/bash
  '
```

---

## ⚙️ 4. 常用运维命令

### 4.1 数据同步 (Sync)
```bash
# 从开发机同步模型数据到 B 电脑 (192.168.10.22)
rsync -av --exclude='models' ./stock-model/ root@192.168.10.22:/opt/hdm/stock-model/
```

### 4.2 容器与任务清理 (Force Clean)
当容器死锁或无法启动时，清理残留任务和进程：
```bash
# 强杀任务和容器
sudo ctr task kill -f kronos 2>/dev/null
sudo ctr container rm kronos 2>/dev/null

# 清理残留的 shim 进程 (解决存储或 FD 占用关键)
sudo ps -ef | grep containerd-shim | awk "{print \\$2}" | xargs -r sudo kill -9
```

### 4.3 GPU 环境检查
```bash
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"
```



