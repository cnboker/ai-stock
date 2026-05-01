#!/bin/bash

APP_DIR="/stock_v3"
CONTAINER_NAME="kronos"

echo "=== 启动 Jetson Kronos 容器 ==="
systemctl start containerd
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
