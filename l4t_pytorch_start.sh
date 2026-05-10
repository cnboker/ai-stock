#!/bin/bash

APP_DIR="/stock_v3"
CONTAINER_NAME="kronos"

echo "=== 启动 Jetson Kronos 容器 ==="
docker run --rm -it --runtime nvidia --network host \
  -v $APP_DIR:/app \
  dustynv/l4t-pytorch:2.2-r35.4.1 bash
  -c '
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
