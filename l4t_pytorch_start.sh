#!/bin/bash
APP_DIR="/stock_v3"
CONTAINER_NAME="kronos"

echo "=== Jetson Kronos 持久化容器 ==="

# 如果容器不存在则创建
if ! docker ps -a | grep -q "$CONTAINER_NAME"; then
    echo "→ 首次创建容器..."
    docker run -d \
      --name "$CONTAINER_NAME" \
      --runtime nvidia \
      --network host \
      -v "$APP_DIR:/app" \
      -w /app \
      -e PIP_INDEX_URL="" \
      -e PIP_EXTRA_INDEX_URL="" \
      dustynv/l4t-pytorch:2.2-r35.4.1 \
      tail -f /dev/null

    sleep 3

    echo "→ 首次初始化 PyPI 和依赖..."
    docker exec "$CONTAINER_NAME" bash -c '
        unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL
        rm -f /usr/pip.conf /root/.config/pip/pip.conf 2>/dev/null || true

        pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple
        pip config set global.extra-index-url https://pypi.org/simple
        pip config set global.trusted-host "mirrors.cloud.tencent.com pypi.org files.pythonhosted.org"

        pip install --no-cache-dir --upgrade "numpy>=1.24.0" "typing-extensions>=4.7.0"

        [ -f requirements.txt ] && pip install --no-cache-dir -r requirements.txt
        [ -f models/Kronos_Source/requirements.txt ] && pip install --no-cache-dir -r models/Kronos_Source/requirements.txt

        echo "=== 首次初始化完成！==="
    ' || echo "初始化有警告，可继续使用"
fi

# 启动容器
docker start "$CONTAINER_NAME" >/dev/null 2>&1

echo "→ 进入容器..."

PID=$(docker inspect --format "{{.State.Pid}}" "$CONTAINER_NAME")

# 使用 Jetson 兼容方式进入
nsenter -t $PID -m -u -i -n -p chroot /new_root /bin/bash
