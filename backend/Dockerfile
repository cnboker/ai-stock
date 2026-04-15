# 使用南京大学或阿里云的镜像源（示例为公共镜像库路径）
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/library/python:3.10-slim

WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip
# 1. 先拷贝 requirements.txt (利用 Docker 缓存层)
COPY requirements.txt .

# 2. 使用清华源加速安装
RUN pip install --no-cache-dir -i https://mirrors.cloud.tencent.com/pypi/simple/ -r requirements.txt

# 3. 再拷贝代码 (这样修改代码不需要重新下载包)
COPY . .

EXPOSE 8000