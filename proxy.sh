#!/bin/bash
# 建议保存为 ~/proxy.sh

# A电脑（代理主机）的IP和端口
PROXY_IP="192.168.10.2"
PROXY_PORT="7890"

function proxy_on() {
    export http_proxy="http://$PROXY_IP:$PROXY_PORT"
    export https_proxy="http://$PROXY_IP:$PROXY_PORT"
    export all_proxy="socks5://$PROXY_IP:$PROXY_PORT"
    # 排除本地和内网地址
    export no_proxy="localhost,127.0.0.1,192.168.10.0/24,192.168.1.0/24"
    
    echo -e "\033[32m[✔] 代理已开启: $PROXY_IP:$PROXY_PORT\033[0m"
    # 测试连接
    curl -I -s --connect-timeout 2 https://www.google.com | head -n 1
}

function proxy_off() {
    unset http_proxy
    unset https_proxy
    unset all_proxy
    echo -e "\033[31m[✘] 代理已关闭\033[0m"
}

# 检查输入参数
case "$1" in
    on)  proxy_on ;;
    off) proxy_off ;;
    *)   echo "使用方法: source proxy.sh [on|off]" ;;
esac
