#!/usr/bin/env bash
# 发布 demo 到公网:http://47.236.93.96/(sinoark-singapore, nginx, /var/www/forkscope)
# 改完本目录的 html 后跑一下这个脚本即可。
set -euo pipefail
cd "$(dirname "$0")"
scp ./*.html sinoark-singapore:/var/www/forkscope/
echo "deployed -> http://47.236.93.96/"
