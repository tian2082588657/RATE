#!/bin/bash
# launcher: 服务器端 nohup 后台启动 v4 全量(share_k=-1)，避免交互式 ssh & 语法问题
cd /TeRed+RATE/code
nohup bash /tmp/run_v4_full_k-1.sh > /tmp/run_v4_full_k-1_outer.log 2>&1 &
echo "LAUNCHED_PID=$!"
sleep 5
echo "--- ps ---"
ps aux | grep -E "e5_f1_eval|run_v4_full_k-1" | grep -v grep | head
echo "--- outer log head ---"
head -10 /tmp/run_v4_full_k-1_outer.log 2>/dev/null
