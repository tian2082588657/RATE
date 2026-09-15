#!/bin/bash
# v4 全量后台 monitor，每 15 分钟输出进度到 /tmp/v4_monitor.log
TARGET_PID=54605
LOG=/tmp/v4_monitor.log
while kill -0 $TARGET_PID 2>/dev/null; do
    echo "==== $(date '+%H:%M:%S') ====" >> $LOG
    echo "v4 全量主任务 PID=$TARGET_PID 在跑" >> $LOG
    ps -p $TARGET_PID -o pid,etime,pcpu,cmd 2>&1 | head -2 >> $LOG
    echo "" >> $LOG
    echo "--- 子任务日志（最后 8 行） ---" >> $LOG
    LATEST=$(ls -t /TeRed+RATE/code/logs/e5_f1_v4_*.log 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        echo "FILE: $LATEST" >> $LOG
        tail -8 "$LATEST" >> $LOG
    fi
    echo "" >> $LOG
    echo "--- 已生成 CSV ---" >> $LOG
    ls -la /TeRed+RATE/code/results/e5_f1_v4_*.csv 2>/dev/null >> $LOG
    echo "" >> $LOG
    sleep 900  # 15 分钟
done
echo "==== $(date '+%H:%M:%S') 进程已退出 ====" >> $LOG
echo "--- 全部子任务日志 + CSV 终态 ---" >> $LOG
ls -la /TeRed+RATE/code/results/e5_f1_v4_*.csv 2>/dev/null >> $LOG