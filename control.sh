#!/bin/bash

# 设置应用名称、虚拟环境路径和日志文件路径
APP_NAME="main.py"
VENV_DIR="venv"    # 你的虚拟环境目录路径
LOG_FILE="run.log"
PID_FILE="main.pid"
PORT=20216

# 根据端口号杀掉占用该端口的所有进程
kill_by_port() {
    pids=$(lsof -t -i:$PORT)  # 查找占用端口的进程ID，可能会返回多个PID
    if [ ! -z "$pids" ]; then
      echo "找到占用端口 $PORT 的进程，PID: $pids"
      for pid in $pids; do
        kill -9 $pid
        echo "进程 $pid 已终止。"
      done
    else
      echo "没有进程占用端口 $PORT。"
    fi
}

# 启动方法
start() {
    # 启动时先检查是否有进程占用端口
    kill_by_port  # 先杀掉占用端口的进程
    
    echo "启动应用 $APP_NAME ..."
    source $VENV_DIR/bin/activate  # 激活虚拟环境
    nohup python $APP_NAME >> $LOG_FILE 2>&1 &
    echo $! > $PID_FILE  # 保存PID
    echo "${APP_NAME} 启动成功，PID: $(cat $PID_FILE)"
}

# 停止方法
stop() {
    # 查找并终止占用端口的进程
    kill_by_port
    rm -f $PID_FILE
    echo "${APP_NAME} 已停止。"
}

# 重启方法
restart() {
  stop
  start
}

# 检查状态
status() {
  pids=$(lsof -t -i:$PORT)
  if [ ! -z "$pids" ]; then
    echo "${APP_NAME} 正在运行，PID: $pids"
  else
    echo "${APP_NAME} 未在运行。"
  fi
}

# 处理命令行参数
case "$1" in
  start)
    start
    ;;
  stop)
    stop
    ;;
  restart)
    restart
    ;;
  status)
    status
    ;;
  *)
    echo "用法: $0 {start|stop|restart|status}"
    exit 1
    ;;
esac

