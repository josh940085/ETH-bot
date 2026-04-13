#!/bin/bash

while true
do
  echo "🔄 更新程式..."
  git pull origin main

  echo "🚀 執行策略..."
  python3 eth.py

  echo "⏱ 等待60秒..."
  sleep 60
done
