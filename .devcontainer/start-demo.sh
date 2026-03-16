#!/usr/bin/env bash
set -euo pipefail
if pgrep -f "demo_visual --bin demo_visual" >/dev/null 2>&1; then
  exit 0
fi
nohup cargo run --release -p demo_visual --bin demo_visual >/tmp/rustorch-demo.log 2>&1 &
sleep 3
echo "RusTorch demo is running at http://127.0.0.1:3003/"
