#!/usr/bin/env bash
# ch02 验收演示:三条验收标准
#
# 本机(Win + git-bash)注意:
#   - 无 exec 位(Windows core.filemode=false),用 `bash scripts/demo_agent.sh` 运行,别用 ./。
#   - `python3` 是应用商店占位 stub(坏),用 `python`;`-X utf8` 让 stdin/stdout 走 UTF-8
#     (否则读响应时按 GBK 解码会吞引号、解析失败),`--no-ensure-ascii` 让中文/emoji 可读。
#   - 内联中文 body 会被 shell 串码(FastAPI 回 400),故 body 走 UTF-8 临时文件 + --data-binary。
set -euo pipefail
BASE=http://localhost:8000
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# 三个请求体写成 UTF-8 文件(heredoc 逐字节落盘,不经 shell 参数串码)
cat > "$TMP/1.json" <<'EOF'
{"user_id":"demo","message":"订单 1001 的物流到哪了"}
EOF
cat > "$TMP/2.json" <<'EOF'
{"user_id":"demo","message":"退货政策是什么"}
EOF
cat > "$TMP/3.json" <<'EOF'
{"user_id":"demo","message":"邮费是多少"}
EOF

post() { curl -s "$BASE/api/agent" -H 'Content-Type: application/json' --data-binary @"$1" | python -X utf8 -m json.tool --no-ensure-ascii; }

echo "== 验收1:订单物流(应选中 query_logistics)=="
post "$TMP/1.json"
echo "== 验收2:退货政策(query_faq 命中)=="
post "$TMP/2.json"
echo "== 验收3:换说法(query_faq 漏召回,预期结果)=="
post "$TMP/3.json"
