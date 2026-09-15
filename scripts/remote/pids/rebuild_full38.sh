#!/usr/bin/env bash
# rebuild_full38.sh — 归拢 38 bin JSON 到统一目录 + TRUNCATE 全量重建 cadets_e5 + backfill
set -e
SRC1=/TeRed+RATE/dataset/darpae5/cadets_json
SRC2=/TeRed+RATE/dataset/darpae5/cadets_json2
SRC3=/TeRed+RATE/dataset/darpae5/cadets_json3
ALL=/TeRed+RATE/dataset/darpae5/cadets_json_all

echo "[full38] 归拢 JSON ..."
mkdir -p "$ALL"
N1=$(ls "$SRC1"/*.json 2>/dev/null | wc -l)
N2=$(ls "$SRC2"/*.json 2>/dev/null | wc -l)
N3=$(ls "$SRC3"/*.json 2>/dev/null | wc -l)
echo "[full38] json 分布: J1=$N1 J2=$N2 J3=$N3"
# 去重拷贝（不同目录不重叠，直接 mv；如目标已存在则跳过）
for S in "$SRC1" "$SRC2" "$SRC3"; do
  for F in "$S"/*.json; do
    [ -e "$F" ] || continue
    B=$(basename "$F")
    [ -e "$ALL/$B" ] || mv "$F" "$ALL/"
  done
done
N=$(ls "$ALL"/*.json | wc -l)
echo "[full38] 统一目录 json 总数: $N"
[ "$N" -lt 30 ] && { echo "[full38] 数量异常，中止"; exit 1; }

echo "[full38] TRUNCATE ..."
docker exec postgres psql -U postgres -d cadets_e5 -c "TRUNCATE event_table, file_node_table, netflow_node_table, subject_node_table;"

echo "[full38] build_db 全量重灌 ..."
~/pids-lite/bin/python ~/pidsmaker/scripts/build_db_cadets.py \
  --raw-dir "$ALL" \
  --db cadets_e5 --host localhost --port 5432 --user postgres --password postgres \
  > ~/build_full38.log 2>&1
echo "[full38] build_db rc=$? (log: ~/build_full38.log)"

echo "[full38] backfill labels ..."
docker exec postgres psql -U postgres -d cadets_e5 <<'SQL'
UPDATE subject_node_table SET path = 'proc:' || node_uuid WHERE path IS NULL OR path = '';
UPDATE file_node_table   SET path = 'file:' || node_uuid WHERE path IS NULL OR path = '';
UPDATE netflow_node_table SET dst_addr = 'nf:' || node_uuid, dst_port = '0',
       src_addr = '', src_port = '0' WHERE dst_addr IS NULL OR dst_addr = '';
SQL

echo "[full38] counts ..."
docker exec postgres psql -U postgres -d cadets_e5 -c "SELECT (SELECT count(*) FROM event_table) AS ev, (SELECT count(*) FROM subject_node_table) AS subj, (SELECT count(*) FROM file_node_table) AS file, (SELECT count(*) FROM netflow_node_table) AS nf;"
echo "[full38] DONE $(date '+%H:%M:%S')"
