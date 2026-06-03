#!/usr/bin/env bash
# Extract docker images from a verl parquet and pre-pull (parallel-N) any
# that aren't already on disk. Eliminates Docker Hub rate-limit NO_CONTAINER
# failures at training time.
#
# Usage: prepull_parquet_images.sh <parquet> [parallel]
set -e
PARQ=${1:?need parquet path}
PAR=${2:-4}   # parallel pulls (Docker Hub rate-limit safe at 4)

# 1) extract image list from parquet → tmp file
TMP=$(mktemp)
/home/temp/venv/bin/python - "$PARQ" > "$TMP" <<'PY'
import pandas as pd, json, sys
df = pd.read_parquet(sys.argv[1])
seen = set()
for _, r in df.iterrows():
    ei = r['extra_info']
    if isinstance(ei, str):
        try: ei = json.loads(ei)
        except: ei = {}
    img = ei.get('docker_image') or ei.get('image_name') or ''
    if img and ':' not in img.split('/')[-1]:
        img = img + ':latest'
    if img:
        seen.add(img)
for img in sorted(seen):
    print(img)
PY

TOTAL=$(wc -l < "$TMP")
echo "extracted $TOTAL unique images -> $TMP"

# 2) filter out images already on disk
NEED=$(mktemp)
while IFS= read -r img; do
  if ! sg docker -c "docker image inspect '$img'" >/dev/null 2>&1; then
    echo "$img" >> "$NEED"
  fi
done < "$TMP"
NTODO=$(wc -l < "$NEED" 2>/dev/null || echo 0)
NHAVE=$((TOTAL - NTODO))
echo "already pulled: $NHAVE; need to pull: $NTODO"

if [ "$NTODO" = 0 ]; then
  echo "all images present — done"
  rm -f "$TMP" "$NEED"
  exit 0
fi

# 3) parallel pull with throttle
echo "pulling $NTODO images, parallelism=$PAR ..."
START=$(date +%s)
cat "$NEED" | xargs -I{} -P "$PAR" bash -c '
  IMG="$1"
  if sg docker -c "docker pull \"$IMG\"" >/dev/null 2>&1; then
    echo "  OK $IMG"
  else
    echo "  FAIL $IMG"
  fi
' _ {}
DUR=$(( ($(date +%s) - START) / 60 ))
echo "done in ${DUR} min"
rm -f "$TMP" "$NEED"
