#!/usr/bin/env bash
# Restore data/ from the Kaggle competition.
#   ./scripts/fetch_data.sh
# Requires: kaggle CLI configured (~/.kaggle/kaggle.json or ./kaggle.json)
set -euo pipefail

COMP="radiology-reporting-harness"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/data"

for f in train.csv test.csv sample_submission.csv; do
  if [ -f "$ROOT/data/$f" ]; then
    echo "skip   $f (already present)"
  else
    echo "fetch  $f"
    kaggle competitions download -c "$COMP" -f "data/$f" -p "$ROOT/data" --unzip \
      || kaggle competitions download -c "$COMP" -f "$f" -p "$ROOT/data"
  fi
done

echo "data/ now contains:"
ls -la "$ROOT/data"
