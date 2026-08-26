#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Label a corpus of chess positions with lc0 depth-zero WDL, batched.
#
# label_zero_wdl.py does the work; this drives it over each Hive-partitioned
# dataset in a corpus directory with settings suited to a rented accelerator,
# and mirrors results as they complete so a killed session loses at most one
# dataset's tail. See DEPTH_ZERO_WDL.md.
#
# Batching is the whole point. Over UCI lc0 evaluates one position per round
# trip: 31 positions/s measured on a Tesla P100, against 74 for the backend at
# batch 1 and 740 at batch 55. --lc0-batch-size forces workers=1, because one
# backend saturates the accelerator far better than four batch-1 processes
# competing for it. BATCH defaults to 64 because that is where the P100
# measurement was taken; the optimum depends on the net and the card, and
# `lc0 backendbench --weights=NET` sweeps it in about a minute.
#
# Resumable. Completed partitions are skipped by their row count and metadata,
# and partial partitions resume from the last checkpoint chunk, so re-running
# after a session dies costs nothing already paid for.
#
# Usage: ./label_corpus_lc0.sh <corpus_dir> <output_dir>
set -euo pipefail

corpus=${1:?Usage: ./label_corpus_lc0.sh <corpus_dir> <output_dir>}
output=${2:?Usage: ./label_corpus_lc0.sh <corpus_dir> <output_dir>}

HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-python3}
HF=${HF:-hf}
LC0=${LC0:?set LC0 to the patched lc0 binary}
LC0_WEIGHTS=${LC0_WEIGHTS:?set LC0_WEIGHTS to the network}
LC0_PYTHON_PATH=${LC0_PYTHON_PATH:-$(dirname "$LC0")}
BACKEND=${BACKEND:-cuda}
BATCH=${BATCH:-64}
CHECKPOINT_ROWS=${CHECKPOINT_ROWS:-25000}
BUCKET=${BUCKET:-}

# The datasets to label. Each is a Hive-partitioned directory of positions.
DATASETS=${DATASETS:-"lc0-selfplay-2m lichess-elite-300k"}

# Check the bindings before touching the corpus: without them the run would
# fail only after the first chunk was already evaluated. Report why the import
# failed rather than assuming, since a missing build flag is only one cause --
# meson's find_installation('python3') may also have built the module against a
# different interpreter than the one running this script, and an extension
# module is only importable by the Python whose ABI tag it carries.
if ! bindings_error=$("$PYTHON" -c "
import sys
sys.path.insert(0, '$LC0_PYTHON_PATH')
import backends
" 2>&1); then
  {
    echo "Cannot import lc0's backends module from $LC0_PYTHON_PATH:"
    echo "$bindings_error" | sed 's/^/  /'
    echo
    echo "This interpreter is $("$PYTHON" -c 'import sys;print(sys.executable)')"
    echo "and loads extension modules tagged $("$PYTHON" -c '
import sysconfig
print(sysconfig.get_config_var("EXT_SUFFIX"))')"
    echo
    found=$(find "$(dirname "$LC0")/.." -name 'backends*.so' -o -name 'backends*.pyd' 2>/dev/null | head -5)
    if [ -n "$found" ]; then
      echo "Modules that were built:"
      echo "$found" | sed 's/^/  /'
      echo
      echo "If the tag above does not match one of these, the bindings were built"
      echo "against a different Python. Rebuild pointing meson at this one:"
      echo "  PATH=\"$(dirname "$("$PYTHON" -c 'import sys;print(sys.executable)')"):\$PATH\" ./build.sh -Dpython_bindings=true"
      echo "Or set LC0_PYTHON_PATH to the directory holding a matching module."
    else
      echo "No backends module near the binary. LC0_PYTHON_PATH defaults to the"
      echo "directory holding \$LC0, so a binary copied out of its build tree"
      echo "leaves it pointing somewhere the module was never installed. Either:"
      echo
      echo "  LC0_PYTHON_PATH=/path/to/lc0/build/release  # if it was built"
      echo "  ./build.sh -Dpython_bindings=true           # if it was not"
      echo
      echo "The module is built to build/release/backends.<tag>.so; find it with"
      echo "  find /path/to/lc0 -name 'backends*.so'"
    fi
  } >&2
  exit 1
fi

for name in $DATASETS; do
  source_dir=$corpus/$name.parquet
  [ -d "$source_dir" ] || { echo "missing $source_dir" >&2; exit 1; }
  destination=$output/$name-lc0-wdl.parquet

  echo "=== $name -> $destination"
  "$PYTHON" "$HERE/label_zero_wdl.py" \
    --input "$source_dir" \
    --output "$destination" \
    --engine "$LC0" \
    --engine-kind lc0 \
    --weights "$LC0_WEIGHTS" \
    --backend "$BACKEND" \
    --lc0-batch-size "$BATCH" \
    --lc0-python-path "$LC0_PYTHON_PATH" \
    --checkpoint-rows "$CHECKPOINT_ROWS"

  # Mirror each dataset as it finishes rather than at the end, so a session
  # that dies later still leaves the completed ones durable. _work holds
  # partial chunks and is local scratch, never published.
  if [ -n "$BUCKET" ]; then
    echo "mirroring $name ..."
    "$HF" sync "$destination" "$BUCKET/$name-lc0-wdl.parquet" \
      --exclude "_work/**" \
      || echo "  mirror failed; results remain in $destination" >&2
  fi
done

echo "Done. Labeled datasets in $output"
