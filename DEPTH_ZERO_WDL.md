# Depth-Zero WDL Evaluation with Metal

This lc0 checkout supports immediate static win/draw/loss evaluation through
the UCI command:

```text
go depth 0
```

The command performs one neural-network evaluation of the current position,
does not search successor positions, and reports the result as UCI WDL values.

## Output format

Example:

```text
info depth 0 seldepth 1 time 16 nodes 1 score cp 31 wdl 259 623 118 tbhits 0 pv b1d2
bestmove b1d2
```

The three WDL integers are ordered as:

```text
win draw loss
```

They sum to 1000. Divide each value by 1000 to obtain probabilities. In the
example above, the static evaluation is:

```text
P(win)  = 0.259
P(draw) = 0.623
P(loss) = 0.118
```

The probabilities are from the perspective of the side to move.

`UCI_ShowWDL` does not need to be enabled for `go depth 0`. WDL remains
controlled by that option for ordinary searched output.

## Implementation

Two narrowly scoped changes enable this behavior:

1. `src/search/classic/wrapper.cc` recognizes an explicit `go depth 0`
   request and preserves depth zero in the response.
2. `src/chess/uciloop.cc` prints genuine depth zero instead of rewriting it to
   depth one, and always exposes WDL for a depth-zero response.

The existing classic-search depth stopper already stops after the root neural
evaluation. No new search algorithm or evaluation path is introduced.

## Build for Apple Metal

Requirements:

- macOS 12.6 or newer
- Apple Command Line Tools or Xcode
- Python 3
- Meson
- Ninja

Create repository-local build tooling if Meson and Ninja are unavailable:

```bash
python3 -m venv .venv
.venv/bin/pip install meson ninja
```

Build lc0 with Metal required and selected as the default backend:

```bash
PATH="$PWD/.venv/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
  ./build.sh \
  -Dgtest=false \
  -Dmetal=enabled \
  -Ddefault_backend=metal
```

The executable is produced at:

```text
build/release/lc0
```

Using `-Dmetal=enabled` is important: configuration fails if the required
Apple Metal frameworks cannot be found instead of silently producing a build
without Metal.

## Install a network

Place a compatible lc0 network in `build/release`. This checkout was verified
with the official 365 MB BT4-it332 network:

```bash
curl --fail --location \
  --output build/release/BT4-it332.pb.gz \
  https://storage.lczero.org/files/networks-contrib/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz
```

Recorded SHA-256:

```text
e6ada9d6c4a769bfab3aa0848d82caeb809aa45f83e6c605fc58a31d21bdd618
```

## Run on the Apple GPU

Start lc0 explicitly with Metal, batch size one, and the desired network:

```bash
cd build/release
./lc0 \
  --weights=BT4-it332.pb.gz \
  --backend=metal \
  --minibatch-size=1
```

For latency measurements or dataset generation where every position must be
evaluated independently, disable the neural cache:

```bash
./lc0 \
  --weights=BT4-it332.pb.gz \
  --backend=metal \
  --minibatch-size=1 \
  --nncache=0
```

With logging directed to stderr, successful accelerator initialization is
identified by:

```text
Initialized metal backend on device Apple M4
```

Metal uses the Apple GPU through Metal Performance Shaders. It does not use
the Apple Neural Engine.

## UCI session

Send a position followed by `go depth 0`:

```text
uci
isready
position startpos
go depth 0
```

For a FEN position:

```text
position fen r1bq1rk1/ppp2ppp/2np1n2/8/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w - - 0 8
go depth 0
```

Wait for `bestmove` before submitting the next position. The WDL label is read
from the preceding `info depth 0` line.

For a long-running labeling process, keep one lc0 process alive. Reusing the
process avoids network loading and Metal graph-compilation overhead.

## Label the published Hugging Face dataset

Install the labeler's Python dependencies:

```bash
python3 -m pip install python-chess pyarrow huggingface-hub
```

Then stream the published Parquet snapshot through the persistent lc0 process:

```bash
python3 scripts/label_zero_wdl.py \
  --hf-dataset Pawitt/zero-evaluator \
  --output zero-evaluator-wdl.parquet \
  --engine-kind lc0 \
  --engine build/release/lc0 \
  --weights build/release/BT4-it332.pb.gz \
  --backend metal
```

`--lc0` remains an alias for `--engine`. A patched Stockfish binary can label
the same dataset without accelerator arguments:

```bash
python3 scripts/label_zero_wdl.py \
  --hf-dataset Pawitt/zero-evaluator \
  --output zero-evaluator-stockfish-wdl.parquet \
  --engine-kind stockfish \
  --engine ../stockfish/src/stockfish
```

For another UCI-compatible depth-zero WDL producer, record its identity
manually when needed:

```bash
python3 scripts/label_zero_wdl.py \
  --input data/ccrl-high-variance-1p5m.parquet \
  --output data/manual-zero-wdl.parquet \
  --engine-kind uci \
  --engine /path/to/engine \
  --engine-name MyTeacher \
  --engine-version experiment-7 \
  --engine-weights network-42
```

Every output row contains `wdl_win`, `wdl_draw`, `wdl_loss`, plus
`wdl_engine_name`, `wdl_engine_version`, and `wdl_engine_weights`. Parquet
metadata and `_manifest.json` additionally record the binary SHA-256, backend,
and weights checksum where applicable.

`--hf-revision` optionally pins a branch, tag, or commit. The script downloads
only the Parquet data through the Hugging Face cache and records the resolved
Hub commit in the output metadata. `--hf-dataset` and local `--input` are
mutually exclusive. lc0 accepts only `metal` and `cuda`; Stockfish and generic
UCI engines do not accept `--backend`. `--uci-option NAME=VALUE` can be
repeated for additional engine settings.

CUDA defaults to four persistent lc0 worker processes so independent root
evaluations can overlap and keep throughput-oriented GPUs occupied. Metal
defaults to one. Override either default with `--workers`; on CUDA, try 2, 4,
and 8 and retain the fastest setting that fits GPU memory. Worker-count changes
do not invalidate completed checkpoints.

Progress is atomically checkpointed every 1,000 positions under the output
dataset's `_work/` directory. Restart the identical command to resume; at most
the currently active 1,000-position block is repeated. Use
`--checkpoint-rows` to trade checkpoint frequency against small-file overhead.
Final partition files are still assembled with 65,536-row groups.

## Verified latency

Test system:

- MacBook Air
- Apple M4, 10 CPU cores
- 16 GB unified memory
- BT4-it332, 365 MB compressed network
- Metal backend
- minibatch size 1
- neural cache disabled

Results across 20 different legal positions:

| Measurement | Latency |
| --- | ---: |
| Backend/network initialization | 423.54 ms |
| First inference and Metal graph compilation | 681.95 ms |
| Warm minimum | 14.57 ms |
| Warm median | 15.86 ms |
| Warm mean | 16.17 ms |
| Warm p90 | 17.27 ms |
| Warm maximum | 23.15 ms |

The warm median corresponds to approximately 62 sequential depth-zero WDL
evaluations per second. Batched throughput is a separate measurement and may
be higher, but batch size one reflects request latency.

## Validation checklist

- The build configuration reports the Metal frameworks as found.
- `./lc0 --help` reports `Backend DEFAULT: metal` and includes `metal` in the
  backend choices.
- Runtime logs contain `Initialized metal backend on device Apple M4` (or the
  actual Apple GPU name).
- `go depth 0` returns exactly one evaluated node.
- The `info` response says `depth 0` and contains `wdl W D L`.
- The WDL integers sum to 1000.

If runtime reports `There was an error initializing the GPU device`, ensure
the process has permission to access Metal. Sandboxed execution environments
may block GPU device creation even when the binary was compiled correctly.

## Batched evaluation

Over UCI lc0 evaluates one position per round trip. On a Tesla P100 that
measured 31 positions per second, against 74 for the backend at batch 1 and 740
at batch 55 in `backendbench`. Most of the batch-1 cost is position setup and
search-tree initialisation, and the rest is the accelerator idling on a batch
of one — very little of it is the network.

`--lc0-batch-size` routes evaluation through lc0's Python bindings instead of
UCI. `GameState.as_input` calls this engine's own `EncodePositionForNN` with
`FillEmptyHistory::FEN_ONLY`, and `Backend.evaluate` accepts as many inputs as
it is given, so batching needs no exported graph and reimplements nothing about
the 112-plane encoding.

Build with the bindings enabled:

```bash
./build.sh -Dpython_bindings=true
```

Then label with a batch:

```bash
python3 scripts/label_zero_wdl.py \
  --input positions.parquet \
  --output labeled.parquet \
  --engine-kind lc0 \
  --engine build/release/lc0 \
  --weights /path/to/network.pb.gz \
  --backend cuda \
  --lc0-batch-size 64 \
  --lc0-python-path build/release
```

`--lc0-batch-size 0` keeps the original UCI path, which remains the default.

Batching sets `workers=1`: one backend saturates the accelerator far better
than several batch-1 processes competing for it. Engine identity still comes
from a short UCI handshake, because the bindings expose no version string and
provenance must keep naming the exact build.

Labels from the batched path were validated against the UCI path over 276
positions in eight strata: identical engine identity, every row summing to
1000, and a maximum deviation of 3 in 1000. The residual lives in lc0's UCI
reporting layer rather than the network — the bindings and an ONNX export of
the same net agree to five decimals.

### Labeling a whole corpus

`scripts/label_corpus_lc0.sh` drives the labeler over each Hive-partitioned
dataset in a directory:

```bash
LC0=build/release/lc0 \
LC0_WEIGHTS=/path/to/network.pb.gz \
BACKEND=cuda \
DATASETS="positions-a positions-b" \
BUCKET=hf://buckets/user/bucket \
./scripts/label_corpus_lc0.sh ./corpus ./labeled
```

It checks the bindings import before touching the corpus, since a build without
`-Dpython_bindings=true` would otherwise fail only once evaluation was under
way. `BATCH` defaults to 64, where the P100 figure was measured; the optimum
depends on the net and the card, and `lc0 backendbench --weights=NET` sweeps it
in about a minute.

Completed datasets are skipped on their row count and metadata, and partial
ones resume from the last checkpoint chunk under `_work/` inside the output, so
re-running after a session dies re-evaluates nothing already paid for. `_work/`
is scratch: readers ignore underscore-prefixed directories and the mirror step
excludes it, so partial chunks are never published.
