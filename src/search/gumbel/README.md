# Gumbel AlphaZero search

Gumbel is the default search. Run `lc0` with any existing lc0 weights/backend.
Use `lc0 classic` to select the classic search explicitly. No training or network
format change is required. For example:

```sh
./build/release/lc0 --weights=/path/to/network.pb.gz --policy-softmax-temp=1
```

Use `go nodes 800` for an 800-simulation search. Without a node or time limit,
the finite-search budget is `GumbelSimulations` (1024 by default). `movetime` and chess clocks impose a
hard time limit between synchronous evaluations. `go depth` caps traversal
length; it does not request an exhaustive search to that depth. `searchmoves`
restricts the root. `stop`, abort, ponder, and infinite commands use the normal
asynchronous lc0 search interface. Infinite/ponder and timed searches without
an explicit node budget continue in growing sequential-halving rounds, retaining
the tree, evaluations, accumulated visits, and fixed root noise. Each round
reconsiders root candidates. This is an anytime extension of the finite-budget
algorithm. Infinite/ponder searches with an explicit node budget finish that
budget and wait for stop/ponderhit before emitting bestmove.

## Policy compatibility

The existing `Backend` converts raw network logits to legal-move probabilities,
using lc0's existing `MoveToNNIndex` mapping and board transform. This adapter
passes that legal-move order unchanged to the C search. `log(p)` supplies logits
up to an additive constant, which cancels in Gumbel ranking and softmax. No
sorting of network output planes, new policy head, or retraining is needed.
Use `--policy-softmax-temp=1` to use the original logit scale; lc0's default
policy temperature otherwise still applies. Very small/zero probabilities are
floored at `FLT_MIN` to keep finite scores after backend underflow.

At the root, fixed per-search Gumbel noise and sequential halving allocate the
simulation budget. The final move maximizes Gumbel + log prior + transformed Q
among the most visited candidates. At interior nodes, deterministic selection
tracks `softmax(log prior + sigma(completed Q))`, rather than PUCT. Unvisited Qs
use the Appendix D mixed value; values are min/max normalized before applying
`(c_visit + max_child_visits) * c_scale`. `gaz_policy()` exposes the improved
policy in the same legal-action order, without Gumbel noise. It is deliberately
not a visit-count distribution and does not alter the network weights.

The algorithm follows the definitions in
[Policy improvement by planning with Gumbel](https://openreview.net/forum?id=bERaNdoegnO)
and the scheduling/completion conventions of
[DeepMind mctx](https://github.com/google-deepmind/mctx).

## Options

| UCI option | Default | Meaning |
| --- | --- | --- |
| GumbelSimulations | 1024 | Finite budget, or initial round for continuing analysis |
| GumbelConsideredMoves | 16 | Maximum initial root candidates |
| GumbelCVisit | 50 | Visit offset in Q transform |
| GumbelCScale | 0.1 | Q transform scale |
| GumbelNoiseScale | 1 | Gumbel noise scale; 0 is deterministic |
| GumbelSeed | 0 | Repeatable per-search noise seed |

The candidate count is capped by the legal-move count and nonzero simulation
budget. The seed is reset for each search for reproducibility. Change the seed
for independent samples across searches.

## Nibbler

The default engine supports Nibbler's forced options: `LogLiveStats`,
`MoveOverheadMs`, `MultiPV`, `ScoreType`, `SmartPruningFactor`,
`UCI_ShowWDL`, and `VerboseMoveStats`. The smart-pruning option is accepted for
compatibility; selection still uses Gumbel sequential halving. Live reports are
sent every 200 ms between evaluations, with root-move priors, visit counts,
Q/raw V, draw probabilities, WDL scores, NPS, and principal variations.
Verbose rows use Nibbler's lc0 format and finish with a `node` summary.
`U` is zero because this search has no PUCT bonus; `S` is the improved-policy
probability. The strongest move is the last verbose row, as Nibbler expects.
Draw probabilities are backed up with Q rather than estimated from Q alone.

## Linux CUDA build

Install a C/C++20 toolchain, Python 3, Meson, Ninja, zlib development headers,
and the NVIDIA CUDA Toolkit (including nvcc, cuBLAS, and CUDA runtime).
With CUDA installed at `/usr/local/cuda`, run from the repository root:

```sh
meson setup build/cuda --buildtype=release \
  -Ddefault_search=gumbel -Ddefault_backend=cuda-auto \
  -Dplain_cuda=true -Dcudnn=false -Dcutlass=false \
  -Dblas=false -Donnx=false -Dmetal=disabled \
  -Dcudnn_include=/usr/local/cuda/include \
  -Dcudnn_libdirs=/usr/local/cuda/lib64
meson compile -C build/cuda
meson test -C build/cuda GumbelSearch GumbelUci --print-errorlogs
./build/cuda/lc0 --weights=/path/to/network.pb.gz --policy-softmax-temp=1
```

This selects the CUDA backend automatically and Gumbel search by default;
cuDNN is not required for this configuration. Adjust the include/library paths
for another CUDA installation. To build for a different GPU than the build
machine, add `-Dnative_cuda=false -Dcc_cuda=XX`, replacing `XX` with that GPU's
compute capability without the decimal point. CUDA compilation and runtime
must be verified on Linux/NVIDIA hardware; the development build was tested
on macOS, including a Metal-enabled release build.

## Implementation and validation

`gumbel.c` is a standalone C11 tree/search implementation with callbacks for
position evaluation and push/pop. `wrapper.cc` bridges lc0's C++ board, neural
backend, options, and UCI lifecycle. Tree values alternate sign at every ply;
terminal positions use lc0's mate, stalemate, material, repetition and fifty-move
adjudication. Trees are rebuilt for each search. This initial implementation is
single-worker, one evaluation at a time, with a maximum traversal depth of 256;
it does not probe Syzygy or generate training artifacts. It makes no strength or
GPU throughput claim relative to lc0's batched classic search.

Meson builds the C core and registers the `gumbel` search alongside `classic`.
The standalone test also runs without gtest:

```sh
meson test -C build/release GumbelSearch GumbelUci --print-errorlogs
# Or directly, including sanitizers:
cc -std=c11 -Wall -Wextra -Werror -pedantic -fsanitize=address,undefined \
  -Isrc src/search/gumbel/gumbel.c src/search/gumbel/gumbel_test.c \
  -lm -o /tmp/gumbel_test
/tmp/gumbel_test
```
