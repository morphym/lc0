#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later

"""Label a partitioned Parquet chess dataset with UCI depth-zero WDL."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
from typing import Sequence
import threading
import time
from pathlib import Path

import chess
import pyarrow as pa
import pyarrow.parquet as pq


WDL_RE = re.compile(r"\bdepth 0\b.*\bwdl (\d+) (\d+) (\d+)\b")
ENGINE_MARKER = "zero-wdl"
WDL_COLUMNS = ("wdl_win", "wdl_draw", "wdl_loss")
# Backends differ in the last permille; anything larger is not rounding.
VERIFY_TOLERANCE = 5
ENGINE_COLUMNS = ("wdl_engine_name", "wdl_engine_version", "wdl_engine_weights")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_input(args: argparse.Namespace) -> tuple[Path, dict[str, str]]:
    if args.input:
        source = Path(args.input).resolve()
        return source, {"type": "local", "path": str(source)}
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "--hf-dataset requires huggingface_hub; install huggingface_hub"
        ) from exc
    snapshot = Path(
        snapshot_download(
            repo_id=args.hf_dataset,
            repo_type="dataset",
            revision=args.hf_revision,
            allow_patterns=("data/**/*.parquet", "data/_manifest.json"),
        )
    )
    source = snapshot / "data"
    return source, {
        "type": "huggingface",
        "dataset": args.hf_dataset,
        "requested_revision": args.hf_revision,
        "resolved_revision": snapshot.name,
    }


class Progress:
    def __init__(self, total: int) -> None:
        self.done = 0
        self.run_done = 0
        self.total = total
        self.started = time.monotonic()
        self.lock = threading.Lock()

    def advance(self, count: int = 1, report: bool = True) -> None:
        with self.lock:
            previous_bucket = self.done // 100
            self.done += count
            if report:
                self.run_done += count
            if not report or self.done // 100 == previous_bucket:
                return
            elapsed = time.monotonic() - self.started
            rate = self.run_done / elapsed
            remaining = (self.total - self.done) / rate
            print(
                f"{self.done:,}/{self.total:,} "
                f"({rate:.2f} positions/s, ETA {remaining / 3600:.2f} h)",
                flush=True,
            )


class UciWdlEngine:
    def __init__(
        self,
        binary: Path,
        kind: str,
        weights: Path | None,
        backend: str | None,
        uci_options: list[str],
        name_override: str | None,
        version_override: str | None,
        weights_override: str | None,
    ) -> None:
        self.binary = binary.resolve()
        self.kind = kind
        self.weights = weights.resolve() if weights else None
        self.backend = backend
        self.stderr_lines: collections.deque[str] = collections.deque(maxlen=200)
        self.backend_ready = threading.Event()
        self.backend_verified = False
        self._cuda_runtime_seen = False
        command = [str(self.binary)]
        if kind == "lc0":
            command.extend(
                [
                    f"--weights={self.weights}",
                    f"--backend={backend}",
                    "--minibatch-size=1",
                    "--nncache=0",
                ]
            )
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert self.process.stdin and self.process.stdout and self.process.stderr
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        uci_name, default_weights = self._initialize(uci_options)
        parsed_name, _, parsed_version = uci_name.partition(" ")
        self.name = name_override or parsed_name
        self.version = version_override or parsed_version or uci_name
        if weights_override:
            self.weights_identity = weights_override
        elif self.weights:
            self.weights_identity = (
                f"{self.weights.name}@sha256:{sha256(self.weights)}"
            )
        else:
            self.weights_identity = default_weights or "none"
        if self.kind != "uci" and ENGINE_MARKER not in uci_name:
            self.close()
            raise RuntimeError(
                f"engine identity {uci_name!r} does not contain {ENGINE_MARKER!r}"
            )

    def _drain_stderr(self) -> None:
        assert self.process.stderr
        for raw in self.process.stderr:
            line = raw.rstrip()
            self.stderr_lines.append(line)
            if (
                self.kind == "lc0"
                and self.backend == "metal"
                and "Initialized metal backend on device" in line
            ):
                self.backend_ready.set()
            if self.kind == "lc0" and self.backend == "cuda":
                if line.startswith("CUDA Runtime version:"):
                    self._cuda_runtime_seen = True
                elif self._cuda_runtime_seen and line.startswith("GPU: "):
                    self.backend_ready.set()

    def _send(self, command: str) -> None:
        if self.process.poll() is not None:
            raise RuntimeError(f"engine exited with status {self.process.returncode}")
        assert self.process.stdin
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _read_until(self, terminator: str) -> list[str]:
        assert self.process.stdout
        lines: list[str] = []
        while True:
            raw = self.process.stdout.readline()
            if raw == "":
                tail = "\n".join(self.stderr_lines)
                raise RuntimeError(f"engine closed stdout before {terminator!r}\n{tail}")
            line = raw.rstrip()
            lines.append(line)
            if line == terminator or line.startswith(terminator + " "):
                return lines

    def _initialize(self, uci_options: list[str]) -> tuple[str, str | None]:
        self._send("uci")
        lines = self._read_until("uciok")
        names = [line.removeprefix("id name ") for line in lines if line.startswith("id name ")]
        if len(names) != 1:
            raise RuntimeError(f"expected one UCI engine identity, received {names}")
        eval_defaults = [
            match.group(1)
            for line in lines
            if (match := re.match(r"option name EvalFile .* default (.+)$", line))
        ]
        if self.kind == "stockfish":
            self._send("setoption name UCI_ShowWDL value true")
            if self.weights:
                self._send(f"setoption name EvalFile value {self.weights}")
        for option in uci_options:
            if "=" not in option:
                raise RuntimeError(f"invalid --uci-option {option!r}; expected NAME=VALUE")
            name, value = option.split("=", 1)
            self._send(f"setoption name {name} value {value}")
        self._send("isready")
        self._read_until("readyok")
        return names[0], eval_defaults[0] if eval_defaults else None

    def evaluate(self, fen: str) -> tuple[int, int, int]:
        board = chess.Board(fen)
        if not board.is_valid():
            raise RuntimeError(f"invalid chess position: {fen!r}")
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            if outcome.winner is None:
                return (0, 1000, 0)
            return (
                (1000, 0, 0)
                if outcome.winner == board.turn
                else (0, 0, 1000)
            )
        self._send("position fen " + fen)
        self._send("go depth 0")
        lines = self._read_until("bestmove")
        matches = [WDL_RE.search(line) for line in lines]
        values = [tuple(map(int, match.groups())) for match in matches if match]
        if len(values) != 1:
            raise RuntimeError(f"expected one depth-zero WDL response for {fen!r}: {lines}")
        wdl = values[0]
        if sum(wdl) != 1000:
            raise RuntimeError(f"invalid WDL sum for {fen!r}: {wdl}")
        if self.kind == "lc0" and not self.backend_verified:
            self.require_accelerator()
        return wdl

    def require_accelerator(self) -> None:
        if not self.backend_ready.wait(timeout=2):
            tail = "\n".join(self.stderr_lines)
            raise RuntimeError(
                f"lc0 did not confirm {self.backend} initialization; "
                "refusing CPU fallback\n"
                + tail
            )
        self.backend_verified = True

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self._send("quit")
                self.process.wait(timeout=5)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.terminate()
                self.process.wait(timeout=5)

    def __enter__(self) -> "UciWdlEngine":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class BatchedLc0Engine:
    """Evaluate lc0 depth-zero WDL in batches through lc0's Python bindings.

    Over UCI lc0 evaluates one position per round trip, which on a Tesla P100
    measured 31 positions per second against 74 for the backend at batch 1 and
    740 at batch 55. Most of batch-1 time is position setup and search-tree
    initialisation, and the rest is the GPU idling on a batch of one.

    The bindings avoid both. ``GameState.as_input`` calls lc0's own
    ``EncodePositionForNN`` with ``FillEmptyHistory::FEN_ONLY`` -- the same code
    the engine uses -- so nothing about the encoding is reimplemented here, and
    ``Backend.evaluate`` takes as many inputs as it is given.

    Engine identity still comes from a UCI handshake, because the bindings
    expose no version string and provenance must keep naming the exact build.
    """

    def __init__(
        self,
        binary: Path,
        weights: Path | None,
        backend: str | None,
        uci_options: list[str],
        name_override: str | None,
        version_override: str | None,
        weights_override: str | None,
        batch_size: int,
        module_path: str | None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        self.kind = "lc0"
        self.batch_size = batch_size
        # One short-lived UCI session records exactly which build produced the
        # labels; the bindings cannot report it.
        with UciWdlEngine(
            binary, "lc0", weights, backend, uci_options,
            name_override, version_override, weights_override,
        ) as probe:
            self.name = probe.name
            self.version = probe.version
            self.weights_identity = probe.weights_identity

        module = _import_lc0_bindings(module_path)
        self._weights = module.Weights(str(weights)) if weights else module.Weights()
        self._backend = module.Backend(weights=self._weights, backend=backend)
        self._module = module

    def evaluate(self, fen: str) -> tuple[int, int, int]:
        return self.evaluate_many([fen])[0]

    def evaluate_many(self, fens: Sequence[str]) -> list[tuple[int, int, int]]:
        results: list[tuple[int, int, int] | None] = [None] * len(fens)
        pending: list[tuple[int, str]] = []
        for index, fen in enumerate(fens):
            terminal = _terminal_wdl(fen)
            if terminal is None:
                pending.append((index, fen))
            else:
                results[index] = terminal
        for start in range(0, len(pending), self.batch_size):
            chunk = pending[start : start + self.batch_size]
            states = [self._module.GameState(fen=fen) for _, fen in chunk]
            # as_input returns owned objects; they must outlive the evaluate
            # call, so the list is held rather than built inline.
            inputs = [state.as_input(self._backend) for state in states]
            outputs = self._backend.evaluate(*inputs)
            if len(outputs) != len(chunk):
                raise RuntimeError(
                    f"lc0 returned {len(outputs)} results for {len(chunk)} inputs"
                )
            for (index, _), output in zip(chunk, outputs):
                results[index] = _wdl_from_q_d(output.q(), output.d())
        if any(result is None for result in results):
            raise RuntimeError("batched lc0 returned an incomplete WDL batch")
        return [result for result in results if result is not None]

    def require_accelerator(self) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self) -> "BatchedLc0Engine":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _import_lc0_bindings(module_path: str | None):
    """Import lc0's ``backends`` module, built with -Dpython_bindings=true."""

    if module_path:
        sys.path.insert(0, module_path)
    try:
        import backends  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "lc0 Python bindings not importable. Build lc0 with "
            "-Dpython_bindings=true and pass --lc0-python-path pointing at the "
            f"directory holding backends*.so ({error})"
        ) from error
    return backends


def _terminal_wdl(fen: str) -> tuple[int, int, int] | None:
    """Exact WDL for a finished game, or None when the engine must evaluate."""

    board = chess.Board(fen)
    if not board.is_valid():
        raise RuntimeError(f"invalid chess position: {fen!r}")
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return (0, 1000, 0)
    return (1000, 0, 0) if outcome.winner == board.turn else (0, 0, 1000)


def _wdl_from_q_d(q: float, d: float) -> tuple[int, int, int]:
    """lc0 reports expected score and draw rate; WDL follows from the pair.

    Draw absorbs the rounding residue so the triple sums to 1000 exactly, which
    is what the UCI layer does.
    """

    win = int(round((1.0 + q - d) / 2.0 * 1000))
    loss = int(round((1.0 - q - d) / 2.0 * 1000))
    win = max(0, min(1000, win))
    loss = max(0, min(1000 - win, loss))
    return (win, 1000 - win - loss, loss)


def add_wdl(
    table: pa.Table,
    values: list[tuple[int, int, int]],
    engine_fields: tuple[str, str, str],
    metadata: dict[bytes, bytes],
) -> pa.Table:
    output_columns = WDL_COLUMNS + ENGINE_COLUMNS
    if any(name in table.column_names for name in output_columns):
        raise RuntimeError("input already contains WDL or engine provenance columns")
    for index, name in enumerate(WDL_COLUMNS):
        table = table.append_column(
            name, pa.array((wdl[index] for wdl in values), type=pa.uint16())
        )
    for name, value in zip(ENGINE_COLUMNS, engine_fields):
        table = table.append_column(
            name, pa.array([value] * table.num_rows, type=pa.string())
        )
    return table.replace_schema_metadata(metadata)


def slice_digest(table: pa.Table) -> bytes:
    """Identify the rows a chunk covers, not merely how many there are."""

    digest = hashlib.sha256()
    for value in table.column("fen").to_pylist():
        digest.update(value.encode())
        digest.update(b"\0")
    return digest.hexdigest().encode()


def chunk_is_valid(
    path: Path, rows: int, metadata: dict[bytes, bytes], digest: bytes | None = None
) -> bool:
    if not path.is_file():
        return False
    parquet = pq.ParquetFile(path)
    actual = parquet.schema_arrow.metadata or {}
    if parquet.metadata.num_rows != rows:
        return False
    if not all(actual.get(key) == value for key, value in metadata.items()):
        return False
    # Row count and source checksum do not pin down which rows a chunk holds:
    # two slices of one file with the same length look identical to both. A
    # resume whose row-group or checkpoint boundaries fall differently would
    # then reuse a chunk whose labels belong to other positions. Chunks written
    # before this check carry no digest and are rejected, since there is no way
    # to tell which rows they cover.
    if digest is not None and actual.get(b"chunk_row_digest") != digest:
        return False
    return True


def output_is_valid(path: Path, rows: int, metadata: dict[bytes, bytes]) -> bool:
    return chunk_is_valid(path, rows, metadata) and (
        pq.ParquetFile(path).schema_arrow.metadata or {}
    ).get(b"zero_wdl_complete") == b"true"


def evaluate_positions(
    fens: list[str], engines: list, progress: Progress
) -> list[tuple[int, int, int]]:
    # A batched engine is one object that consumes the whole chunk, so there is
    # nothing to shard across worker threads.
    if len(engines) == 1 and isinstance(engines[0], BatchedLc0Engine):
        engine = engines[0]
        results = []
        for start in range(0, len(fens), engine.batch_size):
            chunk = fens[start : start + engine.batch_size]
            results.extend(engine.evaluate_many(chunk))
            progress.advance(len(chunk))
        return results

    results: list[tuple[int, int, int] | None] = [None] * len(fens)
    shards: list[list[tuple[int, str]]] = [[] for _ in engines]
    for index, fen in enumerate(fens):
        shards[index % len(engines)].append((index, fen))
    stop = threading.Event()

    def evaluate_shard(engine: UciWdlEngine, shard: list[tuple[int, str]]) -> None:
        try:
            for index, fen in shard:
                if stop.is_set():
                    return
                results[index] = engine.evaluate(fen)
                progress.advance()
        except BaseException:
            stop.set()
            raise

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(engines))
    futures = [
        executor.submit(evaluate_shard, engine, shard)
        for engine, shard in zip(engines, shards)
    ]
    try:
        for future in concurrent.futures.as_completed(futures):
            future.result()
    finally:
        stop.set()
        executor.shutdown(wait=True, cancel_futures=True)
    if any(result is None for result in results):
        raise RuntimeError("engine workers returned an incomplete WDL batch")
    return [result for result in results if result is not None]


def verify_output(
    destination: Path, engines: list, sample_rows: int, seed: int
) -> None:
    """Re-evaluate a sample of finished rows and insist the labels reproduce.

    Every other check here is structural -- row counts, checksums, triples
    summing to 1000, a completion flag -- and a backend returning wrong values
    satisfies all of them. Only asking the engine the same question twice
    catches that, so a labeled file is not accepted until a sample of it
    reproduces.
    """

    import random

    table = pq.read_table(destination, columns=["fen", *WDL_COLUMNS])
    count = min(sample_rows, table.num_rows)
    picks = random.Random(seed).sample(range(table.num_rows), count)
    fens = [table.column("fen")[i].as_py() for i in picks]
    stored = [
        tuple(int(table.column(c)[i].as_py()) for c in WDL_COLUMNS) for i in picks
    ]
    silent = Progress(count)
    silent.report = lambda *a, **k: None  # type: ignore[method-assign]
    again = evaluate_positions(list(fens), engines, silent)

    mismatched = [
        (f, a, b) for f, a, b in zip(fens, stored, again)
        if max(abs(x - y) for x, y in zip(a, b)) > VERIFY_TOLERANCE
    ]
    if mismatched:
        f, a, b = mismatched[0]
        raise RuntimeError(
            f"{destination}: {len(mismatched)} of {count} sampled rows did not "
            f"reproduce within {VERIFY_TOLERANCE}/1000. The engine is returning "
            f"unstable results, so the labels cannot be trusted.\n"
            f"  first mismatch: {f}\n  stored {a} vs re-evaluated {b}"
        )
    print(f"verified {destination}: {count} sampled rows reproduce", flush=True)


def label_file(
    source: Path,
    destination: Path,
    work: Path,
    engines: list[UciWdlEngine],
    engine_fields: tuple[str, str, str],
    provenance: dict[bytes, bytes],
    progress: Progress,
    checkpoint_rows: int,
    verify_rows: int = 0,
) -> None:
    source_hash = sha256(source)
    metadata = dict(pq.ParquetFile(source).schema_arrow.metadata or {})
    metadata.update(provenance)
    metadata[b"source_parquet_sha256"] = source_hash.encode()
    metadata[b"zero_wdl_complete"] = b"false"
    parquet = pq.ParquetFile(source)
    rows = parquet.metadata.num_rows
    final_metadata = dict(metadata)
    final_metadata[b"zero_wdl_complete"] = b"true"
    if output_is_valid(destination, rows, final_metadata):
        # Verify on the skip path too. A file that is structurally complete but
        # holds wrong labels is exactly what resume would otherwise carry
        # forward untouched, run after run.
        if verify_rows:
            verify_output(destination, engines, verify_rows, seed=rows)
        progress.advance(rows, report=False)
        print(f"skip complete {destination}: {rows:,} rows", flush=True)
        return

    work.mkdir(parents=True, exist_ok=True)
    chunks: list[Path] = []
    for group in range(parquet.num_row_groups):
        table = parquet.read_row_group(group)
        legacy_chunk = work / f"row-group-{group:05d}.parquet"
        if chunk_is_valid(legacy_chunk, table.num_rows, metadata, slice_digest(table)):
            chunks.append(legacy_chunk)
            progress.advance(table.num_rows, report=False)
            continue
        for part, offset in enumerate(range(0, table.num_rows, checkpoint_rows)):
            source_part = table.slice(offset, checkpoint_rows)
            chunk = work / f"row-group-{group:05d}-part-{part:05d}.parquet"
            chunks.append(chunk)
            digest = slice_digest(source_part)
            if chunk_is_valid(chunk, source_part.num_rows, metadata, digest):
                progress.advance(source_part.num_rows, report=False)
                continue
            fens = source_part.column("fen").to_pylist()
            wdls = evaluate_positions(fens, engines, progress)
            chunk_metadata = dict(metadata)
            chunk_metadata[b"chunk_row_digest"] = digest
            labeled = add_wdl(source_part, wdls, engine_fields, chunk_metadata)
            temporary = chunk.with_suffix(".tmp")
            pq.write_table(
                labeled, temporary, compression="zstd", compression_level=6
            )
            os.replace(temporary, chunk)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    writer: pq.ParquetWriter | None = None
    buffered: list[pa.Table] = []
    buffered_rows = 0
    try:
        for chunk in chunks:
            table = pq.read_table(chunk).replace_schema_metadata(final_metadata)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary,
                    table.schema,
                    compression="zstd",
                    compression_level=6,
                    write_statistics=True,
                )
            buffered.append(table)
            buffered_rows += table.num_rows
            if buffered_rows >= 65_536:
                combined = pa.concat_tables(buffered)
                writer.write_table(combined.slice(0, 65_536), row_group_size=65_536)
                remainder = combined.slice(65_536)
                buffered = [remainder] if remainder.num_rows else []
                buffered_rows = remainder.num_rows
        if writer is None:
            raise RuntimeError(f"source has no row groups: {source}")
        if buffered:
            combined = pa.concat_tables(buffered)
            writer.write_table(combined, row_group_size=65_536)
        writer.close()
        writer = None
        os.replace(temporary, destination)
    finally:
        if writer is not None:
            writer.close()
    if not output_is_valid(destination, rows, final_metadata):
        raise RuntimeError(f"completed output failed validation: {destination}")
    if verify_rows:
        verify_output(destination, engines, verify_rows, seed=rows)
    print(f"completed {destination}: {rows:,} rows", flush=True)


def run(args: argparse.Namespace) -> None:
    source_root, source_details = resolve_input(args)
    output_root = Path(args.output).resolve()
    binary = Path(args.engine).resolve()
    weights = Path(args.weights).resolve() if args.weights else None
    if source_root == output_root:
        raise RuntimeError("input and output must differ for atomic, recoverable labeling")
    files = sorted(source_root.glob("source_split=*/phase=*/*.parquet"))
    if not files:
        raise RuntimeError(f"no partitioned Parquet files found under {source_root}")
    if args.engine_kind == "lc0" and (weights is None or args.backend is None):
        raise RuntimeError("lc0 requires both --weights and --backend")
    if args.engine_kind != "lc0" and args.backend is not None:
        raise RuntimeError("--backend is only valid with --engine-kind lc0")
    for path in (binary, weights):
        if path is None:
            continue
        if not path.is_file():
            raise RuntimeError(f"missing required file: {path}")

    total = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    workers = (
        args.workers
        if args.workers is not None
        else (4 if args.engine_kind == "lc0" and args.backend == "cuda" else 1)
    )
    if workers < 1:
        raise RuntimeError("--workers must be at least 1")
    progress = Progress(total)
    binary_hash = sha256(binary)
    weights_hash = sha256(weights) if weights else None
    weights_identity = args.engine_weights
    if weights_identity is None and weights is not None:
        weights_identity = f"{weights.name}@sha256:{weights_hash}"
    with contextlib.ExitStack() as stack:
        if args.lc0_batch_size and args.engine_kind == "lc0":
            # Batching replaces process parallelism: one backend saturates the
            # accelerator far better than several batch-1 processes competing
            # for it.
            workers = 1
            engines = [
                stack.enter_context(
                    BatchedLc0Engine(
                        binary,
                        weights,
                        args.backend,
                        args.uci_option,
                        args.engine_name,
                        args.engine_version,
                        weights_identity,
                        args.lc0_batch_size,
                        args.lc0_python_path,
                    )
                )
            ]
        else:
            engines = [
                stack.enter_context(
                    UciWdlEngine(
                        binary,
                        args.engine_kind,
                        weights,
                        args.backend,
                        args.uci_option,
                        args.engine_name,
                        args.engine_version,
                        weights_identity,
                    )
                )
                for _ in range(workers)
            ]
        identities = {
            (engine.name, engine.version, engine.weights_identity)
            for engine in engines
        }
        if len(identities) != 1:
            raise RuntimeError(f"engine worker identity mismatch: {sorted(identities)}")
        engine_fields = next(iter(identities))
        engine_name, engine_version, engine_weights = engine_fields
        provenance = {
            b"zero_wdl_command": b"go depth 0",
            b"zero_wdl_perspective": b"side_to_move",
            b"zero_wdl_scale": b"1000",
            b"zero_wdl_terminal_adjudication": (
                b"python-chess outcome(claim_draw=True)"
            ),
            b"wdl_engine_kind": args.engine_kind.encode(),
            b"wdl_engine_name": engine_name.encode(),
            b"wdl_engine_version": engine_version.encode(),
            b"wdl_engine_weights": engine_weights.encode(),
            b"wdl_engine_binary_sha256": binary_hash.encode(),
            b"wdl_engine_uci_options": json.dumps(
                args.uci_option, separators=(",", ":")
            ).encode(),
        }
        if args.backend:
            provenance[b"wdl_engine_backend"] = args.backend.encode()
        if weights_hash:
            provenance[b"wdl_engine_weights_sha256"] = weights_hash.encode()
        if source_details["type"] == "huggingface":
            provenance[b"source_hf_dataset"] = source_details["dataset"].encode()
            provenance[b"source_hf_revision"] = source_details[
                "resolved_revision"
            ].encode()
        print(
            json.dumps(
                {
                    "rows": total,
                    "engine_kind": args.engine_kind,
                    "engine_name": engine_name,
                    "engine_version": engine_version,
                    "engine_weights": engine_weights,
                    "engine_binary_sha256": binary_hash,
                    "backend": args.backend,
                    "workers": workers,
                },
                indent=2,
            ),
            flush=True,
        )
        for source in files:
            relative = source.relative_to(source_root)
            destination = output_root / relative
            work = output_root / "_work" / relative.parent
            label_file(
                source,
                destination,
                work,
                engines,
                engine_fields,
                provenance,
                progress,
                args.checkpoint_rows,
                args.verify_rows,
            )

    partitions = []
    output_total = 0
    for source in files:
        relative = source.relative_to(source_root)
        destination = output_root / relative
        parquet = pq.ParquetFile(destination)
        rows = parquet.metadata.num_rows
        output_total += rows
        partitions.append(
            {
                "file": relative.as_posix(),
                "rows": rows,
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
    if output_total != total:
        raise RuntimeError(f"output row mismatch: {output_total} != {total}")
    manifest = {
        "format": "VPD1-Parquet-ZeroWDL",
        "rows": output_total,
        "wdl_columns": list(WDL_COLUMNS),
        "engine_columns": list(ENGINE_COLUMNS),
        "wdl_scale": 1000,
        "wdl_perspective": "side_to_move",
        "terminal_adjudication": "python-chess outcome(claim_draw=True)",
        "engine": {
            "kind": args.engine_kind,
            "name": engine_name,
            "version": engine_version,
            "weights": engine_weights,
            "binary_sha256": binary_hash,
            "weights_sha256": weights_hash,
            "backend": args.backend,
            "workers": workers,
            "uci_options": args.uci_option,
        },
        "source": source_details,
        "partitions": partitions,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / "_manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_root / "_manifest.json")
    print(f"complete: {output_total:,} rows", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="local unlabeled Parquet dataset directory")
    source.add_argument(
        "--hf-dataset",
        help="Hugging Face dataset repository, for example Pawitt/zero-evaluator",
    )
    parser.add_argument(
        "--hf-revision",
        default="main",
        help="Hugging Face branch, tag, or commit (default: main)",
    )
    parser.add_argument("--output", required=True, help="labeled Parquet dataset directory")
    parser.add_argument(
        "--engine",
        "--lc0",
        dest="engine",
        required=True,
        help="custom depth-zero-WDL UCI binary (--lc0 is a compatibility alias)",
    )
    parser.add_argument(
        "--engine-kind",
        choices=("lc0", "stockfish", "uci"),
        default="lc0",
        help="engine protocol profile (default: lc0)",
    )
    parser.add_argument("--weights", help="lc0 weights or Stockfish EvalFile")
    parser.add_argument(
        "--backend",
        choices=("cuda", "metal"),
        help="lc0 hardware accelerator; required for lc0",
    )
    parser.add_argument("--engine-name", help="manual Parquet engine-name value")
    parser.add_argument("--engine-version", help="manual Parquet version value")
    parser.add_argument("--engine-weights", help="manual Parquet weights identity")
    parser.add_argument(
        "--uci-option",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="additional UCI option; repeat for multiple options",
    )
    parser.add_argument(
        "--lc0-batch-size",
        type=int,
        default=0,
        help="evaluate lc0 in batches of this size through its Python bindings "
             "instead of one position per UCI round trip; 0 keeps the UCI path. "
             "Throughput saturates near 55 on a P100 and 64 on Apple metal",
    )
    parser.add_argument(
        "--lc0-python-path",
        help="directory holding lc0's backends*.so, built with "
             "-Dpython_bindings=true (default: rely on sys.path)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="persistent engine processes (default: lc0 cuda=4, otherwise=1)",
    )
    parser.add_argument(
        "--checkpoint-rows",
        type=int,
        default=1000,
        help="positions per atomic resume checkpoint (default: 1000)",
    )
    parser.add_argument(
        "--verify-rows",
        type=int,
        default=256,
        help="after each file, re-evaluate this many random labeled rows and "
             "fail if they do not reproduce; 0 disables the check",
    )
    try:
        args = parser.parse_args()
        if args.checkpoint_rows < 1:
            parser.error("--checkpoint-rows must be at least 1")
        if args.lc0_batch_size < 0:
            parser.error("--lc0-batch-size must not be negative")
        if args.lc0_batch_size and args.engine_kind != "lc0":
            parser.error("--lc0-batch-size only applies to --engine-kind lc0")
        run(args)
    except (KeyboardInterrupt, BrokenPipeError):
        print("interrupted; completed position checkpoints are retained", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
