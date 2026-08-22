#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate resumable root-only lc0 MCTS policy targets as sharded Parquet."""

from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import chess
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


SCHEMA_VERSION = 1
RESUME_SCHEMA_VERSION = 1
SPLITS = ("train", "validation", "test")
MOVE_STATS_RE = re.compile(
    r"^info string (?P<move>[a-h][1-8][a-h][1-8][qrbn]?)\s+"
    r"\(\s*\d+\s*\)\s+N:\s*(?P<visits>\d+)\s+"
    r"\(\+\s*\d+\)\s+\(P:\s*(?P<prior>[0-9.]+)%\).*?"
    r"\(Q:\s*(?P<q>-?[0-9.]+)\).*?"
    r"\(V:\s*(?P<v>-?[0-9.]+|-\.-+)\)"
)
WDL_RE = re.compile(r"\bwdl (\d+) (\d+) (\d+)\b")
NODES_RE = re.compile(r"\bnodes (\d+)\b")
CHECKPOINT_RE = re.compile(r"^rows-(\d{9})-(\d{9})\.parquet$")
CASTLING_UCI = {
    "e1h1": "e1g1",
    "e1a1": "e1c1",
    "e8h8": "e8g8",
    "e8a8": "e8c8",
}

OUTPUT_SCHEMA = pa.schema(
    [
        ("fen", pa.string()),
        ("moves", pa.list_(pa.string())),
        ("visits", pa.list_(pa.uint32())),
        ("priors", pa.list_(pa.float32())),
        ("q_values", pa.list_(pa.float32())),
        ("v_values", pa.list_(pa.float32())),
        ("root_wdl_win", pa.uint16()),
        ("root_wdl_draw", pa.uint16()),
        ("root_wdl_loss", pa.uint16()),
        ("best_move", pa.string()),
        ("search_nodes", pa.uint32()),
        ("source_game_id", pa.string()),
        ("source_ply", pa.uint16()),
        ("phase", pa.string()),
    ]
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def input_sha256(path: Path) -> str:
    if path.is_file():
        return sha256(path)
    digest = hashlib.sha256()
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise RuntimeError(f"no Parquet files found under input plan {path}")
    for file in files:
        digest.update(file.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256(file).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def git_identity(binary: Path) -> str:
    for parent in binary.parents:
        if not (parent / ".git").exists():
            continue
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=parent, text=True
            ).strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=parent,
                text=True,
                stdout=subprocess.PIPE,
                check=True,
            ).stdout
            return commit + ("-dirty" if dirty else "")
        except (OSError, subprocess.CalledProcessError):
            break
    return "unknown"


class Progress:
    def __init__(self, total: int, done: int = 0) -> None:
        self.done = done
        self.run_done = 0
        self.total = total
        self.started = time.monotonic()

    @property
    def rate(self) -> float:
        elapsed = time.monotonic() - self.started
        return self.run_done / elapsed if elapsed > 0 else 0.0

    def set_total(self, total: int) -> None:
        self.total = total

    def advance(self, count: int, resumed: bool = False) -> None:
        self.done += count
        if resumed:
            return
        self.run_done += count
        rate = self.rate
        remaining = (self.total - self.done) / rate
        print(
            f"{self.done:,}/{self.total:,} "
            f"({rate:.3f} positions/s, ETA {remaining / 3600:.2f} h)",
            flush=True,
        )


def apportion(total: int, counts: dict[str, int]) -> dict[str, int]:
    denominator = sum(counts.values())
    raw = {key: total * value / denominator for key, value in counts.items()}
    result = {key: math.floor(value) for key, value in raw.items()}
    missing = total - sum(result.values())
    order = sorted(counts, key=lambda key: (raw[key] - result[key], key), reverse=True)
    for key in order[:missing]:
        result[key] += 1
    return result


def write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def checkpoint_path(
    output: Path, split: str, offset: int, rows: int
) -> Path:
    return (
        output
        / "_work"
        / split
        / f"rows-{offset:09d}-{offset + rows:09d}.parquet"
    )


def checkpoint_bounds(path: Path) -> tuple[int, int]:
    match = CHECKPOINT_RE.match(path.name)
    if not match:
        raise RuntimeError(f"invalid checkpoint filename: {path}")
    return int(match.group(1)), int(match.group(2))


def scan_checkpoints(
    tables: dict[str, pa.Table],
    output: Path,
    checkpoint_rows: int,
    provenance: dict[bytes, bytes],
    limits: dict[str, int],
) -> dict[str, dict[int, Path]]:
    found: dict[str, dict[int, Path]] = {split: {} for split in SPLITS}
    for split, table in tables.items():
        for offset in range(0, limits[split], checkpoint_rows):
            rows = min(checkpoint_rows, limits[split] - offset)
            path = checkpoint_path(output, split, offset, rows)
            if checkpoint_valid(path, rows, provenance):
                found[split][offset] = path
    return found


def refresh_resume_state(
    state: dict[str, object],
    completed: dict[str, dict[int, Path]],
    tables: dict[str, pa.Table],
    limits: dict[str, int],
) -> None:
    ranges: dict[str, list[dict[str, object]]] = {}
    completed_rows = 0
    for split in SPLITS:
        items = []
        for offset, path in sorted(completed[split].items()):
            start, end = checkpoint_bounds(path)
            if start != offset:
                raise RuntimeError(f"checkpoint offset mismatch: {path}")
            if start >= limits[split] or end > limits[split]:
                continue
            items.append({"start": start, "end": end, "file": str(path)})
            completed_rows += end - start
        ranges[split] = items
    state["completed_ranges"] = ranges
    state["completed_rows"] = completed_rows

    next_position: dict[str, object] | None = None
    for split in SPLITS:
        end = 0
        for item in ranges[split]:
            if int(item["start"]) != end:
                break
            end = int(item["end"])
        if end < limits[split]:
            row = tables[split].slice(end, 1).to_pylist()[0]
            next_position = {
                "split": split,
                "split_offset": end,
                "source_game_id": row["source_game_id"],
                "source_ply": row["source_ply"],
                "fen": row["fen"],
            }
            break
    state["next_position"] = next_position


class Lc0PolicyEngine:
    def __init__(
        self,
        binary: Path,
        weights: Path,
        backend: str,
        cache_size: int,
        minibatch_size: int,
    ) -> None:
        self.binary = binary.resolve()
        self.weights = weights.resolve()
        self.backend = backend
        self.minibatch_size = minibatch_size
        self.stderr_lines: collections.deque[str] = collections.deque(maxlen=200)
        self.backend_ready = threading.Event()
        self.backend_verified = False
        self._cuda_runtime_seen = False
        self.process = subprocess.Popen(
            [
                str(self.binary),
                f"--weights={self.weights}",
                f"--backend={backend}",
                f"--minibatch-size={minibatch_size}",
                f"--nncache={cache_size}",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert self.process.stdin and self.process.stdout and self.process.stderr
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        self.version = self._initialize()

    def _drain_stderr(self) -> None:
        assert self.process.stderr
        for raw in self.process.stderr:
            line = raw.rstrip()
            self.stderr_lines.append(line)
            if self.backend == "metal" and "Initialized metal backend on device" in line:
                self.backend_ready.set()
            if self.backend == "cuda":
                if line.startswith("CUDA Runtime version:"):
                    self._cuda_runtime_seen = True
                elif self._cuda_runtime_seen and line.startswith("GPU: "):
                    self.backend_ready.set()

    def _send(self, command: str) -> None:
        if self.process.poll() is not None:
            raise RuntimeError(f"lc0 exited with status {self.process.returncode}")
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
                raise RuntimeError(f"lc0 closed stdout before {terminator!r}\n{tail}")
            line = raw.rstrip()
            lines.append(line)
            if line == terminator or line.startswith(terminator + " "):
                return lines

    def _initialize(self) -> str:
        self._send("uci")
        lines = self._read_until("uciok")
        names = [line[8:] for line in lines if line.startswith("id name ")]
        if len(names) != 1:
            raise RuntimeError(f"expected one lc0 UCI identity, received {names}")
        options = (
            ("VerboseMoveStats", "true"),
            ("UCI_ShowWDL", "true"),
            ("SmartPruningFactor", "0"),
            ("HistoryFill", "fen_only"),
            ("Threads", "1"),
            ("DirichletNoiseEpsilon", "0"),
        )
        for name, value in options:
            self._send(f"setoption name {name} value {value}")
        self._send("isready")
        self._read_until("readyok")
        return names[0]

    def evaluate(self, fen: str, nodes: int) -> dict[str, object]:
        board = chess.Board(fen)
        if not board.is_valid():
            raise RuntimeError(f"invalid chess position: {fen!r}")
        legal_moves = {move.uci() for move in board.legal_moves}
        if len(legal_moves) <= 1:
            raise RuntimeError(f"position has only {len(legal_moves)} legal moves: {fen!r}")
        if board.outcome(claim_draw=True) is not None:
            raise RuntimeError(f"terminal position is not eligible: {fen!r}")

        self._send("ucinewgame")
        self._send("isready")
        self._read_until("readyok")
        self._send("position fen " + fen)
        self._send(f"go nodes {nodes}")
        lines = self._read_until("bestmove")
        result = parse_search(lines, fen, legal_moves, nodes)
        if not self.backend_verified:
            self.require_accelerator()
        return result

    def require_accelerator(self) -> None:
        if not self.backend_ready.wait(timeout=2):
            raise RuntimeError(
                f"lc0 did not confirm {self.backend} initialization; refusing CPU fallback\n"
                + "\n".join(self.stderr_lines)
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

    def __enter__(self) -> "Lc0PolicyEngine":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def parse_search(
    lines: list[str], fen: str, legal_moves: set[str], requested_nodes: int
) -> dict[str, object]:
    edges: dict[str, tuple[int, float, float, float | None]] = {}
    latest_wdl: tuple[int, int, int] | None = None
    latest_nodes: int | None = None
    best_move: str | None = None
    for line in lines:
        if not line.startswith("info string "):
            match = WDL_RE.search(line)
            if match:
                latest_wdl = tuple(map(int, match.groups()))
            match = NODES_RE.search(line)
            if match:
                latest_nodes = int(match.group(1))
        match = MOVE_STATS_RE.match(line)
        if match:
            move = match.group("move")
            if move not in legal_moves:
                move = CASTLING_UCI.get(move, move)
            if move in edges:
                raise RuntimeError(f"duplicate root move {move!r} for {fen!r}")
            raw_v = match.group("v")
            edges[move] = (
                int(match.group("visits")),
                float(match.group("prior")) / 100.0,
                float(match.group("q")),
                None if raw_v.startswith("-.") else float(raw_v),
            )
        if line.startswith("bestmove "):
            best_move = line.split()[1]
            if best_move not in legal_moves:
                best_move = CASTLING_UCI.get(best_move, best_move)

    if set(edges) != legal_moves:
        missing = sorted(legal_moves - set(edges))
        extra = sorted(set(edges) - legal_moves)
        raise RuntimeError(
            f"root move set mismatch for {fen!r}; missing={missing}, extra={extra}"
        )
    if latest_wdl is None or sum(latest_wdl) != 1000:
        raise RuntimeError(f"missing or invalid root WDL for {fen!r}: {latest_wdl}")
    if best_move not in legal_moves:
        raise RuntimeError(f"invalid bestmove {best_move!r} for {fen!r}")

    moves = sorted(legal_moves)
    visits = [edges[move][0] for move in moves]
    priors = [edges[move][1] for move in moves]
    q_values = [edges[move][2] for move in moves]
    v_values = [edges[move][3] for move in moves]
    prior_sum = sum(priors)
    # VerboseMoveStats exposes the network's raw policy mass. Lc0 masks illegal
    # actions without renormalizing the remaining legal priors, so their sum can
    # legitimately be below one (substantially so in unusual positions).
    if not math.isfinite(prior_sum) or not 0.0 < prior_sum <= 1.001:
        raise RuntimeError(f"invalid raw root prior sum {prior_sum:.9f} for {fen!r}")
    # Searches stop with multiple NN batches potentially in flight. Their
    # completion can legitimately overshoot the requested budget, so validate
    # root visits against Lc0's authoritative final UCI node counter.
    if latest_nodes is None or latest_nodes < requested_nodes:
        raise RuntimeError(
            f"invalid final node count {latest_nodes} for requested "
            f"{requested_nodes} nodes at {fen!r}"
        )
    if sum(visits) > latest_nodes:
        raise RuntimeError(
            f"root visits {sum(visits)} exceed final UCI nodes {latest_nodes} "
            f"for {fen!r}"
        )
    if visits[moves.index(best_move)] != max(visits):
        raise RuntimeError(f"bestmove {best_move!r} lacks maximum visits for {fen!r}")
    for move, visit, value in zip(moves, visits, v_values):
        if visit > 0 and value is None:
            raise RuntimeError(f"visited move {move!r} has no raw V for {fen!r}")

    return {
        "fen": fen,
        "moves": moves,
        "visits": visits,
        "priors": priors,
        "q_values": q_values,
        "v_values": v_values,
        "root_wdl_win": latest_wdl[0],
        "root_wdl_draw": latest_wdl[1],
        "root_wdl_loss": latest_wdl[2],
        "best_move": best_move,
        "search_nodes": requested_nodes,
    }


def load_plan(path: Path) -> pa.Table:
    dataset = ds.dataset(path, format="parquet", partitioning="hive")
    required = {"fen", "split", "source_game_id", "source_ply", "phase"}
    missing = required - set(dataset.schema.names)
    if missing:
        raise RuntimeError(f"input plan lacks required columns: {sorted(missing)}")
    table = dataset.to_table(columns=sorted(required))
    invalid_splits = set(table["split"].to_pylist()) - set(SPLITS)
    if invalid_splits:
        raise RuntimeError(f"input plan has invalid splits: {sorted(invalid_splits)}")
    invalid_phases = set(table["phase"].to_pylist()) - {
        "opening", "middlegame", "endgame"
    }
    if invalid_phases:
        raise RuntimeError(f"input plan has invalid phases: {sorted(invalid_phases)}")
    if table.num_rows != pc.count_distinct(table["fen"]).as_py():
        raise RuntimeError("input plan contains duplicate FENs")
    assignments: dict[str, str] = {}
    phase_games: set[tuple[str, str]] = set()
    for row in table.select(["source_game_id", "split", "phase"]).to_pylist():
        game = str(row["source_game_id"])
        split = str(row["split"])
        if game in assignments and assignments[game] != split:
            raise RuntimeError(f"source game crosses splits: {game}")
        assignments[game] = split
        phase_game = (game, str(row["phase"]))
        if phase_game in phase_games:
            raise RuntimeError(f"source game has multiple positions in one phase: {game}")
        phase_games.add(phase_game)
    return table


def checkpoint_valid(path: Path, rows: int, metadata: dict[bytes, bytes]) -> bool:
    if not path.is_file():
        return False
    parquet = pq.ParquetFile(path)
    actual = parquet.schema_arrow.metadata or {}
    return parquet.metadata.num_rows == rows and all(
        actual.get(key) == value for key, value in metadata.items()
    )


def evaluate_chunk(
    table: pa.Table,
    engine: Lc0PolicyEngine,
    nodes: int,
    retries: int,
) -> pa.Table:
    output: list[dict[str, object]] = []
    for source in table.to_pylist():
        failure: Exception | None = None
        for _ in range(retries + 1):
            try:
                row = engine.evaluate(source["fen"], nodes)
                row.update(
                    source_game_id=str(source["source_game_id"]),
                    source_ply=int(source["source_ply"]),
                    phase=str(source["phase"]),
                )
                output.append(row)
                failure = None
                break
            except RuntimeError as exc:
                failure = exc
        if failure is not None:
            raise failure
    return pa.Table.from_pylist(output, schema=OUTPUT_SCHEMA)


def write_final_shards(
    split: str,
    chunks: list[Path],
    destination: Path,
    shard_rows: int,
    metadata: dict[bytes, bytes],
) -> list[Path]:
    total_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in chunks)
    shard_count = math.ceil(total_rows / shard_rows)
    outputs: list[Path] = []
    pending: list[pa.Table] = []
    pending_rows = 0
    shard = 0
    destination.mkdir(parents=True, exist_ok=True)

    def flush(rows: int) -> None:
        nonlocal pending, pending_rows, shard
        combined = pa.concat_tables(pending)
        table = combined.slice(0, rows).replace_schema_metadata(metadata)
        output = destination / f"{split}-{shard:05d}-of-{shard_count:05d}.parquet"
        temporary = output.with_suffix(".tmp")
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            compression_level=6,
            row_group_size=8192,
            write_statistics=True,
        )
        os.replace(temporary, output)
        outputs.append(output)
        remainder = combined.slice(rows)
        pending = [remainder] if remainder.num_rows else []
        pending_rows = remainder.num_rows
        shard += 1

    for chunk in chunks:
        table = pq.read_table(chunk).replace_schema_metadata(None)
        pending.append(table)
        pending_rows += table.num_rows
        while pending_rows >= shard_rows:
            flush(shard_rows)
    if pending_rows:
        flush(pending_rows)
    if shard != shard_count:
        raise RuntimeError(f"shard count mismatch for {split}: {shard} != {shard_count}")
    return outputs


def run(args: argparse.Namespace) -> None:
    plan_path = Path(args.input_plan).resolve()
    output = Path(args.output).resolve()
    binary = Path(args.lc0).resolve()
    weights = Path(args.weights).resolve()
    for path in (plan_path, binary, weights):
        if not path.exists():
            raise RuntimeError(f"missing required path: {path}")
    if not binary.is_file() or not weights.is_file():
        raise RuntimeError("--lc0 and --weights must be files")

    plan = load_plan(plan_path)
    plan_hash = input_sha256(plan_path)
    binary_hash = sha256(binary)
    weights_hash = sha256(weights)
    commit = args.lc0_commit or git_identity(binary)
    split_tables = {
        split: plan.filter(pc.equal(plan["split"], split)) for split in SPLITS
    }
    split_sizes = {split: table.num_rows for split, table in split_tables.items()}

    with Lc0PolicyEngine(
        binary, weights, args.backend, args.nncache_size, args.minibatch_size
    ) as engine:
        provenance = {
            b"schema_version": str(SCHEMA_VERSION).encode(),
            b"lc0_version": engine.version.encode(),
            b"lc0_commit": commit.encode(),
            b"lc0_binary_sha256": binary_hash.encode(),
            b"network_filename": weights.name.encode(),
            b"network_sha256": weights_hash.encode(),
            b"nodes_per_position": str(args.nodes).encode(),
            b"history_fill": b"fen_only",
            b"smart_pruning_factor": b"0",
            b"dirichlet_noise": b"false",
            b"threads": b"1",
            b"minibatch_size": str(args.minibatch_size).encode(),
            b"wdl_perspective": b"side_to_move",
            b"wdl_scale": b"1000",
            b"q_perspective": b"side_to_move",
            b"input_plan_sha256": plan_hash.encode(),
        }
        resume_path = output / "resumability.metadata.json"
        adaptive_enabled = args.adaptive_stop_hours is not None
        resume_config = {
            "backend": args.backend,
            "nodes": args.nodes,
            "minibatch_size": args.minibatch_size,
            "checkpoint_rows": args.checkpoint_rows,
            "adaptive_stop_hours": args.adaptive_stop_hours,
            "adaptive_safety_factor": args.adaptive_safety_factor,
            "adaptive_warmup_rows": args.adaptive_warmup_rows,
        }
        provenance_json = {
            key.decode(): value.decode() for key, value in provenance.items()
        }
        existing_work = (output / "_work").exists() and any(
            (output / "_work").rglob("*.parquet")
        )
        if (resume_path.exists() or existing_work) and not args.resume:
            raise RuntimeError(
                f"existing run state found under {output}; pass --resume to continue "
                "it or choose a new --output"
            )
        if resume_path.exists():
            state = json.loads(resume_path.read_text())
            if state.get("schema_version") != RESUME_SCHEMA_VERSION:
                raise RuntimeError("unsupported resumability metadata schema")
            if state.get("input_plan_sha256") != plan_hash:
                raise RuntimeError("resume input plan hash does not match")
            if state.get("config") != resume_config:
                raise RuntimeError("resume settings do not match the existing run")
            if state.get("provenance") != provenance_json:
                raise RuntimeError("resume engine or network provenance does not match")
        else:
            state = {
                "schema_version": RESUME_SCHEMA_VERSION,
                "status": "running",
                "input_plan": str(plan_path),
                "input_plan_sha256": plan_hash,
                "plan_rows": plan.num_rows,
                "split_plan_rows": split_sizes,
                "config": resume_config,
                "provenance": provenance_json,
                "adaptive": {
                    "enabled": adaptive_enabled,
                    "measured_positions_per_second": None,
                    "target_rows": None if adaptive_enabled else plan.num_rows,
                    "target_rows_by_split": None,
                },
                "completed_ranges": {split: [] for split in SPLITS},
                "completed_rows": 0,
                "next_position": None,
            }

        adaptive = state["adaptive"]
        target_rows_value = adaptive.get("target_rows")
        target_rows = int(target_rows_value) if target_rows_value is not None else None
        limits = (
            apportion(target_rows, split_sizes)
            if target_rows is not None
            else dict(split_sizes)
        )
        completed = scan_checkpoints(
            split_tables, output, args.checkpoint_rows, provenance, limits
        )
        resume_base_rows = sum(
            checkpoint_bounds(path)[1] - checkpoint_bounds(path)[0]
            for paths in completed.values()
            for path in paths.values()
        )
        progress = Progress(target_rows or plan.num_rows, done=resume_base_rows)
        refresh_resume_state(state, completed, split_tables, limits)
        write_json_atomic(resume_path, state)
        print(
            json.dumps(
                {
                    "rows": target_rows or plan.num_rows,
                    "plan_rows": plan.num_rows,
                    "lc0_version": engine.version,
                    "lc0_commit": commit,
                    "network_sha256": weights_hash,
                    "backend": args.backend,
                    "nodes_per_position": args.nodes,
                    "minibatch_size": args.minibatch_size,
                    "adaptive_stop_hours": args.adaptive_stop_hours,
                    "resumed_rows": resume_base_rows,
                },
                indent=2,
            ),
            flush=True,
        )
        chunks_by_split: dict[str, list[Path]] = {split: [] for split in SPLITS}
        for split in SPLITS:
            split_table = split_tables[split]
            offset = 0
            while offset < limits[split]:
                source = split_table.slice(
                    offset, min(args.checkpoint_rows, limits[split] - offset)
                )
                checkpoint = checkpoint_path(output, split, offset, source.num_rows)
                chunks_by_split[split].append(checkpoint)
                existing = completed[split].get(offset)
                if existing is not None and checkpoint_valid(
                    existing, source.num_rows, provenance
                ):
                    checkpoint = existing
                    chunks_by_split[split][-1] = checkpoint
                else:
                    labeled = evaluate_chunk(source, engine, args.nodes, args.retries)
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    temporary = checkpoint.with_suffix(".tmp")
                    pq.write_table(
                        labeled.replace_schema_metadata(provenance),
                        temporary,
                        compression="zstd",
                        compression_level=6,
                    )
                    os.replace(temporary, checkpoint)
                    completed[split][offset] = checkpoint
                    progress.advance(source.num_rows)

                    if (
                        adaptive_enabled
                        and target_rows is None
                        and progress.run_done >= args.adaptive_warmup_rows
                    ):
                        capacity = math.floor(
                            progress.rate
                            * args.adaptive_stop_hours
                            * 3600
                            * args.adaptive_safety_factor
                        )
                        target_rows = min(
                            plan.num_rows,
                            max(progress.done, resume_base_rows + capacity),
                        )
                        limits = apportion(target_rows, split_sizes)
                        if limits[split] < offset + source.num_rows:
                            limits[split] = offset + source.num_rows
                            target_rows = sum(limits.values())
                        progress.set_total(target_rows)
                        adaptive["measured_positions_per_second"] = progress.rate
                        adaptive["target_rows"] = target_rows
                        adaptive["target_rows_by_split"] = limits
                        adaptive["warmup_new_rows"] = progress.run_done
                        adaptive["estimated_search_hours"] = (
                            (target_rows - resume_base_rows) / progress.rate / 3600
                        )
                        print(
                            f"adaptive target frozen: {target_rows:,} rows at "
                            f"{progress.rate:.3f} positions/s "
                            f"({adaptive['estimated_search_hours']:.2f} h)",
                            flush=True,
                        )

                refresh_resume_state(state, completed, split_tables, limits)
                write_json_atomic(resume_path, state)
                offset += source.num_rows

        if adaptive_enabled and target_rows is None:
            target_rows = progress.done
            limits = apportion(target_rows, split_sizes)
            adaptive["target_rows"] = target_rows
            adaptive["target_rows_by_split"] = limits
        state["status"] = "finalizing"
        refresh_resume_state(state, completed, split_tables, limits)
        write_json_atomic(resume_path, state)

        final_metadata = dict(provenance)
        final_metadata[b"generation_complete"] = b"true"
        final_files: list[Path] = []
        for split in SPLITS:
            if chunks_by_split[split]:
                final_files.extend(
                    write_final_shards(
                        split,
                        chunks_by_split[split],
                        output,
                        args.shard_rows,
                        final_metadata,
                    )
                )

    partitions = []
    total_rows = 0
    for path in final_files:
        rows = pq.ParquetFile(path).metadata.num_rows
        total_rows += rows
        partitions.append(
            {
                "file": path.name,
                "rows": rows,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    expected_rows = target_rows or plan.num_rows
    if total_rows != expected_rows:
        raise RuntimeError(f"final row mismatch: {total_rows} != {expected_rows}")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "rows": total_rows,
        "lc0_version": engine.version,
        "lc0_commit": commit,
        "lc0_binary_sha256": binary_hash,
        "network_filename": weights.name,
        "network_sha256": weights_hash,
        "nodes_per_position": args.nodes,
        "history_fill": "fen_only",
        "smart_pruning_factor": 0,
        "dirichlet_noise": False,
        "threads": 1,
        "minibatch_size": args.minibatch_size,
        "backend": args.backend,
        "wdl_perspective": "side_to_move",
        "wdl_scale": 1000,
        "q_perspective": "side_to_move",
        "source": args.source_description,
        "split_unit": "source_game_id",
        "input_plan_sha256": plan_hash,
        "adaptive": state["adaptive"],
        "resumability_metadata": resume_path.name,
        "partitions": partitions,
    }
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output / "manifest.json")
    state["status"] = "complete"
    state["next_position"] = None
    write_json_atomic(resume_path, state)
    print(f"complete: {total_rows:,} rows", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-plan", required=True, help="Parquet position plan")
    parser.add_argument("--output", required=True, help="output dataset directory")
    parser.add_argument("--lc0", required=True, help="lc0 executable")
    parser.add_argument("--weights", required=True, help="fixed lc0 network")
    parser.add_argument("--backend", required=True, choices=("cuda", "metal"))
    parser.add_argument("--nodes", type=int, default=400)
    parser.add_argument("--checkpoint-rows", type=int, default=25)
    parser.add_argument("--shard-rows", type=int, default=25_000)
    parser.add_argument("--nncache-size", type=int, default=200_000)
    parser.add_argument(
        "--minibatch-size", type=int, default=32,
        help=(
            "NN batch size; 32 balances accelerator use and short-search "
            "quality; 0 uses the backend recommendation"
        ),
    )
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument(
        "--adaptive-stop-hours",
        type=float,
        help="measure throughput, then cap output rows to this runtime budget",
    )
    parser.add_argument(
        "--adaptive-warmup-rows", type=int, default=100,
        help="newly evaluated rows used for adaptive throughput measurement",
    )
    parser.add_argument(
        "--adaptive-safety-factor", type=float, default=0.95,
        help="fraction of the runtime budget assigned to searches",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume the run recorded by resumability.metadata.json",
    )
    parser.add_argument("--lc0-commit", help="override auto-detected lc0 commit")
    parser.add_argument(
        "--source-description",
        default="mixed position plan",
        help="dataset-level source description",
    )
    args = parser.parse_args()
    for name in ("nodes", "checkpoint_rows", "shard_rows", "nncache_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.minibatch_size < 0:
        parser.error("--minibatch-size cannot be negative")
    if args.retries < 0:
        parser.error("--retries cannot be negative")
    if args.adaptive_stop_hours is not None and args.adaptive_stop_hours <= 0:
        parser.error("--adaptive-stop-hours must be positive")
    if args.adaptive_warmup_rows < 1:
        parser.error("--adaptive-warmup-rows must be positive")
    if not 0 < args.adaptive_safety_factor <= 1:
        parser.error("--adaptive-safety-factor must be in (0, 1]")
    try:
        run(args)
    except (KeyboardInterrupt, BrokenPipeError):
        print("interrupted; completed position checkpoints are retained", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
