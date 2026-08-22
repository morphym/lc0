#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build a deterministic, quota-balanced position plan for policy MCTS."""

from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import heapq
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterator, TextIO

import chess
import chess.pgn
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


PHASES = ("opening", "middlegame", "endgame")
SOURCES = ("zero", "engine", "human", "hard")
PLAN_SCHEMA = pa.schema(
    [
        ("fen", pa.string()),
        ("split", pa.string()),
        ("source_game_id", pa.string()),
        ("source_ply", pa.uint16()),
        ("phase", pa.string()),
        ("source_category", pa.string()),
    ]
)
SEASON_RE = re.compile(r"TCEC Season\s+(\d+)", re.IGNORECASE)
ELITE_EVENT_RE = re.compile(
    r"Swiss|Cup|League 1|Division P|Superfinal|4K", re.IGNORECASE
)


def stable_unit(seed: int, *parts: object) -> float:
    digest = hashlib.blake2b(digest_size=8, person=b"vex-policy-plan")
    digest.update(str(seed).encode())
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8", "replace"))
    value = int.from_bytes(digest.digest(), "big")
    return (value + 1) / (2**64 + 1)


def apportion(total: int, shares: dict[str, float]) -> dict[str, int]:
    raw = {key: total * value / sum(shares.values()) for key, value in shares.items()}
    result = {key: math.floor(value) for key, value in raw.items()}
    missing = total - sum(result.values())
    order = sorted(shares, key=lambda key: (raw[key] - result[key], key), reverse=True)
    for key in order[:missing]:
        result[key] += 1
    return result


class Reservoir:
    """Retain the lowest deterministic weighted priorities with bounded memory."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.heap: list[tuple[float, int, dict[str, object]]] = []
        self.serial = 0

    def add(self, row: dict[str, object], priority: float) -> None:
        if self.capacity == 0:
            return
        item = (-priority, self.serial, row)
        self.serial += 1
        if len(self.heap) < self.capacity:
            heapq.heappush(self.heap, item)
        elif item > self.heap[0]:
            heapq.heapreplace(self.heap, item)

    def rows(self) -> list[dict[str, object]]:
        return [item[2] for item in sorted(self.heap, reverse=True)]


def canonical_key(fen: str) -> str:
    board = chess.Board(fen)
    return " ".join(board.fen(en_passant="fen").split()[:4])


def eligible(board: chess.Board) -> bool:
    # The target receives only a FEN, so repetition history is deliberately absent.
    return (
        board.is_valid()
        and board.halfmove_clock < 100
        and not board.is_game_over(claim_draw=False)
        and board.legal_moves.count() > 1
    )


def classify_phase(board: chess.Board) -> str:
    ply = board.ply()
    pieces = len(board.piece_map())
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(
        board.pieces(chess.QUEEN, chess.BLACK)
    )
    non_pawn = sum(
        len(board.pieces(piece, color)) * value
        for color in (chess.WHITE, chess.BLACK)
        for piece, value in (
            (chess.KNIGHT, 3),
            (chess.BISHOP, 3),
            (chess.ROOK, 5),
            (chess.QUEEN, 9),
        )
    )
    if pieces <= 12 or (queens == 0 and non_pawn <= 16):
        return "endgame"
    developed_material = sum(
        len(board.pieces(piece, color))
        for color in (chess.WHITE, chess.BLACK)
        for piece in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )
    if 8 <= ply <= 20 and developed_material >= 11:
        return "opening"
    return "middlegame"


def split_for(game_id: str, seed: int, train: float, validation: float) -> str:
    value = stable_unit(seed, "split", game_id)
    if value < train:
        return "train"
    if value < train + validation:
        return "validation"
    return "test"


@contextlib.contextmanager
def open_pgn(path: Path) -> Iterator[TextIO]:
    if path.suffix != ".zst":
        with path.open(encoding="utf-8", errors="replace") as source:
            yield source
        return
    process = subprocess.Popen(
        ["zstdcat", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout
    try:
        yield process.stdout
    finally:
        process.stdout.close()
        status = process.wait()
        if status not in (0, -13):
            raise RuntimeError(f"zstdcat failed for {path} with status {status}")


def game_id(prefix: str, game: chess.pgn.Game) -> str:
    digest = hashlib.sha256()
    for key in ("Event", "Site", "Date", "Round", "White", "Black", "Result"):
        digest.update(key.encode() + b"=" + game.headers.get(key, "").encode("utf-8", "replace") + b"\n")
    for move in game.mainline_moves():
        digest.update(move.uci().encode() + b" ")
    return f"{prefix}:{digest.hexdigest()[:24]}"


def positions_from_game(game: chess.pgn.Game, source_game_id: str, seed: int) -> list[dict[str, object]]:
    board = game.board()
    # Keep a few deterministic choices per phase. Creating a FEN and generating
    # every legal move at every ply makes a full historical PGN scan needlessly
    # expensive; eligibility is checked only for these bounded choices.
    choices: dict[str, list[tuple[float, str, int]]] = collections.defaultdict(list)
    for move in game.mainline_moves():
        board.push(move)
        if board.halfmove_clock >= 100:
            continue
        phase = classify_phase(board)
        priority = stable_unit(seed, source_game_id, phase, board.ply(), "within-game")
        bucket = choices[phase]
        if len(bucket) < 4 or priority < bucket[-1][0]:
            bucket.append((priority, board.fen(en_passant="fen"), board.ply()))
            bucket.sort(key=lambda item: item[0])
            del bucket[4:]
    rows = []
    for phase, candidates in choices.items():
        chosen = next(
            ((fen, ply) for _, fen, ply in candidates if eligible(chess.Board(fen))),
            None,
        )
        if chosen is None:
            continue
        fen, ply = chosen
        rows.append(
            {
                "fen": fen,
                "source_game_id": source_game_id,
                "source_ply": ply,
                "phase": phase,
            }
        )
    return rows


def add_pgn_candidates(
    paths: list[Path],
    source: str,
    reservoirs: dict[tuple[str, str], Reservoir],
    seed: int,
    min_tcec_season: int,
    min_human_elo: int,
    max_games: int,
) -> dict[str, int]:
    counts = collections.Counter()
    for path in paths:
        accepted = 0
        with open_pgn(path) as stream:
            while True:
                game = chess.pgn.read_game(stream)
                if game is None:
                    break
                counts["games_read"] += 1
                headers = game.headers
                weight = 1.0
                if source == "engine":
                    match = SEASON_RE.search(headers.get("Event", ""))
                    if not match or int(match.group(1)) < min_tcec_season:
                        continue
                    season = int(match.group(1))
                    # Recent elite competitions dominate; historical games remain
                    # available to provide style and architecture diversity.
                    weight = (2.0 if season >= 20 else 1.0) * (
                        3.0 if ELITE_EVENT_RE.search(headers.get("Event", "")) else 1.0
                    )
                else:
                    if headers.get("Variant", "Standard") != "Standard":
                        continue
                    if headers.get("WhiteTitle") == "BOT" or headers.get("BlackTitle") == "BOT":
                        continue
                    try:
                        if min(int(headers.get("WhiteElo", 0)), int(headers.get("BlackElo", 0))) < min_human_elo:
                            continue
                    except ValueError:
                        continue
                accepted += 1
                counts["games_accepted"] += 1
                identity = game_id("tcec" if source == "engine" else "lichess", game)
                rows = positions_from_game(game, identity, seed)
                for row in rows:
                    phase = str(row["phase"])
                    priority = -math.log(stable_unit(seed, identity, phase, source)) / weight
                    row["source_category"] = source
                    reservoirs[(source, phase)].add(row, priority)
                    counts[f"positions_{phase}"] += 1
                if max_games and accepted >= max_games:
                    break
        print(f"scanned {path}: {accepted:,} qualifying {source} games", flush=True)
    return dict(counts)


def hard_weight(row: dict[str, object]) -> float:
    score = 0
    score += 3 if bool(row.get("in_check")) else 0
    score += 2 if int(row.get("legal_moves", 99)) <= 4 else 0
    score += 2 if abs(int(row.get("material_balance", 0))) >= 5 else 0
    score += 1 if int(row.get("piece_count", 32)) <= 10 else 0
    score += 1 if int(row.get("halfmove_clock", 0)) >= 70 else 0
    return float(score)


def add_parquet_candidates(
    path: Path,
    reservoirs: dict[tuple[str, str], Reservoir],
    seed: int,
) -> dict[str, int]:
    dataset = ds.dataset(path, format="parquet", partitioning="hive")
    required = {
        "fen", "phase", "source_member", "game_number", "ply", "in_check",
        "legal_moves", "material_balance", "piece_count", "halfmove_clock",
    }
    missing = required - set(dataset.schema.names)
    if missing:
        raise RuntimeError(f"zero source lacks columns: {sorted(missing)}")
    counts = collections.Counter()
    for batch in dataset.scanner(columns=sorted(required), batch_size=32768).to_batches():
        for source_row in batch.to_pylist():
            phase = str(source_row["phase"])
            if phase not in PHASES:
                continue
            if int(source_row["legal_moves"]) <= 1 or int(source_row["halfmove_clock"]) >= 100:
                continue
            identity = f"ccrl:{source_row['source_member']}:{source_row['game_number']}"
            base = {
                "fen": str(source_row["fen"]),
                "source_game_id": identity,
                "source_ply": int(source_row["ply"]),
                "phase": phase,
            }
            zero = dict(base, source_category="zero")
            reservoirs[("zero", phase)].add(
                zero, -math.log(stable_unit(seed, identity, phase, "zero"))
            )
            counts["zero"] += 1
            weight = hard_weight(source_row)
            if weight:
                hard = dict(base, source_category="hard")
                reservoirs[("hard", phase)].add(
                    hard,
                    -math.log(stable_unit(seed, identity, phase, "hard")) / weight,
                )
                counts["hard"] += 1
    print(f"scanned {path}: {counts['zero']:,} zero and {counts['hard']:,} hard candidates", flush=True)
    return dict(counts)


def select_rows(
    reservoirs: dict[tuple[str, str], Reservoir],
    quotas: dict[tuple[str, str], int],
    seed: int,
    train_ratio: float,
    validation_ratio: float,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    # Hard first because it is the narrowest pool; broad pools yield duplicates to it.
    for source in ("hard", "human", "engine", "zero"):
        for phase in PHASES:
            need = quotas[(source, phase)]
            if need == 0:
                continue
            accepted = 0
            for row in reservoirs[(source, phase)].rows():
                key = canonical_key(str(row["fen"]))
                if key in seen:
                    continue
                seen.add(key)
                row["split"] = split_for(
                    str(row["source_game_id"]), seed, train_ratio, validation_ratio
                )
                selected.append(row)
                accepted += 1
                if accepted == need:
                    break
            if accepted != need:
                raise RuntimeError(
                    f"insufficient unique {source}/{phase} positions: {accepted} < {need}; "
                    "increase the corpus, lower a filter, or use a larger --reservoir-factor"
                )
    selected.sort(key=lambda row: (str(row["split"]), stable_unit(seed, "order", row["source_game_id"], row["source_ply"])))
    return selected


def run(args: argparse.Namespace) -> None:
    source_shares = {
        "zero": args.zero_share,
        "engine": args.engine_share,
        "human": args.human_share,
        "hard": args.hard_share,
    }
    global_phase_shares = {
        "opening": args.opening_share,
        "middlegame": args.middlegame_share,
        "endgame": args.endgame_share,
    }
    engine_phase_shares = {
        "opening": args.engine_opening_share,
        "middlegame": args.engine_middlegame_share,
        "endgame": args.engine_endgame_share,
    }
    source_totals = apportion(args.size, source_shares)
    global_phase_totals = apportion(args.size, global_phase_shares)
    quotas: dict[tuple[str, str], int] = {
        ("zero", phase): 0 for phase in PHASES
    }
    for source in ("engine", "human", "hard"):
        profile = engine_phase_shares if source == "engine" else global_phase_shares
        for phase, count in apportion(source_totals[source], profile).items():
            quotas[(source, phase)] = count
    residual = {}
    for phase in PHASES:
        residual[phase] = global_phase_totals[phase] - sum(
            quotas[(source, phase)] for source in ("engine", "human", "hard")
        )
    # Integer apportionment can over-allocate a phase for very small smoke tests.
    # Move cells between phases while preserving every non-zero source total.
    while any(value < 0 for value in residual.values()):
        over = next(phase for phase in PHASES if residual[phase] < 0)
        under = max(PHASES, key=lambda phase: residual[phase])
        if residual[under] <= 0:
            raise RuntimeError("unable to reconcile integer source/phase quotas")
        movable = [
            source for source in ("engine", "human", "hard")
            if quotas[(source, over)] > 0
        ]
        source = max(movable, key=lambda name: quotas[(name, over)])
        quotas[(source, over)] -= 1
        quotas[(source, under)] += 1
        residual[over] += 1
        residual[under] -= 1
    for phase in PHASES:
        quotas[("zero", phase)] = residual[phase]
    if sum(quotas[("zero", phase)] for phase in PHASES) != source_totals["zero"]:
        raise RuntimeError("quota rounding failed to preserve the zero-source total")
    reservoirs = {
        key: Reservoir(0 if count == 0 else max(count * args.reservoir_factor, count + 32))
        for key, count in quotas.items()
    }

    stats = {
        "zero_and_hard": add_parquet_candidates(Path(args.zero_parquet), reservoirs, args.seed),
        "engine": add_pgn_candidates(
            [Path(args.tcec_pgn)], "engine", reservoirs, args.seed,
            args.min_tcec_season, args.min_human_elo, args.max_games_per_pgn,
        ),
        "human": add_pgn_candidates(
            [Path(path) for path in args.lichess_pgn], "human", reservoirs, args.seed,
            args.min_tcec_season, args.min_human_elo, args.max_games_per_pgn,
        ),
    }
    rows = select_rows(
        reservoirs, quotas, args.seed, args.train_ratio, args.validation_ratio
    )
    table = pa.Table.from_pylist(rows, schema=PLAN_SCHEMA)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd", compression_level=6)
    temporary.replace(output)

    source_counts = collections.Counter(row["source_category"] for row in rows)
    phase_counts = collections.Counter(row["phase"] for row in rows)
    split_counts = collections.Counter(row["split"] for row in rows)
    manifest = {
        "schema_version": 1,
        "rows": len(rows),
        "seed": args.seed,
        "quotas": {f"{source}/{phase}": count for (source, phase), count in quotas.items()},
        "source_counts": dict(sorted(source_counts.items())),
        "phase_counts": dict(sorted(phase_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "split_unit": "source_game_id",
        "canonical_dedup_fields": "board turn castling en-passant",
        "min_tcec_season": args.min_tcec_season,
        "min_human_elo": args.min_human_elo,
        "inputs": {
            "zero_parquet": str(Path(args.zero_parquet).resolve()),
            "tcec_pgn": str(Path(args.tcec_pgn).resolve()),
            "lichess_pgn": [str(Path(path).resolve()) for path in args.lichess_pgn],
        },
        "scan_stats": stats,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), **manifest}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero-parquet", required=True)
    parser.add_argument("--tcec-pgn", required=True)
    parser.add_argument("--lichess-pgn", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--zero-share", type=float, default=0.30)
    parser.add_argument("--engine-share", type=float, default=0.50)
    parser.add_argument("--human-share", type=float, default=0.10)
    parser.add_argument("--hard-share", type=float, default=0.10)
    parser.add_argument("--opening-share", type=float, default=0.20)
    parser.add_argument("--middlegame-share", type=float, default=0.60)
    parser.add_argument("--endgame-share", type=float, default=0.20)
    parser.add_argument("--engine-opening-share", type=float, default=0.24)
    parser.add_argument("--engine-middlegame-share", type=float, default=0.52)
    parser.add_argument("--engine-endgame-share", type=float, default=0.24)
    parser.add_argument("--train-ratio", type=float, default=0.96)
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument("--min-tcec-season", type=int, default=0)
    parser.add_argument("--min-human-elo", type=int, default=2200)
    parser.add_argument("--reservoir-factor", type=int, default=4)
    parser.add_argument(
        "--max-games-per-pgn", type=int, default=0,
        help="qualifying-game scan cap per PGN; 0 scans all (smoke tests only)",
    )
    args = parser.parse_args()
    if args.size < 1 or args.reservoir_factor < 1 or args.max_games_per_pgn < 0:
        parser.error("size/reservoir factor must be positive and max-games nonnegative")
    if any(value < 0 for value in (
        args.zero_share, args.engine_share, args.human_share, args.hard_share,
        args.opening_share, args.middlegame_share, args.endgame_share,
        args.engine_opening_share, args.engine_middlegame_share, args.engine_endgame_share,
    )):
        parser.error("shares cannot be negative")
    if not math.isclose(args.train_ratio + args.validation_ratio, 0.98):
        parser.error("train-ratio + validation-ratio must equal 0.98 (test is fixed at 0.02)")
    for path in [args.zero_parquet, args.tcec_pgn, *args.lichess_pgn]:
        if not Path(path).exists():
            parser.error(f"missing input: {path}")
    try:
        run(args)
    except (KeyboardInterrupt, BrokenPipeError):
        print("interrupted before the plan was published", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
