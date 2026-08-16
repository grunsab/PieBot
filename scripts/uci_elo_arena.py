#!/usr/bin/env python3
"""Reproducible, resumable PieBot-versus-Stockfish UCI Elo arena.

The arena is deliberately sequential and uses one thread per engine. Every
opening is played twice with colors reversed. Results are atomically persisted
after each game, so an interrupted invocation can be resumed with the exact
same commands, binaries, model, options, openings, seed, and time control.

Install the Python dependency with::

    python -m pip install -r training/nnue/requirements.txt

Example::

    python scripts/uci_elo_arena.py \
      --piebot-command PieBot/target/release/uci \
      --piebot-nnue /absolute/path/to/nnue_quant.nnue \
      --piebot-blend 75 \
      --stockfish-command stockfish --stockfish-elo 2600 \
      --games 100 --time-control 60+0.5 \
      --openings-file gate_results.json \
      --results out/stockfish_2600_arena.json

``--openings-file`` accepts newline-delimited FEN/EPD positions, a JSON list
of positions, or a ``compare_play`` JSON result. For the latter, the final
position from each ``pairing.openings[].positions`` entry is extracted.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import queue
import random
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


STATE_SCHEMA = "piebot-uci-elo-arena-v1"
DEFAULT_TIME_CONTROL = "60+0.5"
DEFAULT_MAX_PLIES = 300
DEFAULT_GAME_WALL_TIME_S = 900.0
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_SEED = 0x5049_4542_4F54

PIEBOT_REQUIRED_OPTIONS = {
    "Threads",
    "Hash",
    "UseNNUE",
    "NNUEQuantFile",
    "EvalBlend",
}
STOCKFISH_REQUIRED_OPTIONS = {
    "Threads",
    "Hash",
    "UCI_LimitStrength",
    "UCI_Elo",
    "Ponder",
    "MultiPV",
    "SyzygyProbeLimit",
}
PYTHON_CHESS_MANAGED_OPTIONS = {"ponder", "multipv"}

# Balanced, mainstream positions are generated from this small built-in suite
# when no external opening file is supplied. External compare_play openings are
# preferred for a large production match.
BUILTIN_OPENING_LINES: tuple[tuple[str, ...], ...] = (
    ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"),
    ("e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6"),
    ("e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4", "g7g6"),
    ("e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6", "e4e5", "f6d7"),
    ("e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4", "c8f5"),
    ("d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7"),
    ("d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6"),
    ("d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "e2e3", "e8g8"),
    ("c2c4", "e7e5", "b1c3", "g8f6", "g1f3", "b8c6", "g2g3", "d7d5"),
    ("c2c4", "g8f6", "b1c3", "e7e5", "g1f3", "b8c6", "g2g3", "f8b4"),
    ("g1f3", "d7d5", "g2g3", "g8f6", "f1g2", "g7g6", "e1g1", "f8g7"),
    ("g1f3", "g8f6", "c2c4", "e7e6", "g2g3", "d7d5", "f1g2", "f8e7"),
)


@dataclasses.dataclass(frozen=True)
class GamePlan:
    game_index: int
    pair_index: int
    opening_id: str
    opening_fen: str
    piebot_color: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class GameSettings:
    initial_time_s: float = 60.0
    increment_s: float = 0.5
    max_plies: int = DEFAULT_MAX_PLIES
    game_wall_time_s: float = DEFAULT_GAME_WALL_TIME_S
    timeout_grace_s: float = 1.0

    def __post_init__(self) -> None:
        if self.initial_time_s <= 0:
            raise ValueError("initial time must be positive")
        if self.increment_s < 0:
            raise ValueError("increment must be non-negative")
        if self.max_plies <= 0:
            raise ValueError("max plies must be positive")
        if self.game_wall_time_s <= 0:
            raise ValueError("game wall-time cap must be positive")
        if self.timeout_grace_s < 0:
            raise ValueError("timeout grace must be non-negative")


class MoveDeadlineExceeded(TimeoutError):
    """Raised after an engine exceeds its chess clock/wall deadline."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return sha256_bytes(encoded)


def piebot_uci_options(
    model_path: Path, *, blend: int, hash_mb: int
) -> tuple[dict[str, Any], str]:
    if not 0 <= blend <= 100:
        raise ValueError("PieBot blend must be between 0 and 100")
    if hash_mb <= 0:
        raise ValueError("PieBot hash size must be positive")
    resolved = model_path.expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise ValueError(f"PieBot NNUE model is not a non-empty file: {resolved}")
    digest = sha256_file(resolved)
    # Dict insertion order matters: load the model before enabling NNUE.
    return (
        {
            "Threads": 1,
            "Hash": hash_mb,
            "NNUEQuantFile": str(resolved),
            "UseNNUE": True,
            "EvalBlend": blend,
        },
        digest,
    )


def stockfish_uci_options(
    *, elo: int, hash_mb: int, full_strength: bool = False
) -> dict[str, Any]:
    """UCI options for the anchor.

    ``full_strength=True`` turns UCI_LimitStrength OFF entirely. This is the
    only configuration that anchors to a rating anyone else can check: the
    limited-strength ladder is self-inconsistent -- the same binary and net
    measured 2146 at rungs 1800/2100 and 2422 at 2400/2700 on the same idle box,
    with disjoint CIs -- whereas full-strength Stockfish has a published
    CCRL 40/15 rating. See evidence/ladder_s6_controlled_and_rung_dependence_20260812.json.
    """
    if hash_mb <= 0:
        raise ValueError("Stockfish hash size must be positive")
    options: dict[str, Any] = {
        "Threads": 1,
        "Hash": hash_mb,
        "Ponder": False,
        "MultiPV": 1,
        "SyzygyProbeLimit": 0,
    }
    if full_strength:
        # UCI_Elo is ignored by Stockfish when LimitStrength is false, but leave
        # it out entirely so a stray value cannot be mistaken for a live rung.
        options["UCI_LimitStrength"] = False
        return options
    if elo <= 0:
        raise ValueError("Stockfish UCI_Elo must be positive")
    options["UCI_LimitStrength"] = True
    options["UCI_Elo"] = elo
    return options


def parse_time_control(raw: str) -> tuple[float, float]:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)\s*", raw)
    if not match:
        raise ValueError("time control must use INITIAL+INCREMENT seconds, for example 60+0.5")
    initial, increment = (float(match.group(1)), float(match.group(2)))
    if initial <= 0 or increment < 0:
        raise ValueError("time control must have positive initial time and non-negative increment")
    return initial, increment


def parse_command(raw: str) -> list[str]:
    command = shlex.split(raw)
    if not command:
        raise ValueError("engine command cannot be empty")
    executable = command[0]
    if os.sep in executable or (os.altsep and os.altsep in executable):
        resolved = Path(executable).expanduser().resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"engine executable is not a file: {resolved}")
        command[0] = str(resolved)
    else:
        found = shutil.which(executable)
        if not found:
            raise ValueError(f"engine executable was not found on PATH: {executable}")
        command[0] = str(Path(found).resolve())
    return command


def command_identity(command: Sequence[str]) -> dict[str, Any]:
    executable = Path(command[0]).resolve(strict=True)
    return {
        "command": list(command),
        "executable": str(executable),
        "executable_sha256": sha256_file(executable),
    }


def _deduplicate(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        value = value.strip()
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _opening_values_from_json(payload: Any) -> list[str]:
    if isinstance(payload, list):
        values: list[str] = []
        for item in payload:
            if isinstance(item, str):
                values.append(item)
            elif isinstance(item, dict) and isinstance(item.get("fen"), str):
                values.append(item["fen"])
        return values
    if not isinstance(payload, dict):
        return []

    pairing = payload.get("pairing", payload)
    openings = pairing.get("openings", []) if isinstance(pairing, dict) else []
    values = []
    if isinstance(openings, list):
        for opening in openings:
            if not isinstance(opening, dict):
                continue
            positions = opening.get("positions")
            if isinstance(positions, list) and positions and isinstance(positions[-1], str):
                values.append(positions[-1])
            elif isinstance(opening.get("fen"), str):
                values.append(opening["fen"])
    return values


def load_opening_fens(path: Path) -> list[str]:
    raw = path.expanduser().read_text(encoding="utf-8")
    values: list[str] = []
    try:
        values = _opening_values_from_json(json.loads(raw))
    except json.JSONDecodeError:
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("fen "):
                line = line[4:].strip()
            values.append(line)
    result = _deduplicate(values)
    if not result:
        raise ValueError(f"no opening FEN/EPD positions found in {path}")
    return result


def _chess_modules():
    try:
        import chess
        import chess.engine
    except ImportError as error:  # pragma: no cover - exercised in dependency-missing installs.
        raise RuntimeError(
            "python-chess is required; install training/nnue/requirements.txt"
        ) from error
    return chess, chess.engine


def normalize_opening_fen(value: str) -> str:
    chess, _ = _chess_modules()
    value = value.strip()
    try:
        board = chess.Board(value)
    except ValueError:
        try:
            board, _operations = chess.Board.from_epd(value)
        except ValueError as error:
            raise ValueError(f"invalid opening FEN/EPD: {value}") from error
    if board.is_game_over(claim_draw=True):
        raise ValueError(f"opening is already terminal: {value}")
    return board.fen()


def builtin_opening_fens(opening_plies: int = 8) -> list[str]:
    if opening_plies < 0:
        raise ValueError("opening plies must be non-negative")
    chess, _ = _chess_modules()
    positions: list[str] = []
    for line in BUILTIN_OPENING_LINES:
        board = chess.Board()
        for uci in line[:opening_plies]:
            try:
                board.push_uci(uci)
            except ValueError as error:  # Fail loudly if the embedded suite is edited incorrectly.
                raise RuntimeError(f"invalid built-in opening move {uci!r} in {line!r}") from error
        positions.append(board.fen())
    return _deduplicate(positions)


def build_game_plans(
    opening_fens: Sequence[str], *, games: int, seed: int
) -> list[GamePlan]:
    if games <= 0:
        raise ValueError("games must be positive")
    if games % 2:
        raise ValueError("paired arena requires an even game count")
    openings = _deduplicate(opening_fens)
    if not openings:
        raise ValueError("at least one opening is required")

    rng = random.Random(seed)
    pair_count = games // 2
    selected: list[str] = []
    while len(selected) < pair_count:
        cycle = list(openings)
        rng.shuffle(cycle)
        selected.extend(cycle)

    plans: list[GamePlan] = []
    for pair_index, fen in enumerate(selected[:pair_count]):
        opening_id = sha256_bytes(fen.encode("utf-8"))[:16]
        for return_game, piebot_color in enumerate(("white", "black")):
            plans.append(
                GamePlan(
                    game_index=pair_index * 2 + return_game,
                    pair_index=pair_index,
                    opening_id=opening_id,
                    opening_fen=fen,
                    piebot_color=piebot_color,
                )
            )
    return plans


def _state_plans(plans: Sequence[GamePlan]) -> list[dict[str, Any]]:
    return [plan.as_dict() for plan in plans]


def load_or_create_state(
    path: Path, config: Mapping[str, Any], plans: Sequence[GamePlan]
) -> dict[str, Any]:
    expected_config = dict(config)
    expected_config_sha = canonical_sha256(expected_config)
    expected_plans = _state_plans(plans)
    if not path.exists():
        now = utc_now()
        return {
            "schema": STATE_SCHEMA,
            "created_at": now,
            "updated_at": now,
            "config": expected_config,
            "config_sha256": expected_config_sha,
            "plans": expected_plans,
            "games": [],
        }

    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot resume invalid arena state {path}: {error}") from error
    if state.get("schema") != STATE_SCHEMA:
        raise ValueError(f"unsupported arena state schema in {path}")
    stored_config = state.get("config")
    if (
        state.get("config_sha256") != canonical_sha256(stored_config)
        or stored_config != expected_config
        or state.get("config_sha256") != expected_config_sha
    ):
        raise ValueError(
            "arena configuration changed; use the original configuration or a new results path"
        )
    if state.get("plans") != expected_plans:
        raise ValueError("arena opening/game plan changed; use a new results path")

    games = state.get("games")
    if not isinstance(games, list):
        raise ValueError("arena state games must be a list")
    plan_by_index = {plan.game_index: plan for plan in plans}
    seen: set[int] = set()
    for record in games:
        if not isinstance(record, dict) or not isinstance(record.get("game_index"), int):
            raise ValueError("arena state contains an invalid game record")
        game_index = record["game_index"]
        if game_index in seen or game_index not in plan_by_index:
            raise ValueError("arena state contains duplicate or unknown game indexes")
        seen.add(game_index)
        plan = plan_by_index[game_index]
        for key in ("pair_index", "opening_id", "piebot_color"):
            if record.get(key) != getattr(plan, key):
                raise ValueError(f"arena state game {game_index} does not match its plan")
        if float(record.get("piebot_score", -1)) not in (0.0, 0.5, 1.0):
            raise ValueError(f"arena state game {game_index} has an invalid score")
    return state


def save_state(path: Path, state: Mapping[str, Any]) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload["updated_at"] = utc_now()
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def pending_plans(state: Mapping[str, Any], plans: Sequence[GamePlan]) -> list[GamePlan]:
    completed = {int(record["game_index"]) for record in state.get("games", [])}
    return [plan for plan in plans if plan.game_index not in completed]


def logistic_elo(score_rate: float) -> float:
    if not 0.0 <= score_rate <= 1.0:
        raise ValueError("score rate must be between zero and one")
    if score_rate == 0.0:
        return -math.inf
    if score_rate == 1.0:
        return math.inf
    return 400.0 * math.log10(score_rate / (1.0 - score_rate))


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a percentile of an empty sample")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = probability * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = rank - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def summarize_results(
    records: Sequence[Mapping[str, Any]], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    scores = [float(record["piebot_score"]) for record in records]
    if any(score not in (0.0, 0.5, 1.0) for score in scores):
        raise ValueError("game scores must be 0, 0.5, or 1")
    wins = sum(score == 1.0 for score in scores)
    draws = sum(score == 0.5 for score in scores)
    losses = sum(score == 0.0 for score in scores)
    score_rate = sum(scores) / len(scores) if scores else 0.5

    by_pair: dict[int, dict[int, float]] = {}
    for record in records:
        by_pair.setdefault(int(record["pair_index"]), {})[
            int(record["game_index"])
        ] = float(record["piebot_score"])
    pair_totals = [
        sum(pair.values())
        for _pair_index, pair in sorted(by_pair.items())
        if len(pair) == 2
    ]
    pair_scores = [total / 2.0 for total in pair_totals]
    pair_score_counts = {f"{half_points / 2:.1f}": 0 for half_points in range(5)}
    for total in pair_totals:
        pair_score_counts[f"{total:.1f}"] += 1
    termination_counts: dict[str, int] = {}
    for record in records:
        termination = str(record.get("termination", "unknown"))
        termination_counts[termination] = termination_counts.get(termination, 0) + 1
    termination_counts = dict(sorted(termination_counts.items()))

    if pair_scores:
        rng = random.Random(seed)
        bootstrap = []
        for _ in range(bootstrap_samples):
            bootstrap.append(
                sum(rng.choice(pair_scores) for _ in pair_scores) / len(pair_scores)
            )
        bootstrap.sort()
        score_ci = [_percentile(bootstrap, 0.025), _percentile(bootstrap, 0.975)]
        elo_ci = [logistic_elo(score_ci[0]), logistic_elo(score_ci[1])]
    else:
        score_ci = [math.nan, math.nan]
        elo_ci = [math.nan, math.nan]

    return {
        "games": len(records),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score_points": sum(scores),
        "score_rate": score_rate,
        "elo_difference": logistic_elo(score_rate),
        "complete_pairs": len(pair_scores),
        "pair_score_counts": pair_score_counts,
        "pentanomial": list(pair_score_counts.values()),
        "termination_counts": termination_counts,
        "bootstrap_samples": bootstrap_samples,
        "score_95_ci": score_ci,
        "elo_95_ci": elo_ci,
    }


def _uci_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def run_uci_preflight(
    command: Sequence[str],
    options: Mapping[str, Any],
    *,
    required_options: set[str],
    failure_markers: Sequence[str] = (),
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    """Perform a raw UCI handshake and retain unsolicited ``info string`` lines.

    python-chess intentionally treats setoption as fire-and-forget. PieBot
    reports a failed NNUE load as an info string and still answers readyok, so a
    raw preflight is required to prevent an accidental PST fallback.
    """

    if timeout_s <= 0:
        raise ValueError("preflight timeout must be positive")
    try:
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except OSError as error:
        raise RuntimeError(f"failed to start UCI engine {command!r}: {error}") from error

    assert process.stdin is not None
    assert process.stdout is not None
    lines: queue.Queue[str | None] = queue.Queue()

    def reader() -> None:
        try:
            for line in process.stdout:
                lines.put(line.rstrip("\r\n"))
        finally:
            lines.put(None)

    reader_thread = threading.Thread(target=reader, daemon=True, name="uci-preflight-reader")
    reader_thread.start()
    transcript: list[str] = []

    def send(line: str) -> None:
        if "\n" in line or "\r" in line:
            raise ValueError("UCI command values cannot contain newlines")
        try:
            process.stdin.write(line + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError(f"UCI engine exited while sending {line!r}") from error

    def read_until(marker: str) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"UCI preflight timed out waiting for {marker}")
            try:
                line = lines.get(timeout=remaining)
            except queue.Empty as error:
                raise RuntimeError(f"UCI preflight timed out waiting for {marker}") from error
            if line is None:
                raise RuntimeError(
                    f"UCI engine exited before {marker}; transcript: {transcript[-10:]}"
                )
            transcript.append(line)
            if line.strip().casefold() == marker.casefold():
                return

    try:
        send("uci")
        read_until("uciok")
        advertised: set[str] = set()
        spin_ranges: dict[str, dict[str, int]] = {}
        for line in transcript:
            match = re.match(r"^option\s+name\s+(.+?)\s+type\s+", line, re.IGNORECASE)
            if match:
                advertised.add(match.group(1).strip())
            spin = re.match(
                r"^option\s+name\s+(.+?)\s+type\s+spin\s+.*?\bmin\s+(-?\d+)\s+max\s+(-?\d+)",
                line,
                re.IGNORECASE,
            )
            if spin:
                spin_ranges[spin.group(1).strip()] = {
                    "min": int(spin.group(2)),
                    "max": int(spin.group(3)),
                }
        advertised_folded = {name.casefold() for name in advertised}
        missing = sorted(name for name in required_options if name.casefold() not in advertised_folded)
        if missing:
            raise RuntimeError(f"UCI engine is missing required options: {', '.join(missing)}")

        # Engines silently clamp out-of-range spin values (Stockfish advertises
        # UCI_Elo ~1320-3190); a clamped rung would corrupt the ladder
        # calibration without any visible failure, so reject it here.
        spin_ranges_folded = {name.casefold(): bounds for name, bounds in spin_ranges.items()}
        for name, value in options.items():
            bounds = spin_ranges_folded.get(name.casefold())
            if bounds is None or isinstance(value, bool) or not isinstance(value, int):
                continue
            if value < bounds["min"] or value > bounds["max"]:
                raise RuntimeError(
                    f"requested {name}={value} is outside the engine's advertised "
                    f"range [{bounds['min']}, {bounds['max']}] and would be "
                    "silently clamped"
                )

        for name, value in options.items():
            send(f"setoption name {name} value {_uci_value(value)}")
        send("isready")
        read_until("readyok")
        folded_transcript = "\n".join(transcript).casefold()
        for marker in failure_markers:
            if marker.casefold() in folded_transcript:
                matching = next(
                    (line for line in transcript if marker.casefold() in line.casefold()), marker
                )
                raise RuntimeError(matching)
        return {
            "id": [line for line in transcript if line.lower().startswith("id ")],
            "advertised_options": sorted(advertised),
            "spin_ranges": spin_ranges,
        }
    finally:
        if process.poll() is None:
            try:
                send("quit")
            except (RuntimeError, ValueError):
                pass
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        reader_thread.join(timeout=1.0)
        process.stdin.close()
        process.stdout.close()


def _safe_close_engine(engine: Any) -> None:
    try:
        engine.close()
    except Exception:
        pass


def _engine_play_with_timeout(
    engine: Any, board: Any, limit: Any, *, game: Any, timeout_s: float
) -> Any:
    completed: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            completed.put((True, engine.play(board, limit, game=game)))
        except Exception as error:
            completed.put((False, error))

    worker = threading.Thread(target=invoke, daemon=True, name="uci-engine-play")
    worker.start()
    worker.join(max(timeout_s, 0.0))
    if worker.is_alive():
        _safe_close_engine(engine)
        raise MoveDeadlineExceeded(f"engine exceeded {timeout_s:.3f}s move deadline")
    ok, value = completed.get_nowait()
    if ok:
        return value
    raise value


def _piebot_score_for_winner(winner: Any, piebot_color: str, chess: Any) -> float:
    if winner is None:
        return 0.5
    piebot_is_white = piebot_color == "white"
    return 1.0 if bool(winner == chess.WHITE) == piebot_is_white else 0.0


def _base_record(plan: GamePlan) -> dict[str, Any]:
    return {
        **plan.as_dict(),
        "started_at": utc_now(),
    }


def startup_forfeit_record(plan: GamePlan, failed_engine: str, error: Exception) -> dict[str, Any]:
    if failed_engine not in ("piebot", "stockfish"):
        raise ValueError("failed engine must be piebot or stockfish")
    record = _base_record(plan)
    record.update(
        {
            "completed_at": utc_now(),
            "duration_s": 0.0,
            "plies": 0,
            "played_plies": 0,
            "piebot_score": 0.0 if failed_engine == "piebot" else 1.0,
            "termination": f"{failed_engine}_start_failure",
            "error": str(error)[:500],
            "moves": [],
            "final_fen": plan.opening_fen,
        }
    )
    return record


def play_game(
    plan: GamePlan,
    settings: GameSettings,
    *,
    piebot_engine: Any,
    stockfish_engine: Any,
) -> dict[str, Any]:
    chess, chess_engine = _chess_modules()
    board = chess.Board(plan.opening_fen)
    piebot_is_white = plan.piebot_color == "white"
    clocks = {chess.WHITE: settings.initial_time_s, chess.BLACK: settings.initial_time_s}
    game_started = time.monotonic()
    wall_deadline = game_started + settings.game_wall_time_s
    opening_ply = board.ply()
    moves: list[str] = []
    record = _base_record(plan)

    def finish(score: float, termination: str, error: Exception | None = None) -> dict[str, Any]:
        duration = time.monotonic() - game_started
        record.update(
            {
                "completed_at": utc_now(),
                "duration_s": round(duration, 6),
                "plies": opening_ply + len(moves),
                "played_plies": len(moves),
                "piebot_score": score,
                "termination": termination,
                "moves": moves,
                "final_fen": board.fen(),
                "final_clocks_s": {
                    "white": round(clocks[chess.WHITE], 6),
                    "black": round(clocks[chess.BLACK], 6),
                },
            }
        )
        if error is not None:
            record["error"] = str(error)[:500]
        return record

    game_token = f"piebot-arena-{plan.game_index}"
    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            return finish(
                _piebot_score_for_winner(outcome.winner, plan.piebot_color, chess),
                f"chess_{outcome.termination.name.lower()}",
            )
        if opening_ply + len(moves) >= settings.max_plies:
            return finish(0.5, "max_plies")
        wall_remaining = wall_deadline - time.monotonic()
        if wall_remaining <= 0:
            return finish(0.5, "game_wall_time_cap")

        turn = board.turn
        is_piebot_turn = bool(turn == chess.WHITE) == piebot_is_white
        engine_name = "piebot" if is_piebot_turn else "stockfish"
        engine = piebot_engine if is_piebot_turn else stockfish_engine
        remaining = clocks[turn]
        if remaining <= 0:
            return finish(0.0 if is_piebot_turn else 1.0, f"{engine_name}_time_forfeit")

        limit = chess_engine.Limit(
            white_clock=clocks[chess.WHITE],
            black_clock=clocks[chess.BLACK],
            white_inc=settings.increment_s,
            black_inc=settings.increment_s,
        )
        allowed = min(remaining + settings.timeout_grace_s, wall_remaining)
        move_started = time.monotonic()
        try:
            result = _engine_play_with_timeout(
                engine, board, limit, game=game_token, timeout_s=allowed
            )
        except MoveDeadlineExceeded as error:
            return finish(
                0.0 if is_piebot_turn else 1.0,
                f"{engine_name}_time_forfeit",
                error,
            )
        except Exception as error:
            return finish(
                0.0 if is_piebot_turn else 1.0,
                f"{engine_name}_engine_crash",
                error,
            )

        elapsed = time.monotonic() - move_started
        if elapsed > remaining + settings.timeout_grace_s:
            return finish(
                0.0 if is_piebot_turn else 1.0,
                f"{engine_name}_time_forfeit",
            )
        move = getattr(result, "move", None)
        if move is None or move not in board.legal_moves:
            return finish(
                0.0 if is_piebot_turn else 1.0,
                f"{engine_name}_invalid_move",
            )
        clocks[turn] = max(0.0, remaining - elapsed) + settings.increment_s
        moves.append(move.uci())
        board.push(move)


def _open_python_chess_engine(
    command: Sequence[str],
    options: Mapping[str, Any],
    *,
    startup_timeout_s: float,
    command_timeout_s: float,
) -> Any:
    _chess, chess_engine = _chess_modules()
    engine = chess_engine.SimpleEngine.popen_uci(list(command), timeout=startup_timeout_s)
    try:
        # Ponder and MultiPV are managed by python-chess and configure() rejects
        # them. They are explicitly pinned during raw preflight; python-chess's
        # play() path itself uses ponder=False and a single PV.
        configurable = {
            name: value
            for name, value in options.items()
            if name.casefold() not in PYTHON_CHESS_MANAGED_OPTIONS
        }
        engine.configure(configurable)
        # SimpleEngine otherwise reuses its startup timeout for play(). With a
        # clock Limit (no movetime), that would incorrectly kill legal thinks
        # after the default 10 seconds. The outer per-move watchdog remains the
        # authoritative dynamic deadline.
        engine.timeout = command_timeout_s
    except Exception:
        _safe_close_engine(engine)
        raise
    return engine


def play_isolated_game(
    plan: GamePlan,
    settings: GameSettings,
    *,
    piebot_command: Sequence[str],
    piebot_options: Mapping[str, Any],
    stockfish_command: Sequence[str],
    stockfish_options: Mapping[str, Any],
    startup_timeout_s: float,
) -> dict[str, Any]:
    try:
        piebot = _open_python_chess_engine(
            piebot_command,
            piebot_options,
            startup_timeout_s=startup_timeout_s,
            command_timeout_s=settings.game_wall_time_s + settings.timeout_grace_s,
        )
    except Exception as error:
        return startup_forfeit_record(plan, "piebot", error)
    try:
        stockfish = _open_python_chess_engine(
            stockfish_command,
            stockfish_options,
            startup_timeout_s=startup_timeout_s,
            command_timeout_s=settings.game_wall_time_s + settings.timeout_grace_s,
        )
    except Exception as error:
        _safe_close_engine(piebot)
        return startup_forfeit_record(plan, "stockfish", error)

    try:
        return play_game(
            plan,
            settings,
            piebot_engine=piebot,
            stockfish_engine=stockfish,
        )
    finally:
        _safe_close_engine(piebot)
        _safe_close_engine(stockfish)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return None
        return "+Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _format_elo(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return f"{value:+.1f}"


def print_summary(summary: Mapping[str, Any], *, stockfish_elo: int) -> None:
    difference = float(summary["elo_difference"])
    ci = summary["elo_95_ci"]
    estimate = stockfish_elo + difference
    estimate_text = _format_elo(estimate).lstrip("+") if math.isfinite(estimate) else _format_elo(estimate)
    print(
        f"score={summary['score_rate']:.3%} "
        f"W-D-L={summary['wins']}-{summary['draws']}-{summary['losses']} "
        f"PieBot-SF={_format_elo(difference)} Elo "
        f"paired-bootstrap-95%=[{_format_elo(float(ci[0]))}, {_format_elo(float(ci[1]))}] "
        f"estimated-PieBot-Elo={estimate_text} vs SF UCI_Elo {stockfish_elo}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--piebot-command",
        default="PieBot/target/release/uci",
        help="PieBot UCI command (shell quoting is honored)",
    )
    parser.add_argument("--piebot-nnue", type=Path, required=True, help="quantized NNUE model")
    parser.add_argument("--piebot-blend", type=int, default=100, help="EvalBlend, 0..100")
    parser.add_argument("--piebot-hash", type=int, default=64, help="PieBot hash MiB")
    parser.add_argument("--stockfish-command", default="stockfish", help="Stockfish command")
    parser.add_argument("--stockfish-elo", type=int, default=2500, help="Stockfish UCI_Elo")
    parser.add_argument("--stockfish-hash", type=int, default=64, help="Stockfish hash MiB")
    parser.add_argument("--games", type=int, default=100, help="even game count")
    parser.add_argument(
        "--time-control", default=DEFAULT_TIME_CONTROL, help="Fischer INITIAL+INCREMENT seconds"
    )
    parser.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    parser.add_argument(
        "--game-wall-time",
        type=float,
        default=DEFAULT_GAME_WALL_TIME_S,
        help="hard wall-time cap per game in seconds",
    )
    parser.add_argument("--timeout-grace", type=float, default=1.0)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument(
        "--openings-file",
        type=Path,
        help="FEN/EPD list, JSON position list, or compare_play JSON output",
    )
    parser.add_argument(
        "--opening-plies",
        type=int,
        default=8,
        help="plies retained from each built-in opening when no file is supplied",
    )
    parser.add_argument("--results", type=Path, required=True, help="atomic resumable JSON state")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        initial_time, increment = parse_time_control(args.time_control)
        settings = GameSettings(
            initial_time_s=initial_time,
            increment_s=increment,
            max_plies=args.max_plies,
            game_wall_time_s=args.game_wall_time,
            timeout_grace_s=args.timeout_grace,
        )
        if args.bootstrap_samples <= 0:
            raise ValueError("bootstrap samples must be positive")
        if args.startup_timeout <= 0:
            raise ValueError("startup timeout must be positive")

        piebot_command = parse_command(args.piebot_command)
        stockfish_command = parse_command(args.stockfish_command)
        piebot_options, model_sha = piebot_uci_options(
            args.piebot_nnue, blend=args.piebot_blend, hash_mb=args.piebot_hash
        )
        stockfish_options = stockfish_uci_options(
            elo=args.stockfish_elo,
            hash_mb=args.stockfish_hash,
            full_strength=args.stockfish_full_strength,
        )

        raw_openings = (
            load_opening_fens(args.openings_file)
            if args.openings_file
            else builtin_opening_fens(args.opening_plies)
        )
        openings = _deduplicate(normalize_opening_fen(fen) for fen in raw_openings)
        plans = build_game_plans(openings, games=args.games, seed=args.seed)

        piebot_preflight = run_uci_preflight(
            piebot_command,
            piebot_options,
            required_options=PIEBOT_REQUIRED_OPTIONS,
            failure_markers=("failed to load NNUEQuantFile",),
            timeout_s=args.startup_timeout,
        )
        stockfish_preflight = run_uci_preflight(
            stockfish_command,
            stockfish_options,
            required_options=STOCKFISH_REQUIRED_OPTIONS,
            timeout_s=args.startup_timeout,
        )

        config = {
            "piebot": {
                **command_identity(piebot_command),
                "uci_id": piebot_preflight["id"],
                "options": piebot_options,
                "model_sha256": model_sha,
            },
            "stockfish": {
                **command_identity(stockfish_command),
                "uci_id": stockfish_preflight["id"],
                "options": stockfish_options,
            },
            "games": args.games,
            "time_control": {
                "initial_s": initial_time,
                "increment_s": increment,
            },
            "max_plies": args.max_plies,
            "game_wall_time_s": args.game_wall_time,
            "timeout_grace_s": args.timeout_grace,
            "concurrency": 1,
            "seed": args.seed,
            "bootstrap_samples": args.bootstrap_samples,
            "openings_sha256": canonical_sha256(openings),
            "openings_count": len(openings),
            "pairing": "same-position-colors-reversed-v1",
        }
        state = load_or_create_state(args.results, config, plans)
        remaining = pending_plans(state, plans)
        print(
            f"arena: {len(state['games'])}/{args.games} complete, {len(remaining)} remaining; "
            f"model_sha256={model_sha}; results={args.results}"
        )

        model_path = Path(str(piebot_options["NNUEQuantFile"]))
        for plan in remaining:
            if sha256_file(model_path) != model_sha:
                raise RuntimeError("PieBot NNUE model changed during the arena")
            record = play_isolated_game(
                plan,
                settings,
                piebot_command=piebot_command,
                piebot_options=piebot_options,
                stockfish_command=stockfish_command,
                stockfish_options=stockfish_options,
                startup_timeout_s=args.startup_timeout,
            )
            state["games"].append(record)
            state["games"].sort(key=lambda game: int(game["game_index"]))
            summary = summarize_results(
                state["games"], bootstrap_samples=args.bootstrap_samples, seed=args.seed
            )
            state["summary"] = _json_safe(summary)
            save_state(args.results, state)
            print(
                f"game {plan.game_index + 1}/{args.games} pair={plan.pair_index + 1} "
                f"PieBot={plan.piebot_color} score={record['piebot_score']} "
                f"termination={record['termination']} plies={record['plies']}"
            )

        summary = summarize_results(
            state["games"], bootstrap_samples=args.bootstrap_samples, seed=args.seed
        )
        state["summary"] = _json_safe(summary)
        save_state(args.results, state)
        print_summary(summary, stockfish_elo=args.stockfish_elo)
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
