"""
analyze_results.py
------------------
Parse the .txt result files produced by run.sh and print a consolidated
summary of every matchup together with per-agent totals.

Usage
-----
    python analyze_results.py
    python analyze_results.py base1_base2.txt agent_base.txt   # explicit files
"""

import re
import sys
import ast
from pathlib import Path
from dataclasses import dataclass

from oxono import Game, State

# ---------------------------------------------------------------------------
# Files emitted by run.sh  (order does not matter)
# ---------------------------------------------------------------------------
DEFAULT_FILES = [
    "base1_base2.txt",
    "base2_base1.txt",
    "agent_base.txt",
    "base_agent.txt",
    "agent_base2.txt",
    "base2_agent.txt",
]

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
RESULTS_HEADER = re.compile(r"=== Results over (\d+) game\(s\) ===")
AGENT_LINE     = re.compile(r"^\s+(\S+)\s+wins:\s+(\d+)\s+\([\d.]+%\)")
DRAWS_LINE     = re.compile(r"^\s+Draws:\s+(\d+)")


def parse_file(path: Path) -> dict | None:
    """
    Return the *final* result block from a manager output file as:
        {"p0": name, "p1": name, "wins_p0": int, "wins_p1": int,
         "draws": int, "n_games": int}
    or None if the file is empty / not yet generated.
    """
    if not path.exists():
        return None

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return None

    # Collect all "=== Results over N ===" blocks; keep the last one.
    blocks = list(RESULTS_HEADER.finditer(text))
    if not blocks:
        return None

    last = blocks[-1]
    n_games = int(last.group(1))
    snippet = text[last.end():]

    lines = snippet.splitlines()
    agent_data = []
    draws = 0

    for line in lines:
        if m := AGENT_LINE.match(line):
            agent_data.append((m.group(1), int(m.group(2))))
        elif m := DRAWS_LINE.match(line):
            draws = int(m.group(1))
        # Stop after we have found both agent lines + draws line
        if len(agent_data) == 2 and draws is not None:
            break

    if len(agent_data) < 2:
        return None

    return {
        "p0":       agent_data[0][0],
        "p1":       agent_data[1][0],
        "wins_p0":  agent_data[0][1],
        "wins_p1":  agent_data[1][1],
        "draws":    draws,
        "n_games":  n_games,
    }


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------
COL = 22


def pct(n, total):
    return f"{100 * n / total:.1f}%" if total else "n/a"


@dataclass
class TimeStats:
    total_time: float = 0.0
    moves: int = 0
    max_move: float = 0.0


def _parse_action_line(line: str):
    """
    Parse a log line of the form:
        ('O', (5, 2), (4, 2)), 299.9995
    Returns (action_tuple, remaining_time_float) or None for non-action lines.
    """
    if ", " not in line:
        return None
    action_raw, remaining_raw = line.rsplit(", ", 1)
    try:
        action = ast.literal_eval(action_raw)
        remaining = float(remaining_raw)
    except (ValueError, SyntaxError):
        return None
    return action, remaining


def parse_log_time_stats(log_file: Path) -> dict[int, TimeStats] | None:
    """
    Parse a single manager log file and return timing stats for players 0 and 1.
    """
    if not log_file.exists():
        return None

    lines = [ln.strip() for ln in log_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return None

    try:
        time_limit = float(lines[0])
    except ValueError:
        return None

    state = State()
    remaining = [time_limit, time_limit]
    stats = {0: TimeStats(), 1: TimeStats()}

    for line in lines[1:]:
        parsed = _parse_action_line(line)
        if parsed is None:
            break
        action, new_remaining = parsed

        current = Game.to_move(state)
        spent = remaining[current] - new_remaining
        if spent < 0:
            spent = 0.0

        s = stats[current]
        s.total_time += spent
        s.moves += 1
        s.max_move = max(s.max_move, spent)

        remaining[current] = new_remaining
        Game.apply(state, action)

    return stats


def parse_matchup_time_stats(log_dir: Path) -> dict[int, TimeStats] | None:
    """
    Aggregate timing stats over all log_*.txt files in a matchup log directory.
    """
    if not log_dir.exists() or not log_dir.is_dir():
        return None

    all_logs = sorted(log_dir.glob("log_*.txt"))
    if not all_logs:
        return None

    agg = {0: TimeStats(), 1: TimeStats()}
    parsed_any = False

    for log_file in all_logs:
        per_game = parse_log_time_stats(log_file)
        if per_game is None:
            continue
        parsed_any = True
        for p in (0, 1):
            agg[p].total_time += per_game[p].total_time
            agg[p].moves += per_game[p].moves
            agg[p].max_move = max(agg[p].max_move, per_game[p].max_move)

    return agg if parsed_any else None


def _fmt_sec(x: float) -> str:
    return f"{x:.3f}s"


def print_time_table(title: str, rows: list[tuple[str, TimeStats]]):
    print(f"\n  {title}")
    print(f"    {'Agent':<{COL}} {'Total':>10}  {'Moves':>6}  {'Avg/move':>10}  {'Max move':>10}")
    print(f"    {'-'*COL} {'-'*10:>10}  {'-'*6:>6}  {'-'*10:>10}  {'-'*10:>10}")
    for agent_name, s in rows:
        avg = (s.total_time / s.moves) if s.moves else 0.0
        print(
            f"    {agent_name:<{COL}} {_fmt_sec(s.total_time):>10}  {s.moves:>6}  "
            f"{_fmt_sec(avg):>10}  {_fmt_sec(s.max_move):>10}"
        )


def print_matchup(label: str, r: dict):
    n = r["n_games"]
    print(f"\n  {label}")
    print(f"    {'Agent':<{COL}} {'Wins':>6}  {'%':>7}")
    print(f"    {'-'*COL} {'------':>6}  {'-------':>7}")
    print(f"    {r['p0']:<{COL}} {r['wins_p0']:>6}  {pct(r['wins_p0'], n):>7}")
    print(f"    {r['p1']:<{COL}} {r['wins_p1']:>6}  {pct(r['wins_p1'], n):>7}")
    print(f"    {'Draws':<{COL}} {r['draws']:>6}  {pct(r['draws'], n):>7}")
    print(f"    Total games: {n}")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(results: list[dict]) -> dict[str, dict]:
    """
    Sum wins/draws/games per unique agent name across all parsed results.
    Returns  { agent_name: {"wins": int, "draws": int, "games": int} }
    """
    totals: dict[str, dict] = {}

    def add(name, wins, draws, games):
        if name not in totals:
            totals[name] = {"wins": 0, "draws": 0, "games": 0}
        totals[name]["wins"]  += wins
        totals[name]["draws"] += draws
        totals[name]["games"] += games

    for r in results:
        add(r["p0"], r["wins_p0"], r["draws"], r["n_games"])
        add(r["p1"], r["wins_p1"], r["draws"], r["n_games"])

    return totals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    files = [Path(f) for f in (sys.argv[1:] or DEFAULT_FILES)]

    parsed = {}
    timing_by_file = {}
    missing = []
    for f in files:
        r = parse_file(f)
        if r is None:
            missing.append(f.name)
        else:
            parsed[f.name] = r
            log_dir = f.with_suffix("")
            timing_by_file[f.name] = parse_matchup_time_stats(log_dir)

    print("=" * 60)
    print("  OXONO — Results summary")
    print("=" * 60)

    if missing:
        print(f"\n  [!] Files not found or empty: {', '.join(missing)}")

    if not parsed:
        print("\n  No results to display.")
        return

    # ---- Per-file matchup details -----------------------------------------
    print("\n--- Per-matchup details ---")
    for fname, r in parsed.items():
        print_matchup(fname, r)

    # ---- Combined matchups (same pair played twice, sides swapped) ---------
    # Group by frozenset of the two agent names
    from collections import defaultdict
    groups: dict[frozenset, list] = defaultdict(list)
    for r in parsed.values():
        key = frozenset([r["p0"], r["p1"]])
        groups[key].append(r)

    combined_pairs = {k: v for k, v in groups.items() if len(v) > 1}
    if combined_pairs:
        print("\n--- Combined (same pair, both sides) ---")
        for pair_set, rs in combined_pairs.items():
            agents = sorted(pair_set)
            a, b = agents[0], agents[1]
            wa = wb = da = db = n = 0
            for r in rs:
                n  += r["n_games"]
                if r["p0"] == a:
                    wa += r["wins_p0"]; wb += r["wins_p1"]
                else:
                    wa += r["wins_p1"]; wb += r["wins_p0"]
                da += r["draws"]
            label = f"{a}  vs  {b}  (combined)"
            fake_r = {"p0": a, "p1": b, "wins_p0": wa, "wins_p1": wb,
                      "draws": da // len(rs), "n_games": n}
            print_matchup(label, fake_r)

    # ---- Global per-agent totals ------------------------------------------
    totals = aggregate(list(parsed.values()))
    print("\n--- Global per-agent totals ---")
    print(f"\n  {'Agent':<{COL}} {'Wins':>6}  {'Draws':>6}  {'Games':>6}  {'Win%':>7}")
    print(f"  {'-'*COL} {'------':>6}  {'------':>6}  {'------':>6}  {'-------':>7}")
    for agent, t in sorted(totals.items(), key=lambda x: -x[1]["wins"]):
        print(
            f"  {agent:<{COL}} {t['wins']:>6}  {t['draws']:>6}  "
            f"{t['games']:>6}  {pct(t['wins'], t['games']):>7}"
        )

    # ---- Timing stats from log directories --------------------------------
    print("\n--- Timing stats (from logs) ---")
    global_time: dict[str, TimeStats] = {}
    had_any_timing = False

    for fname, r in parsed.items():
        tstats = timing_by_file.get(fname)
        if not tstats:
            continue
        had_any_timing = True

        p0 = r["p0"]
        p1 = r["p1"]
        print_time_table(f"{fname}", [(p0, tstats[0]), (p1, tstats[1])])

        for agent_name, s in ((p0, tstats[0]), (p1, tstats[1])):
            if agent_name not in global_time:
                global_time[agent_name] = TimeStats()
            gt = global_time[agent_name]
            gt.total_time += s.total_time
            gt.moves += s.moves
            gt.max_move = max(gt.max_move, s.max_move)

    if not had_any_timing:
        print("\n  No log directories found (expected folder names matching result file stems).")
    else:
        rows = sorted(global_time.items(), key=lambda x: x[1].total_time, reverse=True)
        print_time_table("Global per-agent timing", rows)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
