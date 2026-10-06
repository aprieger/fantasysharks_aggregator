#!/usr/bin/env python3
"""
FantasySharks weekly projection aggregator.

Pulls the weekly projections for every scoring-system option, averages each
player's rank and projected fantasy points across all of them, and writes a
plain-text report grouped by position and sorted by average rank.

Usage:
    python fantasysharks_aggregator.py
    python fantasysharks_aggregator.py -o week5.txt --delay 2
    python fantasysharks_aggregator.py --positions QB PK
    python fantasysharks_aggregator.py --csv projections.csv

Requires: requests, beautifulsoup4, lxml
    pip install requests beautifulsoup4 lxml
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean

import requests
from bs4 import BeautifulSoup

URL_TEMPLATE = (
    "https://www.fantasysharks.com/apps/Projections/WeeklyProjections.php"
    "?l={league}&pos={pos}&RosterSize=&SalaryCap=&Comments=false"
)

# option value -> label, straight from the site's <select>
LEAGUES = {
    "9": "BestBall10s",
    "11": "CBS Sports",
    "4": "Distance PPR",
    "16": "Draft Kings",
    "20": "Draft Predictor",
    "10": "ESPN",
    "17": "FanDuel",
    "7": "Fantasy Insights",
    "21": "FFPC",
    "13": "NFL.com",
    "14": "RTSports",
    "15": "USA Today",
    "12": "Yahoo!",
}

POSITIONS = ["QB", "FLEX", "PK", "D"]

# What each position group is called in the report. FLEX rows carry their own
# Pos column (RB/WR/TE), so they get split out unless --no-split-flex is used.
POSITION_ORDER = ["QB", "RB", "WR", "TE", "FLEX", "PK", "D"]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class Player:
    """One player's numbers accumulated across every scoring system."""

    name: str
    position: str
    teams: list = field(default_factory=list)
    ranks: list = field(default_factory=list)
    points: list = field(default_factory=list)
    sources: list = field(default_factory=list)

    @property
    def team(self) -> str:
        if not self.teams:
            return "--"
        # Most frequently reported team wins; sources occasionally disagree.
        return max(set(self.teams), key=self.teams.count)

    @property
    def avg_rank(self) -> float:
        return mean(self.ranks) if self.ranks else float("inf")

    @property
    def avg_points(self) -> float:
        return mean(self.points) if self.points else 0.0

    @property
    def n_sources(self) -> int:
        return len(self.sources)


# --------------------------------------------------------------------------
# Fetching and parsing
# --------------------------------------------------------------------------

def fetch(session: requests.Session, league: str, pos: str,
          retries: int = 3, timeout: int = 30) -> str | None:
    url = URL_TEMPLATE.format(league=league, pos=pos)
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            if attempt == retries:
                print(f"    ! failed after {retries} tries: {exc}", file=sys.stderr)
                return None
            wait = 2 ** attempt
            print(f"    . retry {attempt}/{retries - 1} in {wait}s ({exc})",
                  file=sys.stderr)
            time.sleep(wait)
    return None


def cell_text(cell) -> str:
    return re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).strip()


FIELD_ALIASES = {
    "rank": {"rank", "#", "no", "no."},
    "pos": {"pos", "position"},
    "name": {"name", "player"},
    "team": {"team", "tm"},          # 'Vs' is the opponent - deliberately excluded
    "points": {"fantasy points", "points", "pts", "fpts"},
}


def header_rows(table):
    """The <tr>s that make up the header block, i.e. everything before data."""
    rows = []
    for row in table.find_all("tr"):
        if row.find("td") is not None:
            break
        if row.find("th") is not None:
            rows.append(row)
    return rows


def build_column_map(table) -> dict[str, int]:
    """Map logical fields to *data-row* column indices.

    The header is a grid, not a list: group headers like 'Rushing' carry
    colspan=3 while 'Rank' carries rowspan=2, so a header cell's position in
    its own <tr> tells you nothing about which data column it labels. This
    walks the grid, honouring both spans, and lets the narrowest cell covering
    a column win (the leaf 'Att' beats the group 'Rushing').
    """
    labels: dict[int, tuple[str, int]] = {}
    occupied: set[tuple[int, int]] = set()

    for r, row in enumerate(header_rows(table)):
        col = 0
        for cell in row.find_all(["th", "td"]):
            while (r, col) in occupied:
                col += 1
            try:
                colspan = max(1, int(cell.get("colspan", 1)))
                rowspan = max(1, int(cell.get("rowspan", 1)))
            except (TypeError, ValueError):
                colspan = rowspan = 1
            text = cell_text(cell).lower()

            for rr in range(r, r + rowspan):
                for cc in range(col, col + colspan):
                    occupied.add((rr, cc))
            for cc in range(col, col + colspan):
                prev = labels.get(cc)
                if prev is None or colspan < prev[1]:
                    labels[cc] = (text, colspan)
            col += colspan

    cols: dict[str, int] = {}
    for idx, (label, _span) in sorted(labels.items()):
        for field_name, aliases in FIELD_ALIASES.items():
            if label in aliases and field_name not in cols:
                cols[field_name] = idx
    return cols


def resolve_row_columns(cells, cols: dict[str, int]) -> dict[str, int]:
    """Repair the column map against an actual data row.

    Used when the header is missing, truncated, or points past the end of the
    row. The structural cues are stable across all four position tables: rank
    is first, the name cell carries class="playerLink", team follows the name,
    and fantasy points is last.
    """
    resolved = {k: v for k, v in cols.items() if v < len(cells)}

    name_idx = resolved.get("name")
    if name_idx is None or not cell_text(cells[name_idx]):
        for i, cell in enumerate(cells):
            classes = cell.get("class") or []
            if "playerLink" in classes or cell.find(
                "a", href=re.compile(r"playerpage", re.I)
            ):
                resolved["name"] = i
                break

    resolved.setdefault("rank", 0)
    resolved.setdefault("points", len(cells) - 1)
    if "name" in resolved and "team" not in resolved:
        team_idx = resolved["name"] + 1
        if team_idx < len(cells):
            resolved["team"] = team_idx
    return resolved


def normalize_name(raw: str) -> str:
    """'Allen, Josh' -> 'Josh Allen'; strip trailing injury/status tags."""
    name = re.sub(r"\s+", " ", raw).strip()
    name = re.sub(r"\s*\((?:Q|D|O|IR|PUP|SUS|NA)\)\s*$", "", name, flags=re.I)
    name = name.strip(" .,*")
    if "," in name:
        last, _, first = name.partition(",")
        name = f"{first.strip()} {last.strip()}".strip()
    return name


def to_float(raw: str) -> float | None:
    cleaned = raw.replace(",", "").replace("+", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    return float(match.group()) if match else None


def parse_table(html: str, requested_pos: str, split_flex: bool):
    """Yield (position, name, team, rank, points) tuples from one page."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id="toolData")
    if table is None:
        return

    base_cols = build_column_map(table)
    fallback_rank = 0

    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 3:
            continue  # header repeats, ad rows, spacers

        cols = resolve_row_columns(cells, base_cols)
        if "name" not in cols:
            continue

        values = [cell_text(c) for c in cells]
        name = normalize_name(values[cols["name"]])
        if not name or name.lower() in {"player", "team", "name", "rank"}:
            continue

        points = to_float(values[cols["points"]])
        if points is None:
            # Header said one thing, the row says another - try the last cell.
            points = to_float(values[-1])
        if points is None:
            continue

        rank_val = to_float(values[cols["rank"]]) if "rank" in cols else None
        rank = int(rank_val) if rank_val is not None else None
        if rank is None:
            fallback_rank += 1
            rank = fallback_rank
        else:
            fallback_rank = rank

        team = values[cols["team"]] if "team" in cols else name
        team = team.upper()[:4] if team else "--"

        position = requested_pos
        if requested_pos == "FLEX" and split_flex:
            pos_idx = cols.get("pos", 1)  # Pos sits between Rank and Name
            if pos_idx < len(values):
                raw_pos = values[pos_idx].upper()
                if raw_pos in {"RB", "WR", "TE"}:
                    position = raw_pos

        yield position, name, team, rank, points


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def aggregate(positions, leagues, delay: float, split_flex: bool):
    players: dict[tuple[str, str], Player] = {}
    session = requests.Session()

    for pos in positions:
        print(f"[{pos}]")
        for league_id, league_name in leagues.items():
            print(f"  -> {league_name} (l={league_id})")
            html = fetch(session, league_id, pos)
            if html is None:
                continue

            found = 0
            for p_pos, name, team, rank, points in parse_table(html, pos, split_flex):
                key = (p_pos, name.lower())
                player = players.get(key)
                if player is None:
                    player = Player(name=name, position=p_pos)
                    players[key] = player
                player.teams.append(team)
                player.ranks.append(rank)
                player.points.append(points)
                player.sources.append(league_name)
                found += 1

            if found == 0:
                print("    ! no rows parsed (page layout may have changed)",
                      file=sys.stderr)
            else:
                print(f"     {found} rows")
            time.sleep(delay)

    return players


def group_and_sort(players, min_sources: int):
    grouped: dict[str, list[Player]] = defaultdict(list)
    for player in players.values():
        if player.n_sources >= min_sources:
            grouped[player.position].append(player)
    for position in grouped:
        grouped[position].sort(key=lambda p: (p.avg_rank, -p.avg_points))
    return grouped


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def write_report(grouped, path: str, total_sources: int) -> None:
    col_widths = (5, 26, 6, 10, 12, 8)
    headers = ("Rank", "Player", "Team", "Avg Rank", "Avg Pts", "Sources")
    header_line = "".join(h.ljust(w) for h, w in zip(headers, col_widths)).rstrip()
    divider = "-" * len(header_line)

    lines = [
        "FANTASYSHARKS WEEKLY PROJECTIONS - AGGREGATED",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Scoring systems polled: {total_sources}",
        "",
    ]

    ordered = [p for p in POSITION_ORDER if p in grouped]
    ordered += [p for p in sorted(grouped) if p not in POSITION_ORDER]

    for position in ordered:
        roster = grouped[position]
        lines.append("=" * len(header_line))
        lines.append(f"{position}  ({len(roster)} players)")
        lines.append("=" * len(header_line))
        lines.append(header_line)
        lines.append(divider)
        for i, p in enumerate(roster, start=1):
            row = (
                str(i),
                p.name[: col_widths[1] - 1],
                p.team,
                f"{p.avg_rank:.2f}",
                f"{p.avg_points:.2f}",
                str(p.n_sources),
            )
            lines.append("".join(v.ljust(w) for v, w in zip(row, col_widths)).rstrip())
        lines.append("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\nWrote {path}")


def write_csv(grouped, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Position", "PosRank", "Player", "Team",
                         "AvgRank", "AvgPts", "Sources"])
        ordered = [p for p in POSITION_ORDER if p in grouped]
        ordered += [p for p in sorted(grouped) if p not in POSITION_ORDER]
        for position in ordered:
            for i, p in enumerate(grouped[position], start=1):
                writer.writerow([position, i, p.name, p.team,
                                 f"{p.avg_rank:.2f}", f"{p.avg_points:.2f}",
                                 p.n_sources])
    print(f"Wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", default="fantasysharks_weekly.txt",
                        help="text report path (default: fantasysharks_weekly.txt)")
    parser.add_argument("--csv", help="also write a CSV to this path")
    parser.add_argument("--positions", nargs="+", default=POSITIONS,
                        choices=POSITIONS, help="positions to pull")
    parser.add_argument("--leagues", nargs="+",
                        help="limit to these option values, e.g. --leagues 9 10 12")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="seconds between requests (default: 1.5)")
    parser.add_argument("--min-sources", type=int, default=1,
                        help="drop players appearing in fewer than N systems")
    parser.add_argument("--no-split-flex", action="store_true",
                        help="keep RB/WR/TE lumped together as FLEX")
    args = parser.parse_args()

    leagues = LEAGUES
    if args.leagues:
        leagues = {k: v for k, v in LEAGUES.items() if k in set(args.leagues)}
        if not leagues:
            print("No valid league option values given.", file=sys.stderr)
            return 1

    players = aggregate(args.positions, leagues, args.delay, not args.no_split_flex)
    if not players:
        print("Nothing collected - check connectivity or the page layout.",
              file=sys.stderr)
        return 1

    grouped = group_and_sort(players, args.min_sources)
    write_report(grouped, args.output, len(leagues))
    if args.csv:
        write_csv(grouped, args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())