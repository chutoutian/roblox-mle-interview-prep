#!/usr/bin/env python3
"""Generate original, fictional Roblox-themed MLE practice data.

This is an original practice exercise, not a Roblox or CodeSignal question.
All players share one prediction cutoff: day_offset == 0. Labels describe the
following week; only sessions with day_offset < 0 belong in model features.
Run from any directory with Python 3; only the standard library is required.
"""

import csv
import math
import random
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "codesignal_roblox"
SEED = 42
PLAYER_COUNT = 1_000
ZERO_HISTORY_USER = "user_0001"
ALL_INVALID_USER = "user_0002"


def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-value))


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rng = random.Random(SEED)
    players = []
    historic_sessions = []

    for player_index in range(1, PLAYER_COUNT + 1):
        user_id = f"user_{player_index:04d}"
        account_age = min(1_460, 1 + int(rng.expovariate(1.0 / 180)))
        country = rng.choices(
            ["US", "GB", "CA", "BR", "IN"], weights=[40, 15, 15, 15, 15]
        )[0]
        device = rng.choices(
            ["mobile", "desktop", "tablet", "console"], weights=[45, 30, 15, 10]
        )[0]
        session_count = rng.randint(0, 10)
        if user_id == ZERO_HISTORY_USER:
            session_count = 0
        elif user_id == ALL_INVALID_USER:
            session_count = 3

        # User-level variation plus lognormal noise gives right-skewed minutes.
        typical_minutes = 28 * rng.lognormvariate(-0.05, 0.35)
        clean_sessions = []
        for _ in range(session_count):
            minutes = round(
                min(240.0, max(1.0, rng.lognormvariate(math.log(typical_minutes), 0.65))),
                2,
            )
            completed = int(rng.random() < 0.1 + 0.8 * sigmoid((minutes - 25) / 14))
            row = {
                "session_id": f"session_{len(historic_sessions) + 1:05d}",
                "user_id": user_id,
                "day_offset": rng.randint(-7, -1),
                "minutes_played": minutes,
                "completed": completed,
            }
            clean_sessions.append(row)
            historic_sessions.append(row)

        average_minutes = (
            sum(row["minutes_played"] for row in clean_sessions) / session_count
            if session_count else 0.0
        )
        completion_rate = (
            sum(row["completed"] for row in clean_sessions) / session_count
            if session_count else 0.0
        )
        device_effect = {"mobile": -0.08, "desktop": 0.15, "tablet": -0.10, "console": 0.07}
        log_odds = (
            -4.8
            + 0.33 * session_count
            + 0.03 * average_minutes
            + 0.7 * completion_rate
            + 0.25 * math.log1p(account_age / 30)
            + device_effect[device]
        )
        returned = int(rng.random() < sigmoid(log_odds))
        players.append({
            "user_id": user_id,
            "account_age_days": account_age,
            "country": country,
            "device": device,
            "returned_next_week": returned,
            # Intentionally unavailable at prediction time: explicit label leakage.
            "next_week_minutes": round(rng.uniform(5, 300), 2) if returned else 0.0,
        })

    # Inject imperfections only AFTER every label has been drawn from clean data.
    # Missingness in player features is independent of the outcome.
    for column, fraction in [("country", 0.05), ("device", 0.05), ("account_age_days", 0.03)]:
        for index in rng.sample(range(PLAYER_COUNT), round(PLAYER_COUNT * fraction)):
            players[index][column] = ""

    invalid_fixture = [
        index for index, row in enumerate(historic_sessions)
        if row["user_id"] == ALL_INVALID_USER
    ]
    eligible = [index for index in range(len(historic_sessions)) if index not in invalid_fixture]
    missing_count = round(len(historic_sessions) * 0.02)
    negative_count = round(len(historic_sessions) * 0.01)
    missing_indices = set(invalid_fixture[:2])
    missing_indices.update(rng.sample(eligible, missing_count - 2))
    negative_indices = {invalid_fixture[2]}
    negative_indices.update(rng.sample(
        [index for index in eligible if index not in missing_indices], negative_count - 1
    ))
    for index in missing_indices:
        historic_sessions[index]["minutes_played"] = ""
    for index in negative_indices:
        historic_sessions[index]["minutes_played"] = -1

    # These are exact copied rows, including session_id, after corruption.
    duplicate_sessions = [dict(row) for row in rng.sample(historic_sessions, 30)]

    # Distinct future rows are a cutoff-filtering exercise, not label inputs.
    # Only returning players can have them; durations fit their next-week totals.
    future_sessions = []
    returning_players = [player for player in players[2:] if player["returned_next_week"] == 1]
    for index, player in enumerate(rng.sample(returning_players, 30), start=1):
        future_sessions.append({
            "session_id": f"future_{index:03d}",
            "user_id": player["user_id"],
            "day_offset": rng.randint(0, 3),
            "minutes_played": min(
                round(rng.lognormvariate(math.log(30), 0.65), 2), player["next_week_minutes"]
            ),
            "completed": rng.randint(0, 1),
        })

    sessions = historic_sessions + duplicate_sessions + future_sessions
    rng.shuffle(sessions)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "players.csv", [
        "user_id", "account_age_days", "country", "device",
        "returned_next_week", "next_week_minutes",
    ], players)
    write_csv(OUTPUT_DIR / "sessions.csv", [
        "session_id", "user_id", "day_offset", "minutes_played", "completed",
    ], sessions)

    positives = sum(player["returned_next_week"] for player in players)
    historical_users = {row["user_id"] for row in historic_sessions}
    print(f"Wrote {OUTPUT_DIR / 'players.csv'}: {len(players):,} players")
    print(f"Positive labels: {positives:,} ({positives / PLAYER_COUNT:.1%})")
    print(f"Wrote {OUTPUT_DIR / 'sessions.csv'}: {len(sessions):,} session rows")
    print(f"Unique historic sessions: {len(historic_sessions):,}; exact duplicates: 30; future rows: 30")
    print(f"Historic missing durations: {missing_count}; negative sentinels: {negative_count}")
    print(f"Zero-history players: {PLAYER_COUNT - len(historical_users)}; guaranteed fixture: {ZERO_HISTORY_USER}")
    print(f"All-invalid-duration fixture: {ALL_INVALID_USER} (3 historic sessions; 0 valid durations)")


if __name__ == "__main__":
    main()
