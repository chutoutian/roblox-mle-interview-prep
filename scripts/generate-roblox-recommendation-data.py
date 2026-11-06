#!/usr/bin/env python3
"""Write the deterministic recommendation mock inputs as two supplied CSVs."""
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "codesignal_roblox_recommendation"
SEED = 42


def main():
    targets = [OUTPUT_DIR / "impressions.csv", OUTPUT_DIR / "experiences.csv"]
    if any(path.exists() for path in targets):
        raise SystemExit("Recommendation CSVs already exist; refusing to overwrite them.")

    rng = np.random.default_rng(SEED)
    experiences = pd.DataFrame({
        "experience_id": np.arange(60),
        "genre": rng.choice(["adventure", "simulation", "puzzle"], 60),
        "experience_age_days": rng.integers(1, 1500, 60),
    })
    rows = []
    for user in range(600):
        device = rng.choice(["mobile", "desktop", "console"])
        for position, experience in enumerate(rng.choice(60, 4, replace=False), start=1):
            minutes = float(rng.gamma(1.5, 20))
            friends = int(rng.poisson(0.7))
            logit = -3.3 + 0.025 * minutes + 0.6 * friends - 0.35 * (position - 1)
            joined = int(rng.random() < 1 / (1 + np.exp(-logit)))
            rows.append({
                "impression_id": len(rows), "user_id": user, "request_id": user,
                "experience_id": experience, "device": device, "position": position,
                "prior_genre_minutes": minutes, "friends_playing": friends,
                "joined": joined,
                "minutes_after_impression": float(rng.uniform(1, 45)) if joined else 0.0,
            })

    impressions = pd.DataFrame(rows)
    impressions.loc[rng.choice(len(impressions), 60, replace=False), "prior_genre_minutes"] = -1
    impressions.loc[rng.choice(len(impressions), 50, replace=False), "device"] = np.nan
    impressions = pd.concat([impressions, impressions.iloc[:20]], ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    impressions.to_csv(targets[0], index=False)
    experiences.to_csv(targets[1], index=False)
    print(f"Wrote {len(impressions)} impressions and {len(experiences)} experiences to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
