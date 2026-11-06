#!/usr/bin/env python3
"""Generate the deterministic synthetic CSV inputs for the softmax mock."""
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "codesignal_roblox_softmax"
SEED = 42


def main():
    rng = np.random.default_rng(SEED)
    n = 1200

    text_length = rng.integers(10, 650, n).astype(float)
    link_count = rng.poisson(0.7, n)
    uppercase = rng.beta(1.4, 6, n)
    repeated = rng.beta(1.3, 5, n)
    age = rng.integers(1, 1500, n).astype(float)

    logits = np.column_stack([
        np.full(n, 1.3),
        -1.4 + 1.3 * link_count + 4 * repeated - 0.001 * age,
        -1.5 + 7 * uppercase + 0.001 * text_length,
    ])
    unnormalized = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = unnormalized / unnormalized.sum(axis=1, keepdims=True)
    class_names = np.array(["clean", "spam", "abuse"])
    labels = np.array([rng.choice(class_names, p=row) for row in probabilities])

    creators = pd.DataFrame({
        "creator_id": np.arange(n),
        "account_age_days": age,
    })
    submissions = pd.DataFrame({
        "submission_id": np.arange(n),
        "creator_id": np.arange(n),
        "text_length": text_length,
        "link_count": link_count,
        "uppercase_ratio": uppercase,
        "repeated_token_ratio": repeated,
        "content_type": rng.choice(["chat", "description", "comment"], n),
        "label": labels,
        "review_minutes": np.where(labels == "clean", 1, 5) + rng.random(n),
    })

    submissions.loc[rng.choice(n, 35, replace=False), "text_length"] = -1
    submissions.loc[rng.choice(n, 40, replace=False), "content_type"] = np.nan
    creators.loc[rng.choice(n, 45, replace=False), "account_age_days"] = np.nan
    submissions = pd.concat([submissions, submissions.iloc[:15]], ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submissions.to_csv(OUTPUT_DIR / "submissions.csv", index=False)
    creators.to_csv(OUTPUT_DIR / "creators.csv", index=False)
    print(f"Wrote {len(submissions)} submission rows and {len(creators)} creator rows to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
