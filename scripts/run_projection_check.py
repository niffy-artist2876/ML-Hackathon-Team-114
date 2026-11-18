"""Run the A-F2 projection check for a real trip.

Example:
    python scripts/run_projection_check.py --inventory C:/path/dataset_inventory.csv \
        --data-root data/raw --partition train --trip-id 00000001
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.loader import read_track
from src.geo.projection import project_dataframe, projection_sanity_check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--partition", default="train")
    parser.add_argument("--trip-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/tables"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = pd.read_csv(args.inventory, dtype={"trip_id": str})
    selected = inventory[
        (inventory["partition"] == args.partition)
        & (inventory["trip_id"].astype(str).str.zfill(8) == str(args.trip_id).zfill(8))
    ]
    if selected.empty:
        raise ValueError(f"Trip {args.partition}/{args.trip_id} is not present in the inventory")

    track = read_track(args.partition, args.trip_id, root=args.data_root)
    check = projection_sanity_check(track)
    projected = project_dataframe(track)
    check.update(
        {
            "partition": args.partition,
            "trip_id": str(args.trip_id).zfill(8),
            "n_points": len(projected),
            "first_x_m": projected.iloc[0]["x_m"],
            "first_y_m": projected.iloc[0]["y_m"],
            "last_x_m": projected.iloc[-1]["x_m"],
            "last_y_m": projected.iloc[-1]["y_m"],
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([check]).to_csv(args.output_dir / "projection_check.csv", index=False)
    print(pd.Series(check).to_string())


if __name__ == "__main__":
    main()
