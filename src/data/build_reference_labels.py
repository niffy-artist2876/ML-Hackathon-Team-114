import time
import pandas as pd
from pathlib import Path
from src.data.loader import list_trips
from src.data.alignment import align_trip

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)


def main(partitions=("train", "student_test")):
    frames, failed = [], []
    for part in partitions:
        for trip in list_trips(part):
            t0 = time.time()
            try:
                df = align_trip(part, trip)
                df.insert(0, "partition", part)
                frames.append(df)
                print(f"{part}/{trip}: {len(df)} pts, "
                      f"{(~df.label_reliable).sum()} unreliable, {time.time() - t0:.1f}s", flush=True)
            except Exception as e:
                failed.append((part, trip, repr(e)))
                print(f"FAILED {part}/{trip}: {e!r}", flush=True)

    labels = pd.concat(frames, ignore_index=True)
    labels.to_csv(OUT / "reference_labels.csv", index=False)

    per_trip = (labels.groupby(["partition", "trip_id"])
                .agg(n_points=("point_idx", "size"),
                     n_unreliable=("label_reliable", lambda s: int((~s).sum())),
                     mean_dist_m=("dist_to_ref_m", "mean"),
                     max_dist_m=("dist_to_ref_m", "max"))
                .reset_index())
    per_trip["frac_unreliable"] = per_trip.n_unreliable / per_trip.n_points
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    per_trip.to_csv("results/tables/alignment_quality_per_trip.csv", index=False)

    print("\ntrips done:", len(per_trip), " failed:", len(failed))
    print("total points:", len(labels), " unreliable:", int((~labels.label_reliable).sum()))
    print(per_trip.sort_values("frac_unreliable", ascending=False).head(8).to_string())
    for f in failed:
        print("FAILED:", f)


if __name__ == "__main__":
    main()
