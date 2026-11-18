import pandas as pd
from pathlib import Path
from src.data.loader import DATA_ROOT, list_trips, read_track, read_route

'''
This module loads the inventory table, which shows trip counts, 
file counts and observation counts. 
'''


EXTS = ["track", "nodes", "arcs", "route"]

def count_lines(path):
    with open(path, "rb") as f:
        return sum(1 for _ in f)

def build_inventory(root=DATA_ROOT, partitions=("train", "student_test")):
    rows = []
    for part in partitions:
        for trip in list_trips(part, root):
            folder = Path(root)/part/trip
            row = {"partition":part, "trip_id":trip}
            for ext in EXTS:
                p = folder/f"{trip}.{ext}"
                row[f"has_{ext}"] = p.exists()
                row[f"size_{ext}_kb"] = round(p.stat().st_size / 1024, 1) if p.exists() else None
                row[f"n_{ext}"] = count_lines(p) if p.exists else None
            rows.append(row)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    inv = build_inventory()
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    inv.to_csv("results/tables/dataset_inventory.csv", index=False)
    try:
        print("dataset_inventory.csv has been made successfully.")
        print(inv.groupby("partition").size())          # expect 80 and 10
        print(inv[[c for c in inv.columns if c.startswith("n_")]].describe().T)
    except Exception as e:
        print(f"An error occurred: {e}")