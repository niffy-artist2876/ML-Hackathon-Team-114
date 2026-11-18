# Data-stage contract and review notes

## Scope and assumptions

Only the 80 train and 10 student-test trips were processed. No evaluator
data is read by the new entry points. The original teammate loader is
unchanged; callers of that low-level loader must still enforce partition rules.

Coordinates are interpreted as WGS84 longitude, latitude, following the
dataset column documentation. The datum itself is an assumption. Trips are
not confined to Germany. Each trip selects a UTM zone/hemisphere from its
median observed coordinates, with longitude-first transformation. A sample
of up to 200 road lengths is compared with WGS84 geodesic distances in
projection_checks.csv. Roads are straight segments between supplied nodes.

Reference labels use an offline dynamic program over ordered route
occurrences. Occurrence indices cannot decrease; stays and skips are allowed.
Repeated arc IDs remain separate occurrences. Spatial Huber loss uses a
fixed 25 m threshold. Start/end occurrences are not forced. This is reference
construction, not a prediction decoder. It does not establish true physical
ground truth, and can misalign loops, outliers, route gaps or incomplete routes.
Spatial alternatives within 2 m of the assigned distance are flagged; this
flag is not a probability or an uncertainty estimate.

## Candidates and features

Exact projected point-to-segment nearest candidates are selected from the
full directed network. A sampled spatial index only accelerates lookup;
final distances use original segment endpoints. Ties use ascending original
arc ID, so the distance-only baseline cannot reliably distinguish opposite
directions with identical geometry. Candidate misses are counted over ALL
observations, including cases without a positive candidate. References are
never inserted to fix misses.

Training-only selection compares k=4,8,16: take the smallest k with <=1%
point-weighted misses, otherwise the lowest miss rate with smaller-k tie
breaking. k=16 was selected but did not meet the target. Student-test
results do not select k. No learned preprocessing is fitted at this stage.

Exact candidate table columns:

| Column | Type | Meaning |
|---|---|---|
| trip_id | string | Eight digits, preserve leading zeroes |
| point_idx | int64 | Original zero-based observation index |
| arc_id | int64 | Original trip-local directed arc row ID |
| distance_m | float64 | Distance to candidate segment |
| distance_gap_m | float64 | Distance minus closest-candidate distance |
| candidate_rank | int64 | Distance rank, ties by arc ID |
| arc_length_m | float64 | Projected endpoint distance |
| projection_fraction | float64 | Closest position on segment, [0,1] |
| heading_cos | float64 | Previous GPS motion vs directed arc, zero if invalid |
| heading_valid | bool | Motion >=3 m, 0<gap<=10 s, nonzero arc length |
| speed_mps | float64 | Previous displacement/gap, zero if invalid |
| speed_valid | bool | 0<previous gap<=10 s |
| time_gap_s | float64 | Previous time gap; zero at first point |
| large_gap | bool | Previous gap >10 s |
| from_out_degree | int64 | Outgoing arcs at segment start |
| to_out_degree | int64 | Outgoing arcs at segment end |
| label | int8 | Candidate equals derived reference arc; offline only |

Use FEATURE_COLUMNS from src.data.features; identifiers and label are not
model inputs. Arc IDs are local to each trip. For inference:

```python
import json
from src.data.features import build_inference_table
config = json.load(open('configs/data_stage.json'))
table = build_inference_table('student_test', '00000000', k=config['selected_k'])
assert 'label' not in table
```

This function never reads `.route`. It fails clearly for invalid/empty input;
it is not yet a production API with fallback/error-response handling.

## Shared folds and remaining gates

Five GroupKFold folds balance GPS observation counts on training trips.
Every candidate row joins the saved fold by trip_id; do not generate new
model-specific folds. Fit any later scaler/imputer only on each training fold.
Use student-test only for the documented comparisons.

Before training, review three Folium maps (trips 00000001, 00000069,
00000090), the large-gap cases, reference connectivity and high miss rates.
Interactive map review was interrupted by a computer-use policy check;
the maps exist, but visual approval has NOT occurred. The figures must not
be described as manually validated. Feature choices, alignment assumptions,
metric definitions and AI disclosure still require team review.

Raw files are never modified. Local processed labels and reference maps
are ignored by Git. The newly observed data.zip is also excluded. No
model comparison, Bayesian tuning, HMM prediction, deployment, commit-history
rewrite or evaluator access is part of this stage.
