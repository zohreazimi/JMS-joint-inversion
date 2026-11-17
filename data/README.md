# Data

## Public file

`survey_geometry.csv` gives the station geometry of the Sangan survey used by the synthetic benchmarks (training model A, held-out model B and the ten random models):

| Column | Unit | Meaning |
|---|---|---|
| `x_local` | m | easting relative to the south-west corner of the survey |
| `y_local` | m | northing relative to the south-west corner of the survey |
| `elevation` | m | ground elevation |

3072 stations on a 20 m grid. It contains no gravity or magnetic values. With this file alone the notebook reproduces every synthetic table and figure of the article exactly.

## Field data (not public)

The Sangan gravity and magnetic field data and the borehole logs are available from the corresponding author (a.afshar@aut.ac.ir) upon reasonable request, owing to industrial and institutional data restrictions. With these files in this folder the notebook also runs Part C (field data):

* `Rsiduals_for_3d_Model.csv`: columns `X`, `Y` (UTM zone 41N, m), `Z` (elevation, m), `Res GRV` (residual Bouguer gravity, mGal), `Res Mag` (residual magnetic anomaly, nT).
* `boreholes.json`: one entry per borehole, `{"X": ..., "Y": ..., "elev": ..., "depth": ..., "dev": ..., "az": ..., "intervals": [[from_m, to_m, class], ...]}` with class 1 = low, 2 = medium, 3 = high magnetite.

To apply the methods to your own survey, provide files in the same formats (see `docs/TUTORIAL.md`).
