# Tutorial

## 1. First run

```
pip install -r requirements.txt
python examples/minimal_example.py
```

The script builds a 20 x 20 station grid, a 50 m mesh and one dense, magnetic block, computes noisy synthetic data, and inverts them with the smooth L2 method and with JMS. It prints the Dice overlap with the true block and the final chi2/N of both data sets (close to 1). Read the script from top to bottom: it is the shortest complete use of the library.

## 2. Comparing all methods on a synthetic model

```
python examples/quick_test_published_models.py
```

This runs L2, L1, MS, MGS, PTSS and JMS on the two test models of Ming et al. (2024) and prints, for each method, the Dice of density and magnetization and the final chi2/N. Compare `outputs/table_paper_tests_dice.csv` with `results/tables/table_paper_tests_dice.csv`.

## 3. Using your own survey

1. Prepare a CSV file with the columns `X, Y, Z, Res GRV, Res Mag` (UTM coordinates in m, elevation in m, residual gravity in mGal, residual magnetic anomaly in nT) on a regular grid. See `data/README.md`.
2. Open `Focusing_Joint_Inversion.ipynb` and, in the first code cell, set
   * `DATA_FILE` to the path of your file,
   * `CELL, DZ, DEPTH` to a mesh suited to your survey (cell size about the station spacing, depth about half the survey width),
   * `INC, DEC, F_TOTAL` to the IGRF values for your survey area and date,
   * `BOUNDS_RHO`, `CHI_MAX` and `Q_MAX` from the petrophysical information of your area.
3. Delete the `outputs/` folder if it contains results of another survey.
4. Run Sections 1-5 and Part C. Part C is the field-data analysis of the article and runs when both `DATA_FILE` and `BOREHOLE_FILE` are present; without borehole logs, invert your data with the library directly as in Section 4 below. Parts A, B, B2, A2 and D are the synthetic tests of the article; they use your survey geometry and can be skipped.
5. Put your boreholes in a JSON file and set `BOREHOLE_FILE` to it (format in `data/README.md`). Each entry has collar UTM coordinates (m), collar elevation (m, `null` to take it from the topography), hole length (m), deviation from vertical and azimuth (degrees), and logged intervals `(from_m, to_m, class)` measured along the hole, with class 1 = low, 2 = medium, 3 = high magnetite.

The models are written to `outputs/models_real.csv` with UTM coordinates (one row per cell; for each method the density contrast, the magnetization and the equivalent effective susceptibility) and can be imported into any 3D viewer or GIS.

## 4. Using the library directly

```python
from jms_lib import *
mesh = DrapedMesh(xs, ys, elev, dx=25.0, dz=25.0, depth=400.0, pad=5)
zs = mesh.station_z(elev, height=1.0); boxes = mesh.boxes()
Ag = gravity_kernel(boxes, xs, ys, zs)
Am = magnetic_kernel(boxes, xs, ys, zs, inc=52.1, dec=2.8)
D = diff_operator(mesh)
Pg = Problem(Ag, d_grav, sigma_g, sensitivity_weight(Ag, sigma_g), (0.0, 2.0), D); Pg.offset = True
Pm = Problem(Am, d_mag, sigma_m, sensitivity_weight(Am, sigma_m), (0.0, 147.0), D); Pm.offset = True
rho, M, info = invert_joint(Pg, Pm, kappa=0.5)          # proposed method
rho_L2, _ = invert(Pg, 'L2'); M_L2, _ = invert(Pm, 'L2')  # smooth reference
```

`rho` and `M` are vectors with one value per cell; `mesh.to_grid(rho)` reshapes them to `(nz, ny, nx)`.

## 5. Choosing kappa

Use the value of the article (0.5) or repeat the selection of Part A on a training model that resembles your target (the notebook tries `KAPPA_GRID = (0.5, 1, 2)` and keeps the value with the highest mean Dice). If you expect bodies that are magnetic but not dense, use the one-way coupling `kappa=(0.0, 0.5)`.
