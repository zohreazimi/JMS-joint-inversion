# User guide

All functions are in `jms_lib.py`. Units: lengths in m, density contrast in g/cm^3, magnetization in A/m, gravity in mGal, magnetic data in nT. The vertical coordinate inside the library is positive downwards from a reference level.

## 1. Data

`load_residuals(path, keep_dy=20.0)`
: reads a CSV with columns `X, Y, Z, Res GRV, Res Mag` and keeps every station whose Y spacing is a multiple of `keep_dy`. Returns `xs, ys, elev, d_grav, d_mag`.

`roughness_sigma(x, y, v)`
: estimates the standard deviation of the high-frequency part of gridded data (difference from the mean of the 8 neighbours, robust MAD estimate). Used as the error floor; the error model of the article is `sigma = sqrt(floor^2 + (p*|d|)^2)` with `p = 0.03` (gravity) and `0.05` (magnetics).

## 2. Mesh

`DrapedMesh(xs, ys, elev, dx, dz, depth, pad=5)`
: `nx x ny` columns of square cells of size `dx`, each with `nz = depth/dz` cells hanging below the ground surface, plus `pad` padding columns on every side. Useful attributes: `n` (number of cells), `shape = (nz, ny, nx)`, `Xc, Yc` (cell centres), `Dc` (depth of the cell centre below ground), `Ec` (elevation of the cell centre), `core` (True inside the survey area). Methods: `boxes()` (prism corners for the kernels), `station_z(elev, height)` (station z for the kernels), `to_grid(m)` (reshape a model to `(nz, ny, nx)`).

`diff_operator(mesh)` / `central_diff_ops(mesh)`
: forward differences (used by MGS and the structural correlation) and central differences (used by PTSS).

## 3. Forward kernels

`gravity_kernel(boxes, xs, ys, zs, chunk=25)`
: dense matrix `A` (stations x cells), gravity in mGal per 1 g/cm^3.

`magnetic_kernel(boxes, xs, ys, zs, inc, dec, inc_m=None, dec_m=None, chunk=25)`
: total-field anomaly in nT per 1 A/m. `inc, dec` give the inducing field; `inc_m, dec_m` the magnetization direction (default: parallel to the field).

Both kernels use closed-form prism formulas valid at any point outside a prism (also below or beside it, as happens under topography). They run in several threads and print their progress. Memory: `4 * n_stations * n_cells` bytes per kernel (about 1.2 GB for the Sangan mesh).

`sensitivity_weight(A, sigma)`
: integrated-sensitivity depth weighting `w_j = (sum_i (A_ij/sigma_i)^2)^(1/4)`, normalized to 1.

## 4. Inverse problem

`Problem(A, d, sigma, w, bounds, D, eps=0.02, name='')`
: data `d` with standard deviations `sigma`, depth weights `w`, physical bounds `(lower, upper)`, difference operator `D` and focusing parameter `eps`. Set `P.offset = True` to estimate a constant base level of residual data analytically. `P.misfit(m)` returns `(chi2/N, normalized residuals)`; `P.predict(m)` returns the predicted data.

`invert(P, kind, target=1.0, tol=0.05, ...)`
: single-property inversion with `kind` in `'L2', 'L1', 'MS', 'MGS'`. Returns `(m, info)`; `info['chi2']` is the final chi2/N, which lies within `target*(1 +/- tol)` unless the target cannot be reached within the bounds.

`invert_joint(Pg, Pm, kappa=0.5, targets=(1.0, 1.0), ...)`
: the proposed JMS inversion of gravity (`Pg`) and magnetic (`Pm`) data together. `kappa` is either one number (symmetric coupling, used in the article, selected value 0.5) or a pair `(kappa_rho, kappa_M)`; `(0, 0.5)` is the one-way variant that avoids spurious density in light magnetic bodies. `kappa = 0` reduces to two independent MS inversions. Returns `(rho, M, info)`.

`ptss_operators(m_smooth, D3, n=3)` and `invert_ptss(P, B_self, B_mut, lam_self, lam_mut, target)`
: implementation of the joint PTSS inversion of Ming et al. (2024), started from the smooth models.

`misfit_floor(P)`
: best attainable chi2/N within the bounds. If it is above 1 (as for the Sangan magnetic data with induced magnetization only), the article uses 1.1 times this value as the common target.

## 5. Metrics

`truth_metrics(m, m_true, mesh, thr, mask)`
: RMSE, correlation, Dice, IoU, centroid error, depth error and volume ratio against a true model; bodies are cells `>= thr`.

`field_metrics(m, P, D, mask)`, `v90(m)`, `hoyer(v)`, `structural_corr(m1, m2, D)`
: measures without a true model: misfit, compactness and structural agreement of two models.

`borehole_trace(bh, X0, Y0)`, `borehole_labels(bh, md)`, `sample_along(mesh, m, x, y, z)`, `auc(score, label)`
: borehole validation: trace of a deviated hole, logged class along it, model values along it, ROC AUC.

## 6. Options of the notebook

The first code cell (`1. Settings`) holds every option: data files (`DATA_FILE`, `BOREHOLE_FILE`, `GEOMETRY_FILE`; without the field data and borehole logs Part C is skipped and the synthetic benchmarks run on `GEOMETRY_FILE`), mesh (`CELL, DZ, DEPTH, PAD`), inducing field (`INC, DEC, F_TOTAL`), bounds (`BOUNDS_RHO`, `CHI_MAX`, `Q_MAX`), error model (`P_GRAV, P_MAG`), focusing parameter `EPS`, candidate `KAPPA_GRID` and `LAM_GRID`, and the random seed. After changing any of them, delete `outputs/results_*.pkl` and `outputs/misfit_floor*.pkl` so that the inversions are recomputed (kernels are recomputed automatically when the mesh or field changes).

## 7. Expected behaviour and run times

* Every inversion ends at the same chi2/N for all methods (printed during the run); the tables report it.
* Laptop run times (single machine, 4 threads): Sangan kernels a few minutes; one inversion of the Sangan problem a few minutes to about half an hour; the complete notebook several hours the first time and a few minutes afterwards (cached results).
* `examples/quick_test_published_models.py` takes about 40 s and must reproduce `results/tables/table_paper_tests_dice.csv`.
* With induced magnetization only and susceptibility below 1.5 SI the Sangan magnetic data cannot be fitted (chi2/N about 6.8); the article therefore allows remanence through the bound on the effective magnetization (`Q_MAX = 1.6`).
