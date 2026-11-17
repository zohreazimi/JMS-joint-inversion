"""Builds Focusing_Joint_Inversion.ipynb and its script version run_all.py from the cells below.
Run:  python build_notebook.py"""
import json, pathlib
HERE = pathlib.Path(__file__).parent

cells = []
def md(s): cells.append(('markdown', s.strip('\n')))
def code(s): cells.append(('code', s.strip('\n')))

md(r"""
# Joint focusing inversion of gravity and magnetic data: Sangan iron-ore deposit
## Proposed Joint Minimum Support (JMS) inversion compared with smooth and established focusing inversions

| Name | Stabilizer | Reference |
|---|---|---|
| `L2` | smooth (minimum L2 norm with depth weighting); the "smooth" method of Ming et al. (2024) | Li & Oldenburg 1996 |
| `PTSS` | joint power-type structural self-constrained inversion (n = 3, self + mutual constraint, started from the smooth result) | Ming et al. 2024 |
| `L1` | L1 norm | Farquharson 2008 |
| `MS` | minimum support | Last & Kubik 1983; Portniaguine & Zhdanov 1999 |
| `MGS` | minimum gradient support | Portniaguine & Zhdanov 1999 |
| **`JMS`** | **proposed: joint minimum support; density and magnetization are solved simultaneously** | this work |

**Proposed method (JMS).** The penalty of each property is

`R_rho = mean( u_rho^2 / (u_rho^2 + kappa*u_M^2 + eps^2) )`  and  `R_M = mean( u_M^2 / (u_M^2 + kappa*u_rho^2 + eps^2) )`

Wherever one property has a body, the penalty of the other property is reduced. Both models are updated **simultaneously in one optimization**.
Gravity constrains depth better and pulls the magnetic body to the right depth; magnetics sharpens the edges. kappa = 0 gives two independent MS inversions.

**Test design (to avoid tuning on the test itself):**
- **Model A (training):** used only to select kappa (JMS) and lambda (PTSS) with a pre-declared criterion: the mean of the density and magnetization Dice.
- **Model B (held-out):** a completely different geometry, not used in any decision. **This is the main accuracy result.**
- **Ten random models:** paired Wilcoxon signed-rank tests.
- **Sangan field data:** validation with four logged boreholes that were not used in any setting.

**Fair-comparison rules:** mesh, solver (L-BFGS-B), physical bounds, depth weighting, eps, error model, data base level and misfit target are identical for all methods. Only the stabilizer differs.
""")

md("## 1. Settings")
code(r"""
# ---------- data files and output folder
# The Sangan field data and borehole logs are not public (available from the corresponding author on reasonable request).
# Without them the notebook runs every synthetic benchmark (Parts A, B, B2, A2, D) on the public survey geometry
# and skips the field-data part (Part C).
# DATA_FILE: field data CSV with columns X, Y, Z, Res GRV, Res Mag. If None, the notebook folder and its
#            sub-folders are searched for such a file.
DATA_FILE     = None
BOREHOLE_FILE = 'data/boreholes.json'          # borehole logs (field-data part only)
GEOMETRY_FILE = 'data/survey_geometry.csv'     # public: station coordinates (local, m) and elevations
OUT_DIR       = 'outputs'

# ---------- mesh (m)
CELL, DZ, DEPTH, PAD = 25.0, 25.0, 400.0, 5     # horizontal cell size, cell thickness, model depth, number of padding cells
SENSOR_H  = 1.0                                  # sensor height above ground

# ---------- inducing geomagnetic field (data are residual total-field anomalies; IGRF and regional removed)
# Field direction used in the paper. IGRF at the survey centre (34.48 N, 60.45 E, 1490 m) for 1990.5: I = 52.13, D = 2.84, F = 48 011 nT.
INC, DEC  = 51.5, 3.0          # inclination and declination (degrees)
F_TOTAL   = 47500.0            # total field intensity (nT), used to convert susceptibility to magnetization

# ---------- physical bounds (petrophysical information for Sangan: susceptibility 0-1.5 SI, density contrast 0-2 g/cm^3)
CHI_MAX    = 1.5
BOUNDS_RHO = (0.0, 2.0)                                   # g/cm³
M_MAX      = CHI_MAX*F_TOTAL*1e-9/(4e-7*3.141592653589793)  # maximum induced magnetization (A/m), about 56.7
BOUNDS_M   = (0.0, M_MAX)                                 # A/m, synthetic models (induced only)
# Field data: massive magnetite usually carries remanence. Effective magnetization = (1 + Q)*chi*H with Koenigsberger ratio Q <= Q_MAX
Q_MAX      = 1.6
BOUNDS_M_REAL = (0.0, (1 + Q_MAX)*M_MAX)                  # about 147 A/m
OFFSET     = True     # estimate the base level of the residual data (one constant per data set, for all methods)

# ---------- error model: sigma = sqrt(floor^2 + (p*|d|)^2); the floor is estimated from the roughness of the data
P_GRAV, P_MAG = 0.03, 0.05

# ---------- methods
EPS        = 0.02                 # focusing parameter eps (identical for all focusing methods)
KAPPA_GRID = (0.5, 1.0, 2.0)      # kappa values tried on the training model A only
BASELINES  = ['L2', 'L1', 'MS', 'MGS']
PTSS_N     = 3                    # gradient power in PTSS (optimal value of Ming et al. 2024)
LAM_GRID   = (1.0, 10.0, 100.0)   # relative weight of the PTSS constraints; selected on model A only, like kappa
ORDER      = ['L2', 'PTSS', 'L1', 'MS', 'MGS', 'JMS']
SEED       = 1

import os, time, pickle, numpy as np, pandas as pd, matplotlib.pyplot as plt
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 300, 'font.size': 9})
def save(fig, name): fig.savefig(os.path.join(OUT_DIR, name), bbox_inches='tight')
def show_table(t):
    try: display(t)
    except NameError: print(t.to_string())
COLORS = {'L2': '#7f7f7f', 'PTSS': '#8c564b', 'L1': '#1f77b4', 'MS': '#2ca02c', 'MGS': '#ff7f0e', 'JMS': '#d62728'}
print(f'Magnetization bound (synthetic, induced): 0 to {M_MAX:.1f} A/m | (Sangan, with remanence Q <= {Q_MAX}): 0 to {BOUNDS_M_REAL[1]:.1f} A/m')
""")

md("## 2. Library (kernels, mesh, common solver, proposed method, metrics)")
code(r"""
# The library is kept in jms_lib.py (same folder as this notebook)
from jms_lib import *
""")

md("## 3. Data and error estimation from the data")
code(r"""
import glob, json
def find_data_file(path):
    # locate the field data file: if no path is given, take the first CSV with the required columns (None if absent)
    if path:
        if os.path.exists(path): return path
        raise FileNotFoundError(f'File not found: {path}')
    need = {'X', 'Y', 'Z', 'Res GRV', 'Res Mag'}
    for f in sorted(glob.glob('**/*.csv', recursive=True)):
        try: cols = {c.strip() for c in pd.read_csv(f, nrows=1).columns}
        except Exception: continue
        if need <= cols: return f
    return None
DATA_FILE = find_data_file(DATA_FILE)
FIELD_DATA = DATA_FILE is not None

if FIELD_DATA:
    print('Field data file:', DATA_FILE)
    dat = load_residuals(DATA_FILE)
    X0, Y0 = dat.X.min(), dat.Y.min()                      # local coordinates: south-west corner = 0
    xs, ys, elev = dat.X.values - X0, dat.Y.values - Y0, dat.Z.values
    d_grav, d_mag = dat['Res GRV'].values, dat['Res Mag'].values
    # error floor from data roughness (difference from the 8-neighbour mean, robust standard deviation)
    FLOOR_G = roughness_sigma(xs, ys, d_grav); FLOOR_M = roughness_sigma(xs, ys, d_mag)
    sig_g = np.sqrt(FLOOR_G**2 + (P_GRAV*d_grav)**2); sig_m = np.sqrt(FLOOR_M**2 + (P_MAG*d_mag)**2)
    if not os.path.exists(GEOMETRY_FILE):                 # write the public geometry file (no data values)
        os.makedirs(os.path.dirname(GEOMETRY_FILE) or '.', exist_ok=True)
        pd.DataFrame(dict(x_local=xs, y_local=ys, elevation=elev)).to_csv(GEOMETRY_FILE, index=False)
else:
    # public mode: station geometry only; the error floors are those estimated from the field data (Section 3 of the paper)
    print('Field data not found: running the synthetic benchmarks on the public survey geometry', GEOMETRY_FILE)
    geo = pd.read_csv(GEOMETRY_FILE)
    X0, Y0 = 0.0, 0.0
    xs, ys, elev = geo.x_local.values, geo.y_local.values, geo.elevation.values
    FLOOR_G, FLOOR_M = 0.05834628177482533, 378.7816919964442   # mGal, nT
print(f'{xs.size} stations | area {np.ptp(xs):.0f} x {np.ptp(ys):.0f} m | elevation {elev.min():.0f}-{elev.max():.0f} m')
print(f'Error floor: gravity {FLOOR_G:.3f} mGal | magnetics {FLOOR_M:.0f} nT')

# borehole logs (field-data part only)
BOREHOLES = {}
if os.path.exists(BOREHOLE_FILE):
    BOREHOLES = {k: dict(v, intervals=[tuple(i) for i in v['intervals']]) for k, v in json.load(open(BOREHOLE_FILE)).items()}
RUN_FIELD = FIELD_DATA and len(BOREHOLES) > 0
print('Part C (Sangan field data):', 'will run' if RUN_FIELD else 'skipped (field data or borehole logs not available)')

# Figure D1: topography, residual gravity, residual magnetics (topography only without field data)
panels = [(elev, 'Topography', 'm', 'terrain')]
if FIELD_DATA: panels += [(d_grav, 'Residual gravity', 'mGal', 'RdYlBu_r'), (d_mag, 'Residual magnetic (TMI)', 'nT', 'RdYlBu_r')]
fig, axs = plt.subplots(1, len(panels), figsize=(5.4*len(panels), 4.4), squeeze=False)
for ax, (v, t, u, cm) in zip(axs[0], panels):
    kw = {} if cm == 'terrain' else dict(vmin=-np.percentile(np.abs(v), 99), vmax=np.percentile(np.abs(v), 99))
    tc = ax.tricontourf(xs, ys, v, 40, cmap=cm, **kw); cb = plt.colorbar(tc, ax=ax, fraction=0.046, pad=0.04); cb.set_label(u)
    ax.set_aspect('equal'); ax.set_title(t); ax.set_xlabel(f'Easting − {X0:.0f} (m)'); ax.set_ylabel(f'Northing − {Y0:.0f} (m)')
fig.suptitle('Figure D1. Survey data'); fig.tight_layout(); save(fig, 'D1_data.png'); plt.show()
""")

md("## 4. Topography-draped mesh and kernel matrices (computed on the first run only, then read from file)")
code(r"""
mesh = DrapedMesh(xs, ys, elev, CELL, DZ, DEPTH, pad=PAD)
zs = mesh.station_z(elev, SENSOR_H); D = diff_operator(mesh)
print(f'Mesh: {mesh.nx} x {mesh.ny} x {mesh.nz} = {mesh.n} cells | memory per kernel about {xs.size*mesh.n*4/1e9:.2f} GB')
kfile = os.path.join(OUT_DIR, f'kernels_{int(CELL)}_{int(DZ)}_{int(DEPTH)}_{INC}_{DEC}.npz')
if os.path.exists(kfile):
    K = np.load(kfile); Ag, Am = K['Ag'], K['Am']; print('Kernels read from file')
else:
    # first run only; needs 2-3 GB of free memory. Progress is printed below.
    print('Computing kernels (first run only)...'); t0 = time.time(); bx = mesh.boxes()
    Ag = gravity_kernel(bx, xs, ys, zs); Am = magnetic_kernel(bx, xs, ys, zs, INC, DEC)
    np.savez(kfile, Ag=Ag, Am=Am); print(f'Kernels computed ({time.time()-t0:.0f} s)')
""")

md("## 5. Common tools: problem set-up, running all methods, metrics and figures")
code(r"""
D3 = central_diff_ops(mesh)   # central differences for the PTSS constraints
def make_problem(A, d, sigma, bounds, name):
    # inverse problem with sensitivity-based depth weighting and base-level estimation
    P = Problem(A, d, sigma, sensitivity_weight(A, sigma), bounds, D, EPS, name); P.offset = OFFSET; return P

def run_all(Pg, Pm, tag, kappas, lams, targets=(1.0, 1.0)):
    # all methods with the same rules; results are cached to file so that re-runs are fast
    f = os.path.join(OUT_DIR, f'results_{tag}.pkl')
    res = pickle.load(open(f, 'rb')) if os.path.exists(f) else {}
    for kind in BASELINES:
        if kind in res: continue
        t0 = time.time(); r, ig = invert(Pg, kind, target=targets[0]); M, im = invert(Pm, kind, target=targets[1])
        res[kind] = (r, M, dict(chi2=(ig['chi2'], im['chi2']), time=time.time() - t0, hist=(ig['hist'], im['hist'])))
        print(f'{tag:9s} {kind:4s} χ²/N = {ig["chi2"]:.3f}, {im["chi2"]:.3f}  ({time.time()-t0:.0f} s)', flush=True)
        pickle.dump(res, open(f, 'wb'))
    for lam in lams:               # method of Ming et al. (2024), started from the smooth result as in their paper
        key = 'PTSS' if len(lams) == 1 else f'PTSS λ={lam:g}'
        if key in res: continue
        t0 = time.time(); r0, M0 = res['L2'][0], res['L2'][1]
        B_rho, B_M = ptss_operators(r0, D3, PTSS_N), ptss_operators(M0, D3, PTSS_N)
        r, ig = invert_ptss(Pg, B_rho, B_M, lam, lam, targets[0])     # gravity: self constraint from rho0, mutual from M0
        M, im = invert_ptss(Pm, B_M, B_rho, lam, lam, targets[1])     # magnetics: self constraint from M0, mutual from rho0
        res[key] = (r, M, dict(chi2=(ig['chi2'], im['chi2']), time=time.time() - t0 + res['L2'][2]['time']))
        print(f'{tag:9s} {key:4s} χ²/N = {ig["chi2"]:.3f}, {im["chi2"]:.3f}  ({time.time()-t0:.0f} s)', flush=True)
        pickle.dump(res, open(f, 'wb'))
    for k in kappas:
        key = 'JMS' if len(kappas) == 1 else f'JMS κ={k}'
        if key in res: continue
        r, M, info = invert_joint(Pg, Pm, kappa=k, targets=targets); res[key] = (r, M, info)
        print(f'{tag:9s} {key:4s} χ²/N = {info["chi2"][0]:.3f}, {info["chi2"][1]:.3f}  ({info["time"]:.0f} s)', flush=True)
        pickle.dump(res, open(f, 'wb'))
    return res

def synth_table(res, Pg, Pm, rho_t, M_t, fp):
    # accuracy table against the true model (cells inside the survey only). fp: regions for the false-body test
    rows = []; thr_r, thr_m = 0.25*rho_t.max(), 0.25*M_t.max()
    for k, (r, M, info) in res.items():
        for prop, m, mt, thr, P in (('density', r, rho_t, thr_r, Pg), ('magnetization', M, M_t, thr_m, Pm)):
            row = dict(Method=k, Property=prop, chi2N=P.misfit(m)[0], **truth_metrics(m, mt, mesh, thr, mesh.core))
            for name, (mask, p) in fp.items():
                if p == prop: row[f'False_{name}'] = float(np.mean(np.clip(m[mask] - mt[mask], 0, None)))
            row['Time_s'] = info['time']; rows.append(row)
    return pd.DataFrame(rows).set_index(['Property', 'Method'])

BETTER = dict(RMSE='min', Corr='max', Dice='max', IoU='max', CentroidErr_m='min', DepthErr_m='min')
def winners(T):
    # winner of each metric and number of first places per method
    out, count = [], {}
    for prop in ('density', 'magnetization'):
        t = T.loc[prop]
        for c, s in BETTER.items():
            w = t[c].idxmin() if s == 'min' else t[c].idxmax(); out.append((prop, c, w)); count[w] = count.get(w, 0) + 1
    print('Winner of each metric:'); [print(f'  {p:13s} {c:14s} → {w}') for p, c, w in out]
    print('Number of first places:', dict(sorted(count.items(), key=lambda kv: -kv[1])))

def bars(T, title, fname, extra=()):
    mets = [('RMSE', 'RMSE ↓'), ('Corr', 'Correlation ↑'), ('Dice', 'Dice ↑'), ('IoU', 'IoU ↑'),
            ('CentroidErr_m', 'Centroid error (m) ↓'), ('DepthErr_m', 'Depth error (m) ↓'), ('VolumeRatio', 'Volume ratio (1 = ideal)')] + list(extra)
    fig, axs = plt.subplots(2, len(mets), figsize=(2.5*len(mets), 6))
    for i, prop in enumerate(('density', 'magnetization')):
        t = T.loc[prop]
        for j, (c, lab) in enumerate(mets):
            ax = axs[i, j]; ax.bar(t.index, t[c], color=[COLORS.get(k.split()[0], '#999') for k in t.index])
            if c == 'VolumeRatio': ax.axhline(1, color='k', ls='--', lw=0.8)
            ax.set_title(lab if i == 0 else '', fontsize=9); ax.tick_params(axis='x', rotation=60, labelsize=8)
            if j == 0: ax.set_ylabel(prop)
    fig.suptitle(title); fig.tight_layout(); save(fig, fname); plt.show()

from scipy.ndimage import map_coordinates
def section(ax, m, axis, idx, vmax, cmap, title, truth=None):
    # vertical section on the draped mesh; black line = outline of the true body (half maximum)
    g = mesh.to_grid(m)
    if axis == 'x': v = g[:, idx, :]; h, he, topo = mesh.xc, mesh.xe, mesh.topo[idx]
    else:           v = g[:, :, idx]; h, he, topo = mesh.yc, mesh.ye, mesh.topo[:, idx]
    te = np.interp(he, h, topo); Ze = te[None, :] - mesh.de[:, None]
    pc = ax.pcolormesh(np.broadcast_to(he[None], Ze.shape), Ze, v, cmap=cmap, vmin=0, vmax=vmax, shading='flat')
    if truth is not None and truth.max() > 0:
        tg = mesh.to_grid(truth); tv = tg[:, idx, :] if axis == 'x' else tg[:, :, idx]
        Zc = topo[None, :] - mesh.dc[:, None]
        if tv.max() > 0: ax.contour(np.broadcast_to(h[None], Zc.shape), Zc, tv, [0.5*truth.max()], colors='k', linewidths=1.1)
    ax.plot(he, te, 'k-', lw=0.8); ax.set_title(title, fontsize=9); return pc

def section_figure(res, order, axis, pos, fname, title, truths=None, vmaxs=None):
    idx = np.argmin(np.abs((mesh.yc if axis == 'x' else mesh.xc) - pos))
    cols = (['TRUE'] if truths else []) + order
    fig, axs = plt.subplots(2, len(cols), figsize=(3.0*len(cols), 5.6), sharey=True, squeeze=False)
    for i, (lab, cmap) in enumerate((('ρ (g/cm³)', 'viridis'), ('M (A/m)', 'magma'))):
        vmax = vmaxs[i]; tr = truths[i] if truths else None
        for j, k in enumerate(cols):
            m = tr if k == 'TRUE' else res[k][i]
            pc = section(axs[i, j], m, axis, idx, vmax, cmap, f'{k} {lab}', tr)
        plt.colorbar(pc, ax=axs[i, :].tolist(), shrink=0.8)
    for ax in axs[:, 0]: ax.set_ylabel('elevation (m)')
    fig.suptitle(f'{title} — {"E–W, y" if axis == "x" else "N–S, x"} = {pos:.0f} m'); save(fig, fname); plt.show()
""")

md(r"""
## Part A: training model (used only to select kappa and lambda)
Two magnetite bodies (a 60 degree dipping sheet and a block; rho = 1.0, M = 45 A/m, about 1.2 SI) and a dense non-magnetic body (rho = 0.4).
Noise follows the error model of the field data, and a constant base level is added so that the data behave like residual data.
""")
code(r"""
Xc, Yc, Dc = mesh.Xc, mesh.Yc, mesh.Dc
def noisy(A, m, floor, p, base, rng):
    d0 = (A @ m.astype(np.float32)).astype(float); s = np.sqrt(floor**2 + (p*d0)**2)
    return d0 + base + s*rng.standard_normal(d0.size), s

# model A
slabA  = (np.abs(Xc - (350 + Dc/np.tan(np.radians(60)))) < 40) & (Yc > 350) & (Yc < 700) & (Dc > 25) & (Dc < 200)
blockA = (Xc > 800) & (Xc < 1000) & (Yc > 550) & (Yc < 800) & (Dc > 50) & (Dc < 175)
denseA = (Xc > 750) & (Xc < 950) & (Yc > 100) & (Yc < 300) & (Dc > 100) & (Dc < 250)
rhoA = np.zeros(mesh.n); rhoA[slabA | blockA] = 1.0; rhoA[denseA] = 0.4
MA = np.zeros(mesh.n); MA[slabA | blockA] = 45.0
fpA = {'M_in_dense_nonmagnetic': (denseA, 'magnetization')}
rng = np.random.default_rng(SEED)
dgA, sgA = noisy(Ag, rhoA, FLOOR_G, P_GRAV, -0.2, rng); dmA, smA = noisy(Am, MA, FLOOR_M, P_MAG, 300.0, rng)
PgA, PmA = make_problem(Ag, dgA, sgA, BOUNDS_RHO, 'g'), make_problem(Am, dmA, smA, BOUNDS_M, 'm')
resA = run_all(PgA, PmA, 'modelA', KAPPA_GRID, LAM_GRID)
""")
code(r"""
TA = synth_table(resA, PgA, PmA, rhoA, MA, fpA)
TA.to_csv(os.path.join(OUT_DIR, 'table_modelA_training.csv')); pd.set_option('display.float_format', lambda v: f'{v:.3f}')
show_table(TA[['chi2N', 'RMSE', 'Corr', 'Dice', 'IoU', 'CentroidErr_m', 'DepthErr_m', 'VolumeRatio']])

# select kappa (JMS) and lambda (PTSS) with a pre-declared criterion: highest mean of density and magnetization Dice on model A
score = {k: 0.5*(TA.loc[('density', f'JMS κ={k}'), 'Dice'] + TA.loc[('magnetization', f'JMS κ={k}'), 'Dice']) for k in KAPPA_GRID}
KAPPA = max(score, key=score.get)
print('JMS  - mean Dice for each kappa:', {k: round(v, 3) for k, v in score.items()}, '-> selected kappa =', KAPPA)
score_p = {l: 0.5*(TA.loc[('density', f'PTSS λ={l:g}'), 'Dice'] + TA.loc[('magnetization', f'PTSS λ={l:g}'), 'Dice']) for l in LAM_GRID}
LAM = max(score_p, key=score_p.get)
print('PTSS - mean Dice for each lambda:', {l: round(v, 3) for l, v in score_p.items()}, '-> selected lambda =', LAM)
""")

md(r"""
## Part B: held-out test model (main accuracy result)
A completely different geometry that was not used in any decision:
1. a magnetite lens dipping 45 degrees to the **west** (rho = 1.5, M = 50)
2. a **deep**, sub-horizontal magnetite sheet at 150-225 m depth (rho = 1.2, M = 40)
3. a **light magnetic** body (rho = 0.1, M = 30): tests that no spurious density is created
4. a **dense non-magnetic** body (rho = 0.6, M = 0): tests that no spurious magnetization is created
""")
code(r"""
lensB  = (np.abs(Xc - (950 - Dc/np.tan(np.radians(45)))) < 35) & (Yc > 200) & (Yc < 450) & (Dc > 50) & (Dc < 250)
sheetB = (Xc > 250) & (Xc < 550) & (Yc > 600) & (Yc < 800) & (Dc > 150) & (Dc < 225)
lightB = (Xc > 600) & (Xc < 700) & (Yc > 700) & (Yc < 850) & (Dc > 25) & (Dc < 100)
denseB = (Xc > 100) & (Xc < 300) & (Yc > 100) & (Yc < 300) & (Dc > 75) & (Dc < 200)
rhoB = np.zeros(mesh.n); rhoB[lensB] = 1.5; rhoB[sheetB] = 1.2; rhoB[lightB] = 0.1; rhoB[denseB] = 0.6
MB = np.zeros(mesh.n); MB[lensB] = 50.0; MB[sheetB] = 40.0; MB[lightB] = 30.0
fpB = {'M_in_dense_nonmagnetic': (denseB, 'magnetization'), 'rho_in_light_magnetic': (lightB, 'density')}
rng = np.random.default_rng(SEED + 100)
dgB, sgB = noisy(Ag, rhoB, FLOOR_G, P_GRAV, -0.3, rng); dmB, smB = noisy(Am, MB, FLOOR_M, P_MAG, -500.0, rng)
PgB, PmB = make_problem(Ag, dgB, sgB, BOUNDS_RHO, 'g'), make_problem(Am, dmB, smB, BOUNDS_M, 'm')
resB = run_all(PgB, PmB, 'modelB', (KAPPA,), (LAM,))

fig, axs = plt.subplots(1, 4, figsize=(18, 4)); ext = [mesh.xe[0], mesh.xe[-1], mesh.ye[0], mesh.ye[-1]]
axs[0].imshow(mesh.to_grid(rhoB).max(0), origin='lower', extent=ext, cmap='viridis'); axs[0].set_title('Model B true ρ (column max)')
axs[1].imshow(mesh.to_grid(MB).max(0), origin='lower', extent=ext, cmap='magma'); axs[1].set_title('Model B true M (column max)')
for ax, v, t in ((axs[2], dgB, 'Synthetic gravity (mGal)'), (axs[3], dmB, 'Synthetic TMI (nT)')):
    tc = ax.tricontourf(xs, ys, v, 40, cmap='RdYlBu_r'); plt.colorbar(tc, ax=ax); ax.set_title(t)
for ax in axs: ax.set_aspect('equal')
fig.suptitle('Figure S1. Held-out test model B on the real survey geometry'); save(fig, 'S1_modelB.png'); plt.show()
""")
code(r"""
TB = synth_table(resB, PgB, PmB, rhoB, MB, fpB)
TB.to_csv(os.path.join(OUT_DIR, 'table_modelB_test.csv'))
show_table(TB[['chi2N', 'RMSE', 'Corr', 'Dice', 'IoU', 'CentroidErr_m', 'DepthErr_m', 'VolumeRatio',
               'False_M_in_dense_nonmagnetic', 'False_rho_in_light_magnetic']])
winners(TB)
bars(TB, 'Figure S2. Held-out model B: accuracy against the true model (all methods at the same χ²/N)', 'S2_modelB_metrics.png')
""")
code(r"""
orderB = ORDER
section_figure(resB, orderB, 'x', 325, 'S3_modelB_EW_lens.png', 'Model B', (rhoB, MB), (1.5, 50))
section_figure(resB, orderB, 'x', 700, 'S4_modelB_EW_sheet.png', 'Model B', (rhoB, MB), (1.5, 50))
section_figure(resB, orderB, 'y', 400, 'S5_modelB_NS.png', 'Model B', (rhoB, MB), (1.5, 50))
""")

md(r"""
## Part B2: why does JMS place spurious density in the light magnetic body?
In the model B accuracy table, the spurious density of JMS in the light magnetic body (rho = 0.1, M = 30) is larger than that of MS. This cell shows the cause numerically:
1. the true gravity signal of this body is very weak (less than twice the error standard deviation), so the gravity data hardly constrain the density there;
2. no method recovers the density of the deep sheet of model B, so part of its gravity signal must be explained elsewhere;
3. JMS concentrates the magnetization of this body in a few cells near the upper bound; there kappa*u_M^2 is tens of times eps^2, so placing density in these cells is very cheap.
Remedy: **one-way coupling** (kappa_rho = 0, kappa_M = 0.5): density is penalized as in MS, while magnetization is still guided by density.
(The main results of the paper keep the symmetric kappa selected on model A; this test is a sensitivity analysis.)
""")
code(r"""
cg_, cm_ = PgB.w/PgB.scale, PmB.w/PmB.scale; rJ, MJ = resB['JMS'][0], resB['JMS'][1]
g_light = (Ag @ (rhoB*lightB).astype(np.float32)).astype(float)
excess = np.where(lightB, np.clip(rJ - rhoB, 0, None), 0.); g_exc = (Ag @ excess.astype(np.float32)).astype(float)
miss = np.where(sheetB, np.clip(rhoB - rJ, 0, None), 0.); g_miss = (Ag @ miss.astype(np.float32)).astype(float)
r_clean = np.where(lightB, np.minimum(rJ, rhoB), rJ)
sel = lightB & (rJ > 0.25); ka = KAPPA
diag = {'true gravity of the light body (max, mGal)': float(np.abs(g_light).max()), 'gravity error standard deviation (mGal)': float(FLOOR_G),
        'gravity of the JMS spurious density (max, mGal)': float(np.abs(g_exc).max()),
        'JMS gravity chi2/N': float(PgB.misfit(rJ)[0]), 'JMS gravity chi2/N without the spurious density': float(PgB.misfit(r_clean)[0]),
        'mean density in the deep sheet: true': float(rhoB[sheetB].mean()), '   JMS': float(rJ[sheetB].mean()), '   MS': float(resB['MS'][0][sheetB].mean()),
        'correlation of spurious-density gravity with missing sheet gravity': float(np.corrcoef(g_exc, g_miss)[0, 1]),
        'cells of the body with JMS density > 0.25': int(sel.sum()), 'mean M in these cells (A/m)': float(MJ[sel].mean()) if sel.any() else np.nan,
        'mean kappa*u_M^2/eps^2 in these cells': float(np.mean(ka*(cm_[sel]*MJ[sel])**2)/EPS**2) if sel.any() else np.nan}
for k, v in diag.items(): print(f'{k:58s} {v:.3f}' if isinstance(v, float) else f'{k:58s} {v}')

# asymmetric coupling test on model B (held-out) and model A (training)
f_as = os.path.join(OUT_DIR, 'results_modelB_asym.pkl'); asym = pickle.load(open(f_as, 'rb')) if os.path.exists(f_as) else {}
for kr in (0.1, 0.0):
    key = f'JMS κρ={kr:g} κM={ka:g}'
    if key not in asym:
        asym[key] = invert_joint(PgB, PmB, kappa=(kr, ka)); pickle.dump(asym, open(f_as, 'wb'))
rows = []
for k, (r, M, _) in {'MS': resB['MS'], f'JMS κ={ka:g} (paper)': resB['JMS'], **asym}.items():
    tr = truth_metrics(r, rhoB, mesh, 0.25*rhoB.max(), mesh.core); tm = truth_metrics(M, MB, mesh, 0.25*MB.max(), mesh.core)
    rows.append(dict(Method=k, Dice_rho=tr['Dice'], Dice_M=tm['Dice'], Cent_rho=tr['CentroidErr_m'], Cent_M=tm['CentroidErr_m'],
                     spurious_rho_light=float(np.mean(np.clip(r[lightB] - rhoB[lightB], 0, None))), spurious_M_dense=float(np.mean(np.clip(M[denseB], 0, None))),
                     chi2_g=PgB.misfit(r)[0], chi2_m=PmB.misfit(M)[0]))
T_asym = pd.DataFrame(rows); T_asym.to_csv(os.path.join(OUT_DIR, 'table_spurious_asym.csv'), index=False); show_table(T_asym.set_index('Method'))

f_aa = os.path.join(OUT_DIR, 'results_modelA_asym.pkl'); asA = pickle.load(open(f_aa, 'rb')) if os.path.exists(f_aa) else {}
if 'k0' not in asA: asA['k0'] = invert_joint(PgA, PmA, kappa=(0.0, ka)); pickle.dump(asA, open(f_aa, 'wb'))
for k, (r, M, _) in (('MS', resA['MS']), (f'JMS κ={ka:g}', resA[f'JMS κ={ka:g}']), (f'JMS κρ=0 κM={ka:g}', asA['k0'])):
    tr = truth_metrics(r, rhoA, mesh, 0.25*rhoA.max(), mesh.core); tm = truth_metrics(M, MA, mesh, 0.25*MA.max(), mesh.core)
    print(f'model A  {k:18s}  Dice rho = {tr["Dice"]:.3f}   Dice M = {tm["Dice"]:.3f}   mean = {0.5*(tr["Dice"] + tm["Dice"]):.3f}')
""")

md(r"""
## Part A2: sensitivity of the proposed method to eps (training model A)
""")
code(r"""
sens = {}
f_s = os.path.join(OUT_DIR, 'results_eps_sensitivity.pkl')
if os.path.exists(f_s): sens = pickle.load(open(f_s, 'rb'))
for eps in (0.01, 0.02, 0.05):
    if eps in sens: continue
    PgA.eps = PmA.eps = eps
    r, M, info = invert_joint(PgA, PmA, kappa=KAPPA); sens[eps] = (r, M); pickle.dump(sens, open(f_s, 'wb'))
PgA.eps = PmA.eps = EPS
rows = []
for eps, (r, M) in sens.items():
    tr, tm = truth_metrics(r, rhoA, mesh, 0.25*rhoA.max(), mesh.core), truth_metrics(M, MA, mesh, 0.25*MA.max(), mesh.core)
    rows.append(dict(eps=eps, Dice_rho=tr['Dice'], Dice_M=tm['Dice'], CentroidErr_rho=tr['CentroidErr_m'], CentroidErr_M=tm['CentroidErr_m']))
T_eps = pd.DataFrame(rows).set_index('eps').sort_index(); T_eps.to_csv(os.path.join(OUT_DIR, 'table_eps_sensitivity.csv')); show_table(T_eps)
""")

md(r"""
## Part D: statistical test on ten random synthetic models
So that the comparison does not depend on one or two models, ten random models are built on the Sangan geometry. Each model has 2-4 magnetite bodies
(dipping sheets with random dip and strike, or blocks) with random depth, thickness, density (0.8-1.8) and magnetization (20-55 A/m).
Half of the models also contain one density-only or magnetization-only body.
All methods are run with the same rules and compared with the paired **Wilcoxon signed-rank test**.
""")
code(r"""
N_RANDOM = 10; STAT_METHODS = ['L2', 'PTSS', 'L1', 'MS']   # MGS was consistently the weakest method in the earlier tests
def random_model(seed):
    rng = np.random.default_rng(1000 + seed); rho = np.zeros(mesh.n); M = np.zeros(mesh.n)
    x0, x1, y0, y1 = xs.min() + 100, xs.max() - 100, ys.min() + 100, ys.max() - 100
    for _ in range(rng.integers(2, 5)):
        top = rng.uniform(25, 150); thick = rng.uniform(60, 180); cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
        L, W = rng.uniform(150, 350), rng.uniform(50, 120); az = rng.uniform(0, np.pi)
        u = (Xc - cx)*np.cos(az) + (Yc - cy)*np.sin(az); v = -(Xc - cx)*np.sin(az) + (Yc - cy)*np.cos(az)
        if rng.random() < 0.6:                                   # dipping sheet
            dip = np.radians(rng.uniform(35, 90)); v = v - (Dc - top)/np.tan(dip)
        body = (np.abs(u) < L/2) & (np.abs(v) < W/2) & (Dc > top) & (Dc < top + thick)
        rho[body] = rng.uniform(0.8, 1.8); M[body] = rng.uniform(20, 55)
    if rng.random() < 0.5:                                       # single-property body (decoy)
        cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1); top = rng.uniform(40, 150)
        body = (np.abs(Xc - cx) < 80) & (np.abs(Yc - cy) < 80) & (Dc > top) & (Dc < top + 120) & (rho == 0)
        if rng.random() < 0.5: rho[body] = rng.uniform(0.4, 0.8)
        else: M[body] = rng.uniform(15, 40)
    return rho, M

stat_rows = []
for s_ in range(N_RANDOM):
    rt, Mt = random_model(s_); rng = np.random.default_rng(2000 + s_)
    dg_, sg_ = noisy(Ag, rt, FLOOR_G, P_GRAV, rng.uniform(-0.5, 0.5), rng); dm_, sm_ = noisy(Am, Mt, FLOOR_M, P_MAG, rng.uniform(-800, 800), rng)
    Pg_, Pm_ = make_problem(Ag, dg_, sg_, BOUNDS_RHO, 'g'), make_problem(Am, dm_, sm_, BOUNDS_M, 'm')
    keep = BASELINES[:]; BASELINES[:] = ['L2', 'L1', 'MS']
    res_ = run_all(Pg_, Pm_, f'random{s_:02d}', (KAPPA,), (LAM,)); BASELINES[:] = keep
    for k in STAT_METHODS + ['JMS']:
        r, M, _ = res_[k]
        tr = truth_metrics(r, rt, mesh, 0.25*rt.max(), mesh.core); tm = truth_metrics(M, Mt, mesh, 0.25*Mt.max(), mesh.core)
        stat_rows.append(dict(model=s_, Method=k, Dice_rho=tr['Dice'], Dice_M=tm['Dice'], Dice_mean=0.5*(tr['Dice'] + tm['Dice']),
                              CentroidErr_rho=tr['CentroidErr_m'], CentroidErr_M=tm['CentroidErr_m'], DepthErr_rho=tr['DepthErr_m'], DepthErr_M=tm['DepthErr_m']))
T_stat = pd.DataFrame(stat_rows); T_stat.to_csv(os.path.join(OUT_DIR, 'table_random_models_all.csv'), index=False)
agg = T_stat.groupby('Method').agg(['mean', 'std']).drop(columns='model').loc[STAT_METHODS + ['JMS']]
agg.columns = [f'{a}_{b}' for a, b in agg.columns]; agg.to_csv(os.path.join(OUT_DIR, 'table_random_models_summary.csv')); show_table(agg)

from scipy.stats import wilcoxon
rows = []
for k in STAT_METHODS:
    for metric, better in (('Dice_rho', 'max'), ('Dice_M', 'max'), ('Dice_mean', 'max'), ('CentroidErr_rho', 'min'), ('CentroidErr_M', 'min')):
        a = T_stat[T_stat.Method == 'JMS'].set_index('model')[metric]; b = T_stat[T_stat.Method == k].set_index('model')[metric]
        ok = a.notna() & b.notna(); diff = (a - b)[ok] if better == 'max' else (b - a)[ok]
        p = wilcoxon(diff, alternative='greater').pvalue if (diff != 0).sum() > 0 else np.nan
        rows.append(dict(vs=k, metric=metric, JMS_better_in=f'{int((diff > 0).sum())}/{int(ok.sum())}', mean_improvement=float(diff.mean()), p_value=p))
T_wil = pd.DataFrame(rows); T_wil.to_csv(os.path.join(OUT_DIR, 'table_random_models_wilcoxon.csv'), index=False); show_table(T_wil)

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for ax, (c, lab) in zip(axs, (('Dice_rho', 'Dice density ↑'), ('Dice_M', 'Dice magnetization ↑'), ('CentroidErr_M', 'Centroid error M (m) ↓'))):
    data = [T_stat[T_stat.Method == k][c].dropna().values for k in STAT_METHODS + ['JMS']]
    bp = ax.boxplot(data, patch_artist=True); ax.set_xticks(range(1, len(data) + 1)); ax.set_xticklabels(STAT_METHODS + ['JMS'])
    for patch, k in zip(bp['boxes'], STAT_METHODS + ['JMS']): patch.set_facecolor(COLORS[k]); patch.set_alpha(0.6)
    ax.set_title(lab)
fig.suptitle(f'Figure S6. {N_RANDOM} random synthetic models on the Sangan geometry'); fig.tight_layout(); save(fig, 'S6_random_models_boxplots.png'); plt.show()
""")

md(r"""
## Part C: Sangan field data
This part runs only when the field data and the borehole logs are available (`RUN_FIELD`); they are available from the corresponding author on reasonable request.

With induced magnetization only and susceptibility <= 1.5 SI, the Sangan magnetic data cannot be fitted (lowest chi2/N about 6.8).
Massive magnetite usually carries remanence, so for the field data the bound is on the **effective magnetization** `(1+Q)*chi_max*H` with `Q <= 1.6`.
First the **best attainable misfit** within these bounds is estimated. If it exceeds 1, the misfit target of every method is set to 1.1 times this value.
""")
code(r"""
Pg, Pm = make_problem(Ag, d_grav, sig_g, BOUNDS_RHO, 'g'), make_problem(Am, d_mag, sig_m, BOUNDS_M_REAL, 'm')
ffile = os.path.join(OUT_DIR, 'misfit_floor_Q.pkl')
if os.path.exists(ffile): floors = pickle.load(open(ffile, 'rb'))
else: floors = (misfit_floor(Pg), misfit_floor(Pm)); pickle.dump(floors, open(ffile, 'wb'))
TARGETS = tuple(max(1.0, 1.1*f) for f in floors)
print(f'Best attainable fit: gravity chi2/N = {floors[0]:.2f}, magnetics chi2/N = {floors[1]:.2f}')
print(f'Misfit target (all methods): gravity {TARGETS[0]:.2f}, magnetics {TARGETS[1]:.2f}')
if floors[1] > 1.5:
    print('Warning: the magnetic data cannot be fully explained with susceptibility <= 1.5 SI and induced magnetization only:'
          ' this indicates higher susceptibility in massive magnetite or remanent magnetization.')
resR = run_all(Pg, Pm, 'realQ', (KAPPA,), (LAM,), TARGETS)
""")
code(r"""
orr = ORDER; rows = []
for k in orr:
    r, M, info = resR[k]; fg, fm = field_metrics(r, Pg, D, mesh.core), field_metrics(M, Pm, D, mesh.core)
    rows.append(dict(Method=k, chi2N_grav=fg['chi2N'], chi2N_mag=fm['chi2N'], RMS_grav_mGal=fg['RMS'], RMS_mag_nT=fm['RMS'],
                     V90_rho=fg['V90'], V90_M=fm['V90'], Frac_M_at_upper=float(np.mean(M[mesh.core] >= 0.99*BOUNDS_M_REAL[1])),
                     StructCorr_rho_M=structural_corr(r, M, D), Time_s=info['time']))
T_real = pd.DataFrame(rows).set_index('Method'); T_real.to_csv(os.path.join(OUT_DIR, 'table_real_metrics.csv')); show_table(T_real)
""")

md(r"""
### Validation with the Sangan boreholes (C-South anomaly)
Magnetite intervals (low = 1, medium = 2, high = 3) are read from `BOREHOLE_FILE` (borehole logging report of the C-South anomaly, 2006, unpublished; not public).
**AUC** is the probability that the model gives a point with medium/high magnetite a larger value than a point without it (0.5 = random, 1 = perfect).
Borehole 314 lies outside the survey and is excluded from the mean.
""")
code(r"""
# the borehole logs are read from BOREHOLE_FILE in Section 3
from scipy.stats import spearmanr
from scipy.interpolate import RegularGridInterpolator
_topo_i = RegularGridInterpolator((mesh.yc, mesh.xc), mesh.topo, bounds_error=False, fill_value=None)
rows, prof = [], {}
for name, bh in BOREHOLES.items():
    if bh['elev'] is None: bh['elev'] = float(_topo_i([[bh['Y'] - Y0, bh['X'] - X0]])[0])
    x, y, z, md_ = borehole_trace(bh, X0, Y0); lab = borehole_labels(bh, md_)
    inside = (x >= xs.min()) & (x <= xs.max()) & (y >= ys.min()) & (y <= ys.max())
    prof[name] = dict(md=md_, inside=inside, M={}, rho={})
    for k in orr:
        M_ = sample_along(mesh, resR[k][1], x, y, z); r_ = sample_along(mesh, resR[k][0], x, y, z)
        prof[name]['M'][k], prof[name]['rho'][k] = M_, r_; ok = ~np.isnan(M_)
        rows.append(dict(Borehole=name, InSurvey=bool(inside.mean() > 0.5), Method=k, AUC_M=auc(M_, lab >= 2), AUC_rho=auc(r_, lab >= 2),
                         Spearman_M=float(spearmanr(M_[ok], lab[ok])[0]) if ok.sum() > 10 and np.ptp(M_[ok]) > 0 else np.nan))
T_bh = pd.DataFrame(rows); T_bh.to_csv(os.path.join(OUT_DIR, 'table_boreholes.csv'), index=False)
show_table(T_bh.pivot(index='Method', columns='Borehole', values='AUC_M').loc[orr])
summary = T_bh[T_bh.InSurvey].groupby('Method')[['AUC_M', 'AUC_rho', 'Spearman_M']].mean().loc[orr]
print('Mean over the boreholes inside the survey:'); show_table(summary); summary.to_csv(os.path.join(OUT_DIR, 'table_boreholes_summary.csv'))

fig, axs = plt.subplots(2, len(BOREHOLES), figsize=(4.2*len(BOREHOLES), 9))
for c, (name, p) in enumerate(prof.items()):
    for r_, (key, lab_) in enumerate((('M', 'M (A/m)'), ('rho', 'ρ (g/cm³)'))):
        ax = axs[r_, c]
        for a, b, k in BOREHOLES[name]['intervals']: ax.axhspan(a, b, color={1: '#ffe9a8', 2: '#ffc04d', 3: '#e07b00'}[k], alpha=0.6, lw=0)
        for k in orr: ax.plot(p[key][k], p['md'], color=COLORS[k], lw=1.6 if k in ('L2', 'JMS') else 0.7, label=k)
        ax.invert_yaxis(); ax.set_xlabel(lab_); ax.set_title(name + ('' if p['inside'].mean() > 0.5 else ' (outside survey)'))
        if c == 0: ax.set_ylabel('along-hole depth (m)')
axs[0, 0].legend(fontsize=7)
fig.suptitle('Figure B1. Model values along boreholes (shading: logged magnetite low / medium / high)'); fig.tight_layout()
save(fig, 'B1_boreholes.png'); plt.show()
""")
code(r"""
# field-data sections through the magnetic maximum and through borehole BH311
imax = np.argmax(d_mag); vr, vm = BOUNDS_RHO[1]*0.75, BOUNDS_M_REAL[1]*0.6
section_figure(resR, orr, 'x', ys[imax], 'R2_real_EW_magmax.png', 'Sangan real data', None, (vr, vm))
section_figure(resR, orr, 'y', xs[imax], 'R3_real_NS_magmax.png', 'Sangan real data', None, (vr, vm))
section_figure(resR, orr, 'x', BOREHOLES['BH311']['Y'] - Y0, 'R4_real_EW_BH311.png', 'Sangan real data (through BH311)', None, (vr, vm))

# horizontal slices at three depths
ks = [np.argmin(np.abs(mesh.dc - z)) for z in (37.5, 137.5, 262.5)]; ext = [mesh.xe[0], mesh.xe[-1], mesh.ye[0], mesh.ye[-1]]
for pi, (prop, vmax, cmap) in enumerate((('ρ', vr, 'viridis'), ('M', vm, 'magma'))):
    fig, axs = plt.subplots(len(ks), len(orr), figsize=(3.1*len(orr), 2.8*len(ks)))
    for r_, k in enumerate(ks):
        for j, name in enumerate(orr):
            im = axs[r_, j].imshow(mesh.to_grid(resR[name][pi])[k], origin='lower', extent=ext, cmap=cmap, vmin=0, vmax=vmax)
            for bn, bh in BOREHOLES.items(): axs[r_, j].plot(bh['X'] - X0, bh['Y'] - Y0, 'c^', ms=4)
            axs[r_, j].set_title(f'{name}  depth {mesh.dc[k]:.0f} m', fontsize=8); axs[r_, j].set_xticks([]); axs[r_, j].set_yticks([])
    plt.colorbar(im, ax=axs.ravel().tolist(), shrink=0.7)
    fig.suptitle(f'Sangan depth slices: {prop} (cyan triangles = borehole collars)'); save(fig, f'R5_real_depthslices_{prop}.png'); plt.show()
""")
md(r"""
### Sangan equivalents of Figs 10 and 11 of Ming et al. (2024)
As in Ming et al. (2024), the iso-level of each model is calibrated at the **logged top of mineralization in a borehole**: the model value in BH311 at the first
medium/high magnetite (188 m along the hole). The top of this iso-surface is then mapped over the whole area.
""")
code(r"""
CAL_BH, CAL_MD = 'BH311', 188.0
bh = BOREHOLES[CAL_BH]; x_, y_, z_, md_ = borehole_trace(bh, X0, Y0); kcal = np.argmin(np.abs(md_ - CAL_MD))
SHOW = ['L2', 'PTSS', 'JMS']; iso = {}
for k in SHOW:
    vals = []
    for i in (0, 1):
        v = sample_along(mesh, resR[k][i], x_[kcal:kcal+1], y_[kcal:kcal+1], z_[kcal:kcal+1])[0]
        mx = np.percentile(resR[k][i][mesh.core], 99.5)
        vals.append(v if (np.isfinite(v) and v > 0.05*mx) else 0.3*mx)   # if the model is negligible at that point: 30% of the maximum
    iso[k] = vals
    print(f'{k:5s}: density iso-level = {vals[0]:.3f} g/cm^3, magnetization iso-level = {vals[1]:.2f} A/m')

# equivalent of Fig. 10: E-W section through BH311 with calibrated iso-surfaces and borehole traces
jy = np.argmin(np.abs(mesh.yc - (bh['Y'] - Y0)))
fig, axs = plt.subplots(2, len(SHOW), figsize=(5.2*len(SHOW), 7.5), sharey=True)
for j, k in enumerate(SHOW):
    for i, (lab, cmap, vmax) in enumerate((('ρ (g/cm³)', 'viridis', BOUNDS_RHO[1]*0.6), ('M (A/m)', 'magma', BOUNDS_M_REAL[1]*0.6))):
        ax = axs[i, j]; pc = section(ax, resR[k][i], 'x', jy, vmax, cmap, f'{k}: {lab}')
        g = mesh.to_grid(resR[k][i])[:, jy, :]; Zc = mesh.topo[jy][None, :] - mesh.dc[:, None]
        ax.contour(np.broadcast_to(mesh.xc[None], Zc.shape), Zc, g, [iso[k][i]], colors='red', linestyles='--', linewidths=1.5)
        for nm, b in BOREHOLES.items():
            xx, yy, zz, mm = borehole_trace(b, X0, Y0); sel = np.abs(yy - (b['Y'] - Y0 + 0)) >= 0
            if abs(b['Y'] - bh['Y']) < 200:
                ax.plot(xx, zz, 'c-', lw=1.5)
                lab_ = borehole_labels(b, mm); ax.plot(xx[lab_ >= 2], zz[lab_ >= 2], 'o', color='gold', ms=2)
                ax.text(xx[0], zz[0] + 8, nm, color='c', fontsize=7, ha='center')
        plt.colorbar(pc, ax=ax, shrink=0.8)
    axs[1, j].set_xlabel('x (m)')
for ax in axs[:, 0]: ax.set_ylabel('elevation (m)')
fig.suptitle('Equivalent of paper Fig. 10 — Sangan E–W section through BH311 (red dashed: borehole-calibrated iso-surface; gold: logged medium/high magnetite)')
fig.tight_layout(); save(fig, 'P1_paperstyle_section_BH311.png'); plt.show()

# equivalent of Fig. 11: depth below ground to the top of the iso-surfaces
def top_depth(m, level):
    g = mesh.to_grid(m) >= level; anyh = g.any(0); k = np.argmax(g, axis=0)
    z = mesh.dc[k].astype(float); z[~anyh] = np.nan; return z
sl = (slice(PAD, -PAD), slice(PAD, -PAD)); extc = [mesh.xe[PAD], mesh.xe[-PAD-1], mesh.ye[PAD], mesh.ye[-PAD-1]]
fig, axs = plt.subplots(2, len(SHOW), figsize=(5*len(SHOW), 8))
for j, k in enumerate(SHOW):
    for i, lab in enumerate(('density iso-surface', 'magnetization iso-surface')):
        z = top_depth(resR[k][i], iso[k][i])[sl]
        im = axs[i, j].imshow(z, origin='lower', extent=extc, cmap='RdYlBu', vmin=0, vmax=DEPTH)
        for nm, b in BOREHOLES.items(): axs[i, j].plot(b['X'] - X0, b['Y'] - Y0, 'k^', ms=5); axs[i, j].text(b['X'] - X0, b['Y'] - Y0 + 20, nm, fontsize=7)
        axs[i, j].set_title(f'{k}: depth to top of {lab}', fontsize=9); plt.colorbar(im, ax=axs[i, j], label='depth below ground (m)')
fig.suptitle('Equivalent of paper Fig. 11 — depth to top of calibrated iso-surfaces (Sangan)'); fig.tight_layout()
save(fig, 'P2_paperstyle_depthmaps.png'); plt.show()

# short table: predicted versus logged depth to the top of magnetite in each borehole
rows = []
for nm, b in BOREHOLES.items():
    xx, yy, zz, mm = borehole_trace(b, X0, Y0); lab_ = borehole_labels(b, mm)
    obs = mm[np.argmax(lab_ >= 2)] if (lab_ >= 2).any() else np.nan
    row = dict(Borehole=nm, Logged_top_med_high_MD=obs)
    for k in SHOW:
        Mv = sample_along(mesh, resR[k][1], xx, yy, zz); hit = Mv >= iso[k][1]
        row[f'Pred_top_{k}'] = mm[np.argmax(hit)] if hit.any() else np.nan
    rows.append(row)
T_top = pd.DataFrame(rows); T_top.to_csv(os.path.join(OUT_DIR, 'table_paperstyle_ore_top.csv'), index=False); show_table(T_top)
""")
md(r"""
### Why does the smooth model score a higher AUC in two boreholes?
For each borehole: number and thickness of medium/high magnetite intervals, and the fraction of the hole and of the ore intervals in which each model is non-zero.
""")
code(r"""
rows = []
for name in ('BH311', 'BH312A', 'BH313'):
    bh = BOREHOLES[name]; x, y, z, md_ = borehole_trace(bh, X0, Y0); lab = borehole_labels(bh, md_); ore = lab >= 2
    iv = [(a, b) for a, b, k in bh['intervals'] if k >= 2]
    row = dict(Borehole=name, n_med_high_intervals=len(iv), ore_fraction_of_hole=float(ore.mean()),
               mean_interval_thickness_m=float(np.mean([b - a for a, b in iv])), ore_top_md=float(md_[ore].min()), ore_bottom_md=float(md_[ore].max()))
    for k in ('L2', 'MS', 'JMS'):
        M_ = sample_along(mesh, resR[k][1], x, y, z); mx = np.percentile(resR[k][1][mesh.core], 99.5); nz = M_ > 0.01*mx; hi_ = M_ >= 0.3*mx
        row[f'{k}_frac_hole_nonzero'] = float(np.nanmean(nz)); row[f'{k}_frac_ore_with_nonzero_M'] = float(np.nanmean(nz[ore]))
        row[f'{k}_highM_md_range'] = f'{md_[hi_].min():.0f}-{md_[hi_].max():.0f}' if hi_.any() else 'none'; row[f'{k}_AUC'] = auc(M_, ore)
    rows.append(row)
T_bhd = pd.DataFrame(rows).set_index('Borehole'); T_bhd.to_csv(os.path.join(OUT_DIR, 'table_borehole_diagnostics.csv')); show_table(T_bhd.T)
""")
code(r"""
# data fit: observed, predicted by smooth and proposed method, normalized residual, histograms of all methods
fig, axs = plt.subplots(2, 5, figsize=(21, 7.5))
for i, (P, u) in enumerate(((Pg, 'mGal'), (Pm, 'nT'))):
    lim = np.percentile(np.abs(P.d), 99)
    P.misfit(resR['L2'][i]); pL2 = P.predict(resR['L2'][i]); P.misfit(resR['JMS'][i]); pJ = P.predict(resR['JMS'][i])
    for j, (v, t) in enumerate(((P.d, 'observed'), (pL2, 'predicted L2'), (pJ, 'predicted JMS'))):
        tc = axs[i, j].tricontourf(xs, ys, v, 40, cmap='RdYlBu_r', vmin=-lim, vmax=lim); axs[i, j].set_title(f'{P.name}: {t}'); plt.colorbar(tc, ax=axs[i, j], label=u)
    tc = axs[i, 3].tricontourf(xs, ys, (pJ - P.d)/P.s, 40, cmap='coolwarm', vmin=-3, vmax=3); axs[i, 3].set_title('JMS normalised residual'); plt.colorbar(tc, ax=axs[i, 3])
    for k in orr:
        P.misfit(resR[k][i]); axs[i, 4].hist((P.predict(resR[k][i]) - P.d)/P.s, bins=np.linspace(-5, 5, 60), histtype='step', color=COLORS[k], label=k, density=True)
    xx = np.linspace(-5, 5, 200); axs[i, 4].plot(xx, np.exp(-xx**2/2)/np.sqrt(2*np.pi), 'k--', label='N(0,1)'); axs[i, 4].legend(fontsize=7); axs[i, 4].set_title('normalised residuals')
for ax in axs[:, :4].ravel(): ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('Figure R6. Data fit (Sangan)'); fig.tight_layout(); save(fig, 'R6_real_datafit.png'); plt.show()

# export the models with UTM coordinates
out = pd.DataFrame(dict(X=mesh.Xc + X0, Y=mesh.Yc + Y0, Elevation=mesh.Ec, Depth=mesh.Dc, InSurvey=mesh.core))
for k in orr: out[f'rho_{k}'] = resR[k][0]; out[f'M_{k}'] = resR[k][1]; out[f'chi_eff_SI_{k}'] = resR[k][1]/(F_TOTAL*1e-9/(4e-7*np.pi))
out.to_csv(os.path.join(OUT_DIR, 'models_real.csv'), index=False)
""")

import textwrap
ic = next(i for i, (t, s_) in enumerate(cells) if t == 'markdown' and s_.lstrip().startswith('## Part C'))
for i in range(ic + 1, len(cells)):
    t, s_ = cells[i]
    if t == 'code':
        s_ = s_.strip('\n')
        cells[i] = (t, '\nif RUN_FIELD:\n' + textwrap.indent(s_, '    ') + '\nelse:\n    print("Field data not available: cell skipped")\n')
code(r"""
# all tables in one Excel file (field-data sheets only when Part C was run)
with pd.ExcelWriter(os.path.join(OUT_DIR, 'tables.xlsx')) as xw:
    TB.to_excel(xw, sheet_name='B_test_accuracy'); TA.to_excel(xw, sheet_name='A_training')
    T_asym.to_excel(xw, sheet_name='B_spurious_asym', index=False)
    if RUN_FIELD:
        T_real.to_excel(xw, sheet_name='Sangan_fit')
        T_bh.to_excel(xw, sheet_name='Sangan_boreholes', index=False); summary.to_excel(xw, sheet_name='Sangan_boreholes_mean')
        T_top.to_excel(xw, sheet_name='Sangan_ore_top_paperstyle', index=False); T_bhd.to_excel(xw, sheet_name='Sangan_borehole_diag')
print('Saved in', OUT_DIR)
""")
nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}},
      "cells": []}
for i, (t, s) in enumerate(cells):
    c = {"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True), "id": f"c{i:02d}"}
    if t == 'code': c.update(execution_count=None, outputs=[])
    nb["cells"].append(c)
(HERE/'Focusing_Joint_Inversion.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
script = "import matplotlib; matplotlib.use('Agg')\n" + "\n\n".join(s for t, s in cells if t == 'code')
(HERE/'run_all.py').write_text(script.replace('plt.show()', 'plt.close(fig)'), encoding='utf-8')
print('built', len(cells), 'cells')
