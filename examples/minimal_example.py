# Minimal example: build a small model, compute synthetic gravity and magnetic data,
# and invert them with the smooth L2 method and with the proposed JMS method.
# Run from the repository root:   python examples/minimal_example.py   (a few seconds)
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jms_lib import *

# 1. Stations: a 20 x 20 grid with 50 m spacing on flat ground (elevation 0 m)
g = np.arange(20)*50.0 + 25.0
X, Y = np.meshgrid(g, g); xs, ys = X.ravel(), Y.ravel(); elev = np.zeros(xs.size)

# 2. Mesh: 50 m cells, 500 m deep, no padding
mesh = DrapedMesh(xs, ys, elev, dx=50.0, dz=50.0, depth=500.0, pad=0)
zs = mesh.station_z(elev, height=1.0)          # station z (positive down from the mesh reference level)
boxes = mesh.boxes()

# 3. Kernels: gravity (mGal per g/cm^3) and total-field magnetic data (nT per A/m), field I = 60, D = 0
Ag = gravity_kernel(boxes, xs, ys, zs)
Am = magnetic_kernel(boxes, xs, ys, zs, inc=60.0, dec=0.0)

# 4. True model: one block with density contrast 1 g/cm^3 and magnetization 40 A/m
body = (abs(mesh.Xc - 500) < 150) & (abs(mesh.Yc - 500) < 150) & (mesh.Dc > 100) & (mesh.Dc < 300)
rho_true, M_true = 1.0*body, 40.0*body

# 5. Synthetic data with 2% noise plus a small floor
rng = np.random.default_rng(0)
dg = Ag @ rho_true.astype(np.float32); dm = Am @ M_true.astype(np.float32)
sg = 0.02*np.abs(dg) + 0.01*np.abs(dg).max(); sm = 0.02*np.abs(dm) + 0.01*np.abs(dm).max()
dg = dg + sg*rng.standard_normal(dg.size); dm = dm + sm*rng.standard_normal(dm.size)

# 6. Inverse problems: depth weighting from the sensitivities, physical bounds, finite-difference operator
D = diff_operator(mesh)
Pg = Problem(Ag, dg, sg, sensitivity_weight(Ag, sg), (0.0, 2.0), D, eps=0.02, name='grav')
Pm = Problem(Am, dm, sm, sensitivity_weight(Am, sm), (0.0, 60.0), D, eps=0.02, name='mag')

# 7. Smooth L2 inversion (each data set separately) and joint JMS inversion (both at once)
r_L2, _ = invert(Pg, 'L2'); M_L2, _ = invert(Pm, 'L2')
r_JMS, M_JMS, info = invert_joint(Pg, Pm, kappa=0.5)

# 8. Accuracy: Dice overlap of the recovered and true bodies (threshold 25% of the true maximum)
for name, r, M in (('L2', r_L2, M_L2), ('JMS', r_JMS, M_JMS)):
    print(f'{name:4s} Dice density = {dice(r >= 0.25, body):.2f}   Dice magnetization = {dice(M >= 10.0, body):.2f}'
          f'   chi2/N = {Pg.misfit(r)[0]:.2f}, {Pm.misfit(M)[0]:.2f}')
