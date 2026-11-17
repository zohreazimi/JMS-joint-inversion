# ==========================================================================================
# jms_lib: topography-draped mesh, gravity and magnetic kernels, inversions (L2, L1, MS, MGS,
# PTSS and the proposed Joint Minimum Support, JMS) and evaluation metrics.
# ==========================================================================================
import numpy as np, pandas as pd, time
import scipy.sparse as sp
from scipy.interpolate import griddata
from scipy.optimize import minimize, Bounds

G_CONST = 6.674e-11          # gravitational constant (SI)
CM, T2NT = 1e-7, 1e9         # mu0/(4 pi) and tesla-to-nanotesla conversion

# ------------------------------------------------------------------ data
def load_residuals(path, keep_dy=20.0):
    """Read a CSV file with columns X, Y, Z, Res GRV, Res Mag.
    The station spacing is 20 m in X and 10 m in Y; keep_dy=20 decimates Y to a 20 m spacing."""
    d = pd.read_csv(path); d.columns = [c.strip() for c in d.columns]
    if keep_dy:
        k = ((d['Y'] - d['Y'].min()) % keep_dy)
        d = d[np.isclose(k, 0) | np.isclose(k, keep_dy)]
    return d.reset_index(drop=True)

def roughness_sigma(x, y, v):
    """Estimate the standard deviation of the high-frequency component (noise + sub-cell heterogeneity) from the data:
    difference of each value from the mean of its 8 neighbours, then a robust (MAD) standard deviation.
    The factor 1/sqrt(1+1/8) makes the estimate exact for white noise."""
    xu, yu = np.unique(x), np.unique(y); ix, iy = np.searchsorted(xu, x), np.searchsorted(yu, y)
    Gd = np.full((yu.size, xu.size), np.nan); Gd[iy, ix] = v; P = np.pad(Gd, 1, constant_values=np.nan)
    import warnings
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)              # grid edges have no neighbours
        nb = np.nanmean(np.stack([P[1+a:1+a+Gd.shape[0], 1+b:1+b+Gd.shape[1]]
                                  for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]), 0)
    r = (Gd - nb)[~np.isnan(Gd)]; r = r[~np.isnan(r)]
    return 1.4826*np.median(np.abs(r - np.median(r)))/np.sqrt(1 + 1/8)

# ------------------------------------------------------------------ topography-draped mesh
class DrapedMesh:
    """Mesh draped from the ground surface: nx*ny columns, each with nz cells hanging below the surface at the column centre.
    Cell k covers the depth interval [k*dz, (k+1)*dz] below the surface, so there are no air cells and no gaps below the stations.
    Arrays are ordered (nz, ny, nx), so difference operators are built as on a regular grid."""
    def __init__(self, xs, ys, elev, dx, dz, depth, pad=5):
        self.dx = self.dy = dx; self.dz = dz
        x0 = xs.min() - (pad + 0.37)*dx; y0 = ys.min() - (pad + 0.37)*dx      # small shift so that no cell edge falls on a station
        self.nx = int(np.ceil((xs.max() + pad*dx - x0)/dx)); self.ny = int(np.ceil((ys.max() + pad*dx - y0)/dx))
        self.nz = int(round(depth/dz)); self.pad = pad
        self.xe = x0 + dx*np.arange(self.nx + 1); self.ye = y0 + dx*np.arange(self.ny + 1); self.de = dz*np.arange(self.nz + 1)
        self.xc = 0.5*(self.xe[1:] + self.xe[:-1]); self.yc = 0.5*(self.ye[1:] + self.ye[:-1]); self.dc = 0.5*(self.de[1:] + self.de[:-1])
        XX, YY = np.meshgrid(self.xc, self.yc)
        t = griddata((xs, ys), elev, (XX, YY), method='linear')
        tn = griddata((xs, ys), elev, (XX, YY), method='nearest'); t[np.isnan(t)] = tn[np.isnan(t)]
        self.topo = t; self.top = float(np.ceil(t.max()))                    # reference level for the downward z coordinate
        self.shape = (self.nz, self.ny, self.nx); self.n = self.nz*self.ny*self.nx
        Dd, Y, X = np.meshgrid(self.dc, self.yc, self.xc, indexing='ij')
        T = np.broadcast_to(self.topo[None], self.shape)
        self.Xc, self.Yc, self.Dc = X.ravel(), Y.ravel(), Dd.ravel()         # Dc: depth of the cell centre below ground
        self.Ec = (T - Dd).ravel()                                           # Ec: elevation of the cell centre
        core = np.zeros((self.ny, self.nx), bool); core[pad:-pad, pad:-pad] = True
        self.core = np.broadcast_to(core[None], self.shape).ravel()          # cells inside the survey area (no padding)
    def to_grid(self, m): return np.asarray(m, float).reshape(self.shape)
    def boxes(self):
        ztop = self.top - (self.Ec + self.dz/2)
        return (self.Xc - self.dx/2, self.Xc + self.dx/2, self.Yc - self.dy/2, self.Yc + self.dy/2, ztop, ztop + self.dz)
    def station_z(self, elev, height=1.0): return self.top - (elev + height)

def diff_operator(mesh):
    """Forward differences between neighbouring cells in x, y and z (in cell-index units). Returns a sparse (n_pairs x n) matrix."""
    idx = np.arange(mesh.n).reshape(mesh.shape); rows, cols, vals = [], [], []; r = 0
    for ax in range(3):
        a = np.moveaxis(idx, ax, 0); i0, i1 = a[:-1].ravel(), a[1:].ravel(); k = np.arange(r, r + i0.size)
        rows += [k, k]; cols += [i0, i1]; vals += [-np.ones(i0.size), np.ones(i0.size)]; r += i0.size
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r, mesh.n))

# ------------------------------------------------------------------ parallel computation with progress display
def _run_chunks(func, nd, chunk, label):
    """Run func(s, e) over blocks of stations in several threads (numpy releases the GIL),
    printing the percentage done and the estimated remaining time."""
    import os, sys
    from concurrent.futures import ThreadPoolExecutor, as_completed
    blocks = [(s, min(s + chunk, nd)) for s in range(0, nd, chunk)]
    nw = max(1, min(4, (os.cpu_count() or 2))); t0 = time.time(); done = 0; last = -1
    with ThreadPoolExecutor(max_workers=nw) as ex:
        futs = [ex.submit(func, s, e) for s, e in blocks]
        for f in as_completed(futs):
            f.result(); done += 1; pct = int(100*done/len(blocks))
            if pct // 5 != last // 5 or done == len(blocks):
                el = time.time() - t0; rem = el/done*(len(blocks) - done)
                print(f'\r  {label}: {pct:3d}%  elapsed {el/60:4.1f} min | remaining ~{rem/60:4.1f} min ({nw} threads)', end='', flush=True)
                last = pct
    print(flush=True)

# ------------------------------------------------------------------ kernels (closed-form prism formulas, valid at any point outside the prism)
# With topography some cells lie above nearby stations; these formulas are correct for points above, beside and below a prism
# (checked against point-mass, dipole and numerical-integration solutions to better than 0.1%).
def _corners(boxes, x0, y0, z0):
    X1, X2, Y1, Y2, Z1, Z2 = boxes
    for xi, sx in ((X1, -1), (X2, 1)):
        for yj, sy in ((Y1, -1), (Y2, 1)):
            for zk, sz in ((Z1, -1), (Z2, 1)):
                yield xi[None] - x0, yj[None] - y0, zk[None] - z0, sx*sy*sz
def _sl(v): return np.log(np.maximum(v, 1e-12))

def gravity_kernel(boxes, xs, ys, zs, dtype=np.float32, chunk=25):
    """A[i,j] = gz (mGal) at station i due to cell j with a density contrast of 1 g/cm^3 (z positive downwards)."""
    A = np.empty((xs.size, boxes[0].size), dtype=dtype)
    def work(s, e):
        tot = 0.0
        for x, y, z, sg in _corners(boxes, xs[s:e, None], ys[s:e, None], zs[s:e, None]):
            r = np.sqrt(x*x + y*y + z*z)
            with np.errstate(divide='ignore', invalid='ignore'):
                t = z*np.arctan(x*y/(z*r)) - x*_sl(r + y) - y*_sl(r + x)
            tot = tot + sg*np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        A[s:e] = (G_CONST*1000.0*1e5*tot).astype(dtype)
    _run_chunks(work, xs.size, chunk, 'gravity kernel')
    return A

def dircos(inc, dec):
    i, d = np.radians(inc), np.radians(dec); return np.array([np.cos(i)*np.cos(d), np.cos(i)*np.sin(d), np.sin(i)])

def magnetic_kernel(boxes, xs, ys, zs, inc, dec, inc_m=None, dec_m=None, dtype=np.float32, chunk=25):
    """A[i,j] = total-field anomaly (nT) at station i due to cell j with a magnetization of 1 A/m.
    Magnetization is assumed parallel to the inducing field (inc, dec). dT = f.B with B = (mu0/4 pi) T M (T: prism tensor)."""
    f = dircos(inc, dec); md = dircos(inc if inc_m is None else inc_m, dec if dec_m is None else dec_m)
    A = np.empty((xs.size, boxes[0].size), dtype=dtype)
    def work(s, e):
        T = dict(xx=0., yy=0., zz=0., xy=0., xz=0., yz=0.)
        for x, y, z, sg in _corners(boxes, xs[s:e, None], ys[s:e, None], zs[s:e, None]):
            r = np.sqrt(x*x + y*y + z*z)
            with np.errstate(divide='ignore', invalid='ignore'):
                v = dict(xx=np.arctan(y*z/(x*r)), yy=np.arctan(x*z/(y*r)), zz=np.arctan(x*y/(z*r)),
                         xy=-_sl(z + r), xz=-_sl(y + r), yz=-_sl(x + r))
            for k in T: T[k] = T[k] + sg*np.nan_to_num(v[k], nan=0.0, posinf=0.0, neginf=0.0)
        M = [[T['xx'], T['xy'], T['xz']], [T['xy'], T['yy'], T['yz']], [T['xz'], T['yz'], T['zz']]]
        out = sum(f[a]*M[a][b]*md[b] for a in range(3) for b in range(3))
        A[s:e] = (-CM*T2NT*out).astype(dtype)
    _run_chunks(work, xs.size, chunk, 'magnetic kernel')
    return A

def sensitivity_weight(A, sigma):
    """Integrated-sensitivity depth weighting: w_j = (sum_i (A_ij/sigma_i)^2)^(1/4), normalized to a maximum of 1.
    Equivalent to the z^(-beta/2) weighting on flat ground, but also valid under topography (Portniaguine & Zhdanov 2002)."""
    S = np.zeros(A.shape[1])
    for i in range(0, A.shape[0], 500):
        a = A[i:i+500].astype(np.float64); S += ((a/sigma[i:i+500, None])**2).sum(0)
    w = np.sqrt(np.sqrt(S)); return w/w.max()

# ------------------------------------------------------------------ inverse problem (common to all methods)
class Problem:
    """Dimensionless data misfit: phi_d = chi2/N = mean(((A m - d)/sigma)^2). The target is 1 for every method.
    Stabilizers act on the dimensionless model u = w*m/scale (scale = largest absolute bound),
    so all terms are dimensionless and comparable and no regularization weight carries units."""
    def __init__(self, A, d, sigma, w, bounds, D, eps=0.02, name=''):
        self.A, self.d, self.s, self.w, self.D, self.eps, self.name = A, d, sigma, w, D, eps, name
        self.lo, self.hi = bounds; self.scale = max(abs(bounds[0]), abs(bounds[1])); self.N = d.size
        self.offset = False; self.c = 0.0; self._w2 = 1/sigma**2
    def predict(self, m, with_offset=True):
        p = (self.A @ m.astype(self.A.dtype)).astype(np.float64)
        return p + (self.c if (self.offset and with_offset) else 0.0)
    def misfit(self, m):
        # If offset is enabled, a constant (base level of the residual data) is estimated analytically and optimally.
        # After regional removal the zero level of residual data is arbitrary; this constant compensates for it (identically for all methods).
        p = (self.A @ m.astype(self.A.dtype)).astype(np.float64)
        if self.offset: self.c = float(np.sum((self.d - p)*self._w2)/np.sum(self._w2))
        r = (p + (self.c if self.offset else 0.0) - self.d)/self.s; return float(np.mean(r**2)), r
    def grad_misfit(self, r): return (2.0/self.N)*(self.A.T @ (r/self.s).astype(self.A.dtype)).astype(np.float64)
    def R(self, m, kind):
        """Value and gradient of the stabilizer. kind: L2 (smooth), L1, MS (minimum support),
        MGS (minimum gradient support). The proposed JMS stabilizer is implemented separately in invert_joint."""
        c = self.w/self.scale; u = c*m; e = self.eps; n = m.size
        if kind == 'L2':
            return np.mean(u**2), 2*u*c/n
        if kind == 'L1':
            q = np.sqrt(u**2 + e**2); return np.mean(q), (u/q)*c/n
        if kind == 'MS':                                        # Last & Kubik 1983; Portniaguine & Zhdanov 1999
            q = u**2 + e**2; return np.mean(u**2/q), (2*u*e**2/q**2)*c/n
        if kind == 'MGS':                                       # Portniaguine & Zhdanov 1999
            g = self.D @ u; q = g**2 + e**2
            return np.sum(g**2/q)/n, (self.D.T @ (2*g*e**2/q**2))*c/n
        raise ValueError(kind)

def invert(P, kind, target=1.0, tol=0.05, cool=0.5, inner=150, max_stages=40, n_bisect=5, verbose=False):
    """Common solver for all methods:
    1) the optimization variable is u = w*m/scale (depth-weighted space = depth preconditioning) with per-cell physical bounds;
    2) continuation in beta: start from a large beta, multiply by cool at each stage, solve each stage with warm-started L-BFGS-B;
    3) once chi2/N crosses the target, beta is refined by log-bisection until |chi2/N - 1| <= tol.
    All methods therefore end with the same bounds, weighting, noise model, solver and data misfit; only the stabilizer differs."""
    c = P.w/P.scale; m = np.zeros(P.A.shape[1]); bnd = Bounds(P.lo*c, P.hi*c)
    def solve(beta, m_start):
        def fun(u):
            x = u/c; fd, r = P.misfit(x); fr, gr = P.R(x, kind)
            return fd + beta*fr, (P.grad_misfit(r) + beta*gr)/c
        u = minimize(fun, m_start*c, jac=True, method='L-BFGS-B', bounds=bnd, options=dict(maxiter=inner, maxcor=20)).x
        x = np.clip(u/c, P.lo, P.hi); return x, P.misfit(x)[0]
    f0 = P.misfit(m)[0]; r0 = P.R(np.full(m.size, P.scale*0.1), kind)[0]
    beta = 10*f0/max(r0, 1e-12); hist = []; t0 = time.time(); prev = None
    for _ in range(12):            # if the first stage already overfits the data, increase beta until chi2 is above the target
        m_try, fd_try = solve(beta, m)
        if fd_try >= target*(1 - tol): break
        hist.append((beta, fd_try)); beta *= 4.0
    for st in range(max_stages):
        m_new, fd = solve(beta, m); hist.append((beta, fd))
        if verbose: print(f'  [{P.name}:{kind}] stage {st:2d} β={beta:9.3e} χ²/N={fd:8.3f} ({time.time()-t0:4.0f}s)', flush=True)
        if fd <= target*(1 + tol):
            if fd >= target*(1 - tol) or prev is None: m = m_new; break
            hi_b, hi_m, lo_b = prev[0], prev[1], beta; best = (abs(fd - target), m_new, beta, fd)
            for _ in range(n_bisect):
                bm = np.sqrt(hi_b*lo_b); mm, fm = solve(bm, hi_m); hist.append((bm, fm))
                if verbose: print(f'  [{P.name}:{kind}]   bisect β={bm:9.3e} χ²/N={fm:8.3f}', flush=True)
                if abs(fm - target) < best[0]: best = (abs(fm - target), mm, bm, fm)
                if abs(fm - target) <= tol*target: break
                if fm > target: hi_b, hi_m = bm, mm
                else: lo_b = bm
            _, m, beta, fd = best; break
        prev = (beta, m_new); m = m_new; beta *= cool
        f = [h[1] for h in hist[-3:]]
        if len(f) == 3 and (max(f) - min(f)) < 0.02*min(f):         # chi2 no longer decreases: misfit floor on this mesh
            if verbose: print(f'  [{P.name}:{kind}] χ²/N stagnated at {fd:.3f}', flush=True)
            break
    return m, dict(beta=beta, chi2=fd, hist=hist, time=time.time() - t0)

# ------------------------------------------------------------------ quantitative metrics
def v90(m):
    """Fraction of the volume holding 90% of the sum of |property| (smaller = more compact)."""
    a = np.sort(np.abs(m))[::-1]; c = np.cumsum(a); return float(np.searchsorted(c, 0.9*c[-1]) + 1)/a.size
def hoyer(v):
    """Hoyer sparsity between 0 and 1 (1 = sparsest / most compact)."""
    v = np.abs(np.asarray(v)).ravel(); n = v.size; l1, l2 = v.sum(), np.sqrt((v**2).sum())
    return float((np.sqrt(n) - l1/(l2 + 1e-30))/(np.sqrt(n) - 1))
def dice(a, b): s = a.sum() + b.sum(); return float(2*(a & b).sum()/s) if s else 1.0
def iou(a, b): u = (a | b).sum(); return float((a & b).sum()/u) if u else 1.0

def truth_metrics(m, mt, mesh, thr, mask=None):
    """Metrics against the true model (synthetic tests only). Body = cells >= thr (same threshold for the true and recovered models)."""
    mask = np.ones(m.size, bool) if mask is None else mask; m, mt = m[mask], mt[mask]
    X, Y, Dd = mesh.Xc[mask], mesh.Yc[mask], mesh.Dc[mask]
    bt, be = mt >= thr, m >= thr
    out = dict(RMSE=float(np.sqrt(np.mean((m - mt)**2))), Corr=float(np.corrcoef(m, mt)[0, 1]),
               Dice=dice(be, bt), IoU=iou(be, bt), V90=v90(m), PeakRatio=float(m.max()/mt.max()))
    if be.any():
        ct = np.array([X[bt].mean(), Y[bt].mean(), Dd[bt].mean()])
        ce = np.array([np.average(X[be], weights=m[be]), np.average(Y[be], weights=m[be]), np.average(Dd[be], weights=m[be])])
        out.update(CentroidErr_m=float(np.linalg.norm(ce - ct)), DepthErr_m=float(abs(ce[2] - ct[2])),
                   VolumeRatio=float(be.sum()/bt.sum()))
    else:
        out.update(CentroidErr_m=np.nan, DepthErr_m=np.nan, VolumeRatio=0.0)
    return out

def field_metrics(m, P, D, mask=None):
    """Metrics without a true model (field data): data fit and model compactness."""
    mask = np.ones(m.size, bool) if mask is None else mask
    fd, r = P.misfit(m); res = P.predict(m) - P.d; g = D @ m
    return dict(chi2N=fd, RMS=float(np.sqrt(np.mean(res**2))), RelMisfit=float(np.linalg.norm(res)/np.linalg.norm(P.d)),
                V90=v90(m[mask]), Hoyer_model=hoyer(m[mask]), Hoyer_grad=hoyer(g), Max=float(m.max()))

def structural_corr(m1, m2, D):
    """Correlation of the gradients of two models (1 = identical structure) over cell pairs where at least one gradient is significant."""
    g1, g2 = D @ m1, D @ m2
    ok = (np.abs(g1) > 0.05*np.abs(g1).max()) | (np.abs(g2) > 0.05*np.abs(g2).max())
    return float(np.corrcoef(g1[ok], g2[ok])[0, 1])

# ------------------------------------------------------------------ borehole validation
def borehole_trace(bh, X0, Y0, step=1.0):
    """Trace of a deviated borehole with constant inclination from vertical and azimuth. Returns local x, y, elevation and measured depth."""
    md = np.arange(0, bh['depth'] + step/2, step); th, az = np.radians(bh['dev']), np.radians(bh['az'])
    x = bh['X'] - X0 + md*np.sin(th)*np.sin(az); y = bh['Y'] - Y0 + md*np.sin(th)*np.cos(az)
    return x, y, bh['elev'] - md*np.cos(th), md

def borehole_labels(bh, md):
    """Magnetite class along the hole from the logged intervals: 0 none, 1 low, 2 medium, 3 high (highest class where intervals overlap)."""
    c = np.zeros(md.size, int)
    for a, b, k in bh['intervals']: c[(md >= a) & (md <= b)] = np.maximum(c[(md >= a) & (md <= b)], k)
    return c

def sample_along(mesh, m, x, y, z):
    """Model value in the cell containing each point of the trace (NaN outside the mesh)."""
    i = np.floor((x - mesh.xe[0])/mesh.dx).astype(int); j = np.floor((y - mesh.ye[0])/mesh.dy).astype(int)
    ok = (i >= 0) & (i < mesh.nx) & (j >= 0) & (j < mesh.ny); out = np.full(x.size, np.nan)
    dd = mesh.topo[np.clip(j, 0, mesh.ny-1), np.clip(i, 0, mesh.nx-1)] - z
    k = np.floor(dd/mesh.dz).astype(int); ok &= (k >= 0) & (k < mesh.nz)
    g = mesh.to_grid(m); out[ok] = g[k[ok], j[ok], i[ok]]; return out

def auc(score, label):
    """Threshold-free ROC AUC (probability that a mineralized point scores higher than a barren point)."""
    from scipy.stats import rankdata
    ok = ~np.isnan(score); s, l = score[ok], label[ok].astype(bool)
    if l.sum() == 0 or (~l).sum() == 0: return np.nan
    r = rankdata(s); return float((r[l].sum() - l.sum()*(l.sum() + 1)/2)/(l.sum()*(~l).sum()))

# ------------------------------------------------------------------ PTSS (Ming et al. 2024), for comparison
def ptss_operators(m_inv, D3, n=3):
    """Power-type structural constraint of Ming et al. (2024): B = ((grad m_inv)^n x), their eqs 8-12 and 24-34.
    The gradient power is taken as |grad m|^(n-1) grad m (direction preserved). D3 = (Dx, Dy, Dz) central differences on the grid."""
    Dx, Dy, Dz = D3; gx, gy, gz = Dx @ m_inv, Dy @ m_inv, Dz @ m_inv
    gm = np.sqrt(gx**2 + gy**2 + gz**2); s = gm.max() if gm.max() > 0 else 1.0
    gx, gy, gz, gm = gx/s, gy/s, gz/s, gm/s; f = gm**(n - 1); px, py, pz = (sp.diags(v*f) for v in (gx, gy, gz))
    return [(py @ Dz - pz @ Dy).tocsr(), (pz @ Dx - px @ Dz).tocsr(), (px @ Dy - py @ Dx).tocsr()]

def central_diff_ops(mesh):
    """Central differences (eqs 13-15 of Ming et al. 2024) on the logical grid of the mesh."""
    def d1(k):
        if k == 1: return sp.csr_matrix((1, 1))
        D = sp.lil_matrix((k, k))
        for i in range(k):
            if i == 0: D[i, 0], D[i, 1] = -1, 1
            elif i == k - 1: D[i, k - 2], D[i, k - 1] = -1, 1
            else: D[i, i - 1], D[i, i + 1] = -0.5, 0.5
        return D.tocsr()
    I = lambda k: sp.identity(k, format='csr')
    return (sp.kron(sp.kron(I(mesh.nz), I(mesh.ny)), d1(mesh.nx), format='csr'),
            sp.kron(sp.kron(I(mesh.nz), d1(mesh.ny)), I(mesh.nx), format='csr'),
            sp.kron(sp.kron(d1(mesh.nz), I(mesh.ny)), I(mesh.nx), format='csr'))

def invert_ptss(P, B_self, B_mut, lam_self=1.0, lam_mut=1.0, target=1.0, verbose=False):
    """Joint PTSS inversion for one property (eq. 35 of Ming et al. 2024): L2 stabilizer + self constraint + mutual constraint.
    Weights are dimensionless and relative (lambda*||B||^2 scaled to the L2 term); beta follows the same stopping rule as all methods."""
    BB = [(B, lam_self) for B in B_self] + [(B, lam_mut) for B in B_mut]
    c = P.w/P.scale; n = c.size
    tr_l2 = np.sum(c**2)/n                                         # effective size of the L2 term
    norm = [tr_l2/max(float((B.multiply(B)).sum())/n, 1e-30) for B, _ in BB]
    def R(m):
        u = c*m; val = np.mean(u**2); grad = 2*u*c/n; x = m/P.scale
        for (B, lam), k in zip(BB, norm):
            b = B @ x; val += lam*k*np.sum(b**2)/n; grad += lam*k*2*(B.T @ b)/P.scale/n
        return val, grad
    old = P.R
    P.R = lambda m, kind: R(m)
    try: m, info = invert(P, 'PTSS', target=target, verbose=verbose)
    finally: P.R = old
    return m, info

# ------------------------------------------------------------------ proposed method: Joint Minimum Support (JMS)
def invert_joint(Pg, Pm, kappa=2.0, targets=(1.0, 1.0), tol=0.05, cool=0.5, inner=150, max_stages=40, verbose=False):
    """Proposed JMS (Joint Minimum Support) inversion. Density and magnetization are solved simultaneously. Penalties:
         R_rho = mean( u_rho^2 / (u_rho^2 + kappa*u_M^2 + eps^2) ),   R_M = mean( u_M^2 / (u_M^2 + kappa*u_rho^2 + eps^2) )
    (u = dimensionless weighted model). Wherever one property has support, the penalty of the other is reduced, and both
    are updated in one optimization. kappa = 0 gives two independent MS inversions. Each property has its own beta, tuned
    separately so that both chi2/N reach their targets (same stopping rule as the other methods)."""
    # kappa may be a number or a pair (kappa_rho, kappa_M): kappa_rho = guidance of density by magnetization, kappa_M = the reverse
    kr, km = (kappa, kappa) if np.isscalar(kappa) else kappa
    cg, cm = Pg.w/Pg.scale, Pm.w/Pm.scale; n = cg.size; e2 = Pg.eps**2
    lo = np.r_[Pg.lo*cg, Pm.lo*cm]; hi = np.r_[Pg.hi*cg, Pm.hi*cm]; bnd = Bounds(lo, hi)
    def fun(u, bg, bm):
        ug, um = u[:n], u[n:]; fg, rg = Pg.misfit(ug/cg); fm, rm = Pm.misfit(um/cm)
        a, b = ug*ug, um*um; qg = a + kr*b + e2; qm = b + km*a + e2
        Rg, Rm = np.mean(a/qg), np.mean(b/qm)
        gg = Pg.grad_misfit(rg)/cg + (bg*2*ug*(kr*b + e2)/qg**2 - bm*b*2*km*ug/qm**2)/n
        gm = Pm.grad_misfit(rm)/cm + (bm*2*um*(km*a + e2)/qm**2 - bg*a*2*kr*um/qg**2)/n
        return fg + fm + bg*Rg + bm*Rm, np.r_[gg, gm]
    u = np.zeros(2*n)
    bg = 10*Pg.misfit(np.zeros(n))[0]; bm = 10*Pm.misfit(np.zeros(n))[0]    # R <= 1, so start with the stabilizer term dominant
    step = {'g': np.log(1/cool), 'm': np.log(1/cool)}; last = {'g': 0, 'm': 0}; hist = []; t0 = time.time(); best = None
    for st in range(max_stages):
        u = minimize(fun, u, args=(bg, bm), jac=True, method='L-BFGS-B', bounds=bnd, options=dict(maxiter=inner, maxcor=20)).x
        rho, M = np.clip(u[:n]/cg, Pg.lo, Pg.hi), np.clip(u[n:]/cm, Pm.lo, Pm.hi)
        fg, fm = Pg.misfit(rho)[0], Pm.misfit(M)[0]; hist.append((bg, bm, fg, fm))
        if verbose: print(f'  [JMS] stage {st:2d} βg={bg:8.2e} βm={bm:8.2e} χ²/N g={fg:7.3f} m={fm:7.3f} ({time.time()-t0:4.0f}s)', flush=True)
        tg, tm = targets
        score = abs(np.log(fg/tg)) + abs(np.log(fm/tm))
        if best is None or score < best[0]: best = (score, rho.copy(), M.copy(), fg, fm)
        ok = [abs(fg/tg - 1) <= tol, abs(fm/tm - 1) <= tol]
        if all(ok): break
        for key, f, ok_, tt in (('g', fg, ok[0], tg), ('m', fm, ok[1], tm)):
            if ok_: continue
            d = -1 if f > tt else 1                       # chi2 too large -> decrease beta
            if last[key] and d != last[key]: step[key] /= 2   # direction change -> halve the step (bisection)
            last[key] = d
            if key == 'g': bg *= np.exp(d*step['g'])
            else: bm *= np.exp(d*step['m'])
        if max(step.values()) < 0.02: break
    _, rho, M, fg, fm = best
    return rho, M, dict(chi2=(fg, fm), hist=hist, time=time.time() - t0)


def misfit_floor(P, kind='L2'):
    """Best attainable misfit within the given bounds (very small beta). If it exceeds 1, the misfit target of every method
    is set to 1.1 times this floor, so that the stopping rule stays identical and attainable for all methods."""
    m, info = invert(P, kind, target=1e-3); return info['chi2']
