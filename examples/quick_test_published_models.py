# Quick synthetic test: all six methods on the two published test models of Ming et al. (2024)
# (a single prism, and two bodies), flat ground, 20 x 20 x 15 cells of 500 m.
# Run from the repository root:   python examples/quick_test_published_models.py
# Output: outputs/table_paper_tests_dice.csv (compare with results/tables/table_paper_tests_dice.csv)
import os, sys, numpy as np, warnings, time; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jms_lib import *
os.makedirs('outputs', exist_ok=True)
def dice_(a,b): s=a.sum()+b.sum(); return 2*(a&b).sum()/s if s else 1.0
km=1000.; g=np.arange(20)*500+250.; X,Y=np.meshgrid(g,g); xs,ys=X.ravel(),Y.ravel()
mesh=DrapedMesh(xs,ys,np.zeros(xs.size),500.,500.,7500.,pad=0)        # flat ground, cell grid of the published models
# keep the paper's exact cell grid: 0..10 km
mesh.xe=np.arange(21)*500.; mesh.ye=mesh.xe.copy(); mesh.xc=g; mesh.yc=g
Dd,YY,XX=np.meshgrid(mesh.dc,g,g,indexing='ij'); mesh.Xc,mesh.Yc,mesh.Dc=XX.ravel(),YY.ravel(),Dd.ravel(); mesh.Ec=-mesh.Dc
mesh.topo=np.zeros((20,20)); mesh.top=0.0; mesh.core=np.ones(mesh.n,bool)
zs=np.full(xs.size,-1e-3); bx=mesh.boxes()
Ag=gravity_kernel(bx,xs,ys,zs,chunk=100); Am=magnetic_kernel(bx,xs,ys,zs,90.,0.,chunk=100)
D=diff_operator(mesh); D3=central_diff_ops(mesh)
def box(x1,x2,y1,y2,z1,z2): return (mesh.Xc>x1)&(mesh.Xc<x2)&(mesh.Yc>y1)&(mesh.Yc<y2)&(mesh.Dc>z1)&(mesh.Dc<z2)
tests={'Prism (paper Fig.6)':box(3*km,7*km,3*km,7*km,2*km,4.5*km),
       'Two bodies (paper Fig.7-9)':box(1.5*km,4.5*km,3.5*km,6.5*km,2*km,5*km)|box(6.5*km,8.5*km,4*km,6*km,1*km,3*km)}
rows=[]
for tname,body in tests.items():
    mt=body.astype(float); rng=np.random.default_rng(0)
    g0,m0=(Ag@mt.astype(np.float32)).astype(float),(Am@mt.astype(np.float32)).astype(float)
    noise=0.05 if 'Two' in tname else 0.0
    sg=np.maximum(0.05*np.abs(g0),0.01*np.abs(g0).max()); sm=np.maximum(0.05*np.abs(m0),0.01*np.abs(m0).max())
    dg=g0+noise*np.abs(g0)*rng.standard_normal(g0.size); dm=m0+noise*np.abs(m0)*rng.standard_normal(m0.size)
    mk=lambda A,d,s,n: Problem(A,d,s,sensitivity_weight(A,s),(0.,1.2),D,0.02,n)
    Pg,Pm=mk(Ag,dg,sg,'g'),mk(Am,dm,sm,'m'); res={}
    for k in ['L2','L1','MS','MGS']:
        res[k]=(invert(Pg,k)[0],invert(Pm,k)[0])
    r0,M0=res['L2']; Br,BM=ptss_operators(r0,D3,3),ptss_operators(M0,D3,3)
    res['PTSS']=(invert_ptss(Pg,Br,BM,1.,1.)[0],invert_ptss(Pm,BM,Br,1.,1.)[0])
    r,M,_=invert_joint(Pg,Pm,kappa=0.5); res['JMS']=(r,M)
    for k,(r,M) in res.items():
        rows.append((tname,k,dice_(r>=0.25,mt>=0.25),dice_(M>=0.25,mt>=0.25),Pg.misfit(r)[0],Pm.misfit(M)[0]))
        print(f'{tname:27s} {k:5s} Dice ρ={rows[-1][2]:.3f}  Dice M={rows[-1][3]:.3f}   χ²/N {rows[-1][4]:.2f} {rows[-1][5]:.2f}',flush=True)
import pandas as pd
pd.DataFrame(rows,columns=['Test','Method','Dice_rho','Dice_M','chi2N_g','chi2N_m']).to_csv('outputs/table_paper_tests_dice.csv',index=False)
