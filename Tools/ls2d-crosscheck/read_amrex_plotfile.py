"""Minimal reader for 2D single-level AMReX plotfiles (double precision, native byte order).
usage: python read_amrex_plotfile.py <pltdir> [var]   -> prints area of var>0 and writes <pltdir>_<var>.npy
Header format: see amrex/Docs (Header: version, nvars, names, dim, time, finest, prob_lo/hi, refratio, domains, ...;
Level_0/Cell_H: FAB layout; Level_0/Cell_D_xxxxx: 'FAB ...' header line + raw doubles, component-major per box).
"""
import sys, os, re, numpy as np

def read_plotfile(d):
    with open(os.path.join(d, "Header")) as f: L = [l.rstrip("\n") for l in f]
    nvar = int(L[1]); names = L[2:2+nvar]; k = 2+nvar
    dim = int(L[k]); time = float(L[k+1]); finest = int(L[k+2]); k += 3
    plo = [float(x) for x in L[k].split()]; phi = [float(x) for x in L[k+1].split()]; k += 2
    k += 1                                    # ref ratios
    dom = L[k]; k += 1                        # domains
    m = re.findall(r"\((-?\d+),(-?\d+)\)", dom)[:2]
    lo = (int(m[0][0]), int(m[0][1])); hi = (int(m[1][0]), int(m[1][1]))
    nx, ny = hi[0]-lo[0]+1, hi[1]-lo[1]+1
    # level 0 cell data
    with open(os.path.join(d, "Level_0", "Cell_H")) as f: H = f.read().split("\n")
    # lines: version, how, ncomp, nghost, "(nbox 0", boxes..., ")", nfab, FabOnDisk lines
    nbox = int(H[4].split()[0].strip("("))
    boxes = []; i = 5
    for b in range(nbox):
        mm = re.findall(r"\((-?\d+),(-?\d+)\)", H[i]); i += 1
        boxes.append(((int(mm[0][0]), int(mm[0][1])), (int(mm[1][0]), int(mm[1][1]))))
    i += 1  # ')'
    nfab = int(H[i]); i += 1
    fabs = []
    for b in range(nfab):
        parts = H[i].split(); i += 1          # "FabOnDisk: Cell_D_00000 offset"
        fabs.append((parts[1], int(parts[2])))
    data = {n: np.full((ny, nx), np.nan) for n in names}
    for (blo, bhi), (fn, off) in zip(boxes, fabs):
        with open(os.path.join(d, "Level_0", fn), "rb") as f:
            f.seek(off); hdr = f.readline()   # FAB header line
            bnx, bny = bhi[0]-blo[0]+1, bhi[1]-blo[1]+1
            arr = np.fromfile(f, dtype="<f8", count=bnx*bny*nvar).reshape(nvar, bny, bnx)
        for c, n in enumerate(names):
            data[n][blo[1]-lo[1]:bhi[1]-lo[1]+1, blo[0]-lo[0]:bhi[0]-lo[0]+1] = arr[c]
    dx = (phi[0]-plo[0])/nx; dy = (phi[1]-plo[1])/ny
    return names, time, plo, phi, dx, dy, data

if __name__ == "__main__":
    d = sys.argv[1]; var = sys.argv[2] if len(sys.argv) > 2 else "phi"
    names, t, plo, phi, dx, dy, data = read_plotfile(d)
    a = data[var]; assert not np.isnan(a).any()
    # sub-cell sharp area (bilinear, 10x10), same estimator as ls2d volume_sharp
    ny, nx = a.shape; ns = 10; cnt = 0
    ap = np.pad(a, 1, mode="wrap")
    for sj in range(ns):
        for si in range(ns):
            fx = (si+0.5)/ns-0.5; fy = (sj+0.5)/ns-0.5
            i0 = 0 if fx < 0 else 1; j0 = 0 if fy < 0 else 1; tx = fx+1 if fx < 0 else fx; ty = fy+1 if fy < 0 else fy
            v = ((1-tx)*(1-ty)*ap[j0:j0+ny, i0:i0+nx] + tx*(1-ty)*ap[j0:j0+ny, i0+1:i0+1+nx]
                 + (1-tx)*ty*ap[j0+1:j0+1+ny, i0:i0+nx] + tx*ty*ap[j0+1:j0+1+ny, i0+1:i0+1+nx])
            cnt += (v > 0).sum()
    print(f"{d}: t={t} vars={names} nx={nx} ny={ny} area_sharp={cnt*dx*dy/(ns*ns):.6f} area_cells={(a>0).sum()*dx*dy:.6f} min={a.min():.4g} max={a.max():.4g}")
    np.save(os.path.basename(d.rstrip('/'))+"_"+var+".npy", a)
