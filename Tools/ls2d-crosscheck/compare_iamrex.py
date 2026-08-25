"""Diagnostics from an AMReX plotfile for code-to-code comparison with ls2d.
usage: python compare_iamrex.py <pltdir> [--tg nu]      (--tg: Taylor-Green L2 error of velocity/pressure)
Prints: time, phi>0 area, interface ymin/ymax (zero contour of phi, linear interpolation along columns),
        kinetic energy, and for --tg the L2 errors.
"""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from read_amrex_plotfile import read_plotfile

d = sys.argv[1]; names, t, plo, phi_, dx, dy, data = read_plotfile(d)
u, v = data["x_velocity"], data["y_velocity"]; rho = data.get("density"); phi = data.get("phi")
ny, nx = u.shape
x = plo[0] + (np.arange(nx) + 0.5) * dx; y = plo[1] + (np.arange(ny) + 0.5) * dy
ke = 0.5 * (rho * (u**2 + v**2)).sum() * dx * dy if rho is not None else 0.5 * (u**2 + v**2).sum() * dx * dy
out = f"{d}: t={t:.6f} KE={ke:.6e}"
if phi is not None:
    a = phi; ymin, ymax = 1e30, -1e30
    for i in range(nx):
        col = a[:, i]
        for j in range(ny - 1):
            if col[j] * col[j + 1] <= 0 and col[j] != col[j + 1]:
                yy = y[j] + dy * col[j] / (col[j] - col[j + 1]); ymin = min(ymin, yy); ymax = max(ymax, yy)
    out += f" area_pos={(a > 0).sum() * dx * dy:.6f} interface_ymin={ymin:.5f} interface_ymax={ymax:.5f}"
if "--tg" in sys.argv:
    nu = float(sys.argv[sys.argv.index("--tg") + 1]); X, Y = np.meshgrid(x, y); dcy = math.exp(-8 * math.pi**2 * nu * t)
    ue = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) * dcy; ve = -np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y) * dcy
    out += f" L2_error_u={math.sqrt(((u - ue)**2).sum() * dx * dy):.6e} Linf_u={np.abs(u - ue).max():.6e}"
print(out)
