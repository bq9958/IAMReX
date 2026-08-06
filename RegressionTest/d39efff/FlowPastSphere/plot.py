#!/usr/bin/env python3
"""FlowPastSphere -- drag coefficient vs particle Reynolds number.

Reads one IB_Particle_0.csv per Re<NNN>/ subdirectory, extracts the steady-state
drag coefficient, and compares it against the Schiller-Naumann correlation and
whatever digitised literature curves live in ../../ref_data/.

Drag force
----------
From Docs/IAMReX_documentation/source/Results.rst:

    F_D = -rho_f * sum_l F_l^{n+1/2} dV_l  +  rho_f * d/dt( int_Vp u dV )

Both terms are already in IB_Particle_0.csv, so no plotfile is needed:

  * `Fx` is sum_l F_l dV_l.  ForceSpreading_cir() does `fxP *= dv` in place on
    the marker's Fx_Marker (DiffusedIB_Parallel.cpp), and ib_force is the
    ReduceSum of those, so the dV_l factor is already folded in.
  * `SumUx` is (sum_u_new - sum_u_old)/dt, i.e. d/dt( int_Vp u dV ), written by
    WriteIBForceAndMoment().

    C_D = rho_f * (-Fx + SumUx) / (0.5 * rho_f * U^2 * pi * r^2)

Reynolds number
---------------
    Re = U * D / nu,  D = 2 * particle_inputs.radius,  nu = ns.vel_visc_coef

taken from each subdirectory's own inputs file, not from the folder name, so a
hand-edited inputs cannot silently disagree with its label.

Usage
-----
    python3 plot.py                      # scan ./Re*/, write Cd_vs_Re.png
    python3 plot.py --tail-frac 0.2      # average over the last 20% of the run
    python3 plot.py --out other.png
"""

import argparse
import csv
import glob
import math
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "ref_data"))

# The commit under test is the name of the directory two levels up, optionally
# prefixed with a date ("d39efff" or "2026-08-04_d39efff").  Deriving it beats
# hard-coding: renaming the directory is then the only thing to do.
COMMIT = os.path.basename(os.path.dirname(HERE)).split("_")[-1]

# Okabe-Ito: colourblind-safe by construction, the standard categorical set for
# scientific figures.  Assigned in fixed order by *source identity* and never
# cycled.  Reynolds number is a magnitude, not an identity, so the per-Re
# history lines get a sequential ramp instead (see below).
C_PRESENT = "#0072B2"   # blue      -- IAMReX, this commit
C_SN = "#D55E00"        # vermillion-- Schiller-Naumann correlation
C_REF = ["#009E73", "#CC79A7", "#E69F00", "#56B4E9"]  # literature files


# ----------------------------------------------------------------------
# inputs parsing
# ----------------------------------------------------------------------
def parse_inputs(path):
    """Minimal AMReX ParmParse reader: {key: [tokens]}, later wins."""
    table = {}
    with open(path) as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key and " " not in key:
                table[key] = value.split()
    return table


def scalar(table, key, default=None):
    try:
        return float(table[key][0])
    except (KeyError, IndexError, ValueError):
        return default


# ----------------------------------------------------------------------
# one run
# ----------------------------------------------------------------------
class Run:
    def __init__(self, directory, tail_frac):
        self.dir = directory
        self.label = os.path.basename(directory.rstrip("/"))
        self.ok = False
        self.note = ""

        inputs = glob.glob(os.path.join(directory, "inputs*"))
        if not inputs:
            self.note = "no inputs file"
            return
        cfg = parse_inputs(inputs[0])

        self.nu = scalar(cfg, "ns.vel_visc_coef")
        self.rho = scalar(cfg, "ns.fluid_rho", 1.0)
        radius = scalar(cfg, "particle_inputs.radius")
        vel = cfg.get("xlo.velocity")
        self.U = float(vel[0]) if vel else None

        if None in (self.nu, radius, self.U) or self.nu <= 0:
            self.note = "inputs missing vel_visc_coef / radius / xlo.velocity"
            return

        self.D = 2.0 * radius
        self.area = math.pi * radius ** 2
        self.Re = self.U * self.D / self.nu

        csv_path = os.path.join(directory, "IB_Particle_0.csv")
        if not os.path.exists(csv_path):
            self.note = "IB_Particle_0.csv not found (has the job run?)"
            return

        t, cd = [], []
        denom = 0.5 * self.rho * self.U ** 2 * self.area
        with open(csv_path) as fh:
            for row in csv.DictReader(fh):
                try:
                    fx = float(row["Fx"])
                    sumux = float(row["SumUx"])
                    t.append(float(row["time"]))
                except (KeyError, ValueError):
                    continue
                # F_D = rho*(-sum F_l dV_l) + rho*d/dt(int u dV)
                cd.append(self.rho * (-fx + sumux) / denom)

        if len(t) < 2:
            self.note = f"only {len(t)} row(s) in CSV -- run did not progress"
            return

        self.t = np.asarray(t)
        self.cd = np.asarray(cd)

        n_tail = max(1, int(round(tail_frac * len(self.cd))))
        tail = self.cd[-n_tail:]
        self.cd_mean = float(np.mean(tail))
        self.cd_std = float(np.std(tail))
        self.n_tail = n_tail
        self.ok = True

        # A short run is the usual reason a sweep looks wrong: the tutorial
        # inputs ships max_step = 2 as a CI smoke test.
        if len(self.t) < 50:
            self.note = (f"only {len(self.t)} steps -- almost certainly not at "
                         f"steady state (raise max_step in inputs)")
        elif self.cd_mean != 0.0 and abs(self.cd_std / self.cd_mean) > 0.02:
            self.note = (f"Cd still varying by {100*self.cd_std/abs(self.cd_mean):.1f}% "
                         f"over the averaging window")


def schiller_naumann(re):
    """C_D = (24/Re)(1 + 0.15 Re^0.687), Schiller & Naumann (1933)."""
    re = np.asarray(re, dtype=float)
    return (24.0 / re) * (1.0 + 0.15 * re ** 0.687)


def load_ref_data(pattern):
    """Load two-column `Re Cd` .dat files; '#' starts a comment."""
    out = []
    for path in sorted(glob.glob(pattern)):
        rows = []
        with open(path) as fh:
            for raw in fh:
                line = raw.split("#", 1)[0].strip()
                if not line:
                    continue
                parts = re.split(r"[,\s]+", line)
                if len(parts) < 2:
                    continue
                try:
                    rows.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
        if rows:
            arr = np.asarray(sorted(rows))
            out.append((os.path.splitext(os.path.basename(path))[0],
                        arr[:, 0], arr[:, 1]))
    return out


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.path.join(HERE, "Re*"),
                    help="glob for the per-Re run directories")
    ap.add_argument("--ref-data",
                    default=os.path.join(REF_DATA_DIR, "FlowPastSphere*.dat"),
                    help="glob for literature .dat files (two columns: Re Cd)")
    ap.add_argument("--tail-frac", type=float, default=0.1,
                    help="fraction of the time series to average for the "
                         "steady-state Cd (default: 0.1)")
    ap.add_argument("--out", default=os.path.join(HERE, "Cd_vs_Re.png"))
    args = ap.parse_args()

    dirs = sorted(d for d in glob.glob(args.runs) if os.path.isdir(d))
    if not dirs:
        sys.exit(f"no run directories matched {args.runs}")

    runs = [Run(d, args.tail_frac) for d in dirs]
    good = sorted((r for r in runs if r.ok), key=lambda r: r.Re)

    # ---- text report ------------------------------------------------
    print(f"{'run':<10} {'Re':>8} {'Cd':>10} {'+/-':>9} {'S-N':>10} {'dev':>8}")
    print("-" * 60)
    for r in runs:
        if not r.ok:
            print(f"{r.label:<10} {'-':>8} {'-':>10} {'-':>9} {'-':>10} {'-':>8}"
                  f"   [{r.note}]")
            continue
        sn = float(schiller_naumann(r.Re))
        dev = 100.0 * (r.cd_mean - sn) / sn
        print(f"{r.label:<10} {r.Re:>8.1f} {r.cd_mean:>10.4f} "
              f"{r.cd_std:>9.4f} {sn:>10.4f} {dev:>7.1f}%"
              + (f"   [{r.note}]" if r.note else ""))
    print()

    if not good:
        sys.exit("no usable runs -- nothing to plot")

    # ---- figure -----------------------------------------------------
    plt.rcParams.update({
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })
    fig, (ax_hist, ax_cd) = plt.subplots(1, 2, figsize=(11, 4.4))

    # Left: convergence history.  Re is a magnitude, so it gets a sequential
    # single-hue ramp (light = low Re) rather than categorical hues, and each
    # line is direct-labelled instead of carrying a legend box.
    ramp = plt.get_cmap("Blues")
    for i, r in enumerate(good):
        shade = 0.35 + 0.6 * (i / max(1, len(good) - 1))
        ax_hist.plot(r.t, r.cd, lw=1.6, color=ramp(shade))
        ax_hist.annotate(f"Re={r.Re:g}", xy=(r.t[-1], r.cd[-1]),
                         xytext=(4, 0), textcoords="offset points",
                         va="center", fontsize=8, color=ramp(shade))
        # shade the window the steady-state average was taken over
        ax_hist.axvspan(r.t[-r.n_tail], r.t[-1], color=ramp(shade), alpha=0.06,
                        lw=0)
    ax_hist.set_xlabel("time")
    ax_hist.set_ylabel(r"$C_D$")
    ax_hist.set_title("Convergence to steady state", loc="left")
    if good:
        top = np.percentile(np.concatenate([r.cd for r in good]), 99)
        ax_hist.set_ylim(0, max(top * 1.15, 1e-6))

    # Right: Cd vs Re.  Log-log -- the drag curve spans decades.
    re_all = [r.Re for r in good]
    re_line = np.logspace(math.log10(min(re_all) / 1.6),
                          math.log10(max(re_all) * 1.6), 200)
    ax_cd.plot(re_line, schiller_naumann(re_line), color=C_SN, lw=2.0,
               label="Schiller-Naumann (1933)")

    for i, (name, ref_re, ref_cd) in enumerate(load_ref_data(args.ref_data)):
        ax_cd.plot(ref_re, ref_cd, "s--", ms=6, lw=1.4, mfc="none",
                   color=C_REF[i % len(C_REF)], label=name)

    ax_cd.plot(re_all, [r.cd_mean for r in good], "o", ms=8, color=C_PRESENT,
               mec="white", mew=1.2, label="IAMReX (this commit)", zorder=5)
    yerr = [r.cd_std for r in good]
    if any(e > 0 for e in yerr):
        ax_cd.errorbar(re_all, [r.cd_mean for r in good], yerr=yerr,
                       fmt="none", ecolor=C_PRESENT, elinewidth=1.2, capsize=3,
                       zorder=4)

    # Re spans a decade or more, so x stays logarithmic.  Cd is on a linear
    # axis anchored at zero: it is a magnitude, and a linear axis that does not
    # include the origin would visually exaggerate the gap to Schiller-Naumann.
    ax_cd.set_xscale("log")
    ax_cd.set_ylim(bottom=0)
    ax_cd.set_xlabel(r"$Re_p = U D_p / \nu$")
    ax_cd.set_ylabel(r"$C_D$")
    ax_cd.set_title("Drag coefficient", loc="left")
    ax_cd.legend(loc="upper right")

    unconverged = [r.label for r in good if r.note]
    failed = [r.label for r in runs if not r.ok]
    caption = []
    if failed:
        caption.append("no data: " + ", ".join(failed))
    if unconverged:
        caption.append("not converged: " + ", ".join(unconverged))
    if caption:
        fig.text(0.01, 0.005, "  |  ".join(caption), fontsize=8, color="#B00020")

    fig.suptitle(f"FlowPastSphere -- commit {COMMIT}", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(args.out, dpi=200)
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
