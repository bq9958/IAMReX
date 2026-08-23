#!/usr/bin/env python3
"""Monodisperse -- total z-direction IB force on all particles vs iteration step.

Reads every ``IB_Particle_*.csv`` produced by IAMReX's
``WriteIBForceAndMoment()`` for the CPU run in this directory, and sums the
z-component of the immersed-boundary force across all particles at each
time step.

Total IB force (per particle, z-component)
------------------------------------------
This is the quantity the Monodisperse validation in
Docs/IAMReX_documentation/source/Results.rst compares against theory:
"time series of total IB force for all particles".

    F_IB,z = rho_f * sum_l F_l^{n+1/2} dV_l

which is exactly the ``Fz`` column of the CSV (the dV_l factor is folded
in by ``ForceSpreading_cir()``, which does ``fxP *= dv`` in place on the
marker's Fx_Marker, and ib_force is the ReduceSum of those).  So per
particle::

    F_IB,z = rho_f * Fz

and the total over the suspension is::

    F_IB,z_total(t) = sum_p rho_f * Fz_p

The force is reported as a positive magnitude (|F_IB,z_total|).

Why the d/dt( int_Vp u dV ) term is NOT included
------------------------------------------------
Results.rst also gives the hydrodynamic drag on a particle as

    F_D = -rho_f * sum_l F_l^{n+1/2} dV_l  +  rho_f * d/dt( int_Vp u dV )

i.e. ``rho_f * (-Fz + SumUz)``.  That form belongs to the FlowPastSphere
case (single particle, no body force), where the second term accounts for
the unsteady acceleration of the fluid inside the particle.  Here the flow
is driven by a uniform body force in z and reaches a steady state, so the
second term should vanish; it does so in x and y (sum_p SumUx, SumUy ~ 3
out of ~2000) but carries a systematic ~+240 offset in the driving
direction z.

The reason is that ``SumUz = (sum_u_new - sum_u_old)/dt`` is not a clean
telescoping time difference: ``UpdateParticles()`` reads S_new right after
``level_project()`` on the finest level, but ``post_timestep()`` then runs
``level_sync()`` / ``MLsyncProject()``, which adds a further correction to
the finest-level velocity.  So the S_new sampled here is not the S_old of
the next step, and the missing sync increment -- random in x and y,
systematic along the forcing direction -- accumulates into that offset.
Including it would bias the total by ~12%.  It does not affect the
dynamics of this case, where every particle is fixed (TL = RL = 0) and
sum_u / sum_t are pure diagnostics.

Theoretical prediction
----------------------
For a monodisperse suspension subject to a uniform pressure gradient
dp/dz, the total IB force balances the pressure-gradient body force over
the domain volume:

    F_theory = (dp/dz) * L_x * L_y * L_z

where L_x, L_y, L_z are the domain extents (taken from
``geometry.prob_hi`` in the inputs file).  Here dp/dz = 1 by default
(override with --dpdz).

Only the CPU run has data here, so only the CPU directory is scanned.

Usage
-----
    python3 plot.py                      # use ./CPU, write drag_vs_step.png
    python3 plot.py --dir CPU            # specify the run directory
    python3 plot.py --dpdz 1.0           # pressure gradient dp/dz
    python3 plot.py --avg-window 5000    # steady-state averaging window
    python3 plot.py --out other.png
"""

import argparse
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Commit hash = parent directory name, optionally date-prefixed
# ("d39efff" or "2026-08-04_d39efff").  Derived rather than hard-coded so
# renaming the directory is the only thing to do.
COMMIT = os.path.basename(os.path.dirname(HERE)).split("_")[-1]

# Okabe-Ito colourblind-safe palette.
C_Z = "#009E73"   # green -- z component (measured drag)
C_TH = "#D55E00"  # vermillion -- theoretical prediction


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
# one particle file
# ----------------------------------------------------------------------
def load_particle(path, rho):
    """Return (istep, time, F_IB,z) arrays.

    F_IB,z = rho * Fz, i.e. the per-particle z-component of the total IB
    force.  Missing / unparsable rows are skipped.
    """
    istep, t, fibz = [], [], []
    with open(path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                fz = float(row["Fz"])
                istep.append(int(row["iStep"]))
                t.append(float(row["time"]))
            except (KeyError, ValueError):
                continue
            fibz.append(rho * fz)
    return np.asarray(istep), np.asarray(t), np.asarray(fibz)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="CPU",
                    help="run directory holding IB_Particle_*.csv "
                         "(default: CPU, relative to this script)")
    ap.add_argument("--inputs", default=None,
                    help="inputs file for rho_f lookup (default: search --dir)")
    ap.add_argument("--dpdz", type=float, default=1.0,
                    help="pressure gradient dp/dz for the theoretical "
                         "prediction F = (dp/dz)*Lx*Ly*Lz (default: 1.0)")
    ap.add_argument("--avg-window", type=int, default=2000,
                    help="number of trailing steps used for the reported "
                         "steady-state average (default: 2000)")
    ap.add_argument("--out", default=os.path.join(HERE, "drag_vs_step.png"))
    args = ap.parse_args()

    run_dir = args.dir
    if not os.path.isabs(run_dir):
        run_dir = os.path.join(HERE, run_dir)
    if not os.path.isdir(run_dir):
        sys.exit(f"run directory not found: {run_dir}")

    # ---- rho_f from inputs -----------------------------------------
    inputs_path = args.inputs
    if not inputs_path:
        found = (glob.glob(os.path.join(run_dir, "inputs*"))
                 + glob.glob(os.path.join(run_dir, "*inputs*")))
        if not found:
            sys.exit("no inputs file found in {} (use --inputs)".format(run_dir))
        inputs_path = found[0]
    cfg = parse_inputs(inputs_path)
    rho = scalar(cfg, "ns.fluid_rho", 1.0)
    if rho is None:
        rho = 1.0
        print("warning: ns.fluid_rho not found in {}; using 1.0".format(
            inputs_path))

    # ---- domain extents for the theoretical prediction --------------
    prob_hi = cfg.get("geometry.prob_hi")
    if prob_hi and len(prob_hi) >= 3:
        Lx, Ly, Lz = (float(prob_hi[0]), float(prob_hi[1]),
                      float(prob_hi[2]))
    else:
        Lx = Ly = Lz = None
        print("warning: geometry.prob_hi not found in {}; "
              "theoretical line will be omitted".format(inputs_path))
    F_theory = (args.dpdz * Lx * Ly * Lz
                if None not in (Lx, Ly, Lz) else None)
    if F_theory is not None:
        print(f"domain: Lx={Lx:g}  Ly={Ly:g}  Lz={Lz:g}")
        print(f"theoretical F = dp/dz * Lx*Ly*Lz = "
              f"{args.dpdz:g} * {Lx:g} * {Ly:g} * {Lz:g} = {F_theory:g}")

    # ---- gather all particle files ---------------------------------
    csv_paths = sorted(
        glob.glob(os.path.join(run_dir, "IB_Particle_*.csv")),
        key=lambda p: int(
            os.path.basename(p).split("_")[2].split(".")[0]))
    if not csv_paths:
        sys.exit("no IB_Particle_*.csv files in {}".format(run_dir))
    print(f"found {len(csv_paths)} particle files in {run_dir}")

    # ---- sum z-drag across all particles ---------------------------
    # Each CSV is one particle's time history.  All particles share the
    # same time grid (single simulation), so we accumulate per-step sums.
    # Use the first file to set the step axis; assume all files match.
    istep0, t0, fibz0 = load_particle(csv_paths[0], rho)
    total_z = fibz0.copy()
    counts = np.ones_like(t0, dtype=float)
    for p in csv_paths[1:]:
        istep, t, fibz = load_particle(p, rho)
        if len(istep) != len(istep0) or not np.array_equal(istep, istep0):
            # align on intersection of iStep (in case runs differ in length)
            common = np.intersect1d(istep0, istep, assume_unique=True)
            if len(common) == 0:
                continue
            mask_a = np.isin(istep0, common)
            mask_b = np.isin(istep, common)
            total_z[mask_a] += fibz[mask_b]
            counts[mask_a] += 1.0
        else:
            total_z += fibz
            counts += 1.0

    n_particles = counts.astype(int)
    # Report magnitude (the IB force opposes the driving force, so the sign
    # of the raw sum just follows the forcing direction).
    total_z_mag = np.abs(total_z)
    print(f"steps: {istep0[0]} .. {istep0[-1]}  ({len(istep0)} rows)")
    print(f"particles summed per step (min/max): "
          f"{n_particles.min()} / {n_particles.max()}")
    print(f"final total |F_IB,z| = {total_z_mag[-1]:+.6g}")

    window = min(args.avg_window, len(total_z_mag))
    avg = total_z_mag[-window:].mean()
    std = total_z_mag[-window:].std()
    print(f"steady-state average over last {window} steps: "
          f"{avg:.6g} +/- {std:.3g}")
    if F_theory:
        print(f"relative error vs theory: "
              f"{(avg - F_theory) / F_theory * 100:+.2f}%")

    # ---- figure ----------------------------------------------------
    plt.rcParams.update({
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })
    fig, ax = plt.subplots(figsize=(8, 4.6))

    ax.plot(istep0, total_z_mag, lw=1.6, color=C_Z,
            label=r"$|F_{IB,z}|$ (measured)")
    if F_theory is not None:
        ax.axhline(F_theory, ls="--", lw=1.8, color=C_TH,
                   label=r"$({\Delta p}/{\Delta z})\,L_x L_y L_z$ "
                         f"= {F_theory:g}")

    ax.set_xlabel("iteration step")
    ax.set_ylabel(r"total IB force  $|\sum_p F_{IB,z,p}|$")
    ax.set_title("Total z-direction IB force on all particles -- CPU run",
                 loc="left")
    ax.legend(loc="best")

    fig.suptitle(
        f"Monodisperse -- commit {COMMIT}  "
        f"({len(csv_paths)} particles, $\\rho_f$={rho:g})",
        x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=200)
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
