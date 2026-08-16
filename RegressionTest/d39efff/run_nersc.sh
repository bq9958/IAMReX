#!/usr/bin/env bash
#
# Build and submit the regression cases for one commit on Perlmutter.
#
# The commit is taken from this directory's own name, so renaming the directory
# is all it takes to retarget -- there is no second place to keep in sync.
#
#   ./run_nersc.sh build              # compile every case (login node)
#   ./run_nersc.sh submit             # sbatch every run
#   ./run_nersc.sh all                # build, then submit
#   ./run_nersc.sh build FlowPastSphere
#   ./run_nersc.sh submit FlowPastSphere/Re100
#   ./run_nersc.sh submit FallingSphere/tenCate/1.5
#
# Build on a login node -- compiling under sbatch burns the allocation.
#
# The executable lands in the case directory (e.g. FlowPastSphere/amr3d.gnu.MPI.ex)
# and each <run>/job.slurm runs it via ../../amr3d.gnu.x86-milan.MPI.ex (for
# FlowPastSphere/Re*/) or ../../../amr3d.gnu.x86-milan.MPI.ex (for the deeper
# FallingSphere/<set>/<Re>/), so every run shares one binary but gets its own
# working directory.  That separation is load-bearing: IB_Particle_*.csv is
# opened with std::ios::app, so two runs in one directory would silently
# concatenate their output.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Directory name is the commit, optionally prefixed with a date:
#   d39efff  or  2026-08-04_d39efff
COMMIT="$(basename "$HERE")"
COMMIT="${COMMIT##*_}"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

# Cases to act on when none are named on the command line.
DEFAULT_CASES=(FlowPastSphere FallingSphere)

# Print the comment block at the top of this file, however long it happens to be.
usage() {
    awk 'NR > 1 { if (!/^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
    exit 2
}

# ----------------------------------------------------------------------
check_commit() {
    local head
    head="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
    if [[ "$head" != "$COMMIT"* && "$COMMIT" != "$head"* ]]; then
        cat >&2 <<EOF
error: this directory records commit $COMMIT but the checkout is at $head.

  Building now would produce an executable that does not match the inputs,
  scripts and reference figure stored here -- which defeats the point of
  keeping one directory per commit.

  Either check the source out first:
      git -C "$REPO_ROOT" checkout $COMMIT
  or, if you really mean to build a different version against these inputs:
      FORCE_COMMIT=1 $0 $*
EOF
        [[ "${FORCE_COMMIT:-0}" == "1" ]] || exit 3
        echo "warning: FORCE_COMMIT=1, building $head against $COMMIT inputs" >&2
    fi
}

check_deps() {
    local missing=0
    for dep in amrex AMReX-Hydro; do
        if [[ ! -d "$REPO_ROOT/../$dep" ]]; then
            echo "error: missing $(cd "$REPO_ROOT/.." && pwd)/$dep" >&2
            missing=1
        fi
    done
    [[ $missing -eq 0 ]] || exit 2
}

do_build() {
    module load PrgEnv-gnu
    check_deps
    check_commit
    for case_name in "$@"; do
        local dir="$HERE/$case_name"
        [[ -f "$dir/GNUmakefile" ]] || { echo "error: no GNUmakefile in $dir" >&2; exit 2; }
        echo "==> building $case_name"
        make -C "$dir" -j16
        ls -la "$dir"/amr*.ex
    done
}

do_submit() {
    for target in "$@"; do
        # Either a case (submit all of its runs) or a single run directory.
        if [[ -f "$HERE/$target/job.slurm" ]]; then
            local runs=("$HERE/$target")
        else
            mapfile -t runs < <(find "$HERE/$target" -mindepth 2 \
                                     -name job.slurm -printf '%h\n' | sort)
        fi
        [[ ${#runs[@]} -gt 0 ]] || { echo "error: no job.slurm under $target" >&2; exit 2; }
        for run in "${runs[@]}"; do
            echo "==> sbatch ${run#$HERE/}"
            ( cd "$run" && sbatch job.slurm )
        done
    done
}

# ----------------------------------------------------------------------
[[ $# -ge 1 ]] || usage
action="$1"; shift
targets=("$@")
[[ ${#targets[@]} -gt 0 ]] || targets=("${DEFAULT_CASES[@]}")

case "$action" in
    build)  do_build  "${targets[@]}" ;;
    submit) do_submit "${targets[@]}" ;;
    all)    do_build "${targets[@]}"; do_submit "${targets[@]}" ;;
    *)      usage ;;
esac

echo
echo "done. after the jobs finish:"
echo "    python3 FlowPastSphere/plot.py"
echo "    python3 FallingSphere/plot.py   (if available)"
