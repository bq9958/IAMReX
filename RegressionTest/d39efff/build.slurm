#!/bin/bash
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 01:00:00
#SBATCH -J build-amr3d
#SBATCH -o build-amr3d.out
#SBATCH -A m3766
#
# Build the GPU (CUDA) and the CPU binary of amr3d for this commit.
#
#   sbatch build.slurm                # both, incremental
#   sbatch build.slurm gpu            # GPU only
#   sbatch build.slurm cpu            # CPU only
#   sbatch build.slurm clean          # wipe both configs first, then build both
#   sbatch build.slurm clean gpu      # wipe the GPU config first, then build it
#
# Products -- the two names never collide, so both live here side by side:
#
#   amr3d.gnu.MPI.CUDA.ex        <- GPU build, used by */GPU/job.slurm
#   amr3d.gnu.x86-milan.MPI.ex   <- CPU build, used by */CPU/job.slurm
#
# The ".x86-milan" suffix is not decoration: amrex/Tools/GNUMake/Make.defs
# appends $(CRAY_CPU_TARGET) only when USE_CUDA != TRUE, so that name *is*
# the marker of a non-CUDA build.  Hence craype-x86-milan is loaded
# explicitly below -- without it the CPU binary gets a different name and
# the CPU job.slurm files stop finding it.
#
# nvcc does not need a GPU to compile, so both builds run on a CPU node in
# the shared queue and you are charged only for the cores requested.  NERSC
# would rather you built on a login node (run_nersc.sh does); use this when
# you want the build queued, logged and reproducible instead.
#
# If the shared QOS is not available to your account, swap the two lines:
#   #SBATCH -q shared  /  #SBATCH -n 1 -c 32
#   #SBATCH -q regular /  #SBATCH -N 1                (charges a whole node)

set -euo pipefail

# sbatch copies this script into a spool directory before running it, so
# ${BASH_SOURCE[0]} points at /var/spool/slurmd/job*/slurm_script, not at
# the tree.  SLURM_SUBMIT_DIR is the directory sbatch was invoked from;
# fall back to the script's own location when run outside Slurm.
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$HERE"

if [[ ! -f GNUmakefile ]]; then
    echo "error: no GNUmakefile in $HERE" >&2
    echo "       submit from the commit directory: cd <...>/d39efff && sbatch build.slurm" >&2
    exit 2
fi

# -c counts logical CPUs (SMT2), so physical cores = SLURM_CPUS_PER_TASK / 2.
JOBS="${JOBS:-$(( ${SLURM_CPUS_PER_TASK:-32} / 2 ))}"

# lmod is not always initialised in a batch shell.
if ! command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /opt/cray/pe/lmod/lmod/init/bash
fi

# ----------------------------------------------------------------------
# Arguments: an optional "clean" keyword plus zero or more targets.
# ----------------------------------------------------------------------
CLEAN=0
TARGETS=()
for arg in "$@"; do
    case "$arg" in
        clean)    CLEAN=1 ;;
        gpu|cpu)  TARGETS+=("$arg") ;;
        *)        echo "error: unknown argument '$arg' (want: clean, gpu, cpu)" >&2
                  exit 2 ;;
    esac
done
[[ ${#TARGETS[@]} -gt 0 ]] || TARGETS=(gpu cpu)

# ----------------------------------------------------------------------
# Module stacks.  Two constraints actually bite here:
#
#   * cray-mpich/9.1.0's libmpi_gtl_cuda.so has NEEDED libcudart.so.13, so
#     it pairs with cudatoolkit/13.x and nothing else.  Pairing it with
#     cudatoolkit/12.x gives a wall of "undefined reference to
#     cudaXxx@libcudart.so.13" at link time.  The CUDA 12 alternative is
#     cray-mpich/9.0.1 (its GTL wants libcudart.so.12) with
#     cudatoolkit/12.4 and gcc-native<=13.2.
#
#   * nvcc's host-compiler cap lives in crt/host_config.h: CUDA 13.x
#     accepts gcc <= 15, so gcc-native/14 is fine.  (CUDA 12.4 caps at 13.)
#
#   * amrex/Tools/GNUMake/comps/nvcc.mak refuses nvcc < 12.2 outright.
# ----------------------------------------------------------------------
GPU_MODULES=(PrgEnv-gnu gcc-native/14 cray-mpich/9.1.0 cudatoolkit/13.2 craype-accel-nvidia80)

# The CPU build goes through the Cray wrappers (Make.nersc sets CC=cc,
# CXX=CC when USE_CUDA != TRUE), and craype-accel-nvidia80 makes those
# wrappers inject -lmpi_gtl_cuda.  Keep it out.
CPU_MODULES=(PrgEnv-gnu gcc-native/14 cray-mpich/9.1.0 craype-x86-milan)

# ----------------------------------------------------------------------
# One build, in a subshell so its module changes do not leak to the next.
# ----------------------------------------------------------------------
build() (
    local tag="$1" use_cuda="$2"; shift 2
    local -a mods=("$@")

    echo
    echo "======================================================================"
    echo "==  $tag  (USE_CUDA=$use_cuda, make -j$JOBS)"
    echo "======================================================================"

    module reset
    module load "${mods[@]}"
    [[ "$tag" == cpu ]] && module unload craype-accel-nvidia80 2>/dev/null || true
    module list

    if [[ "$CLEAN" == 1 ]]; then
        # cleanconfig wipes only this configuration's objects and binary;
        # "make clean" would nuke the other build too.
        echo "--- make USE_CUDA=$use_cuda cleanconfig"
        make USE_CUDA="$use_cuda" cleanconfig
    fi

    local t0 t1
    t0=$(date +%s)
    make -j "$JOBS" USE_CUDA="$use_cuda"
    t1=$(date +%s)
    echo "--- $tag build finished in $(( t1 - t0 ))s"
)

for t in "${TARGETS[@]}"; do
    case "$t" in
        gpu) build gpu TRUE  "${GPU_MODULES[@]}" ;;
        cpu) build cpu FALSE "${CPU_MODULES[@]}" ;;
    esac
done

# ----------------------------------------------------------------------
echo
echo "======================================================================"
echo "==  products"
echo "======================================================================"
ls -la "$HERE"/amr3d*.ex 2>/dev/null || { echo "no executable produced"; exit 1; }

for exe in "$HERE"/amr3d*.ex; do
    echo
    echo "--- $(basename "$exe")"
    # Any "not found" here means the runtime module stack at submit time
    # will not match the one this was linked against.
    ldd "$exe" 2>/dev/null | grep -iE "cudart|mpi_gtl|libmpi|not found" || true
done

echo
echo "done."
echo "  GPU runs expect: amr3d.gnu.MPI.CUDA.ex"
echo "  CPU runs expect: amr3d.gnu.x86-milan.MPI.ex"
