#!/usr/bin/env bash
# Build LAMMPS for the two NequIP interfaces.
#
#   ./build_lammps.sh check                                        # preflight only
#   ./build_lammps.sh pair  /global/scratch/users/$USER/software   # pair_style nequip/allegro
#   ./build_lammps.sh mliap /global/scratch/users/$USER/software   # pair_style mliap unified
#
# Two SEPARATE SOURCE TREES on purpose, not two build directories: patch_lammps.sh appends
# `find_package(Torch REQUIRED)` to lammps/cmake/CMakeLists.txt and copies the pair-style
# sources into src/, so every build from a patched tree links libtorch. An ML-IAP build in
# that tree would hold libtorch twice -- once linked into lmp, once imported by the Python
# interpreter lmp embeds -- which is a well-known way to crash. The mliap mode refuses to
# build from a patched tree for that reason.
#
# Env overrides:
#   JOBS=32                 parallel build jobs
#   KOKKOS=off              pair mode only: build without Kokkos (see below)
#   KOKKOS_ARCH_OVERRIDE=HOPPER90
#   CMAKE_EXTRA="-D..."     extra cmake flags
#
# Sources, all read for this script:
#   https://github.com/mir-group/pair_nequip_allegro   (README, patch_lammps.sh,
#       .github/workflows/tests.yml -- whose GPU cmake line these flags follow, and which
#       is tested against CUDA 12.6, so a 12.x toolchain is the proven configuration)
#   https://nequip.readthedocs.io/en/latest/integrations/lammps/mliap.html
#   https://nequip.readthedocs.io/en/latest/integrations/lammps/pair_styles.html
set -uo pipefail

MODE="${1:-}"
PREFIX="${2:-/global/scratch/users/$USER/software}"
JOBS="${JOBS:-$( (nproc 2>/dev/null || sysctl -n hw.ncpu) )}"
KOKKOS="${KOKKOS:-on}"

case "$MODE" in
    pair|mliap|check) ;;
    *) echo "usage: $0 {pair|mliap|check} [install-prefix]"; exit 2 ;;
esac

PY=$(command -v python)
say() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# ------------------------------------------------------------------ preflight ----
say "preflight"

command -v cmake >/dev/null || die "cmake not found (module load cmake?)"
command -v git   >/dev/null || die "git not found"

TORCH_CUDA=$("$PY" -c "import torch; print(torch.version.cuda or 'none')" 2>/dev/null) \
    || die "torch not importable in this python ($PY)"
"$PY" - <<'EOF'
import torch
print(f"  python           : {torch.__file__.split('/lib/')[0]}")
print(f"  torch            : {torch.__version__}")
print(f"  torch CUDA       : {torch.version.cuda}")
print(f"  CUDA available   : {torch.cuda.is_available()}")
print(f"  cmake prefix     : {torch.utils.cmake_prefix_path}")
abi = torch._C._GLIBCXX_USE_CXX11_ABI
print(f"  CXX11 ABI        : {abi}")
if not abi:
    print("  !! ABI is False: with Kokkos this build will not link against these wheels.")
    print("     Use a cxx11-abi libtorch, or torch built from source (see the pair README).")
EOF

NVCC_CUDA=none
if command -v nvcc >/dev/null; then
    NVCC_CUDA=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p')
    echo "  nvcc             : $NVCC_CUDA  ($(command -v nvcc))"
else
    echo "  nvcc             : NOT FOUND -- 'module load cuda' before building for GPU."
    echo "                     The CUDA bundled with pip torch cannot compile LAMMPS."
fi

# The one mismatch that reliably breaks the build: two CUDA *majors* in one binary. libtorch
# pulls in its own cudart, Kokkos compiles against the toolkit's, and a major-version
# difference between them is not supported. Minor differences within a major are fine --
# CUDA guarantees minor-version compatibility.
if [[ "$TORCH_CUDA" != none && "$NVCC_CUDA" != none ]]; then
    if [[ "${TORCH_CUDA%%.*}" != "${NVCC_CUDA%%.*}" ]]; then
        echo
        echo "  !!!! CUDA MAJOR MISMATCH: torch is built for CUDA $TORCH_CUDA, nvcc is $NVCC_CUDA."
        echo "       Align them before building. Either load a CUDA ${TORCH_CUDA%%.*}.x module, or"
        echo "       reinstall torch for CUDA ${NVCC_CUDA%%.*}.x keeping the same torch version, e.g."
        echo "         pip install torch==<same version>+cu${NVCC_CUDA/./} \\"
        echo "             --index-url https://download.pytorch.org/whl/cu${NVCC_CUDA/./}"
        echo "       Check which wheels exist first:"
        echo "         https://download.pytorch.org/whl/cu${NVCC_CUDA/./}/torch/"
        [[ "$MODE" != check ]] && die "refusing to build with mismatched CUDA majors (set FORCE=1 to override)"
    else
        echo "  CUDA majors      : match (torch $TORCH_CUDA / nvcc $NVCC_CUDA)"
    fi
fi
[[ -n "${FORCE:-}" ]] && echo "  (FORCE=1 set: CUDA checks are advisory)"

# Kokkos needs the exact GPU architecture; getting it wrong costs a full rebuild.
KOKKOS_ARCH=""
if command -v nvidia-smi >/dev/null; then
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
    case "$CC" in
        70) KOKKOS_ARCH=VOLTA70 ;;  75) KOKKOS_ARCH=TURING75 ;;
        80) KOKKOS_ARCH=AMPERE80 ;; 86) KOKKOS_ARCH=AMPERE86 ;;
        89) KOKKOS_ARCH=ADA89 ;;    90) KOKKOS_ARCH=HOPPER90 ;;
        100|103|120) KOKKOS_ARCH=BLACKWELL${CC} ;;
    esac
    echo "  GPU              : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) (cc $CC)"
    echo "  Kokkos arch      : ${KOKKOS_ARCH:-<unmapped; set KOKKOS_ARCH_OVERRIDE=>}"
fi
KOKKOS_ARCH="${KOKKOS_ARCH_OVERRIDE:-$KOKKOS_ARCH}"

if [[ "$MODE" == check ]]; then
    echo
    echo "  prefix would be  : $PREFIX"
    echo "preflight only; nothing built."
    exit 0
fi

mkdir -p "$PREFIX" || die "cannot create $PREFIX"
PREFIX=$(cd "$PREFIX" && pwd)
echo "  prefix           : $PREFIX"

# ----------------------------------------------------------------- pair styles ----
if [[ "$MODE" == pair ]]; then
    SRC="$PREFIX/lammps-pair"
    PATCHREPO="$PREFIX/pair_nequip_allegro"

    say "sources"
    # The pair styles require the 10 Sep 2025 LAMMPS release or newer, so take the tip.
    [[ -d "$SRC" ]]       || git clone --depth=1 https://github.com/lammps/lammps "$SRC" || die "clone lammps"
    [[ -d "$PATCHREPO" ]] || git clone --depth=1 https://github.com/mir-group/pair_nequip_allegro "$PATCHREPO" || die "clone pair repo"

    say "patch"
    if grep -q "find_package(Torch REQUIRED)" "$SRC/cmake/CMakeLists.txt"; then
        echo "  already patched, skipping"
    else
        ( cd "$PATCHREPO" && ./patch_lammps.sh "$SRC" ) || die "patch_lammps.sh failed"
    fi

    say "configure"
    FLAGS=(
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_PREFIX_PATH="$("$PY" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
        # what lets lmp load .nequip.pt2 (AOTInductor) files at all; without it only
        # TorchScript works, which nequip refuses to emit on torch >= 2.10
        -DNEQUIP_AOT_COMPILE=ON
        # torch's CMake hunts for an MKL it does not need; point it anywhere that exists
        -DMKL_INCLUDE_DIR=/tmp
    )
    if [[ "$KOKKOS" == on ]]; then
        # Kokkos is recommended (best GPU performance, and the only GPU-resident path for
        # pair_allegro), and when present the pair styles require it in its DEFAULT
        # double-double precision -- so no precision flags are passed here.
        # pair_nequip_allegro's CI builds OpenMP alongside Kokkos. The README calls the
        # two "mutually exclusive", which is about which one you use at run time, not about
        # what may be compiled in.
        FLAGS+=(-DPKG_KOKKOS=ON -DPKG_OPENMP=yes -DKokkos_ENABLE_OPENMP=ON)
        if [[ -n "$KOKKOS_ARCH" && "$NVCC_CUDA" != none ]]; then
            # Kokkos CUDA needs nvcc_wrapper as the C++ compiler, as the nequip ML-IAP docs
            # spell out for the same kind of build
            FLAGS+=(-DKokkos_ENABLE_CUDA=ON "-DKokkos_ARCH_${KOKKOS_ARCH}=ON"
                    -DCMAKE_CXX_COMPILER="$SRC/lib/kokkos/bin/nvcc_wrapper")
        fi
    else
        echo "  KOKKOS=off: building without Kokkos. The pair styles still run the model on"
        echo "  the GPU through libtorch; only the LAMMPS-side acceleration is missing."
    fi

    cmake -B "$SRC/build" -S "$SRC/cmake" "${FLAGS[@]}" ${CMAKE_EXTRA:-} \
        || die "cmake configure failed"

    say "build ($JOBS jobs)"
    cmake --build "$SRC/build" -j "$JOBS" || die "build failed"

    echo
    echo "  lmp: $SRC/build/lmp"
    echo "  export LMP=$SRC/build/lmp"
fi

# --------------------------------------------------------------------- ML-IAP ----
if [[ "$MODE" == mliap ]]; then
    SRC="$PREFIX/lammps-mliap"

    say "python dependencies"
    # cython is pinned by the nequip docs. cupy must match the CUDA *torch* was built for,
    # since it shares device memory with torch at runtime.
    CUDA_MAJOR="${TORCH_CUDA%%.*}"
    echo "  pip install 'cython==3.0.11' cupy-cuda${CUDA_MAJOR}x"
    "$PY" -m pip install -q "cython==3.0.11" "cupy-cuda${CUDA_MAJOR}x" \
        || echo "  !! cupy-cuda${CUDA_MAJOR}x failed; check which cupy wheel matches CUDA $TORCH_CUDA"

    say "sources"
    # A CLEAN tree: this one must not carry the pair-style patch.
    [[ -d "$SRC" ]] || git clone --depth=1 https://github.com/lammps/lammps "$SRC" || die "clone lammps"
    if grep -q "find_package(Torch REQUIRED)" "$SRC/cmake/CMakeLists.txt"; then
        die "$SRC is patched for the pair styles; ML-IAP needs an unpatched tree"
    fi

    say "configure"
    # ML-IAP in the NequIP framework is a KOKKOS integration -- the docs are explicit that
    # the wrapper targets the KOKKOS package -- so Kokkos is not optional here.
    FLAGS=(
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_CXX_COMPILER="$SRC/lib/kokkos/bin/nvcc_wrapper"
        -DPKG_KOKKOS=ON
        -DKokkos_ENABLE_CUDA=ON
        -DBUILD_MPI=ON
        -DPKG_ML-IAP=ON
        -DPKG_ML-SNAP=ON
        -DMLIAP_ENABLE_PYTHON=ON
        -DPKG_PYTHON=ON
        -DBUILD_SHARED_LIBS=ON
    )
    [[ -n "$KOKKOS_ARCH" ]] && FLAGS+=("-DKokkos_ARCH_${KOKKOS_ARCH}=ON")

    cmake -B "$SRC/build-mliap" -S "$SRC/cmake" "${FLAGS[@]}" ${CMAKE_EXTRA:-} \
        || die "cmake configure failed"

    say "build ($JOBS jobs)"
    cmake --build "$SRC/build-mliap" -j "$JOBS" || die "build failed"

    say "install the python module"
    # ML-IAP drives the model through an embedded interpreter, so LAMMPS' own python
    # bindings must be importable from the environment that runs it
    ( cd "$SRC/build-mliap" && make install-python ) || echo "  !! make install-python failed"

    echo
    echo "  lmp: $SRC/build-mliap/lmp"
    echo "  export LMP_MLIAP=$SRC/build-mliap/lmp"
    echo
    echo "  ML-IAP torch.compiles the model at run time. To stop it recompiling every run:"
    echo "    export TORCHINDUCTOR_AUTOGRAD_CACHE=1"
    echo "    export TORCHINDUCTOR_FX_GRAPH_CACHE=1"
    echo "    export TORCHINDUCTOR_CACHE_DIR=$PREFIX/inductor-cache"
fi
