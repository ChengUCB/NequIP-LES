"""Shared definitions for the MD timing benchmark: sizes, engines, models, CSV.

The sweep is driven by three tables here, so `time_md.py`, `time_lammps.py`,
`run_timing.sh` and `plot.py` cannot disagree about what is being measured.
"""

import csv
import math
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent / "tests"
sys.path.insert(0, str(TESTS))  # reuse tests/engines.py rather than duplicating it

MODELS_DIR = HERE / "models"
DERIVED_DIR = MODELS_DIR / "_derived"
ARTEFACTS = HERE / "artefacts"
CSV_PATH = HERE / "timing.csv"
BASE_XYZ = HERE / "water.xyz"

# ---------------------------------------------------------------------- MD setup ----
# NVE: no thermostat, so nothing but the model and the neighbour list contributes to the
# per-step cost. Same integrator, timestep and initial velocities in every engine.
# overridable from the environment so a dry run can be cheap; the defaults are what a real
# sweep should use, and every row records the values it ran with
STEPS = int(os.environ.get("STEPS", 100))      # timed steps
WARMUP = int(os.environ.get("WARMUP", 100))    # run before the clock starts, never timed
REPEATS = int(os.environ.get("REPEATS", 3))    # timed segments per point; median reported
TIMESTEP_FS = 1.0
TEMPERATURE_K = 300.0
SEED = 12345

# The neighbour-list backend is fixed for every ASE/torch-sim row and recorded in the CSV:
# that path rebuilds the list on the CPU every step, and the backend moves the total
# materially, so a comparison across rows is only meaningful at a fixed choice.
NL_BACKEND = os.environ.get("TIMING_NL_BACKEND", "matscipy")

# ------------------------------------------------------------------------- sizes ----
# 192-atom base cell (12.417 A), roughly geometric, stopping near 10,000 atoms, kept as
# close to cubic as the ladder allows so neither the neighbour list nor the k-grid is
# skewed by one long axis.
SIZES = [
    (1, 1, 1), (2, 1, 1), (2, 2, 1), (3, 2, 1), (2, 2, 2),
    (3, 2, 2), (3, 3, 2), (4, 3, 2), (4, 3, 3), (4, 4, 3),
]

DL = 2.0  # the les_args value the models are trained with; sets the physical k cutoff


def n_max_for(cell_lengths, nxyz, dl=DL):
    """Smallest `N_max` whose integer grid still covers the |k| <= 2*pi/dl sphere.

    `dl` sets the physical cutoff; `N_max` only has to reach it, which needs
    `N_max >= L_max / dl`. Below that the sphere is silently truncated.
    """
    longest = max(L * n for L, n in zip(cell_lengths, nxyz))
    return math.ceil(longest / dl)


# ------------------------------------------------------------------------ engines ----
# The deployment paths worth timing: ASE with every accelerator kernel that works there, and
# LAMMPS plain. Both take a compiled artefact and both pass a cell, which a periodic Ewald sum
# needs. LAMMPS gets no accelerator row -- see the note below.
#
# (engine name, target, modifier or None, runner)
ENGINES = {
    "nequip": [
        ("ase-eager",       None,          None,                      "md"),
        ("ase-aoti",        "ase",         None,                      "md"),
        ("ase-aoti-oeq",    "ase",         "enable_OpenEquivariance", "md"),
        ("ase-aoti-cueq",   "ase",         "enable_CuEquivariance",   "md"),
        ("lammps",          "pair_nequip", None,                      "lammps"),
    ],
    "allegro": [
        ("ase-eager",       None,          None,                              "md"),
        ("ase-aoti",        "ase",         None,                              "md"),
        ("ase-aoti-cueq",   "ase",         "enable_CuEquivarianceContracter", "md"),
        ("ase-aoti-triton", "ase",         "enable_TritonContracter",         "md"),
        ("lammps",          "pair_nequip", None,                              "lammps"),
    ],
}

# Deliberately absent, each for a reason established in tests/ and the docs:
#
#   torchsim (--target batch)   torch-sim is not installed in the target environment
#   ase-eager / ase-compile     training-time paths, not deployment
#   --target pair_allegro       passes no cell, so a periodic LES model is refused at export
#   --mode torchscript          nequip refuses it on torch >= 2.10
#   LAMMPS ML-IAP               passes the model no positions and no cell
#
#   accelerator kernels         a different question (which kernel backend is fastest) from what
#                               LES costs, and unavailable where it would matter. Under a LAMMPS
#                               pair style, OpenEquivariance and cuEquivariance artefacts export
#                               but fail to load with `Could not find schema for ...` (measured),
#                               since their operators are registered from Python. OpenEquivariance
#                               does have a LAMMPS route -- ML-IAP -- but that one passes no
#                               positions and no cell, so it runs SR models only and can never
#                               produce the LES rows this benchmark compares against. The one
#                               modifier that works under a pair style, enable_TritonContracter,
#                               is Allegro-only.
#
# To bring any of them back, add the row here; nothing else needs to change.
# ("lammps-triton",   "pair_nequip", "enable_TritonContracter", "lammps"),   allegro only
# ("torchsim",        "batch",       None,                      "md"),
# ("ase-compile",     None,          None,                      "md"),      train-time compile
#
# The ASE accelerator rows need their package importable at export and at run time:
#   enable_OpenEquivariance         -> openequivariance
#   enable_CuEquivariance*          -> cuequivariance_torch
#   enable_TritonContracter         -> triton
# A missing one makes `prepare.py` log a failed export and the sweep record status=error for
# that engine; the rest of the matrix is unaffected.


# The `nmax-scaled` policy exists to answer a different question -- what LES costs at fixed
# accuracy -- and the answer does not depend on the engine, so it runs on two engines only.
# SR is exempt entirely: with no Ewald sum, `N_max` cannot affect it.
SCALED_ENGINES = {"ase-aoti", "lammps"}
POLICIES = ["fixed", "scaled"]

VARIANTS = ["sr", "les", "les_u"]


def model_files():
    """(path, backbone, variant) for each checkpoint in models/, by filename convention."""
    out = []
    for p in sorted(MODELS_DIR.glob("*.ckpt")):
        stem = p.stem
        for backbone in ("nequip", "allegro"):
            if stem.startswith(backbone + "_"):
                variant = stem[len(backbone) + 1:]
                if variant in VARIANTS:
                    out.append((p, backbone, variant))
                break
    return out


def artefact_path(model_stem, target, modifier=None, n_max=None):
    name = model_stem
    if n_max is not None:
        name += f"_nmax{n_max}"
    name += f"_{target}"
    if modifier:
        name += f"_{modifier}"
    return ARTEFACTS / f"{name}.nequip.pt2"


def compile_cmd():
    """`nequip-compile`, through the TF32 wrapper when it is present: on torch 2.13 + CUDA
    every export otherwise dies reading a legacy cuDNN flag."""
    fix = TESTS / "compile_tf32fix.py"
    return [sys.executable, str(fix)] if fix.exists() else ["nequip-compile"]


# --------------------------------------------------------------------------- CSV ----
FIELDS = [
    "model", "backbone", "variant", "engine", "accel", "nmax_policy", "n_max", "k_vectors",
    "natoms", "nx", "ny", "nz", "steps", "repeat", "wall_s", "s_per_step", "steps_per_s",
    "atom_steps_per_s", "nl_backend", "dtype", "device", "gpu_name", "torch_version", "status",
]


def append_row(row):
    new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def done_keys():
    """(model, engine, policy, nx, ny, nz, repeat) already in the CSV, so a re-run resumes."""
    if not CSV_PATH.exists():
        return set()
    with open(CSV_PATH, newline="") as fh:
        return {
            (r["model"], r["engine"], r["nmax_policy"],
             r["nx"], r["ny"], r["nz"], r["repeat"])
            for r in csv.DictReader(fh)
        }


def gpu_name():
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
    except Exception:
        return "cpu"
