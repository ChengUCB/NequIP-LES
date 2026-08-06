"""The long-range energy must end up inside `per_atom_energy` without changing anything else.

    pytest tests/test_lr_energy_per_atom.py

What is checked, in the order it matters:
  1. the per-atom array sums to `total_energy`      <- the bug this fixes
  2. `total_energy` itself is untouched
  3. batched and unbatched (LAMMPS pair targets pass no batch) agree
  4. the module survives TorchScript and torch.export, which deployment needs
"""

import pytest
import torch

from nequip.data import AtomicDataDict

from nequip_les import _keys
from nequip_les.nn._lr_energy_per_atom import DistributeLREnergyPerAtom


def _data(num_atoms_per_frame, e_lr, dtype=torch.float64, with_batch=True):
    """A minimal dict with just the fields the module reads."""
    n_total = sum(num_atoms_per_frame)
    data = {
        AtomicDataDict.PER_ATOM_ENERGY_KEY: torch.arange(
            n_total, dtype=dtype
        ).unsqueeze(-1),
        _keys.LR_ENERGY_KEY: torch.tensor(e_lr, dtype=dtype),
    }
    if with_batch:
        data[AtomicDataDict.BATCH_KEY] = torch.cat(
            [torch.full((n,), i, dtype=torch.long)
             for i, n in enumerate(num_atoms_per_frame)]
        )
    return data


def _per_frame_sum(per_atom, batch, n_frames):
    return torch.stack(
        [per_atom[batch == i].sum() for i in range(n_frames)]
    )


@pytest.mark.parametrize(
    "num_atoms_per_frame, e_lr",
    [
        ([5], [-4.035135]),            # single frame, the LAMMPS case
        ([4, 7], [1.5, -0.25]),        # uneven batch
        ([3, 3, 3], [0.0, -75.868619, 2.0]),
    ],
)
def test_sums_to_total(num_atoms_per_frame, e_lr):
    data = _data(num_atoms_per_frame, e_lr)
    sr_per_frame = _per_frame_sum(
        data[AtomicDataDict.PER_ATOM_ENERGY_KEY].reshape(-1),
        data[AtomicDataDict.BATCH_KEY],
        len(e_lr),
    )
    # what the model computes as `total_energy`: short-range sum + long-range
    total = sr_per_frame + torch.tensor(e_lr, dtype=torch.float64)

    out = DistributeLREnergyPerAtom()(dict(data))

    new_per_frame = _per_frame_sum(
        out[AtomicDataDict.PER_ATOM_ENERGY_KEY].reshape(-1),
        data[AtomicDataDict.BATCH_KEY],
        len(e_lr),
    )
    # 1. the per-atom array now sums to the total, per frame
    torch.testing.assert_close(new_per_frame, total)
    # 2. the long-range energy field itself is left alone
    torch.testing.assert_close(out[_keys.LR_ENERGY_KEY],
                               torch.tensor(e_lr, dtype=torch.float64))


def test_no_batch_matches_single_frame_batch():
    """The LAMMPS pair targets pass no batch key; the result must not depend on that."""
    with_batch = DistributeLREnergyPerAtom()(_data([6], [-1.25], with_batch=True))
    without = DistributeLREnergyPerAtom()(_data([6], [-1.25], with_batch=False))
    torch.testing.assert_close(
        with_batch[AtomicDataDict.PER_ATOM_ENERGY_KEY],
        without[AtomicDataDict.PER_ATOM_ENERGY_KEY],
    )


def test_share_is_even():
    data = _data([4], [8.0])
    before = data[AtomicDataDict.PER_ATOM_ENERGY_KEY].clone()
    out = DistributeLREnergyPerAtom()(dict(data))
    delta = out[AtomicDataDict.PER_ATOM_ENERGY_KEY] - before
    torch.testing.assert_close(delta, torch.full_like(delta, 2.0))


def test_dtype_follows_per_atom_energy():
    """A float64 cell/energy must not silently promote a float32 model's energies."""
    data = _data([5], [1.0], dtype=torch.float32)
    data[_keys.LR_ENERGY_KEY] = data[_keys.LR_ENERGY_KEY].to(torch.float64)
    out = DistributeLREnergyPerAtom()(data)
    assert out[AtomicDataDict.PER_ATOM_ENERGY_KEY].dtype == torch.float32


def test_torchscript():
    """LAMMPS pair styles on torch < 2.10 load TorchScript, so it has to script."""
    scripted = torch.jit.script(DistributeLREnergyPerAtom())
    data = _data([4, 2], [1.0, -1.0])
    eager = DistributeLREnergyPerAtom()(dict(data))
    out = scripted(dict(data))
    torch.testing.assert_close(
        out[AtomicDataDict.PER_ATOM_ENERGY_KEY],
        eager[AtomicDataDict.PER_ATOM_ENERGY_KEY],
    )


def test_export():
    """AOTInductor goes through torch.export; a scatter here would break that."""

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = DistributeLREnergyPerAtom()

        def forward(self, per_atom, e_lr, batch):
            out = self.m({
                AtomicDataDict.PER_ATOM_ENERGY_KEY: per_atom,
                _keys.LR_ENERGY_KEY: e_lr,
                AtomicDataDict.BATCH_KEY: batch,
            })
            return out[AtomicDataDict.PER_ATOM_ENERGY_KEY]

    per_atom = torch.arange(6, dtype=torch.float64).unsqueeze(-1)
    e_lr = torch.tensor([1.0, 2.0], dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    dyn = {0: torch.export.Dim.DYNAMIC}
    exported = torch.export.export(
        Wrap(), (per_atom, e_lr, batch),
        dynamic_shapes=({0: torch.export.Dim.DYNAMIC, 1: None}, dyn, dyn),
    )
    torch.testing.assert_close(
        exported.module()(per_atom, e_lr, batch), Wrap()(per_atom, e_lr, batch)
    )
