"""
`nequip-compile`, working around a torch 2.13 TF32 inconsistency.

nequip disables TF32 the way torch documents for >= 2.9, with the new per-backend API:

    torch.backends.fp32_precision = "ieee"        # nequip/utils/global_state.py

but `torch.export` still reads the *legacy* flag on its way in:

    torch/export/_trace.py     orig_cudnn_flag = torch.backends.cudnn.set_flags(False)
    torch/backends/cudnn/...   torch._C._get_cudnn_allow_tf32()

and that legacy getter refuses to answer once the new API has been used:

    RuntimeError: PyTorch is checking whether allow_tf32 is enabled for cuDNN without a
    specific operator name, but the current flag(s) indicate that cuDNN conv and cuDNN
    RNN have different TF32 flags.

So every `nequip-compile` run fails, on CPU and CUDA alike, while training is unaffected
(it never goes through torch.export). Measured on torch 2.13.0:

    default state                       legacy getter OK
    conv = rnn = "ieee"  (new API)      raises      <- what nequip sets
    conv = rnn = "tf32"                 OK
    legacy cudnn.allow_tf32 = False     OK

The fix here is deliberately narrow: make the legacy getter answer instead of raising,
by reading the per-backend flag it should have been reading. Nothing about the model or
about nequip's TF32 choice changes.

Usage: same arguments as nequip-compile.
"""

import sys

import torch

_orig_get = torch._C._get_cudnn_allow_tf32


def _get_cudnn_allow_tf32() -> bool:
    try:
        return _orig_get()
    except RuntimeError:
        # the aggregate is unavailable; conv is the flag torch.export actually cares about
        return torch.backends.cudnn.conv.fp32_precision == "tf32"


torch._C._get_cudnn_allow_tf32 = _get_cudnn_allow_tf32

from nequip.scripts.compile import main  # noqa: E402  (import after the patch)

if __name__ == "__main__":
    sys.exit(main())
