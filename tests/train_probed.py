"""
`nequip-train`, plus proof of whether train-time compilation actually happened.

Checking the config is not enough. `CompileGraphModel.forward` falls back to the
eager model for any batch with fewer than two frames, nodes or edges, so a config
that says `compile_mode: compile` can train without ever compiling -- which is
exactly how this suite once passed while real training crashed.

So wrap the one function that only runs when a graph is really traced,
`nequip.utils.fx._nequip_make_fx`, and report what it produced:

    TRACED n_calls=2 nodes=2423

run_all.sh requires that line for a `compiled` config and requires its ABSENCE for
an `eager` or `legacy` one -- so a config that quietly compiles when it should not
is a failure too.

Usage: same arguments as nequip-train.

    python train_probed.py -cn nequip_water_compiled --config-dir configs
"""

import sys

import nequip.utils.fx as _fx

_calls = 0
_nodes = 0
_orig = _fx._nequip_make_fx


def _probe(model, inputs):
    global _calls, _nodes
    gm = _orig(model, inputs)
    _calls += 1
    _nodes = len(list(gm.graph.nodes))
    return gm


_fx._nequip_make_fx = _probe

from nequip.scripts.train import main  # noqa: E402  (import after patching)


if __name__ == "__main__":
    try:
        rc = main()
    finally:
        # stdout, and flushed, so the shell can grep it even when nequip-train fails
        if _calls:
            print(f"TRACED n_calls={_calls} nodes={_nodes}", flush=True)
        else:
            print("NOT-TRACED (ran eagerly)", flush=True)
    sys.exit(rc)
