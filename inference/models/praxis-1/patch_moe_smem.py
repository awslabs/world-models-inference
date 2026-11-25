# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-time patch: cap the Triton MoE GEMM pipeline depth on small-shared-memory GPUs.

Upstream praxis (``praxis/kernels/moe_gemm_triton.py``) launches its grouped and
SwiGLU MoE GEMMs with ``num_stages=3`` and 128x128x64 tiles. Three stages need more
than 101,376 B of shared memory per block, so on an RTX PRO 6000 Blackwell or an
L40S-class GPU the kernel fails to launch with ``triton OutOfResources``. Two stages
fit. H100 (227 KB) and B200 have room for three and are left exactly as upstream.

Only the software-pipelining depth changes: the same tiles, the same arithmetic,
the same results. The patch is applied by the Dockerfile and fails the image build
if the upstream text it anchors on has moved, rather than shipping an unpatched
kernel that crashes at the first rollout.

Usage: python patch_moe_smem.py <path to praxis/kernels/moe_gemm_triton.py>
"""

from __future__ import annotations

import sys
from pathlib import Path

# 3 stages of 128x128x64 bf16 tiles need this much; below it, use 2.
SMEM_THRESHOLD = 147_456

LIMIT_DEF = (
    "# Shared memory per block on this GPU. Below "
    f"{SMEM_THRESHOLD} B (RTX PRO 6000, L40S) three pipeline\n"
    "# stages do not fit; the world-model-inference image caps them at two.\n"
    "_SMEM_LIMIT = (\n"
    "    torch.cuda.get_device_properties(0).shared_memory_per_block_optin\n"
    "    if torch.cuda.is_available()\n"
    "    else 1 << 20\n"
    ")\n\n"
)
ANCHOR_DEFAULT = "DEFAULT_CONFIG = {"
CFG_LINES = (
    "    cfg = {**DEFAULT_CONFIG, **(config or {})}\n",
    "    cfg = {**SWIGLU_CONFIG, **(config or {})}\n",
)
CAP = f'    if _SMEM_LIMIT < {SMEM_THRESHOLD}:\n        cfg["num_stages"] = min(cfg["num_stages"], 2)\n'
MARKER = "_SMEM_LIMIT ="


def patch(source: str) -> str:
    """Return ``source`` with the shared-memory cap applied. Raises on unknown text."""
    if MARKER in source:
        raise ValueError("already patched")
    if source.count(ANCHOR_DEFAULT) != 1:
        raise ValueError(f"expected exactly one {ANCHOR_DEFAULT!r}")
    out = source.replace(ANCHOR_DEFAULT, LIMIT_DEF + ANCHOR_DEFAULT, 1)
    for line in CFG_LINES:
        if out.count(line) != 1:
            raise ValueError(f"expected exactly one {line.strip()!r}")
        out = out.replace(line, line + CAP, 1)
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    path = Path(argv[1])
    try:
        path.write_text(patch(path.read_text()))
    except ValueError as exc:
        print(f"patch_moe_smem: {path}: {exc}; upstream changed, re-check the patch", file=sys.stderr)
        return 1
    print(f"patch_moe_smem: capped MoE GEMM num_stages at 2 below {SMEM_THRESHOLD} B smem in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
