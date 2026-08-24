"""A/B isolate: time the mxfp8 variable-K wgrad FlyDSL kernel DIRECTLY on synthetic operands.
Run twice (PYTHONPATH=MegaMoE vs PYTHONPATH=Primus-Turbo) to see if the identical-source kernel
runs at different speed under the two package trees (compiled-kernel / flydsl / _C env difference).
Timing only (random scales -> wrong output, right shapes/work)."""
import os
import torch

import primus_turbo  # noqa: F401
from primus_turbo.flydsl.grouped_gemm.mxfp8_grouped_kernel import (
    grouped_gemm_mxfp8_variable_k_flydsl_kernel,
)

torch.cuda.set_device(0)
dev = "cuda"
G, OUT_M, OUT_N = 32, 7168, 2048
# group sizes (multiples of BM=256). PT_GROUPS: "balanced" = 32x2048 (m_pad 65536, power-of-2);
# "mixed" = 30x2048 + 2x2304 (m_pad 66048, matches source's real routing distribution).
_MODE = os.environ.get("PT_GROUPS", "balanced")
if _MODE == "mixed":
    sizes = [2048] * 30 + [2304] * 2
else:
    sizes = [2048] * 32
M_total = sum(sizes)
offs = torch.tensor([0] + list(torch.tensor(sizes).cumsum(0).tolist()), device=dev, dtype=torch.int64)

lhs = torch.randint(0, 240, (OUT_M, M_total), device=dev, dtype=torch.uint8).view(torch.float8_e5m2)
rhs = torch.randint(0, 240, (OUT_N, M_total), device=dev, dtype=torch.uint8).view(torch.float8_e5m2)
lhs_s = torch.randint(124, 130, (OUT_M, M_total // 32), device=dev, dtype=torch.uint8).view(torch.float8_e8m0fnu)
rhs_s = torch.randint(124, 130, (OUT_N, M_total // 32), device=dev, dtype=torch.uint8).view(torch.float8_e8m0fnu)


def f():
    return grouped_gemm_mxfp8_variable_k_flydsl_kernel(
        lhs, lhs_s, rhs, rhs_s, offs, OUT_M, OUT_N, G, out_dtype=torch.bfloat16, num_cu=None
    )


for _ in range(30):
    f()
torch.cuda.synchronize()
s, e = torch.cuda.Event(True), torch.cuda.Event(True)
s.record()
for _ in range(50):
    f()
e.record()
torch.cuda.synchronize()
ms = s.elapsed_time(e) / 50
tflops = 2 * M_total * OUT_M * OUT_N / (ms * 1e-3) / 1e12
print(f"GEMM {ms:.3f} ms  {tflops:.0f} TFLOPS   pkg={os.path.dirname(primus_turbo.__file__)}")
