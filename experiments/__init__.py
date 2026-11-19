import os

# fixes the BLAS, OpenMP and torch thread count (LCMM_THREADS, default 2) for reproducible numbers
N = int(os.environ.setdefault("LCMM_THREADS", "2"))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(N)

import torch  # noqa: E402

torch.set_num_threads(N)
torch.set_num_interop_threads(N)
