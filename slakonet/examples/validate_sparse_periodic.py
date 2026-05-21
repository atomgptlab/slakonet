"""Validate periodic sparse H(k)/S(k) + near-gap solver vs dense.

Bulk Si, 2x2x2 MP grid. For each k: dense reference = eighb on the
dense periodic hs_matrix; sparse = solve_near_gap on the sparse complex
H(k)/S(k). Compare the k eigenvalues nearest an interior sigma.

Run in slakonet's native float32 (the dense Periodic path mixes
float32 cell-translation tensors; forcing float64 hits a dtype bug
unrelated to this work).
"""

import numpy as np
import torch
from ase.build import bulk

from slakonet.atoms import Geometry, Periodic
from slakonet.basis import Basis
from slakonet.optim import default_model
from slakonet.utils import create_feeds, generate_shell_dict_upto_Z65, eighb
from slakonet.slaterkoster import hs_matrix
from slakonet.sparse_sk import hs_matrix_sparse, solve_near_gap

CUTOFF = 10.0
KGRID = 2          # 2x2x2 Monkhorst-Pack
KNEAR = 6          # near-gap states per k

model = default_model()
shell_dict = generate_shell_dict_upto_Z65(model=model)
skfs = model.get_updated_skfs()

si = bulk("Si", "diamond", a=5.43)
geometry = Geometry.from_ase_atoms([si])
basis = Basis(geometry.atomic_numbers, shell_dict)
h_feed = create_feeds(skfs, shell_dict, "H")
s_feed = create_feeds(skfs, shell_dict, "S")

# --- dense periodic reference -------------------------------------------
per = Periodic(
    geometry, geometry.cell, cutoff=CUTOFF,
    kpoints=torch.tensor([[KGRID, KGRID, KGRID]]),
)
kfrac = np.asarray(per.kpoints).reshape(-1, 3)
nk = kfrac.shape[0]

Hd = hs_matrix(per, basis, h_feed, cutoff=CUTOFF)
Sd = hs_matrix(per, basis, s_feed, cutoff=CUTOFF)
if Hd.dim() == 4:           # (1, Norb, Norb, nk)
    Hd, Sd = Hd[0], Sd[0]

dense_bands = []
for ik in range(nk):
    ev, _ = eighb(Hd[..., ik], Sd[..., ik], scheme="chol")
    dense_bands.append(np.sort(ev.real.detach().numpy().flatten()))
dense_bands = np.stack(dense_bands, 0)          # (nk, Norb)

# interior target: midpoint of the widest gap of the k-averaged spectrum
mean_spec = dense_bands.mean(0)
lo, hi = len(mean_spec) // 3, 2 * len(mean_spec) // 3
g = lo + int(np.argmax(np.diff(mean_spec[lo : hi + 1])))
sigma = float(0.5 * (mean_spec[g] + mean_spec[g + 1]))

# --- sparse per-k --------------------------------------------------------
print(f"bulk Si  Norb={dense_bands.shape[1]}  nk={nk}  "
      f"sigma={sigma:+.4f} Ha")
max_err = 0.0
for ik in range(nk):
    kp = kfrac[ik]
    Hk = hs_matrix_sparse(geometry, basis, h_feed, cutoff=CUTOFF, kpoint=kp)
    Sk = hs_matrix_sparse(geometry, basis, s_feed, cutoff=CUTOFF, kpoint=kp)
    herm = abs((Hk - Hk.conj().t()).coalesce().values().abs().max().item()) \
        if Hk._nnz() else 0.0
    got = solve_near_gap(Hk, Sk, k=KNEAR, sigma=sigma)
    ref = dense_bands[ik][np.argsort(np.abs(dense_bands[ik] - sigma))[:KNEAR]]
    ref.sort()
    e = float(np.abs(got - ref).max())
    max_err = max(max_err, e)
    print(
        f"  k={np.array2string(kp, precision=2):<22} "
        f"H(k) herm={herm:.1e}  max|Δeig|={e:.2e}"
    )

ok = max_err < 1e-4    # float32 path
print(f"\nmax|Δeig| over all k = {max_err:.2e}  -> "
      f"{'PASS' if ok else 'FAIL'}")
