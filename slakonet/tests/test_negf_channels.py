"""NEGF validation: for a pristine two-probe geometry the Landauer
transmission must equal the number of propagating Bloch channels of the
electrode at the same energy and transverse k.

``T(E) = N_channels(E)`` is exact for a defect-free device, so this is a
parameter-free check of the surface Green functions, the self-energy
signs and the Fisher-Lee trace all at once.
"""

import numpy as np
import torch

from slakonet.negf import SlakoNetNEGF, real_space_blocks


def _generalized_eigvals(H, S):
    """Eigenvalues of H psi = e S psi for Hermitian H, positive-definite S."""
    L = np.linalg.cholesky(S)
    Li = np.linalg.inv(L)
    return np.linalg.eigvalsh(Li @ H @ Li.conj().T)


def channels_from_blocks(h00, s00, h01, s01, energies, nk=2001):
    ks = np.linspace(0.0, 1.0, nk, endpoint=False)
    bands = np.empty((nk, h00.shape[0]))
    for i, k in enumerate(ks):
        ph = np.exp(2j * np.pi * k)
        H = h00 + h01 * ph + h01.conj().T * np.conj(ph)
        S = s00 + s01 * ph + s01.conj().T * np.conj(ph)
        bands[i] = np.sort(_generalized_eigvals(H, S))

    counts = np.empty(len(energies))
    for j, e in enumerate(energies):
        sgn = np.sign(bands - e)
        cross = np.sum(sgn[:-1] * sgn[1:] < 0) + np.sum(sgn[-1] * sgn[0] < 0)
        counts[j] = cross / 2.0
    return counts


def check_case(
    model,
    device_atoms,
    elec_atoms,
    axis,
    kt=(0.0, 0.0),
    energies=None,
    eta=1e-5,
):
    """Return (energies, T_negf, N_channels) for one transverse k."""
    negf = SlakoNetNEGF(
        model,
        device_atoms,
        elec_atoms,
        axis=axis,
        elec_axis=axis,
        eta=eta,
        verbose=False,
    )
    if energies is None:
        energies = np.arange(-8, 8.001, 0.05)

    eb = real_space_blocks(negf._calc_elec, axis, kt, negf.n_r)
    h00 = eb[0][0].numpy().astype(complex)
    s00 = eb[0][1].numpy().astype(complex)
    h01 = eb[1][0].numpy().astype(complex)
    s01 = eb[1][1].numpy().astype(complex)

    T = negf._transmission_single_k(energies, kt)
    N = channels_from_blocks(h00, s00, h01, s01, energies)
    return energies, T, N


if __name__ == "__main__":  # pragma: no cover - manual validation entry point
    import sys

    sys.path.insert(0, "/home/kamalch/Software/slako312/translakonet")
    from translakonet.fdf import read_struct_fdf
    from slakonet.optim import default_model

    root = (
        sys.argv[1]
        if len(sys.argv) > 1
        else ("/home/kamalch/Software/slako312/tln_runs/graphene")
    )
    axis = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    kt_b = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

    dev = read_struct_fdf(f"{root}/transport/STRUCT_DEVICE.fdf")
    elec = read_struct_fdf(f"{root}/electrode/STRUCT_ELEC.fdf")
    model = default_model()

    kt = (0.0, kt_b) if axis == 2 else (0.0, 0.0)
    E, T, N = check_case(model, dev, elec, axis, kt=kt)

    # away from band edges T must be integer and equal to N
    bad = np.abs(T - N)
    print(f"energies      : {len(E)}")
    print(f"max |T - N|   : {bad.max():.4f} at E = {E[np.argmax(bad)]:+.2f}")
    print(f"median |T - N|: {np.median(bad):.2e}")
    print(f"fraction within 0.05: {np.mean(bad < 0.05):.3f}")
    for i in range(0, len(E), 20):
        print(f"  E={E[i]:+6.2f}  T={T[i]:8.4f}  N={N[i]:5.1f}")
