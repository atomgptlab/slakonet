"""Non-collinear exchange of alpha-Fe2O3 (hematite): J_iso, J_ani and DMI.

slakonet gives the tight-binding Hamiltonian, TB2J the exchange tensor:

1. converge the collinear spin-polarized model (atomic Stoner I + anion
   dependent Hubbard U on the Fe 3d shell + SCC, all defaults; nothing to
   tune) in the antiferromagnetic order measured by neutron scattering;
2. build the spinor Hamiltonian

       H(R) = Hbar(R) x 1 + Delta(R) x (n . sigma) + delta_{R,0} zeta L.S

   with the spin axis n along x, y and z (zeta_3d of Fe from NIST atomic
   spectra; ``slakonet.tb2j_ncl.SOC_ZETA_CM1``);
3. run TB2J ``ExchangeNCL`` for each axis and merge the three runs into the
   full tensor (isotropic J, symmetric anisotropic J_ani, DMI vector D).

SOC is applied once, on top of the converged collinear potential
(force-theorem / one-shot SOC, as in TB2J's split-SOC workflow).

Expected result (default settings, ~80 s on 4 CPU cores; Fe 3.95 mu_B):

    shell  d (A)  JS2 (meV)  |D| (meV)  max|J_ani| (meV)
    J1     2.90    +10.9      0.03       0.002    face-sharing pair
    J2     2.97    +11.9      0          0.004    inversion centre
    J3     3.36    +42.2      0.38       0.012    INS: +32.0
    J4     3.70    +29.2      0.40       0.007    INS: +25.0
    J5     3.98     -0.7      0          0.002    inversion centre

JS2 = -2 J_iso converts TB2J's J_iso (normalised spins, J > 0 = FM) to the
per-bond J S^2 of the neutron-scattering (INS) fits, where > 0 = AFM. J3
and J4 dominate, as in the INS fit (here 1.3x and 1.2x too large). D
vanishes on the bonds with an inversion centre and D_ij = -D_ji, as
symmetry requires. J1 and J2 come out AFM, as in published DFT studies
(the INS fit makes them weakly FM). With ``--no-soc``, D = J_ani = 0
exactly and J_iso is unchanged to 0.03 meV.

Usage:
    python fe2o3_soc_dmi.py              # full tensor, 3 spin axes
    python fe2o3_soc_dmi.py --no-soc     # check: SOC off -> D = J_ani = 0
    python fe2o3_soc_dmi.py --quick      # coarser meshes, for a smoke test

Results are written to ``TB2J_results_Fe2O3_soc/`` (exchange.out,
TB2J.pickle, Multibinit/Vampire inputs). Requires TB2J (``pip install
TB2J``).
"""

from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
from ase import Atoms
from ase.spacegroup import crystal
from TB2J.io_exchange import SpinIO

from slakonet.optim import default_model
from slakonet.tb2j_ncl import gen_exchange_slakonet_soc


def hematite() -> Atoms:
    """Rhombohedral primitive cell of alpha-Fe2O3 (R-3c, 4 Fe + 6 O).

    Hexagonal setting a = 5.0356 A, c = 13.7489 A; Fe 12c z = 0.35530,
    O 18e x = 0.30590 (Blake et al., Am. Mineral. 51, 123 (1966)).
    """
    return crystal(
        ["Fe", "O"],
        basis=[(0.0, 0.0, 0.35530), (0.30590, 0.0, 0.25)],
        spacegroup=167,
        cellpar=[5.0356, 5.0356, 13.7489, 90, 90, 120],
        primitive_cell=True,
    )


def hematite_moments(atoms, m=4.0) -> np.ndarray:
    """Initial moments of the experimental AFM order (+ - - + along c).

    The face-sharing Fe pair (shortest Fe-Fe, 2.90 A) is antiparallel and
    the edge-sharing in-plane pairs (2.97 A) are parallel. Among the three
    balanced up/down assignments of the 4 Fe, pick the one that does that.
    """
    fe = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "Fe"]
    d = atoms.get_all_distances(mic=True)
    pairs = sorted((d[i, j], i, j) for i, j in itertools.combinations(fe, 2))
    face = pairs[0][1:]  # shortest Fe-Fe bond
    edge = [p[1:] for p in pairs if abs(p[0] - 2.97) < 0.05]
    for up in itertools.combinations(fe, 2):
        s = {i: (1 if i in up else -1) for i in fe}
        if s[face[0]] != s[face[1]] and all(s[i] == s[j] for i, j in edge):
            moments = np.zeros(len(atoms))
            moments[fe] = [m * s[i] for i in fe]
            return moments
    raise RuntimeError("could not identify the hematite magnetic order")


def summarize(path, atoms, max_shells=5):
    """Print J_iso, J_ani and D per distance shell of the merged result.

    The merged tensor is read back from ``path/TB2J.pickle``.
    """
    spinio = SpinIO.load_pickle(path)
    cell = atoms.get_cell()
    pos = atoms.get_positions()
    ind = spinio.index_spin  # atom index -> spin index (-1: not magnetic)
    atom_of = {s: i for i, s in enumerate(ind) if s >= 0}

    shells = {}
    for key, J in spinio.exchange_Jdict.items():
        R, i, j = key
        d = np.linalg.norm(pos[atom_of[j]] + np.dot(R, cell) - pos[atom_of[i]])
        shells.setdefault(round(d, 2), []).append(key)

    print(
        f"\n{'shell':>5} {'d (A)':>6} {'n':>3} {'J_iso':>8} {'JS2':>8}"
        f" {'max|Jani|':>9} {'|D| mean':>9}   D of one bond (meV)"
    )
    for n, d in enumerate(sorted(shells)[:max_shells], start=1):
        keys = shells[d]
        J = np.mean([spinio.exchange_Jdict[k] for k in keys]) * 1e3
        jani = max(np.abs(spinio.Jani_dict[k]).max() for k in keys) * 1e3
        D = np.array([spinio.dmi_ddict[k] for k in keys]) * 1e3
        Dn = np.linalg.norm(D, axis=1)
        print(
            f"{'J%d' % n:>5} {d:6.2f} {len(keys):3d} {J:+8.3f} {-2 * J:+8.2f}"
            f" {jani:9.4f} {Dn.mean():9.4f}   {np.round(D[0], 4)}"
        )

    # symmetry check: D_ij(R) = -D_ji(-R)
    worst = 0.0
    for (R, i, j), D in spinio.dmi_ddict.items():
        back = spinio.dmi_ddict.get((tuple(-np.array(R)), j, i))
        if back is not None:
            worst = max(worst, np.abs(np.asarray(D) + np.asarray(back)).max())
    print(f"\nmax |D_ij + D_ji| = {worst * 1e3:.2e} meV (should be ~0)")


def main():
    p = argparse.ArgumentParser(
        description="alpha-Fe2O3 exchange tensor with SOC"
    )
    p.add_argument(
        "--no-soc",
        action="store_true",
        help="switch SOC off (D and J_ani must vanish)",
    )
    p.add_argument("--quick", action="store_true", help="coarse meshes")
    p.add_argument("--out", default="TB2J_results_Fe2O3_soc")
    args = p.parse_args()

    atoms = hematite()
    m0 = hematite_moments(atoms)
    print(
        f"alpha-Fe2O3: {atoms.get_chemical_formula()}, "
        f"initial Fe moments {m0[m0 != 0].tolist()}"
    )

    kmesh = (3, 3, 3) if args.quick else (5, 5, 5)
    t0 = time.time()
    _, spin_model = gen_exchange_slakonet_soc(
        atoms,
        ["Fe"],
        model=default_model(),
        initial_moments=m0,
        scf_kmesh=kmesh,  # collinear self-consistency
        kmesh=kmesh,  # TB2J Green's function
        nz=30 if args.quick else 60,
        Rcut=4.1,  # up to J5
        soc_strength=0.0 if args.no_soc else 1.0,
        output_path=args.out,
        verbose=False,
    )
    fe = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "Fe"]
    print(
        f"done in {time.time() - t0:.0f} s; converged Fe moments (mu_B): "
        f"{np.round(spin_model.moments[fe], 2).tolist()}"
    )
    summarize(args.out, spin_model.atoms)
    print(f"\nFull TB2J output: {args.out}/exchange.out")


if __name__ == "__main__":
    main()
