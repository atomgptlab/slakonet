"""Automatic classification of magnetic 3d sites from the structure alone.

Used to pick the Stoner parameter of a site from a fitted table without
user input:

* ``site_environment``: "ionic" when most nearest neighbours of the site are
  anions (O, N, halogens, chalcogens), otherwise "metallic".
* ``formal_oxidation``: oxidation state of a transition-metal species from
  charge balance with fixed valences of the other elements, plus a few
  structural rules for ambiguous cases (S-S / Se-Se dimers, P-P bonds in
  P2S6 units, Te/As/Sb/Bi/P/Si as cations when bonded to O). Returns None
  when an element has no known valence or the result is not a sensible
  integer; callers then fall back to element-level values.
* ``spin_only_moment``: high-spin spin-only moment (mu_B) of a d^n ion.
"""

from __future__ import annotations

import numpy as np

ANIONS = {
    "O": -2,
    "S": -2,
    "Se": -2,
    "Te": -2,
    "N": -3,
    "F": -1,
    "Cl": -1,
    "Br": -1,
    "I": -1,
}
FIXED_CATIONS = {
    "H": 1,
    "Li": 1,
    "Na": 1,
    "K": 1,
    "Rb": 1,
    "Cs": 1,
    "Be": 2,
    "Mg": 2,
    "Ca": 2,
    "Sr": 2,
    "Ba": 2,
    "Zn": 2,
    "Cd": 2,
    "Sc": 3,
    "Y": 3,
    "La": 3,
    "Ce": 3,
    "Pr": 3,
    "Nd": 3,
    "Sm": 3,
    "Eu": 3,
    "Gd": 3,
    "Tb": 3,
    "Dy": 3,
    "Ho": 3,
    "Er": 3,
    "Tm": 3,
    "Yb": 3,
    "Lu": 3,
    "Al": 3,
    "Ga": 3,
    "In": 3,
    "B": 3,
    "Ti": 4,
    "Zr": 4,
    "Hf": 4,
    "Sn": 4,
    "Ge": 4,
    "Nb": 5,
    "Ta": 5,
    "Mo": 6,
    "W": 6,
}
# cation valence of elements that are anions or cations depending on bonding
OXO_CATIONS = {
    "Te": 6,
    "As": 5,
    "Sb": 5,
    "Bi": 3,
    "P": 5,
    "Si": 4,
    "Se": 4,
    "S": 6,
}
OXO_BOND = {"Bi": 2.6}  # M-O bond cutoff (Angstrom); default 2.2
PNICTIDE_ANIONS = {"As": -3, "Sb": -3, "Bi": -3, "P": -3}
TRANSITION_3D = ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"]
D_ELECTRONS_NEUTRAL = {
    "Sc": 3,
    "Ti": 4,
    "V": 5,
    "Cr": 6,
    "Mn": 7,
    "Fe": 8,
    "Co": 9,
    "Ni": 10,
    "Cu": 11,
    "Zn": 12,
}


def _neighbours(atoms, i, factor=1.15):
    """Indices of the nearest-neighbour shell of atom i (within factor x the
    nearest distance), with periodic images."""
    from ase.neighborlist import neighbor_list

    ii, jj, dd = neighbor_list("ijd", atoms, 6.0)
    sel = ii == i
    if not sel.any():
        return np.array([], int), np.array([])
    d = dd[sel]
    keep = d <= factor * d.min()
    return jj[sel][keep], d[keep]


def site_environment(atoms, i):
    """ "ionic" or "metallic" for atom i."""
    nb, _ = _neighbours(atoms, i)
    symbols = np.array(atoms.get_chemical_symbols())
    if len(nb) == 0:
        return "metallic"
    return (
        "ionic"
        if np.mean([s in ANIONS for s in symbols[nb]]) > 0.5
        else "metallic"
    )


def _bonded(atoms, i, element, cutoff):
    from ase.neighborlist import neighbor_list

    ii, jj = neighbor_list("ij", atoms, cutoff)
    symbols = atoms.get_chemical_symbols()
    return any(symbols[j] == element for j in jj[ii == i])


def ligand_state(atoms, k):
    """Fixed valence of a non-transition-metal atom k from its bonding, or
    None when it is not determined by the rules below."""
    s = atoms.get_chemical_symbols()[k]
    if s in ("S", "Se") and _bonded(atoms, k, s, 2.45):
        return -1  # persulfide-like dimer
    if s in OXO_CATIONS and _bonded(atoms, k, "O", OXO_BOND.get(s, 2.2)):
        return OXO_CATIONS[s]
    if s == "P" and _bonded(atoms, k, "P", 2.4):
        return 4  # P2S6 / P2Se6 units
    if s in ANIONS:
        return ANIONS[s]
    if s in PNICTIDE_ANIONS:
        return PNICTIDE_ANIONS[s]
    return FIXED_CATIONS.get(s)


def formal_oxidation(atoms, element):
    """Formal oxidation state of ``element`` (a 3d metal) by charge balance
    with the fixed valences of all other atoms, or None."""
    symbols = atoms.get_chemical_symbols()
    counts = {s: symbols.count(s) for s in set(symbols)}
    if element not in counts:
        return None
    total = 0.0
    for k, s in enumerate(symbols):
        if s == element:
            continue
        v = ligand_state(atoms, k)
        if v is None:
            return None
        total += v
    ox = -total / counts[element]
    if abs(ox - round(ox)) > 0.05 or not 1 <= round(ox) <= 5:
        return None
    return int(round(ox))


def d_count(element, oxidation):
    """d electrons of the ion: valence (4s + 3d) electrons minus oxidation."""
    return D_ELECTRONS_NEUTRAL[element] - oxidation if oxidation else None


def spin_only_moment(element, oxidation):
    """High-spin spin-only moment 2S (mu_B) of the ion, or None."""
    n = d_count(element, oxidation)
    if n is None or not 0 <= n <= 10:
        return None
    return float(n if n <= 5 else 10 - n)


# ---------------------------------------------------------------------------
# Oxidation-state atlas (Wyckoff-resolved, Alexandria), optional
# ---------------------------------------------------------------------------
# Checkout of the oxidation-state atlas; set OXSTATE_ATLAS_DIR to use it.
ATLAS_DIR_DEFAULT = None
_ATLAS_CACHE = {}


def _atlas_modules():
    """Import the oxidation-state atlas package (from OXSTATE_ATLAS_DIR or
    the Python path); None if unavailable."""
    import os
    import sys

    root = os.environ.get("OXSTATE_ATLAS_DIR", ATLAS_DIR_DEFAULT)
    if root and root not in sys.path and os.path.isdir(root):
        sys.path.append(root)
    try:
        from oxstate_atlas.alexandria import wyckoff_grammar
        from oxstate_atlas.atlas import load_atlas, to_prior
        from oxstate_atlas.sitegrammar import parse_sites
    except ImportError:
        return None
    return root, wyckoff_grammar, load_atlas, to_prior, parse_sites


def _atlas_prior(root, load_atlas, to_prior, symbols):
    """Anion-conditioned prior (O, then S, then N) when available, else the
    global Alexandria atlas."""
    import os

    folder = os.path.join(root, "results", "alexandria", "atlases")
    tag = next((f"ANION_{a}" for a in ("O", "S", "N") if a in symbols), "ALL")
    path = os.path.join(folder, f"{tag}_element_ox_summary.csv")
    if not os.path.exists(path):
        path = os.path.join(folder, "ALL_element_ox_summary.csv")
    if path not in _ATLAS_CACHE:
        _ATLAS_CACHE[path] = to_prior(load_atlas(path), p_min=0.01)
    return _ATLAS_CACHE[path]


def atlas_oxidation(atoms, budget=20000):
    """{atom index: oxidation state} from the Wyckoff-resolved atlas, or None.

    Every crystallographic orbit is a variable. Ligand orbits whose valence
    is fixed by ``ligand_state`` keep that value; 3d metals may only take
    positive states; everything else draws from the atlas prior. The
    exact-neutral assignment with the largest sum of log-probabilities wins.
    """
    from itertools import product
    from math import log

    mods = _atlas_modules()
    if mods is None:
        return None
    root, wyckoff_grammar, load_atlas, to_prior, parse_sites = mods
    symbols = atoms.get_chemical_symbols()
    prior = _atlas_prior(root, load_atlas, to_prior, set(symbols))
    sr = wyckoff_grammar(
        atoms.cell.array.tolist(), atoms.get_scaled_positions(), atoms.numbers
    )
    tokens = parse_sites(sr.sites)
    candidates = []
    for tok, members in zip(tokens, sr.members):
        fixed = {ligand_state(atoms, k) for k in members}
        if (
            tok.symbol not in TRANSITION_3D
            and len(fixed) == 1
            and None not in fixed
        ):
            candidates.append([(fixed.pop(), 1.0)])
            continue
        cand = [(q, p) for q, p in prior.get(tok.symbol, []) if p > 0]
        if tok.symbol in TRANSITION_3D:
            cand = [(q, p) for q, p in cand if q > 0]
        if not cand:
            return None
        candidates.append(cand)
    solutions = []
    for n, combo in enumerate(product(*candidates)):
        if n >= budget:
            break
        if sum(t.multiplicity * q for t, (q, _) in zip(tokens, combo)):
            continue
        solutions.append((sum(log(p) for _, p in combo), combo))
    if not solutions:
        return None
    # near-best solutions (prior within a factor 10) are ranked by how well
    # the Shannon radii of the 3d states reproduce the observed M-L distances
    best_lp = max(lp for lp, _ in solutions)
    near = [c for lp, c in solutions if lp >= best_lp - log(10.0)]
    if len(near) > 1:

        def mismatch(combo):
            err = 0.0
            for tok, members, (q, _) in zip(tokens, sr.members, combo):
                if tok.symbol in TRANSITION_3D:
                    e = _bond_length_error(atoms, members[0], q)
                    err += 1.0 if e is None else e
            return err

        near.sort(key=mismatch)
    best = near[0]
    return {k: q for members, (q, _) in zip(sr.members, best) for k in members}


def _ligand_shell(atoms, i):
    """Nearest-neighbour vectors of atom i and its dominant anion (or None)."""
    from ase.neighborlist import neighbor_list

    ii, jj, D = neighbor_list("ijD", atoms, 6.0)
    sel = ii == i
    if not sel.any():
        return np.zeros((0, 3)), None
    d = np.linalg.norm(D[sel], axis=1)
    keep = d <= 1.15 * d.min()
    vecs, nb = D[sel][keep], jj[sel][keep]
    symbols = atoms.get_chemical_symbols()
    anions = [symbols[j] for j in nb if symbols[j] in ANIONS]
    anion = max(set(anions), key=anions.count) if anions else None
    return vecs, anion


def _bond_length_error(atoms, i, q, spin=None):
    """|observed mean M-L distance - Shannon prediction| for atom i in state q."""
    mods = _atlas_physics()
    vecs, anion = _ligand_shell(atoms, i)
    if mods is None or anion is None or len(vecs) == 0:
        return None
    pred = mods.predicted_bond_length(
        atoms.get_chemical_symbols()[i],
        q,
        len(vecs),
        anion,
        ANIONS[anion],
        spin,
    )
    if pred is None:
        return None
    return abs(float(np.linalg.norm(vecs, axis=1).mean()) - pred)


def _atlas_physics():
    if _atlas_modules() is None:
        return None
    try:
        from oxstate_atlas import physics
    except ImportError:
        return None
    return physics


def site_details(atoms, i):
    """Class of atom i plus its local geometry and spin state.

    Returns a dict: element, env, ox, cn, geometry, spin ("HS", "LS" or
    None) and m_expected (unpaired electrons of that spin state, mu_B).
    The spin state is read off the bond length for octahedral d4-d7 ions
    in oxides and fluorides (Shannon high- vs low-spin radii). With softer
    anions (S, Se, Cl, ...) covalent shortening makes that test call known
    high-spin ions low spin (FeCl2, FePS3, MnS, RbMnCl3), so high spin is
    assumed there, as for tetrahedral and all other sites.
    """
    element, env, ox = site_class(atoms, i)
    info = {
        "element": element,
        "env": env,
        "ox": ox,
        "cn": None,
        "geometry": None,
        "spin": None,
        "m_expected": spin_only_moment(element, ox) if ox else None,
    }
    physics = _atlas_physics()
    vecs, anion = _ligand_shell(atoms, i)
    if physics is None or len(vecs) == 0:
        return info
    cn, geometry = physics.coordination_geometry(vecs)
    info.update(cn=cn, geometry=geometry)
    n = d_count(element, ox) if ox else None
    if n is None or anion is None:
        return info
    spin = "HS"
    if (
        4 <= n <= 7
        and geometry in ("octahedral", "square_planar")
        and anion in ("O", "F")
    ):
        mean_d = float(np.linalg.norm(vecs, axis=1).mean())
        spin = (
            physics.spin_from_bond_length(
                element, ox, cn, anion, ANIONS[anion], mean_d
            )
            or "HS"
        )
    opts = physics.spin_state_options(element, ox)
    info["spin"] = spin
    if opts is not None:
        info["m_expected"] = float(opts[0] if spin == "HS" else opts[1])
    return info


ANION_CLASS = {
    "O": "O",
    "F": "F",
    "Cl": "halide",
    "Br": "halide",
    "I": "halide",
    "S": "chalcogenide",
    "Se": "chalcogenide",
    "Te": "chalcogenide",
}


def site_anion_class(atoms, i):
    """Ligand class of atom i: "O", "F", "halide" (Cl, Br, I),
    "chalcogenide" (S, Se, Te), "metallic" (non-ionic environment) or
    "other" (ionic with another dominant anion, e.g. N)."""
    if site_environment(atoms, i) != "ionic":
        return "metallic"
    anion = _ligand_shell(atoms, i)[1]
    return ANION_CLASS.get(anion, "other")


def site_class(atoms, i):
    """(element, environment, oxidation or None) of atom i.

    Oxidation: charge balance with fixed valences when every other atom has
    one (reliable for oxides, halides, thiophosphates); otherwise the
    Wyckoff-resolved oxidation-state atlas (mixed valence, intermetallics),
    when the atlas package is available."""
    element = atoms.get_chemical_symbols()[i]
    env = site_environment(atoms, i)
    ox = formal_oxidation(atoms, element)
    if ox is None:
        states = atlas_oxidation(atoms)
        ox = states.get(i) if states else None
        if ox is not None and not 1 <= ox <= 5:
            ox = None
    return element, env, ox
