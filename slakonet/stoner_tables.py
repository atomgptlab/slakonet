"""Stoner parameters I_d fitted to experimental exchange constants.

Provenance
----------
Fitted on 2026-09-30 with ``examples/fit_stoner_table.py`` (grouping
``element_ox``) to 124 experimental couplings |J S^2| >= 1 meV of 61
compounds of ExpMagDB (INS spin-wave fits and MAGNDATA structures), each
computed with slakonet_v1a + TB2J (all bands kept) in its experimental
collinear order, without SCC. Site classes come from
``slakonet.magnetic_sites`` (charge balance, else the Wyckoff-resolved
oxidation-state atlas). Leave-one-compound-out cross-validation:

    ===========================  ==========  ================
    statistic                    atomic I    this table (LOO)
    ===========================  ==========  ================
    no physical solution         66 / 124    27 / 124
    correct sign                 45 / 124    83 / 124
    within a factor of 2         20 / 124    28 / 124
    median |J_calc / J_ref|      0.00        0.27
    mean absolute error (meV)    11.7        10.5
    ===========================  ==========  ================

The model (rigid Stoner exchange on non-SCC tight binding) limits the
accuracy: even the best I per compound puts only ~40 % of the couplings
within a factor of 2. Treat J from this table as semi-quantitative.
("Fe", "metallic") came out at the upper edge of the fitted range
(0.21 Ha) and gave unphysical bcc Fe (m = 4.2 mu_B), so it was removed.
Metallic sites without a (element, "metallic") entry use the atomic value:
the element-level entries are dominated by ionic compounds.

Keys go from specific to general: (element, environment, oxidation),
(element, environment), (element). Only groups with >= 3 compounds are
kept; Co and V (2 and 1 compounds) fall back to the atomic values of
``slakonet.magnetism.DEFAULT_STONER_I``. Values in Hartree, d shell.
"""

from __future__ import annotations

FITTED_STONER_I_D = {
    ("Cr",): 0.0447,
    ("Cr", "ionic"): 0.0527,
    ("Cr", "ionic", 3): 0.0452,
    ("Mn",): 0.0652,
    ("Mn", "ionic"): 0.0652,
    ("Mn", "ionic", 2): 0.0826,
    ("Mn", "ionic", 3): 0.0230,
    ("Mn", "metallic"): 0.0908,
    ("Mn", "metallic", 2): 0.0908,
    ("Mn", "metallic", 4): 0.0652,
    ("Fe",): 0.1581,
    ("Fe", "ionic"): 0.1581,
    ("Fe", "ionic", 2): 0.1581,
    ("Fe", "ionic", 3): 0.0708,
    ("Ni",): 0.1294,
    ("Ni", "ionic"): 0.1294,
    ("Ni", "ionic", 2): 0.1294,
    ("Cu",): 0.0644,
    ("Cu", "ionic"): 0.0644,
    ("Cu", "ionic", 2): 0.0644,
}


def fitted_site_stoner_I(atoms, indices=None):
    """{atom index: I_d (Ha)} from the fitted table for the 3d atoms that it
    covers (``indices`` restricts the atoms considered). Atoms of elements
    outside the table are left out, so the caller falls back to the atomic
    values."""
    from slakonet.magnetic_sites import site_class

    symbols = atoms.get_chemical_symbols()
    covered = {k[0] for k in FITTED_STONER_I_D}
    out = {}
    for i in range(len(atoms)) if indices is None else indices:
        if symbols[i] not in covered:
            continue
        element, env, ox = site_class(atoms, i)
        keys = [(element, env, ox), (element, env)]
        if env == "ionic":  # element-level values come from ionic compounds
            keys.append((element,))
        for key in keys:
            if key in FITTED_STONER_I_D:
                out[i] = FITTED_STONER_I_D[key]
                break
    return out
