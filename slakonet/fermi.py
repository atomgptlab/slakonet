"""Code associated with electronic/finite temperature."""

from typing import Union, Tuple, Optional, Callable
from numbers import Real
from numpy import sqrt, pi
import torch
from torch import Tensor
from slakonet.basis import Basis
from slakonet.utils import float_like
from slakonet.utils import psort

_Scheme = Callable[[Tensor, Tensor, float_like], Tensor]


def _smearing_preprocessing(
    eigenvalues: Tensor, fermi_energy: Tensor, kT: float_like
) -> Tuple[Tensor, Tensor]:
    """Abstracts repetitive code from the smearing functions.

    Arguments:
        eigenvalues: Eigen-energies, i.e. orbital-energies.
        fermi_energy: The Fermi energy.
        kT: Electronic temperature.

    Returns:
        fermi_energy: Processed fermi energy tensor.
        kT: Processed kT tensor.

    """
    # These must be tensors for code to be batch agnostic & device safe
    assert isinstance(eigenvalues, Tensor), "eigenvalues must be a tensor"
    assert isinstance(fermi_energy, Tensor), "fermi_energy must be a tensor"

    # Shape fermi_energy so that there is one entry per row (repeat for kT).
    if fermi_energy.ndim == 1 and len(fermi_energy) != 1:
        # fermi_energy = fermi_energy.view(-1, 1)
        fermi_energy = fermi_energy.view(-1, *[1] * (eigenvalues.ndim - 1))

    # Ensure kT is a tensor & is shaped correctly if multiple values passed
    if not isinstance(kT, Tensor):
        kT = torch.tensor(
            kT, dtype=eigenvalues.dtype, device=eigenvalues.device
        )

    if kT.ndim >= 1 and len(kT) != 1:
        kT = kT.view(-1, *[1] * (eigenvalues.ndim - 1))

    # kT cannot be allowed to be true zero, otherwise nan's will occur.
    kT = torch.max(torch.tensor(torch.finfo(eigenvalues.dtype).tiny), kT)

    return fermi_energy, kT


def fermi_smearing(
    eigenvalues: Tensor, fermi_energy: Tensor, kT: float_like
) -> Tensor:
    r"""Fractional orbital occupancies due to Fermi-Dirac smearing.

    Using Fermi-Dirac smearing, orbital occupancies are calculated via:

    .. math::

        f_i = \frac{1}{1 + exp\left ( \frac{\epsilon_i - E_f}{kT}\right )}

    where ε, :math:`E_f` & :math:`kT` are the eigenvalues, fermi-energies and
    electronic temperatures respectively.

    Arguments:
        eigenvalues: Eigen-energies, i.e. orbital-energies.
        fermi_energy: The Fermi energy.
        kT: Electronic temperature.

    Returns:
        occupancies: Occupancies of the orbitals, or total electron count if
            total=True.

    Notes:
        ``eigenvalues`` may be a single value, an array of values or a tensor.
        If a tensor is passed then multiple ``fermi_energy`` and ``kT`` values
        may be passed if desired.

        If multiple systems are passed, smearing will be applied to all eigen
        values present, irrespective of whether they are real or fake (caused
        by packing).

    Warnings:
        Gradients resulting from this function can be ill defined, i.e. nan.
    """
    # Developers Notes: it might be worth trying to resolve the gradient
    # stability issue associated with this function.
    fermi_energy, kT = _smearing_preprocessing(eigenvalues, fermi_energy, kT)
    # Calculate and return the occupancies values via the Fermi-Dirac method
    # return 1.0 / (1.0 + torch.exp((eigenvalues - fermi_energy) / kT))
    vals = 1.0 / (1.0 + torch.exp((eigenvalues - fermi_energy) / kT))
    vals[eigenvalues.eq(0)] = 0.0
    return vals


def gaussian_smearing(
    eigenvalues: Tensor, fermi_energy: Tensor, kT: float_like
) -> Tensor:
    r"""Fractional orbital occupancies due to Gaussian smearing.

    Using Gaussian smearing, orbital occupancies are calculated via:

    .. math::

        f_i = frac{\textit{erfc}\left( \frac{\epsilon_i - E_f}{kT} \right)}{2}

    where ε, :math:`E_f` & :math:`kT` are the eigenvalues, fermi-energies and
    electronic temperatures respectively.

    Arguments:
        eigenvalues: Eigen-energies, i.e. orbital-energies.
        fermi_energy: The Fermi energy.
        kT: Electronic temperature.

    Returns:
        occupancies: Occupancies of the orbitals.

    Notes:
        ``eigenvalues`` may be a single value, an array of values or a tensor.
        If a tensor is passed then multiple ``fermi_energy`` and ``kT`` values
        may be passed if desired.

        If multiple systems are passed, smearing will be applied to all eigen
        values present, irrespective of whether they are real or fake (caused
        by packing).

    Warnings:
        Gradients will become unstable if a `kT` value of zero is used.

    """
    fermi_energy, kT = _smearing_preprocessing(eigenvalues, fermi_energy, kT)
    # Calculate and return the occupancies values via the Gaussian method
    return torch.erfc((eigenvalues - fermi_energy) / kT) / 2


def fermi_search(
    eigenvalues=[],
    n_electrons=None,
    k_weights=None,
    kT=0.01,
    tol=1e-6,
    max_iter=100,
):
    """
    Computes Fermi energy using Fermi-Dirac distribution, including k-point weights.
    Args:
        eigenvalues: Tensor [..., kpoints, orbitals]
        n_electrons: float
        k_weights: Tensor [..., kpoints]
    """
    # Get device from eigenvalues tensor
    device = eigenvalues.device

    # Ensure n_electrons is a tensor on the correct device
    if not isinstance(n_electrons, torch.Tensor):
        n_electrons = torch.tensor(
            n_electrons, device=device, dtype=eigenvalues.dtype
        )
    else:
        n_electrons = n_electrons.to(device)

    # Ensure k_weights is on the correct device
    if k_weights is not None:
        k_weights = k_weights.to(device)

    orig_shape = eigenvalues.shape[:-2]
    eig = eigenvalues.reshape(
        *orig_shape, -1, eigenvalues.shape[-1]
    )  # [..., kpoints, orbitals]
    with torch.enable_grad():
        mu = eig.mean(dim=(-1, -2), keepdim=True).clone().requires_grad_(True)

        for i in range(max_iter):
            occ = fermi_dirac(eig, mu, kT)  # [..., kpoints, orbitals]
            weighted_occ = occ * k_weights.unsqueeze(
                -1
            )  # [..., kpoints, orbitals]
            total_occ = 2.0 * weighted_occ.sum(
                dim=(-1, -2), keepdim=True
            )  # spin degeneracy

            loss = (total_occ - n_electrons) ** 2
            grad = torch.autograd.grad(loss.sum(), mu, create_graph=True)[0]

            if torch.max(torch.abs(grad)) < tol:
                break

            mu = (
                (mu - loss / (grad + 1e-12))
                .detach()
                .clone()
                .requires_grad_(True)
            )

    return mu.squeeze(-1)


def fermi_dirac(
    epsilon: torch.Tensor, ef: torch.Tensor, kT: float
) -> torch.Tensor:
    """
    Compute Fermi-Dirac occupation numbers.

    Args:
        epsilon (torch.Tensor): Energy levels [nkpts, nbands]
        ef (torch.Tensor): Fermi level (scalar or broadcastable)
        kT (float): Electronic temperature in Hartree

    Returns:
        torch.Tensor: Occupation numbers
    """
    return 1.0 / (torch.exp((epsilon - ef) / kT) + 1.0)
