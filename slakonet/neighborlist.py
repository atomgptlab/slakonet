"""Torch-native periodic neighbor list with differentiable edges.

Vendored from ALIGNN's ``alignn.torch_graph_builder`` (the pure-torch,
DGL-free graph builder) so that SlaKoNet's sparse Slater-Koster path
has no hard dependency on the ALIGNN package. Only the neighbor-list
primitives are copied; the upstream module also builds line graphs.

Upstream: https://github.com/atomgptlab/alignn
  -> alignn/torch_graph_builder.py  (functions torch_neighbor_list,
     _torch_periodic_shifts, _topk_per_source)

The edges are differentiable functions of atomic positions and the
lattice -- ``r = pos[dst] - pos[src] + shift @ lattice`` -- so autograd
flows back to both (forces via -dE/dx, stress via dE/dL). An optional
matscipy fast path is used for topology when available; the default
pure-torch path is memory-chunked over source atoms.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


def _torch_periodic_shifts(
    lattice: torch.Tensor, cutoff: float
) -> torch.Tensor:
    """Integer shift vectors (K, 3) covering a ``cutoff`` sphere."""
    with torch.no_grad():
        recip = 2 * math.pi * torch.linalg.inv(lattice).T
        recip_len = torch.linalg.norm(recip, dim=1)
        n_max = torch.ceil(
            cutoff * recip_len / (2 * math.pi)
        ).to(torch.long)
    ranges = [
        torch.arange(
            -int(n.item()), int(n.item()) + 1, device=lattice.device
        )
        for n in n_max
    ]
    return torch.cartesian_prod(*ranges).to(lattice.dtype)


def _topk_per_source(
    src: torch.Tensor,
    keys: torch.Tensor,
    max_neighbors: int,
    num_nodes: int,
) -> torch.Tensor:
    """Return indices keeping the K smallest ``keys`` per src node."""
    device = src.device
    order_key = torch.argsort(keys)
    src_k = src[order_key]
    order_src = torch.argsort(src_k, stable=True)
    perm = order_key[order_src]
    src_sorted = src[perm]
    E = perm.numel()
    starts = torch.searchsorted(
        src_sorted, torch.arange(num_nodes, device=device)
    )
    within = torch.arange(E, device=device) - starts[src_sorted]
    return perm[within < max_neighbors]


def torch_neighbor_list(
    positions: torch.Tensor,
    lattice: torch.Tensor,
    cutoff: float,
    max_neighbors: Optional[int] = None,
    atoms=None,
    use_matscipy_topology: bool = False,
    self_tol: float = 1e-8,
    chunk_size: int = 512,
):
    """Torch-native periodic neighbor list with differentiable edges.

    Memory-chunked over source atoms: peak memory is O(K * chunk * N)
    instead of O(K * N^2), which also sidesteps torch's INT_MAX limit
    on torch.where for very large boolean tensors.

    Returns:
        (src, dst, shift, r) -- source/destination atom indices, the
        integer cell-shift per edge, and the differentiable
        displacement vector ``r = pos[dst]-pos[src]+shift@lattice``.
    """
    dtype = positions.dtype
    device = positions.device
    num_nodes = int(positions.shape[0])

    used_matscipy = False
    if use_matscipy_topology and atoms is not None:
        try:
            from matscipy.neighbours import neighbour_list as _mnl

            i_np, j_np, S_np = _mnl(
                "ijS", atoms.ase_converter(), float(cutoff)
            )
            src = torch.from_numpy(np.ascontiguousarray(i_np)).to(
                device=device, dtype=torch.long
            )
            dst = torch.from_numpy(np.ascontiguousarray(j_np)).to(
                device=device, dtype=torch.long
            )
            shift = torch.from_numpy(np.ascontiguousarray(S_np)).to(
                device=device, dtype=dtype
            )
            used_matscipy = True
        except ImportError:
            pass

    if not used_matscipy:
        shifts = _torch_periodic_shifts(lattice, cutoff)  # (K, 3)
        with torch.no_grad():
            offs = shifts @ lattice  # (K, 3) cartesian
            c2 = float(cutoff) * float(cutoff)

            # Dynamically shrink chunk for very large systems so that
            # (K * chunk * N) bool tensor stays well under INT_MAX.
            K = int(shifts.shape[0])
            max_elems = 2**30  # ~1.07e9, safe
            max_chunk_by_int = max(
                1, max_elems // max(K * num_nodes, 1)
            )
            eff_chunk = max(1, min(chunk_size, max_chunk_by_int))

            src_chunks, dst_chunks, shift_chunks = [], [], []
            for i0 in range(0, num_nodes, eff_chunk):
                i1 = min(i0 + eff_chunk, num_nodes)
                # (K, chunk, N, 3)
                rvec = (
                    positions[None, None, :, :]
                    + offs[:, None, None, :]
                    - positions[None, i0:i1, None, :]
                )
                dist2 = rvec.pow(2).sum(-1)  # (K, chunk, N)
                mask = (dist2 <= c2) & (dist2 > self_tol)
                del rvec, dist2
                k_idx, i_local, j_idx = torch.where(mask)
                del mask
                src_chunks.append((i_local + i0).to(torch.long))
                dst_chunks.append(j_idx.to(torch.long))
                shift_chunks.append(shifts[k_idx])
                del k_idx, i_local, j_idx

            src = (
                torch.cat(src_chunks)
                if src_chunks
                else torch.empty(0, dtype=torch.long, device=device)
            )
            dst = (
                torch.cat(dst_chunks)
                if dst_chunks
                else torch.empty(0, dtype=torch.long, device=device)
            )
            shift = (
                torch.cat(shift_chunks)
                if shift_chunks
                else torch.empty((0, 3), dtype=dtype, device=device)
            )

    # Differentiable displacement vectors -- the autograd bridge.
    r = positions[dst] - positions[src] + shift @ lattice

    if (
        max_neighbors is not None
        and max_neighbors > 0
        and src.numel() > 0
    ):
        with torch.no_grad():
            dist = r.norm(dim=1)
        keep = _topk_per_source(
            src, dist, int(max_neighbors), num_nodes
        )
        src, dst, shift, r = (
            src[keep], dst[keep], shift[keep], r[keep],
        )

    return src, dst, shift, r
