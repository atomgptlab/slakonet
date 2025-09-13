#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 17:32:49 2021

@author: gz_fan
"""
from typing import List
from numbers import Real
import bisect
import torch
import numpy as np
from slakonet.utils import pack

Tensor = torch.Tensor


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class PolyInterpU:
    """Polynomial interpolation method with uniform grid points.

    The boundary condition will use `poly_to_zero` function, which make the
    polynomial values smoothly converge to zero at the boundary.

    Arguments:
        xx: Grid points for interpolation, 1D Tensor.
        yy: Values to be interpolated at each grid point.
        tail: Distance to smooth the tail.
        delta_r: Delta distance for 1st, 2nd derivative.
        n_interp: Number of total interpolation grid points.
        n_interp_r: Number of right side interpolation grid points.

    Attributes:
        xx: Grid points for interpolation, 1D Tensor.
        yy: Values to be interpolated at each grid point.
        delta_r: Delta distance for 1st, 2nd derivative.
        tail: Distance to smooth the tail.
        n_interp: Number of total interpolation grid points.
        n_interp_r: Number of right side interpolation grid points.
        grid_step: Distance between each gird points.

    Notes:
        The `PolyInterpU` class, which is taken from the DFTB+, assumes a
        uniform grid. Here, the yy and xx arguments are the values to be
        interpolated and their associated grid points respectively. The tail
        end of the spline is smoothed to zero, meaning that extrapolated
        points will rapidly, but smoothly, decay to zero.
    """

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"xx.shape={tuple(self.xx.shape)}, "
            f"yy.shape={tuple(self.yy.shape)}, "
            f"tail={self.tail}, "
            f"delta_r={self.delta_r}, "
            f"n_interp={self.n_interp}, "
            f"n_interp_r={self.n_interp_r}, "
            f"grid_step={self.grid_step.item():.3e}, "
            f"device={self._device})"
        )

    def __init__(
        self,
        xx: Tensor,
        yy: Tensor,
        tail: Real = 1.0,
        delta_r: Real = 1e-5,
        n_interp: int = 8,
        # n_interp: int = 12,
        n_interp_r: int = 4,
    ):
        self.xx = xx
        self.yy = yy
        self.delta_r = delta_r

        self.tail = tail
        self.n_interp = n_interp
        self.n_interp_r = n_interp_r
        self.grid_step = xx[1] - xx[0]

        # Device type of the tensor in this class
        self._device = xx.device

        # Check xx is uniform & that len(xx) > n_interp
        dxs = xx[1:] - xx[:-1]
        # print("dxs",dxs,dxs.shape)
        # print("self.grid_step",self.grid_step)
        # print("torch.full_like(dxs, self.grid_step)",torch.full_like(dxs, self.grid_step),torch.full_like(dxs, self.grid_step).shape)
        # check_1 = torch.allclose(dxs, torch.full_like(dxs, self.grid_step))
        check_1 = torch.allclose(
            dxs, torch.full_like(dxs, self.grid_step), atol=1e-6, rtol=1e-5
        )
        assert check_1, "Grid points xx are not uniform"
        if len(xx) < n_interp:
            raise ValueError(
                f"`n_interp` ({n_interp}) exceeds the number of"
                f"data points `xx` ({len(xx)})."
            )

    def __call__(self, rr: Tensor) -> Tensor:
        """Get interpolation according to given rr.
        Arguments:
            rr: interpolation points for single and batch.
        Returns:
            result: Interpolation values with given rr.
        """
        n_grid_point = len(self.xx)  # -> number of grid points
        r_max = (n_grid_point - 1) * self.grid_step + self.tail
        ind = torch.floor(rr / self.grid_step).long().to(self._device)
        # result = torch.zeros(*rr.shape, self.yy.shape[-1], device=self._device)
        result = (
            torch.zeros(rr.shape)
            if self.yy.dim() == 1
            else torch.zeros(rr.shape[0], *self.yy.shape[1:])
        )

        # => polynomial fit
        if (ind <= n_grid_point).any():

            _mask = ind <= n_grid_point

            # get the index of rr in grid points
            ind_last = (ind[_mask] + self.n_interp_r + 1).long()
            ind_last[ind_last > n_grid_point] = n_grid_point
            ind_last[ind_last < self.n_interp + 1] = self.n_interp + 1

            # gather xx and yy for both single and batch
            xa = (
                ind_last.unsqueeze(1)
                - self.n_interp
                + torch.arange(self.n_interp, device=self._device)
            ) * self.grid_step

            if self.yy.dim() <= 2:  # -> all rr shares the same integral (yy)
                yb = torch.stack(
                    [
                        self.yy[ii - self.n_interp - 1 : ii - 1]
                        for ii in ind_last
                    ]
                ).to(self._device)
                # ind = torch.arange(self.n_interp).repeat(len(ind_last)) + \
                #     ind_last.repeat_interleave(self.n_interp)
                # yb = self.yy[ind].reshape(len(ind_last), self.n_interp, -1)
            elif self.yy.dim() == 3:
                assert self.yy.shape[1] == rr.shape[0], (
                    "each distance "
                    + "corresponding to different integrals, the size should"
                    + f" be same, but get {self.yy.shape[1]}, {rr.shape[0]}"
                )
                yb = torch.stack(
                    [
                        self.yy[il - self.n_interp - 1 : il - 1, ii]
                        for ii, il in enumerate(ind_last)
                    ]
                ).to(self._device)
            elif self.yy.dim() == 4:
                yb = torch.stack(
                    [
                        self.yy[il - self.n_interp - 1 : il - 1]
                        for il in ind_last
                    ]
                ).to(self._device)
            result[_mask] = poly_interp(xa, yb, rr[_mask])

        # Beyond the grid => extrapolation with polynomial of 5th order
        max_ind = n_grid_point - 1 + int(self.tail / self.grid_step)
        is_tail = ind.masked_fill(
            ind.ge(n_grid_point) * ind.le(max_ind), -1
        ).eq(-1)
        if is_tail.any():
            # dr = rr[is_tail] - r_max
            # ilast = n_grid_point

            # # get grid points and grid point values
            # xa = (ilast - self.n_interp + torch.arange(
            #     self.n_interp, device=self._device)) * self.grid_step
            # yb = self.yy[ilast - self.n_interp - 1: ilast - 1]
            # xa = xa.repeat(dr.shape[0]).reshape(dr.shape[0], -1)
            # yb = yb.unsqueeze(0).repeat_interleave(dr.shape[0], dim=0)

            # # get derivative
            # y0 = poly_interp(xa, yb, xa[:, self.n_interp - 1] - self.delta_r)
            # y2 = poly_interp(xa, yb, xa[:, self.n_interp - 1] + self.delta_r)
            # y1 = self.yy[ilast - 2]
            # y1p = (y2 - y0) / (2.0 * self.delta_r)
            # y1pp = (y2 + y0 - 2.0 * y1) / (self.delta_r * self.delta_r)

            # # result[is_tail] = poly_to_zero2(
            # #     dr, -1.0 * self.tail, -1.0 / self.tail, y1, y1p, y1pp)
            # print('result', result.shape, 'result[is_tail]', result[is_tail].shape,
            #       poly5_zero(y1, y1p, y1pp, dr, -1.0 * self.tail).shape)
            # result[is_tail] = poly5_zero(y1, y1p, y1pp, dr, -1.0 * self.tail)

            dr = rr[is_tail] - r_max

            # For input integrals, it will be 2D, such as (nsize) * (pp0, pp1),
            # initial dr is 1D and will result in errors
            dr = dr.repeat(self.yy.shape[1], 1).T if self.yy.dim() == 2 else dr
            ilast = n_grid_point

            # get grid points and grid point values
            xa = (
                ilast - self.n_interp + torch.arange(self.n_interp)
            ) * self.grid_step
            yb = self.yy[ilast - self.n_interp - 1 : ilast - 1]
            xa = xa.repeat(dr.shape[0]).reshape(dr.shape[0], -1)
            yb = yb.unsqueeze(0).repeat_interleave(dr.shape[0], dim=0)

            # get derivative
            y0 = poly_interp_2d(
                xa, yb, xa[:, self.n_interp - 1] - self.delta_r
            )
            y2 = poly_interp_2d(
                xa, yb, xa[:, self.n_interp - 1] + self.delta_r
            )
            y1 = self.yy[ilast - 2]
            y1p = (y2 - y0) / (2.0 * self.delta_r)
            y1pp = (y2 + y0 - 2.0 * y1) / (self.delta_r * self.delta_r)

            if y1pp.dim() == 3:  # -> compression radii, not good
                dr = dr.repeat(y1pp.shape[1], y1pp.shape[2], 1).transpose(
                    -1, 0
                )
            elif y1pp.dim() == 4:  # -> compression radii, not good
                dr = dr.repeat(y1pp.shape[1], y1pp.shape[2], 1, 1).permute(
                    -1, 0, 1, 2
                )

            result[is_tail] = poly5_zero(y1, y1p, y1pp, dr, -1.0 * self.tail)

        return result


def vcr_poly_to_zero(
    xx: Tensor, yy: Tensor, n_grid: Tensor, ninterp=8, delta_r=1e-5, tail=1.0
):
    """Smooth the tail with input xx and yy with various compression radii.

    Arguments:
        xx:
        yy:

    """
    assert xx.dim() == 2
    assert yy.dim() == 3
    assert n_grid.shape[0] == xx.shape[0]
    ny1, ny2, ny3 = yy.shape

    _yy = yy.permute(0, -1, -2)  # -> permute distance dim from last to second
    ilast = n_grid.clone()
    incr = xx[0, 1] - xx[0, 0]

    # get grid points and grid point values
    xa = (ilast.unsqueeze(1) - ninterp + torch.arange(ninterp)) * incr
    yb = torch.stack(
        [_yy[ii, il - ninterp - 1 : il - 1] for ii, il in enumerate(ilast)]
    )

    # return smooth grid points in the tail for each SKF
    dr = -torch.linspace(4, 0, 5) * incr

    # get derivative
    y0 = poly_interp(xa, yb, xa[:, ninterp - 1] - delta_r)
    y2 = poly_interp(xa, yb, xa[:, ninterp - 1] + delta_r)
    y1 = _yy[torch.arange(_yy.shape[0]), ilast - 2]
    y1p = (y2 - y0) / (2.0 * delta_r)
    y1pp = (y2 + y0 - 2.0 * y1) / (delta_r * delta_r)
    integral_tail = poly_to_zero(
        dr,
        -1.0 * tail,
        -1.0 / tail,
        y1.unsqueeze(-1),
        y1p.unsqueeze(-1),
        y1pp.unsqueeze(-1),
    ).permute(0, 2, 1)

    if _yy.shape[1] == max(n_grid):
        ynew = torch.zeros(ny1, ny3 + integral_tail.shape[1], ny2)
        ynew[:ny1, :ny3, :ny2] = _yy[:ny1, :ny3, :ny2]
    else:
        ynew = _yy.clone()

    for ii, iy in enumerate(_yy):  # -> add tail
        ynew[ii, n_grid[ii] : n_grid[ii] + 5] = integral_tail[ii]

    return ynew.permute(0, -1, -2)


def poly5_zero(
    y0: Tensor, y0p: Tensor, y0pp: Tensor, xx: Tensor, dx: Tensor
) -> Tensor:
    """Get integrals if beyond the grid range with 5th polynomial."""
    dx1 = y0p * dx
    dx2 = y0pp * dx * dx
    dd = 10.0 * y0 - 4.0 * dx1 + 0.5 * dx2
    ee = -15.0 * y0 + 7.0 * dx1 - 1.0 * dx2
    ff = 6.0 * y0 - 3.0 * dx1 + 0.5 * dx2
    xr = xx / dx
    yy = ((ff * xr + ee) * xr + dd) * xr * xr * xr
    return yy


def poly_interp(xp: Tensor, yp: Tensor, rr: Tensor) -> Tensor:
    """Interpolation with given uniform grid points.
    Arguments:
        xp: The grid points, 2D Tensor, first dimension is for different
            system and second is for the corresponding grids in each system.
        yp: The values at the gird points.
        rr: Points to be interpolated.
    Returns:
        yy: Interpolation values corresponding to input rr.
    Notes:
        The function `poly_interp` is designed for both single and multi
        systems interpolation. Therefore xp will be 2D Tensor.
    """
    assert xp.dim() == 2, "xp is not 2D Tensor"
    device = xp.device
    nn0, nn1 = xp.shape[0], xp.shape[1]
    index_nn0 = torch.arange(nn0, device=device)
    icl = torch.zeros(nn0, device=device).long()
    cc, dd = yp.clone(), yp.clone()
    dxp = abs(rr - xp[index_nn0, icl])

    # find the most close point to rr (single atom pair or multi pairs)
    _mask, ii = torch.zeros(len(rr), device=device) == 0.0, 0.0
    _dx_new = abs(rr - xp[index_nn0, 0])
    while (_dx_new < dxp).any():
        ii += 1
        assert ii < nn1 - 1, "index ii range from 0 to %s" % nn1 - 1
        _mask = _dx_new < dxp
        icl[_mask] = ii
        dxp[_mask] = abs(rr - xp[index_nn0, ii])[_mask]

    yy = yp[index_nn0, icl]

    for mm in range(nn1 - 1):
        for ii in range(nn1 - mm - 1):
            r_tmp0 = xp[index_nn0, ii] - xp[index_nn0, ii + mm + 1]

            # use transpose to realize div: (N, M, K) / (N)
            r_tmp1 = (
                (cc[index_nn0, ii + 1] - dd[index_nn0, ii]).transpose(0, -1)
                / r_tmp0
            ).transpose(0, -1)
            cc[index_nn0, ii] = (
                (xp[index_nn0, ii] - rr) * r_tmp1.transpose(0, -1)
            ).transpose(0, -1)
            dd[index_nn0, ii] = (
                (xp[index_nn0, ii + mm + 1] - rr) * r_tmp1.transpose(0, -1)
            ).transpose(0, -1)
        if (2 * icl < nn1 - mm - 1).any():
            _mask = 2 * icl < nn1 - mm - 1
            yy[_mask] = (yy + cc[index_nn0, icl])[_mask]
        else:
            _mask = 2 * icl >= nn1 - mm - 1
            yy[_mask] = (yy + dd[index_nn0, icl - 1])[_mask]
            icl[_mask] = icl[_mask] - 1

    return yy


def poly_to_zero2(
    xx: Tensor,
    dx: Tensor,
    inv_dist: Tensor,
    y0: Tensor,
    y0p: Tensor,
    y0pp: Tensor,
) -> Tensor:
    """Get interpolation if beyond the grid range with 5th order polynomial.
    Arguments:
        y0: Values to be interpolated at each grid point.
        y0p: First derivative of y0.
        y0pp: Second derivative of y0.
        xx: Grid points.
        dx: The grid point range for y0 and its derivative.
    Returns:
        yy: The interpolation values with given xx points in the tail.
    Notes:
        The function `poly_to_zero` realize the interpolation of the points
        beyond the range of grid points, which make the polynomial values
        smoothly converge to zero at the boundary. The variable dx determines
        the point to be zero. This code is consistent with the function
        `poly5ToZero` in DFTB+.
    """
    dx1 = y0p * dx
    dx2 = y0pp * dx * dx
    dd = 10.0 * y0 - 4.0 * dx1 + 0.5 * dx2
    ee = -15.0 * y0 + 7.0 * dx1 - 1.0 * dx2
    ff = 6.0 * y0 - 3.0 * dx1 + 0.5 * dx2
    xr = xx * inv_dist
    yy = ((ff * xr + ee) * xr + dd) * xr * xr * xr

    return yy
