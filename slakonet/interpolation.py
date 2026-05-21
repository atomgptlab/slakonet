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

# torch.set_default_dtype(torch.float32)


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
        # result = (
        #    torch.zeros(rr.shape)
        #    if self.yy.dim() == 1
        #    else torch.zeros(rr.shape[0], *self.yy.shape[1:])
        # )
        # Allocate result on same device & dtype as yy
        if self.yy.dim() == 1:
            result = torch.zeros(
                rr.shape,
                device=self._device,
                dtype=self.yy.dtype,
            )
        else:
            result = torch.zeros(
                rr.shape[0],
                *self.yy.shape[1:],
                device=self._device,
                dtype=self.yy.dtype,
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


class CubicSplineInterpU:
    """C2-continuous natural cubic spline on a uniform grid.

    Drop-in replacement for ``PolyInterpU`` with the same call signature.
    Eliminates the discontinuous polynomial-window shifts of ``PolyInterpU``
    that show up as zigzag noise (~0.1-1 eV) in geometry-scan total energies
    and equation-of-state curves.

    The interior region [xx[0], xx[-1]] is evaluated with a natural cubic
    spline. The tail region (xx[-1], xx[-1] + tail) decays smoothly to zero
    via :func:`poly5_zero`, matching :class:`PolyInterpU`'s convention so
    integrals tabulated to small SK-grid endings still go to zero smoothly.

    Coefficients are precomputed once at construction with
    ``scipy.interpolate.CubicSpline`` (no autograd through the table) and
    evaluated in pure torch so gradients flow through the query distances.

    Arguments:
        xx: Uniform grid points, 1D Tensor.
        yy: Tabulated values. Supports ``yy.dim()`` in {1, 2, 3, 4}; the
            leading axis must match ``xx``.
        tail: Distance over which to smooth values past the last grid point
            to zero (Bohr).
        delta_r: Step used to estimate first/second derivatives at the last
            grid point for the tail decay.

    Notes:
        - At grid points the values are identical to ``yy``.
        - Both first and second derivatives are continuous everywhere, which
          is what removes the EOS zigzag.
        - For SKF tables the natural BC (y'' = 0 at endpoints) is fine
          because the H/S integrals are tabulated well beyond their physical
          support; the endpoint values are already small.
    """

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"xx.shape={tuple(self.xx.shape)}, "
            f"yy.shape={tuple(self.yy.shape)}, "
            f"tail={self.tail}, "
            f"grid_step={self.grid_step.item():.3e}, "
            f"device={self._device})"
        )

    def __init__(
        self,
        xx: Tensor,
        yy: Tensor,
        tail: Real = 1.0,
        delta_r: Real = 1e-5,
        n_interp: int = 8,        # accepted for signature compat; unused
        n_interp_r: int = 4,      # accepted for signature compat; unused
    ):
        from scipy.interpolate import CubicSpline

        self.xx = xx
        self.yy = yy
        self.tail = tail
        self.delta_r = delta_r
        self.n_interp = n_interp
        self.n_interp_r = n_interp_r
        self.grid_step = xx[1] - xx[0]
        self._device = xx.device

        dxs = xx[1:] - xx[:-1]
        assert torch.allclose(
            dxs, torch.full_like(dxs, self.grid_step), atol=1e-6, rtol=1e-5
        ), "Grid points xx are not uniform"
        if len(xx) < 4:
            raise ValueError(
                f"Cubic spline needs >= 4 grid points, got {len(xx)}"
            )

        x_np = xx.detach().cpu().numpy().astype(np.float64)
        y_np = yy.detach().cpu().numpy().astype(np.float64)
        cs = CubicSpline(x_np, y_np, bc_type="natural", axis=0)
        # cs.c has shape (4, N-1, *yy.shape[1:]); index 0 is highest power.
        coef = torch.from_numpy(np.ascontiguousarray(cs.c))
        coef = coef.to(device=self._device, dtype=yy.dtype)
        self._coef_a = coef[0]   # (rr-x_i)^3
        self._coef_b = coef[1]   # (rr-x_i)^2
        self._coef_c = coef[2]   # (rr-x_i)^1
        self._coef_d = coef[3]   # constant

    def __call__(self, rr: Tensor) -> Tensor:
        device = self._device
        dtype = self.yy.dtype
        xx = self.xx
        n_grid = xx.shape[0]
        x_max = xx[-1]
        r_max = x_max + self.tail

        out_shape = (rr.shape[0],) + tuple(self.yy.shape[1:])
        result = torch.zeros(out_shape, device=device, dtype=dtype)

        # Region 1: rr <= x_max  -> cubic spline
        mask_in = rr <= x_max
        if mask_in.any():
            r_in = rr[mask_in]
            idx = torch.bucketize(r_in.detach(), xx) - 1
            idx = torch.clamp(idx, 0, n_grid - 2)
            x_lo = xx[idx]
            dx = (r_in - x_lo).to(dtype)

            a = self._coef_a[idx]
            b = self._coef_b[idx]
            c = self._coef_c[idx]
            d = self._coef_d[idx]

            # broadcast dx against trailing dims of a/b/c/d
            while dx.dim() < a.dim():
                dx = dx.unsqueeze(-1)

            result[mask_in] = d + dx * (c + dx * (b + dx * a))

        # Region 2: x_max < rr < r_max  -> 5th-order smooth-to-zero tail
        mask_tail = (rr > x_max) & (rr < r_max)
        if mask_tail.any():
            # values & derivatives at x_max, taken from the spline itself
            # to guarantee continuity at the seam
            d_lo = xx[-1] - xx[-2]
            a_end = self._coef_a[-1]
            b_end = self._coef_b[-1]
            c_end = self._coef_c[-1]
            d_end = self._coef_d[-1]
            y_end = d_end + d_lo * (c_end + d_lo * (b_end + d_lo * a_end))
            yp_end = c_end + d_lo * (2 * b_end + d_lo * 3 * a_end)
            ypp_end = 2 * b_end + 6 * a_end * d_lo

            r_t = rr[mask_tail]
            dr = (r_t - x_max).to(dtype)
            # broadcast dr to trailing dims
            while dr.dim() < y_end.dim() + 1:
                dr = dr.unsqueeze(-1)
            # poly5_zero expects scalar dx (= -tail) as anchor distance
            tail_val = poly5_zero(
                y_end.unsqueeze(0).expand(r_t.shape[0], *y_end.shape),
                yp_end.unsqueeze(0).expand(r_t.shape[0], *yp_end.shape),
                ypp_end.unsqueeze(0).expand(r_t.shape[0], *ypp_end.shape),
                dr,
                torch.tensor(-float(self.tail), dtype=dtype, device=device),
            )
            result[mask_tail] = tail_val

        # Region 3: rr >= r_max -> already zero
        return result


def get_default_interpolator():
    """Return the default SK interpolator class.

    Set ``SLAKONET_INTERPOLATOR=poly`` to keep the legacy
    :class:`PolyInterpU`. Default is the C2-continuous
    :class:`CubicSplineInterpU`, which removes the EOS/strain zigzag caused
    by polynomial-window shifts at SKF grid boundaries.
    """
    import os

    name = os.environ.get("SLAKONET_INTERPOLATOR", "spline").lower()
    if name in ("poly", "polyinterpu"):
        return PolyInterpU
    return CubicSplineInterpU


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
    target_dtype = rr.dtype
    device = xp.device
    rr = rr.to(device)
    yp = yp.to(device)
    xp = xp.to(dtype=target_dtype)
    yp = yp.to(dtype=target_dtype)
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


def poly_interp_2d(xp: Tensor, yp: Tensor, rr: Tensor) -> Tensor:
    """Interpolate from DFTB+ (lib_math) with uniform grid.

    Arguments:
        xp: 2D tensor, 1st dimension if batch size, 2nd is grid points.
        yp: 2D tensor of integrals.
        rr: interpolation points.
    """
    nn0, nn1 = xp.shape[0], xp.shape[1]
    index_nn0 = torch.arange(nn0)
    icl = torch.zeros(nn0).long()
    cc, dd = yp.clone(), yp.clone()
    dxp = abs(rr - xp[index_nn0, icl])

    # find the most close point to rr (single atom pair or multi pairs)
    _mask, ii = torch.zeros(len(rr)) == 0, 0
    dxNew = abs(rr - xp[index_nn0, 0])
    while (dxNew < dxp).any():
        ii += 1
        assert ii < nn1 - 1  # index ii range from 0 to nn1 - 1
        _mask = dxNew < dxp
        icl[_mask] = ii
        dxp[_mask] = abs(rr - xp[index_nn0, ii])[_mask]

    yy = yp[index_nn0, icl]

    for mm in range(nn1 - 1):
        for ii in range(nn1 - mm - 1):
            rtmp0 = xp[index_nn0, ii] - xp[index_nn0, ii + mm + 1]

            # use transpose to realize div: (N, M, K) / (N)
            rtmp1 = (
                (cc[index_nn0, ii + 1] - dd[index_nn0, ii]).transpose(0, -1)
                / rtmp0
            ).transpose(0, -1)
            cc[index_nn0, ii] = (
                (xp[index_nn0, ii] - rr) * rtmp1.transpose(0, -1)
            ).transpose(0, -1)
            dd[index_nn0, ii] = (
                (xp[index_nn0, ii + mm + 1] - rr) * rtmp1.transpose(0, -1)
            ).transpose(0, -1)
        if (2 * icl < nn1 - mm - 1).any():
            _mask = 2 * icl < nn1 - mm - 1
            yy[_mask] = (yy + cc[index_nn0, icl])[_mask]
        else:
            _mask = 2 * icl >= nn1 - mm - 1
            yy[_mask] = (yy + dd[index_nn0, icl - 1])[_mask]
            icl[_mask] = icl[_mask] - 1
    return yy
