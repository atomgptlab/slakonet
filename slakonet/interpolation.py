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


class MultiVarInterp:
    """Multivariate interpolation."""

    def __init__(self, x: List[Tensor], y: Tensor, **kwargs):
        # test x is ascending
        # test different idm is the same
        self.x, self.y = self._check(x, y, **kwargs)

    def __call__(self, x_new, distances):
        """"""
        x_new = x_new.unsqueeze(1) if x_new.dim() == 1 else x_new
        assert x_new.shape[1] + 1 == len(self.x), "input dimension error"

        _y = self.y

        for ii, ix in enumerate(x_new.T):
            _y = self._1d_linear(ix, self.x[ii], _y, ii == 0)

        return self._1d_linear(distances, self.x[-1], _y, False)

    def _1d_linear(self, points, grid, values, is_1st_dim):
        ind_lef = torch.searchsorted(grid, points)
        mask_max = ind_lef.ge(len(grid) - 1)
        ind_rig = ind_lef + 1

        # to make sure index donot exceed shape of the shape x
        ind_lef[mask_max] = len(grid) - 1
        ind_rig[mask_max] = len(grid) - 1

        # get the nearest grid point
        x_ind_lef = grid[ind_lef]
        x_ind_rig = grid[ind_rig]
        grid_len = x_ind_rig - x_ind_lef

        if is_1st_dim:
            y_lef, y_rig = values[ind_lef], values[ind_rig]
        else:
            mask0 = torch.arange(len(points))
            y_lef, y_rig = values[mask0, ind_lef], values[mask0, ind_rig]

        ratio_lef = (points - x_ind_lef) / grid_len
        ratio_rig = (x_ind_rig - points) / grid_len

        return (y_lef.T * ratio_lef + y_rig.T * ratio_rig).T

    def _check(self, x, y, **kwargs):
        assert isinstance(x, list), "input x should be list"
        for ix in x:
            assert isinstance(
                ix, Tensor
            ), f"grid point should be torch.Tensor, but get {type(x)}"
        assert isinstance(
            y, Tensor
        ), f"values y should be torch.Tensor, but get {type(y)}"

        self._n_dim_values = y.dim()
        self._n_dim_inter = len(x)

        return x, y


class BicubInterp:
    """Vectorized bicubic interpolation method designed for molecule.

    The bicubic interpolation is designed to interpolate the integrals of
    whole molecule. The xmesh, ymesh are the grid points and they are the same,
    therefore only xmesh is needed here.
    The zmesh is a 4D or 5D Tensor. The 1st, 2nd dimensions are corresponding
    to the pairwise atoms in molecule. The 3rd and 4th are corresponding to
    the xmesh and ymesh. The 5th dimension is optional. For bicubic
    interpolation of single integral such as ss0 orbital, it is 4D Tensor.
    For bicubic interpolation of all the orbital integrals, zmesh is 5D Tensor.

    Arguments:
        xmesh: 1D Tensor.
        zmesh: 2D or 3D Tensor, 2D is for single integral with vrious
            compression radii, 3D is for multi integrals.

    References:
        .. [wiki] https://en.wikipedia.org/wiki/Bicubic_interpolation
    """

    def __init__(self, xmesh: Tensor, zmesh: Tensor, hs_grid=None):
        """Get interpolation with two variables."""
        assert (
            zmesh.shape[0] == zmesh.shape[1]
        ), f"1D and 2D shape of zmesh are not same, get {zmesh.shape[:2]}"
        if zmesh.dim() < 2 or zmesh.dim() > 4:
            raise ValueError(f"zmesh should be 2, 3, or 4D, get {zmesh.dim()}")
        elif zmesh.dim() == 2:
            assert hs_grid is None, "Can not interpolate 2D tensor for hs_grid"
            zmesh = zmesh.unsqueeze(0)  # -> single to batch
        elif zmesh.dim() == 3:
            if hs_grid is not None:
                zmesh = zmesh.unsqueeze(-1)
        elif zmesh.dim() == 4:
            zmesh = zmesh.permute(-2, 0, 1, -1)

        self.xmesh = xmesh
        self.zmesh = zmesh
        self.hs_grid = hs_grid

    def __call__(self, xnew: Tensor, distances=None):
        """Calculate bicubic interpolation.

        Arguments:
            xnew: The points to be interpolated for the first dimension and
                second dimension.
        """
        self.xi = xnew if xnew.dim() == 2 else xnew.unsqueeze(0)
        self.batch = self.xi.shape[0]  # number of atom pairs
        self.arange_batch = torch.arange(self.batch)

        if self.hs_grid is not None:  # with DFTB+ distance interpolation
            assert distances is not None, (
                "if hs_grid is not None, " + "distances is expected"
            )

            # original dims: vcr1, vcr2, distances, n_orb_pairs
            # permute dims: distances, vcr1, vcr2, n_orb_pairs
            zmesh = self.zmesh  # .permute([-2, 0, 1, -1])

            ski = PolyInterpU(self.hs_grid, zmesh)
            zmesh = ski(distances)
        else:
            zmesh = self.zmesh

        coeff = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [-3.0, 3.0, -2.0, -1.0],
                [2.0, -2.0, 1.0, 1.0],
            ]
        )
        coeff_ = torch.tensor(
            [
                [1.0, 0.0, -3.0, 2.0],
                [0.0, 0.0, 3.0, -2.0],
                [0.0, 1.0, -2.0, 1.0],
                [0.0, 0.0, -1.0, 1.0],
            ]
        )

        # get the nearest grid points, 1st and second neighbour indices of xi
        self._get_indices()

        # this is to transfer x to fraction and its square, cube
        x_fra = (self.xi - self.xmesh[self.nx0]) / (
            self.xmesh[self.nx1] - self.xmesh[self.nx0]
        )

        # xx0 = torch.stack([self.xmesh[..., 0][self.nx0[..., 0]],
        #                    self.xmesh[..., 1][self.nx0[..., 1]]]).T
        # xx1 = torch.stack([self.xmesh[..., 0][self.nx1[..., 0]],
        #                    self.xmesh[..., 1][self.nx1[..., 1]]]).T
        # x_fra = (self.xi - xx0) / (xx1 - xx0)

        xmat = torch.stack([x_fra**0, x_fra**1, x_fra**2, x_fra**3])

        # get four nearest grid points values, each will be: [natom, natom, 20]
        f00, f10, f01, f11 = self._fmat0th(zmesh)

        # get four nearest grid points derivative over x, y, xy
        f02, f03, f12, f13, f20, f21, f30, f31, f22, f23, f32, f33 = (
            self._fmat1th(zmesh, f00, f10, f01, f11)
        )
        fmat = torch.stack(
            [
                torch.stack([f00, f01, f02, f03]),
                torch.stack([f10, f11, f12, f13]),
                torch.stack([f20, f21, f22, f23]),
                torch.stack([f30, f31, f32, f33]),
            ]
        )

        pdim = [2, 0, 1] if fmat.dim() == 3 else [2, 3, 0, 1]
        a_mat = torch.matmul(torch.matmul(coeff, fmat.permute(pdim)), coeff_)

        return torch.stack(
            [
                torch.matmul(
                    torch.matmul(xmat[:, i, 0], a_mat[i]), xmat[:, i, 1]
                )
                for i in range(self.batch)
            ]
        )

    def _get_indices(self):
        """Get indices and repeat indices."""
        self.nx0 = torch.searchsorted(self.xmesh, self.xi.detach()) - 1
        # self.nx0 = torch.stack([
        #     torch.searchsorted(self.xmesh[..., 0], self.xi.detach()[..., 0]) - 1,
        #     torch.searchsorted(self.xmesh[..., 1], self.xi.detach()[..., 1]) - 1]).T

        # get all surrounding 4 grid points indices and repeat indices
        self.nind = torch.tensor([ii for ii in range(self.batch)])
        self.nx1 = torch.clamp(
            torch.stack([ii + 1 for ii in self.nx0]), 0, len(self.xmesh) - 1
        )
        self.nx_1 = torch.clamp(torch.stack([ii - 1 for ii in self.nx0]), 0)
        self.nx2 = torch.clamp(
            torch.stack([ii + 2 for ii in self.nx0]), 0, len(self.xmesh) - 1
        )

    def _fmat0th(self, zmesh: Tensor):
        """Construct f(0/1, 0/1) in fmat."""
        f00 = zmesh[self.arange_batch, self.nx0[..., 0], self.nx0[..., 1]]
        f10 = zmesh[self.arange_batch, self.nx1[..., 0], self.nx0[..., 1]]
        f01 = zmesh[self.arange_batch, self.nx0[..., 0], self.nx1[..., 1]]
        f11 = zmesh[self.arange_batch, self.nx1[..., 0], self.nx1[..., 1]]
        return f00, f10, f01, f11

    def _fmat1th(
        self, zmesh: Tensor, f00: Tensor, f10: Tensor, f01: Tensor, f11: Tensor
    ):
        """Get the 1st derivative of four grid points over x, y and xy."""
        f_10 = zmesh[self.arange_batch, self.nx_1[..., 0], self.nx0[..., 1]]
        f_11 = zmesh[self.arange_batch, self.nx_1[..., 0], self.nx1[..., 1]]
        f0_1 = zmesh[self.arange_batch, self.nx0[..., 0], self.nx_1[..., 1]]
        f02 = zmesh[self.arange_batch, self.nx0[..., 0], self.nx2[..., 1]]
        f1_1 = zmesh[self.arange_batch, self.nx1[..., 0], self.nx_1[..., 1]]
        f12 = zmesh[self.arange_batch, self.nx1[..., 0], self.nx2[..., 1]]
        f20 = zmesh[self.arange_batch, self.nx2[..., 0], self.nx0[..., 1]]
        f21 = zmesh[self.arange_batch, self.nx2[..., 0], self.nx1[..., 1]]

        # calculate the derivative: (F(1) - F(-1) / (2 * grid)
        fy00 = ((f01 - f0_1).T / (self.nx1[..., 1] - self.nx_1[..., 1])).T
        fy01 = ((f02 - f00).T / (self.nx2[..., 1] - self.nx0[..., 1])).T
        fy10 = ((f11 - f1_1).T / (self.nx1[..., 1] - self.nx_1[..., 1])).T
        fy11 = ((f12 - f10).T / (self.nx2[..., 1] - self.nx0[..., 1])).T
        fx00 = ((f10 - f_10).T / (self.nx1[..., 0] - self.nx_1[..., 0])).T
        fx01 = ((f20 - f00).T / (self.nx2[..., 0] - self.nx0[..., 0])).T
        fx10 = ((f11 - f_11).T / (self.nx1[..., 0] - self.nx_1[..., 0])).T
        fx11 = ((f21 - f01).T / (self.nx2[..., 0] - self.nx0[..., 0])).T
        fxy00, fxy11 = fy00 * fx00, fx11 * fy11
        fxy01, fxy10 = fx01 * fy01, fx10 * fy10

        return (
            fy00,
            fy01,
            fy10,
            fy11,
            fx00,
            fx01,
            fx10,
            fx11,
            fxy00,
            fxy01,
            fxy10,
            fxy11,
        )


class BSpline:
    """Providing routines for multivariate interpolation with tensor product
    splines.

    The number of nodes has to be at least 4 in every dimension. Otherwise a
    singularity error will occur during  calculation.

    Arguments:
        f_nodes (Tensor): Values at the given sites.
        x_nodes (Tensor): x-nodes of the interpolation data.
        *args (Tensor): y-, z-, ... nodes. Every set of nodes is a
            tensor.

    Notes:
        The theory of the univariate b-splines were taken from [3]_. The
        mathematical background about the calculations of the coefficients
        during a multivariate interpolation were taken from [4]_.

    References:
        .. [3] Boor, C.d. 2001, "A Practical Guide to Splines". Rev. ed.
           Springer-Verlag
        .. [4] Floater, M.S. (2007) "Tensor Product Spline Surfaces"[Online].
           Available at: https://www.uio.no/studier/emner/matnat/ifi/nedlagte-emner/INF-MAT5340/v07/undervisningsmateriale/kap7.pdf
           (Accessed: 01 December 2020)
    """

    def __init__(
        self, f_nodes: torch.Tensor, x_nodes: torch.Tensor, *args: torch.Tensor
    ):
        # Dimensions will be called the following: x, y, z, a, b, c, d, ...
        # f_nodes has shape (x, y, z, a, b, ...)
        self.f_nodes = f_nodes

        self.nodes = [x_nodes, *args]
        self.device = self.f_nodes.device

        self.num = len(self.nodes)

        # Permute f_nodes: (z, a, b, ... , x, y)
        if self.f_nodes.dim() == len(self.nodes) + 1:
            self.f_nodes = [
                _if.permute(tuple(torch.arange(self.num).roll(-2)))
                for _if in f_nodes
            ]
        elif self.f_nodes.dim() == len(self.nodes):
            self.f_nodes = [
                f_nodes.permute(tuple(torch.arange(self.num).roll(-2)))
            ]
        else:
            raise ValueError("do not support f_nodes dims")

        # Calculate a list containing the knot-vectors of every dimension
        self.knot_vec = [self.get_knot_vector(nodes) for nodes in self.nodes]

        # Calculate a list containing tensors of b-splines for every
        # dimension.
        self.bsplines = [
            self.get_bsplines(
                self.knot_vec[ii], self.nodes[ii], self.nodes[ii], 4
            )
            for ii in range(self.num)
        ]

        # Create cyclic permutation for the coefficients.
        roll_permutation = torch.arange(self.num).roll(-1)

        _dd, d_list = self.f_nodes, []
        for idd in _dd:
            for ii in self.bsplines:
                idd = torch.solve(idd, ii.T).solution.permute(
                    *roll_permutation
                )
            d_list.append(idd)

        # c has now the following shape: (z, a, b, ... , x, y)
        self.cc = d_list

    def get_knot_vector(self, nodes: torch.Tensor) -> torch.Tensor:
        """Calculates the corresponding knot vector for the given nodes.

        Arguments:
            nodes (Tensor): Nodes of which the knot vector should be
                calculated.

        Returns:
            tt (Tensor): Knot vector for the given nodes.
        """

        tt = torch.zeros((len(nodes) + 4,), device=self.device)
        tt[0:4] = nodes[0]
        tt[4:-4] = nodes[2:-2]
        tt[-4:] = nodes[-1]

        return tt

    def get_bsplines(
        self, tt: torch.Tensor, xx: torch.Tensor, nodes: torch.Tensor, kk: int
    ) -> torch.Tensor:
        r"""Calculates a tensor containing the values of the b-splines at the
        given sites.

        Assume that: :math:`\text{len}(x) = m, \text{len}(x_{\text{nodes}}) =
        n`. The tensor has dimensions: :math:`(m, n)`. The b-splines are
        row-vectors. Hence the tensor has the following structure:

        .. math::

           \begin{matrix}
           B_1(x_1) & ... & B_1(x_m)\\
           \vdots &  & \vdots\\
           B_n(x_1) & ... & B_n(x_m)
           \end{matrix}

        Arguments:
            tt (Tensor): Knot vector of the corresponding nodes.
            xx (Tensor): Values where the b-splines should be evaluated.
            nodes (Tensor): Interpolation nodes.
            kk (int): Order of the b-splines. Order of `k` means that the
                b-splines have degree `k`-1.

        Returns:
            b_tensor (Tensor): Tensor containing the values of the
                b-splines at the corresponding x-values.
        """

        j_num = torch.arange(0, len(tt) - kk, device=self.device)

        # Calculate a tensor containing the b-splines for every dimension
        b_tensor = [self._b_spline(tt, xx, nodes, jj, kk) for jj in j_num]

        b_tensor = torch.stack(b_tensor)

        return b_tensor

    def _b_spline(
        self,
        tt: torch.Tensor,
        xx: torch.Tensor,
        nodes: torch.Tensor,
        jj: int,
        kk: int,
    ) -> torch.Tensor:
        r"""Calculates the b-spline for a given knot vector.

        It calculates the y-values of the `j`th b-spline of order `k`.
        The b-spline will be calculated for the knot vector `t`. The
        calculation follows the recurrence relation:

        .. math::

           B_{j, 1} = 1 if t_j \leq x < t_{jj + 1}, 0 \text{otherwise}
           B_{j, k} = \frac{x - t_j}{t_{j + k - 1} - t_j} B_{j, k-1}
           + (1 - \frac{x - t_{j + 1}}{t_{j + k} -
           t_{j + 1}} B_{j + 1, k-1}

        Arguments:
            tt (Tensor): Knot vector. Tensor containing the knots. You
                need at least `k`+1 knots.
            xx (Tensor): Tensor containing the x-values where the
                b-spline should be evaluated.
            jj (int): Specifies which b-spline should be calculated.
            kk (int): Specifies the order of the b-spline. Order of `k` means
                that the calculated polynomial has degree `k-1`.

        Returns:
            yy (Tensor): Tensor containing the y-values of the `j`th
                b-spline of order `k` for the corresponding x-values.
        """

        t1 = tt[jj]
        t2 = tt[jj + 1]
        t3 = tt[jj + kk - 1]
        t4 = tt[jj + kk]

        if kk == 1:
            yy = torch.where((tt[jj] <= xx) & (xx < tt[jj + 1]), 1, 0)
            if jj == len(tt) - 4:
                yy = torch.where((tt[jj] <= xx) & (xx <= tt[jj + 1]), 1, 0)
            if len(nodes) == 2 and jj == 1:
                yy = torch.where((tt[jj] <= xx) & (xx <= tt[jj + 1]), 1, 0)
        else:
            # Here the recursion formula will be executed. The 'if' and 'else'
            # blocks ensure that one avoid the division by zero.
            if tt[jj + kk - 1] == tt[jj] and tt[jj + kk] == tt[jj + 1]:
                yy = self._b_spline(tt, xx, nodes, jj + 1, kk - 1)
            elif tt[jj + kk - 1] == tt[jj] and tt[jj + kk] != tt[jj + 1]:
                yy = (1 - (xx - t2) / (t4 - t2)) * self._b_spline(
                    tt, xx, nodes, jj + 1, kk - 1
                )
            elif tt[jj + kk - 1] != tt[jj] and tt[jj + kk] == tt[jj + 1]:
                yy = (xx - t1) / (t3 - t1) * self._b_spline(
                    tt, xx, nodes, jj, kk - 1
                ) + self._b_spline(tt, xx, nodes, jj + 1, kk - 1)
            else:
                yy = (xx - t1) / (t3 - t1) * self._b_spline(
                    tt, xx, nodes, jj, kk - 1
                ) + (1 - (xx - t2) / (t4 - t2)) * self._b_spline(
                    tt, xx, nodes, jj + 1, kk - 1
                )
        return yy

    def __call__(
        self, x_new: torch.Tensor, *args: torch.Tensor, grid=True
    ) -> torch.Tensor:
        """Evaluates the spline function at the desired sites.

        Arguments:
            x_new (Tensor): x-values, where you want to evaluate the spline
                function.
            *args (Tensor): New values for the other dimensions.
            grid (bool): You can decide whether you want to evaluate the
                results on a grid or not. If `grid=True` (default) the grid is
                spanned by the input tensors. If `grid=False` the spline
                function will be evaluated at single points specified by the
                rows of one single input tensor with dimension of 2 or by the
                values of multiple tensors.

        Returns:
            ff (Tensor): Tensor containing the values at the given sites.
        """

        if len(x_new.size()) == 1:
            new_vals = [x_new, *args]
        else:
            new_vals = [x_new[:, ii] for ii in range(x_new.shape[1])]
        inverted_num = torch.arange(0, self.num).flip(0)

        # matrices contains the b-spline tensors but in inverted order.
        matrices = [
            self.get_bsplines(
                self.knot_vec[ii], new_vals[ii], self.nodes[ii], 4
            ).T
            for ii in inverted_num
        ]

        # permutation1: cyclic permutation.
        # Permute dd so it has shape: (y, z, a, b, ..., x)
        # dd = self.cc.permute(*torch.arange(self.num).roll(1))
        _dd = [ic.permute(*torch.arange(self.num).roll(1)) for ic in self.cc]
        _d_list = []

        for dd in _dd:
            permutation = [-2, *range(self.num - 2), -1]
            for ii in range(self.num):
                if ii == self.num - 1:
                    # For the last multiplication permute dd so the shapes
                    # matches properly
                    dd = dd.transpose(-1, -2)

                dd = torch.matmul(matrices[ii], dd)
                dd = dd.permute(permutation)

                # to get the diagonal values, instead of [n_batch, n_batch ...]
                if ii > 0:
                    permutation = [-2, *range(self.num - 2 - ii), -1]
                    _mask = torch.arange(dd.shape[0])
                    dd = dd[_mask, _mask]

            _d_list.append(dd)

        # Final permutation of f so it has the same shape as f_nodes in the
        # input: (x, y, z, a, b, ...).
        ff = _d_list  # dd

        for _if in ff:
            if not grid:
                for ii in range(self.num - 1):
                    _if = torch.diagonal(_if)

        return torch.stack(ff).T


class Spline1d:
    """Polynomial natural (linear, cubic) non-periodic spline.

    Arguments:
        x: 1D Tensor variable.
        y: 1D (single) or 2D (batch) Tensor variable.

    Keyword Args:
        kind: Define spline method, 'cubic' or 'linear'.
        abcd: 0th, 1st, 2nd and 3rd order parameters in cubic spline.

    References:
        .. [wiki] https://en.wikipedia.org/wiki/Spline_(mathematics)

    Examples:
        >>> import tbmalt.common.maths.interpolator as interp
        >>> import torch
        >>> x = torch.linspace(1, 10, 10)
        >>> y = torch.sin(x)
        >>> fit = interp.Spline1d(x, y)
        >>> fit(torch.tensor([3.5]))
        >>> tensor([-0.3526])
        >>> torch.sin(torch.tensor([3.5]))
        >>> tensor([-0.3508])
    """

    def __init__(self, xx: Tensor, yy: Tensor, **kwargs):
        self.xp, self.yp = xx, yy.T
        self.kind = kwargs.get("kind", "cubic")

        if self.kind == "cubic":
            if kwargs.get("abcd") is not None:
                self.aa, self.bb, self.cc, self.dd = kwargs.get("abcd")
            else:
                self.aa, self.bb, self.cc, self.dd = self._get_abcd()
            self.abcd = pack([self.aa, self.bb, self.cc, self.dd])
        elif self.kind != "linear":
            raise NotImplementedError("%s not implemented" % self.kind)

    def __call__(self, xnew: Tensor):
        """Evaluate the polynomial linear or cubic spline.
        Arguments:
            xnew: 0D Tensor.
        Returns:
            ynew: 0D Tensor.
        """
        # according to the order to choose spline method
        self.xnew = xnew
        self.knot = [ii in self.xp for ii in self.xnew]

        # boundary condition of xnew,  xp[0] < xnew
        assert self.xnew.ge(self.xp[0]).all()
        self.mask = self.xnew.lt(self.xp[-1])

        # get the nearest grid point index of d in x
        self.dind = [
            bisect.bisect(self.xp.detach().numpy(), ii.detach().numpy()) - 1
            for ii in self.xnew[self.mask]
        ]

        if self.kind == "cubic":
            return self._cubic()
        elif self.kind == "linear":
            return self._linear()

    def _linear(self):
        """Calculate linear interpolation."""
        return (
            self.yp[self.ind]
            + (self.xnew - self.xp[self.dind])
            / (self.xp[1:] - self.xp[:-1])[self.dind]
            * (self.yp[..., 1:] - self.yp[..., :-1])[self.ind]
        )

    def _cubic(self):
        """Calculate cubic spline interpolation."""
        aa, bb, cc, dd = self.abcd[0], self.abcd[1], self.abcd[2], self.abcd[3]

        # calculate a, b, c, d parameters, need input x and y
        dx = self.xnew[self.mask] - self.xp[self.dind]
        _intergal = (
            aa[..., self.dind]
            + bb[..., self.dind] * dx
            + cc[..., self.dind] * dx**2
            + dd[..., self.dind] * dx**3
        )
        _intergal = (
            _intergal.transpose(-1, 0) if _intergal.dim() > 1 else _intergal
        )
        _y = torch.zeros(self.xnew.shape[0], *_intergal.shape[1:])
        _y[self.mask] = _intergal
        return _y

    def _get_abcd(self):
        """Get parameter aa, bb, cc, dd for cubic spline interpolation."""
        # get the first dim of x
        nx = self.xp.shape[0]
        assert nx > 3  # the length of x variable must > 3

        # get the differnce between grid points
        dxp = self.xp[1:] - self.xp[:-1]
        dyp = self.yp[..., 1:] - self.yp[..., :-1]

        # get b, c, d from reference website: step 3~9, first calculate c
        A = torch.zeros(nx, nx)
        A.diagonal()[1:-1] = 2 * (dxp[:-1] + dxp[1:])  # diag
        A[torch.arange(nx - 1), torch.arange(nx - 1) + 1] = dxp  # off-diag
        A[torch.arange(nx - 1) + 1, torch.arange(nx - 1)] = dxp
        A[0, 0], A[-1, -1] = 1.0, 1.0
        A[0, 1], A[1, 0] = 0.0, 0.0  # natural condition
        A[-1, -2], A[-2, -1] = 0.0, 0.0

        B = torch.zeros(*self.yp.shape)
        B[..., 1:-1] = 3 * (dyp[..., 1:] / dxp[1:] - dyp[..., :-1] / dxp[:-1])
        B = B.permute(1, 0) if B.dim() == 2 else B

        cc = torch.linalg.lstsq(A, B).solution
        cc = cc.permute(1, 0) if cc.dim() == 2 else cc.unsqueeze(0)
        bb = dyp / dxp - dxp * (cc[..., 1:] + 2 * cc[..., :-1]) / 3
        dd = (cc[..., 1:] - cc[..., :-1]) / (3 * dxp)
        return self.yp, bb, cc, dd


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class PolyInterpUNew:
    """Polynomial interpolation class to mimic the structure in your parameters"""

    def __init__(
        self,
        xx_shape,
        yy_shape,
        tail=1.0,
        delta_r=1e-05,
        n_interp=8,
        n_interp_r=4,
        grid_step=2.000e-02,
        device="cpu",
    ):
        self.xx = torch.linspace(
            0, (xx_shape[0] - 1) * grid_step, xx_shape[0], device=device
        )
        self.yy = torch.randn(yy_shape, device=device)
        self.tail = tail
        self.delta_r = delta_r
        self.n_interp = n_interp
        self.n_interp_r = n_interp_r
        self.grid_step = grid_step
        self.device = device

    def __call__(self, r):
        """Interpolate at distance r"""
        # Simple linear interpolation for demonstration
        idx = torch.clamp((r / self.grid_step).long(), 0, len(self.xx) - 2)
        alpha = (r / self.grid_step) - idx.float()

        y1 = self.yy[idx]
        y2 = self.yy[idx + 1]
        return y1 + alpha.unsqueeze(-1) * (y2 - y1)


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


def smooth_tail_batch(
    xx: Tensor, yy: Tensor, n_grid: Tensor, ninterp=8, delta_r=1e-5, tail=1.0
):
    """Smooth the tail with input xx and yy."""
    assert xx.dim() == 2
    assert yy.dim() == 3
    assert n_grid.shape[0] == xx.shape[0]

    ilast = n_grid.clone()
    incr = xx[0, 1] - xx[0, 0]

    # get grid points and grid point values
    xa = (ilast.unsqueeze(1) - ninterp + torch.arange(ninterp)) * incr
    yb = torch.stack(
        [yy[ii, il - ninterp - 1 : il - 1] for ii, il in enumerate(ilast)]
    )

    # return smooth grid points in the tail for each SKF
    dr = -torch.linspace(4, 0, 5) * incr

    # get derivative
    y0 = poly_interp_2d(xa, yb, xa[:, ninterp - 1] - delta_r)
    y2 = poly_interp_2d(xa, yb, xa[:, ninterp - 1] + delta_r)
    y1 = yy[torch.arange(yy.shape[0]), ilast - 2]
    y1p = (y2 - y0) / (2.0 * delta_r)
    y1pp = (y2 + y0 - 2.0 * y1) / (delta_r * delta_r)
    # dr = dr.unsqueeze(1).unsqueeze(2).repeat(1, y0.shape[0], y0.shape[1])
    integral_tail = poly5_zero(
        y1.unsqueeze(-1), y1p.unsqueeze(-1), y1pp.unsqueeze(-1), dr, -tail
    ).permute(0, 2, 1)

    for ii, iy in enumerate(yy):  # -> add tail
        yy[ii, n_grid[ii] : n_grid[ii] + 5] = integral_tail[ii]

    return yy


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


def poly_to_zero(xx: Tensor, yy: Tensor, ninterp=8, delta_r=1e-5, tail=1.0):
    """Smooth the tail with input xx and yy with various compression radii.

    Arguments:
        xx:
        yy:

    """
    assert xx.dim() == 1
    assert yy.dim() == 2
    ny1, ny2 = yy.shape

    # _yy = yy.permute(0, -1, -2)  # -> permute distance dim from last to second
    ilast = len(xx)
    incr = xx[1] - xx[0]  # uniform grid points

    # get grid points and grid point values
    xa = (ilast - ninterp + torch.arange(ninterp)).repeat(ny1, 1) * incr
    yb = yy[..., ilast - ninterp - 1 : ilast - 1]

    # return smooth grid points in the tail for each SKF
    dr = -torch.linspace(4, 0, 5).repeat(1, ny1).T * incr

    # get derivative
    y0 = poly_interp_2d(xa, yb, xa[..., ninterp - 1] - delta_r)
    y2 = poly_interp_2d(xa, yb, xa[..., ninterp - 1] + delta_r)
    y1 = yy[..., ilast - 2]
    y1p = (y2 - y0) / (2.0 * delta_r)
    y1pp = (y2 + y0 - 2.0 * y1) / (delta_r * delta_r)
    integral_tail = poly5_zero(y1, y1p, y1pp, dr, -1.0 * tail)

    size_tail = integral_tail.T.shape[-1]
    ynew = torch.zeros(yy.shape[0], yy.shape[1] + size_tail)

    ynew[..., : yy.shape[1]] = yy
    ynew[..., -size_tail:] = integral_tail.T

    return ynew


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
    y0 = poly_interp_2d(xa, yb, xa[:, ninterp - 1] - delta_r)
    y2 = poly_interp_2d(xa, yb, xa[:, ninterp - 1] + delta_r)
    y1 = _yy[torch.arange(_yy.shape[0]), ilast - 2]
    y1p = (y2 - y0) / (2.0 * delta_r)
    y1pp = (y2 + y0 - 2.0 * y1) / (delta_r * delta_r)
    integral_tail = poly5_zero(
        y1.unsqueeze(-1),
        y1p.unsqueeze(-1),
        y1pp.unsqueeze(-1),
        dr,
        -1.0 * tail,
    ).permute(0, 2, 1)

    if _yy.shape[1] == max(n_grid):
        ynew = torch.zeros(ny1, ny3 + integral_tail.shape[1], ny2)
        ynew[:ny1, :ny3, :ny2] = _yy[:ny1, :ny3, :ny2]
    else:
        ynew = _yy.clone()

    for ii, iy in enumerate(_yy):  # -> add tail
        ynew[ii, n_grid[ii] : n_grid[ii] + 5] = integral_tail[ii]

    return ynew.permute(0, -1, -2)
