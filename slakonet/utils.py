from typing import Tuple, Union, Literal, Optional
import torch
import numpy as np
from numbers import Real
from functools import reduce, partial
from typing import Optional, Any, Tuple, List, Union
from collections import namedtuple
import torch

# from tbmalt.common import bool_like
Tensor = torch.Tensor
__sort = namedtuple("sort", ("values", "indices"))
Sliceable = Union[List[Tensor], Tuple[Tensor]]
bool_like = Union[Tensor, bool]
float_like = Union[Tensor, float]


def split_by_size(
    tensor: Tensor, sizes: Union[Tensor, List[int]], dim: int = 0
) -> Tuple[Tensor]:
    """Splits a tensor into chunks of specified length.

    This function takes a tensor & splits it into `n` chunks, where `n` is the
    number of entries in ``sizes``. The length of the `i'th` chunk is defined
    by the `i'th` element of ``sizes``.

    Arguments:
        tensor: Tensor to be split.
        sizes: Size of each chunk.
        dim: Dimension along which to split ``tensor``.

    Returns:
        chunked: Tuple of tensors viewing the original ``tensor``.

    Examples:
        Tensors can be sequentially split into multiple sub-tensors like so:

        >>> from tbmalt.common import split_by_size
        >>> a = torch.arange(10)
        >>> print(split_by_size(a, [2, 2, 2, 2, 2]))
        (tensor([0, 1]), tensor([2, 3]), tensor([4, 5]), tensor([6, 7]), tensor([8, 9]))
        >>> print(split_by_size(a, [5, 5]))
        tensor([0, 1, 2, 3, 4]), tensor([5, 6, 7, 8, 9]))
        >>> print(split_by_size(a, [1, 2, 3, 4]))
        (tensor([0]), tensor([1, 2]), tensor([3, 4, 5]), tensor([6, 7, 8, 9]))

    Notes:
        The resulting tensors ``chunked`` are views of the original tensor and
        not copies. This was created as no analog existed natively within the
        pytorch framework. However, this will eventually be removed once the
        pytorch function `split_with_sizes` becomes operational.

    Raises:
        AssertionError: If number of elements requested via ``split_sizes``
            exceeds the number of elements present in ``tensor``.
    """
    # Looks like it returns a tuple rather than a list
    if dim < 0:  # Shift dim to be compatible with torch.narrow
        dim += tensor.dim()

    # Ensure the tensor is large enough to satisfy the chunk declaration.
    size_match = tensor.shape[dim] == sum(sizes)
    assert size_match, (
        "Sum of split sizes fails to match tensor length "
        "along specified dim"
    )

    # Identify the slice positions
    splits = torch.cumsum(torch.tensor([0, *sizes]), dim=0)[:-1]

    # Return the sliced tensor. use torch.narrow to avoid data duplication
    return tuple(
        torch.narrow(tensor, dim, start, length)
        for start, length in zip(splits, sizes)
    )


def pargsort(
    tensor: Tensor, mask: Optional[bool_like] = None, dim: int = -1
) -> Tensor:
    """Returns indices that sort packed tensors while ignoring padding values.

    Returns the indices that sorts the elements of ``tensor`` along ``dim`` in
    ascending order by value while ensuring padding values are shuffled to the
    end of the dimension.

    Arguments:
        tensor: the input tensor.
        mask: a boolean tensor which is True & False for "real" & padding
            values restively. [DEFAULT=None]
        dim: the dimension to sort along. [DEFAULT=-1]

    Returns:
        out: ``indices`` which along the dimension ``dim``.

    Notes:
        This will redirect to `torch.argsort` if no ``mask`` is supplied.
    """
    if mask is None:
        return torch.argsort(tensor, dim=dim)
    else:
        # A secondary sorter is used to reorder the primary sorter so that padding
        # values are moved to the end.
        n = tensor.shape[dim]
        s1 = tensor.argsort(dim)
        s2 = (
            torch.arange(n, device=tensor.device) + (~mask.gather(dim, s1) * n)
        ).argsort(dim)
        return s1.gather(dim, s2)


def psort(
    tensor: Tensor, mask: Optional[bool_like] = None, dim: int = -1
) -> __sort:
    """Sort a packed ``tensor`` while ignoring any padding values.

    Sorts the elements of ``tensor`` along ``dim`` in ascending order by value
    while ensuring padding values are shuffled to the end of the dimension.

    Arguments:
        tensor: the input tensor.
        mask: a boolean tensor which is True & False for "real" & padding
            values restively. [DEFAULT=None]
        dim: the dimension to sort along. [DEFAULT=-1]

    Returns:
        out: A namedtuple of (values, indices) is returned, where the values
             are the sorted values and indices are the indices of the elements
             in the original input tensor.

    Notes:
        This will redirect to `torch.sort` if no ``mask`` is supplied.
    """
    if mask is None:
        return torch.sort(tensor, dim=dim)
    else:
        indices = pargsort(tensor, mask, dim)
        return __sort(tensor.gather(dim, indices), indices)


def pack(
    tensors: Sliceable,
    axis: int = 0,
    value: Any = 0,
    size: Optional[Union[Tuple[int], torch.Size]] = None,
    return_mask: bool = False,
) -> Union[Tensor, Optional[Tensor]]:
    """Pad and pack a sequence of tensors together.

    Pad a list of variable length tensors with zeros, or some other value, and
    pack them into a single tensor.

    Arguments:
        tensors: List of tensors to be packed, all with identical dtypes.
        axis: Axis along which tensors should be packed; 0 for first axis -1
            for the last axis, etc. This will be a new dimension. [DEFAULT=0]
        value: The value with which the tensor is to be padded. [DEFAULT=0]
        size: Size of each dimension to which tensors should be padded. This
            to the largest size encountered along each dimension.
        return_mask: If True, a mask identifying the padding values is
            returned. [DEFAULT=False]

    Returns:
        packed_tensors: Input tensors padded and packed into a single tensor.
        mask: A tensor that can mask out the padding values. A False value in
            ``mask`` indicates the corresponding entry in ``packed_tensor`` is
            a padding value.

    Notes:
        ``packed_tensors`` maintains the same order as ``tensors``. This
        is faster & more flexible than the internal pytorch pack & pad
        functions (at this particularly task).

        If a ``tensors`` is a `torch.tensor` it will be immedatly returned.
        This helps with batch agnostic programming.

    Examples:
        Multiple tensors can be packed into a single tensor like so:

        >>> from tbmalt.common.batch import pack
        >>> import torch
        >>> a, b, c = torch.rand(2,2), torch.rand(3,3), torch.rand(4,4)
        >>> abc_packed_a = pack([a, b, c])
        >>> print(abc_packed_a.shape)
        torch.Size([3, 4, 4])
        >>> abc_packed_b = pack([a, b, c], axis=1)
        >>> print(abc_packed_b.shape)
        torch.Size([4, 3, 4])
        >>> abc_packed_c = pack([a, b, c], axis=-1)
        >>> print(abc_packed_c.shape)
        torch.Size([4, 4, 3])

        An optional mask identifying the padding values can also be returned:

        >>> packed, mask = pack([torch.tensor([1.]),
        >>>                      torch.tensor([2., 2.]),
        >>>                      torch.tensor([3., 3., 3.])],
        >>>                     return_mask=True)
        >>> print(packed)
        tensor([[1., 0., 0.],
                [2., 2., 0.],
                [3., 3., 3.]])
        >>> print(mask)
        tensor([[ True, False, False],
                [ True,  True, False],
                [ True,  True,  True]])

    """
    # If "tensors" is already a Tensor then return it immediately as there is
    # nothing more that can be done. This helps with batch agnostic
    # programming.
    if isinstance(tensors, Tensor):
        return tensors

    # Gather some general setup info
    count, device, dtype = len(tensors), tensors[0].device, tensors[0].dtype

    # Identify the maximum size, if one was not specified.
    if size is None:
        size = torch.tensor([i.shape for i in tensors]).max(0).values

    # Tensor to pack into, filled with padding value.
    padded = torch.full((count, *size), value, dtype=dtype, device=device)

    if return_mask:  # Generate the mask if requested.
        mask = torch.full(
            (count, *size), False, dtype=torch.bool, device=device
        )

    # Loop over & pack "tensors" into "padded". A proxy index "n" must be used
    # for assignments rather than a slice to prevent in-place errors.
    for n, source in enumerate(tensors):
        # Slice operations not elegant but they are dimension agnostic & fast.
        padded[(n, *[slice(0, s) for s in source.shape])] = source
        if return_mask:  # Update the mask if required.
            mask[(n, *[slice(0, s) for s in source.shape])] = True

    # If "axis" was anything other than 0, then "padded" must be permuted.
    if axis != 0:
        # Resolve relative any axes to their absolute equivalents to maintain
        # expected slicing behaviour when using the insert function.
        axis = padded.dim() + 1 + axis if axis < 0 else axis

        # Build a list of axes indices; but omit the axis on which the data
        # was concatenated (i.e. 0).
        ax = list(range(1, padded.dim()))

        ax.insert(axis, 0)  # Re-insert the concatenation axis as specified

        padded = padded.permute(ax)  # Perform the permeation

        if return_mask:  # Perform permeation on the mask is present.
            mask = mask.permute(ax)

    # Return the packed tensor, and the mask if requested.
    return (padded, mask) if return_mask else padded


class _SymEigB(torch.autograd.Function):
    # State that this can solve for multiple systems and that the first
    # dimension should iterate over instance of the batch.
    r"""Solves standard eigenvalue problems for real symmetric matrices.

    This solves standard eigenvalue problems for real symmetric matrices, and
    can apply conditional or Lorentzian broadening to the eigenvalues during
    the backwards pass to increase gradient stability.

    Notes:
        Results from backward passes through eigen-decomposition operations
        tend to suffer from numerical stability [*]_  issues when operating
        on systems with degenerate eigenvalues. Fortunately,  the stability
        of such operations can be increased through the application of eigen
        value broadening. However, such methods will induce small errors in
        the returned gradients as they effectively mutate  the eigen-values
        in the backwards pass. Thus, it is important to be aware that while
        increasing the extent of  broadening will help to improve stability
        it will also increase the error in the gradients.

        Two different broadening methods have been  implemented within this
        class. Conditional broadening as described by Seeger [MS2019]_, and
        Lorentzian as detailed by Liao [LH2019]_. During the forward pass the
        `torch.linalg.eigh` function is used to calculate both the eigenvalues &
        the eigenvectors (U & :math:`\lambda` respectively). The gradient
        is then calculated following:

        .. math:: \bar{A} = U (\bar{\Lambda} + sym(F \circ (U^t \bar{U}))) U^T

        Where bar indicates a value's gradient passed in from  the previous
        layer, :math:`\Lambda` is the diagonal matrix associated with the
        :math:`\bar{\lambda}` values,  :math:`\circ`  is the so  called
        Hadamard product, sym is the symmetrisation operator and F is:

        .. math:: F_{i, j} = \frac{I_{i \ne j}}{h(\lambda_i - \lambda_j)}

        Where, for conditional broadening, h is:

        .. math:: h(t) = max(|t|, \epsilon)sgn(t)

        and for, Lorentzian broadening:

        .. math:: h(t) = \frac{t^2 + \epsilon}{t}

        The advantage of conditional broadening is that is is only applied
        when it is needed, thus the errors induced in the gradients will be
        restricted to systems whose gradients would be nan's otherwise.
        The Lorentzian method, on the other hand, will apply broadening to
        all systems, irrespective of whether or not it is necessary. Note
        that if the h function is a unity operator then this is identical
        to a standard backwards pass through an eigen-solver.


        .. [*] Where stability is defined as the propensity of a function to
               return nan values or some raise an error.

    References:
        .. [MS2019] Seeger, M., Hetzel, A., Dai, Z., & Meissner, E. Auto-
                    Differentiating Linear Algebra. ArXiv:1710.08717 [Cs,
                    Stat], Aug. 2019. arXiv.org,
                    http://arxiv.org/abs/1710.08717.
        .. [LH2019] Liao, H.-J., Liu, J.-G., Wang, L., & Xiang, T. (2019).
                    Differentiable Programming Tensor Networks. Physical
                    Review X, 9(3).
        .. [Lapack] www.netlib.org/lapack/lug/node54.html (Accessed 10/08/2020)

    """

    # Note that 'none' is included only for testing purposes
    KNOWN_METHODS = ["cond", "lorn", "none"]

    @staticmethod
    def forward(
        ctx, a: Tensor, method: str = "cond", factor: float = 1e-12
    ) -> Tuple[Tensor, Tensor]:
        """Calculate the eigenvalues and eigenvectors of a symmetric matrix.

        Finds the eigenvalues and eigenvectors of a real symmetric
        matrix using the torch.linalg.eigh function.

        Arguments:
            a: A real symmetric matrix whose eigenvalues & eigenvectors will
                be computed.
            method: Broadening method to used, available options are:

                    - "cond" for conditional broadening.
                    - "lorn" for Lorentzian broadening.

                See class doc-string for more info on these methods.
                [DEFAULT='cond']
            factor: Degree of broadening (broadening factor). [Default=1E-12]

        Returns:
            w: The eigenvalues, in ascending order.
            v: The eigenvectors.

        Notes:
            The ctx argument is auto-parsed by PyTorch & is used to pass data
            from the .forward() method to the .backward() method. This is not
            normally described in the docstring but has been done here to act
            as an example.

        Warnings:
            Under no circumstances should `factor` be a torch.tensor entity.
            The `method` and `factor` parameters MUST be passed as positional
            arguments and NOT keyword arguments.

        """
        # Check that the method is of a known type
        if method not in _SymEigB.KNOWN_METHODS:
            raise ValueError("Unknown broadening method selected.")

        # Compute eigen-values & vectors using torch.linalg.eigh.
        w, v = torch.linalg.eigh(a)

        # Save tensors that will be needed in the backward pass
        ctx.save_for_backward(w, v)

        # Save the broadening factor and the selected broadening method.
        ctx.bf, ctx.bm = factor, method

        # Store dtype/device to prevent dtype/device mixing
        ctx.dtype, ctx.device = a.dtype, a.device

        # Return the eigenvalues and eigenvectors
        return w, v

    @staticmethod
    def backward(ctx, w_bar: Tensor, v_bar: Tensor) -> Tuple[Tensor, Tensor]:
        """Evaluates gradients of the eigen decomposition operation.

        Evaluates gradients of the matrix from which the eigenvalues
        and eigenvectors were taken.

        Arguments:
            w_bar: Gradients associated with the the eigenvalues.
            v_bar: Gradients associated with the eigenvectors.

        Returns:
            a_bar: Gradients associated with the `a` tensor.

        Notes:
            See class doc-string for a more detailed description of this
            method.

        """
        # Equation to variable legend
        #   w <- λ
        #   v <- U

        # __Preamble__
        # Retrieve eigenvalues (w) and eigenvectors (v) from ctx
        w, v = ctx.saved_tensors

        # Retrieve, the broadening factor and convert to a tensor entity
        if not isinstance(ctx.bf, Tensor):
            bf = torch.tensor(ctx.bf, dtype=ctx.dtype, device=ctx.device)
        else:
            bf = ctx.bf

        # if bf is complex
        if bf.dtype in (torch.complex32, torch.complex64, torch.complex128):
            _bf = bf.real
        else:
            _bf = bf

        # Retrieve the broadening method
        bm = ctx.bm

        # Form the eigenvalue gradients into diagonal matrix
        lambda_bar = w_bar.diag_embed()

        # Identify the indices of the upper triangle of the F matrix
        tri_u = torch.triu_indices(*v.shape[-2:], 1)

        # Construct the deltas
        deltas = w[..., tri_u[1]] - w[..., tri_u[0]]

        # Apply broadening
        if bm == "cond":  # <- Conditional broadening
            deltas = (
                1
                / torch.where(torch.abs(deltas) > _bf, deltas, _bf)
                * torch.sign(deltas)
            )
        elif bm == "lorn":  # <- Lorentzian broadening
            deltas = deltas / (deltas**2 + bf)
        elif bm == "none":  # <- Debugging only
            deltas = 1 / deltas
        else:  # <- Should be impossible to get here
            raise ValueError(f"Unknown broadening method {bm}")

        # Construct F matrix where F_ij = v_bar_j - v_bar_i; construction is
        # done in this manner to avoid 1/0 which can cause intermittent and
        # hard-to-diagnose issues.
        F = torch.zeros(
            *w.shape, w.shape[-1], dtype=ctx.dtype, device=w_bar.device
        )
        # Upper then lower triangle
        # if bf is complex
        if bf.dtype in (torch.complex32, torch.complex64, torch.complex128):
            deltas = torch.tensor(deltas, dtype=bf.dtype)
        F[..., tri_u[0], tri_u[1]] = deltas
        F[..., tri_u[1], tri_u[0]] -= F[..., tri_u[0], tri_u[1]]

        # Construct the gradient following the equation in the doc-string.
        a_bar = (
            v
            @ (lambda_bar + sym(F * (v.transpose(-2, -1) @ v_bar)))
            @ v.transpose(-2, -1)
        )

        # Return the gradient. PyTorch expects a gradient for each parameter
        # (method, bf) hence two extra Nones are returned
        return a_bar, None, None


def merge(tensors: Sliceable, value: Any = 0, axis: int = 0) -> Tensor:
    """Merge two or more packed tensors into a single packed tensor.

    Arguments:
        tensors: Packed tensors which are to be merged.
        value: Value with which the tensor were/are to be padded. [DEFAULT=0]
        axis: Axis along which ``tensors`` are to be stacked. [DEFAULT=0]

    Returns:
        merged: The tensors ``tensors`` merged along the axis ``axis``.

    Warnings:
        Care must be taken to ensure the correct padding value is specified as
        erroneous behaviour may otherwise ensue. As the correct padding value
        cannot be reliably detected in situ it defaults to zero.
    """

    # Merging is performed along the 0'th axis internally. If a non-zero axis
    # is requested then tensors must be reshaped during input and output.
    if axis != 0:
        tensors = [t.transpose(0, axis) for t in tensors]

    # Tensor to merge into, filled with padding value.
    shapes = torch.tensor([i.shape for i in tensors])
    merged = torch.full(
        (shapes.sum(0)[0], *shapes.max(0).values[1:]),
        value,
        dtype=tensors[0].dtype,
        device=tensors[0].device,
    )

    n = 0  # <- batch dimension offset
    for src, size in zip(tensors, shapes):  # Assign values to tensor
        merged[(slice(n, size[0] + n), *[slice(0, s) for s in size[1:]])] = src
        n += size[0]

    # Return the merged tensor, transposing back as required
    return merged if axis == 0 else merged.transpose(0, axis)


def _eig_sort_out(
    w: Tensor, v: Tensor, ghost: bool = True
) -> Tuple[Tensor, Tensor]:
    """Move ghost eigen values/vectors to the end of the array.

    Discuss the difference between ghosts (w=0) and auxiliaries (w=1)

    Performing and eigen-decomposition operation on a zero-padded packed
    tensor results in the emergence of ghost eigen-values/vectors. This can
    cause issues downstream, thus they are moved to the end here which means
    they can be easily clipped off should the user wish to do so.

    Arguments:
        w: The eigen-values.
        v: The eigen-vectors.
        ghost: Ghost-eigen-vlaues are assumed to be 0 if True, else assumed to
            be 1. If zero padded then this should be True, if zero padding is
            turned into identity padding then False should be used. This will
            also change the ghost eigenvalues from 1 to zero when appropriate.
            [DEFAULT=True]

    Returns:
        w: The eigen-values, with ghosts moved to the end.
        v: The eigen-vectors, with ghosts moved to the end.

    """
    val = 0 if ghost else 1

    # Create a mask that is True when an eigen value is zero/one
    mask = torch.eq(w, val)
    # and its associated eigen vector is a column of a identity matrix:
    # i.e. all values are 1 or 0 and there is only a single 1. This will
    # just all zeros if columns are not one-hot.
    is_one = torch.eq(v, 1)  # <- precompute
    mask &= torch.all(torch.eq(v, 0) | is_one, dim=1)
    mask &= torch.sum(is_one, dim=1) <= 1  # <- Only a single "1" at most.

    # Convert any auxiliary eigenvalues into ghosts
    if not ghost:
        w = w - mask.type(w.dtype)

    # Pull out the indices of the true & ghost entries and cat them together
    # so that the ghost entries are at the end.
    # noinspection PyTypeChecker
    indices = torch.cat(
        (torch.stack(torch.where(~mask)), torch.stack(torch.where(mask))),
        dim=-1,
    )

    # argsort fixes the batch order and stops eigen-values accidentally being
    # mixed between different systems. As PyTorch's argsort is not stable, i.e.
    # it dose not respect any order already present in the data, numpy's argsort
    # must be used for now.
    ####sorter = np.argsort(indices[0].cpu(), kind="stable")
    sorter = torch.argsort(indices[0], stable=True)
    ####sorter = np.argsort(indices[0].cpu(), kind="stable")

    # Apply sorter to indices; use a tuple to make 1D & 2D cases compatible
    sorted_indices = tuple(indices[..., sorter])

    # Fix the order of the eigen values and eigen vectors.
    w = w[sorted_indices].reshape(w.shape)
    # Reshaping is needed to allow sorted_indices to be used for 2D & 3D
    v = v.transpose(-1, -2)[sorted_indices].reshape(v.shape).transpose(-1, -2)

    # Return the eigenvalues and eigenvectors
    return w, v


def eighb(
    a: Tensor,
    b: Tensor = None,
    scheme: Literal["chol", "lowd"] = "chol",
    broadening_method: Optional[Literal["cond", "lorn"]] = "cond",
    factor: float = 1e-12,
    sort_out: bool = True,
    aux: bool = True,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    r"""Solves general & standard eigen-problems, with optional broadening.

    Solves standard and generalised eigenvalue problems of the from Az = λBz
    for a real symmetric matrix ``a`` and can apply conditional or Lorentzian
    broadening to the eigenvalues during the backwards pass to increase
    gradient stability. Multiple  matrices may be passed in batch major form,
    i.e. the first axis iterates over entries.

    Arguments:
        a: Real symmetric matrix whose eigen-values/vectors will be computed.
        b: Complementary positive definite real symmetric matrix for the
            generalised eigenvalue problem.
        scheme: Scheme to covert generalised eigenvalue problems to standard
            ones:

                - "chol": Cholesky factorisation. [DEFAULT='chol']
                - "lowd": Löwdin orthogonalisation.

            Has no effect on solving standard problems.

        broadening_method: Broadening method to used:

                - "cond": conditional broadening. [DEFAULT='cond']
                - "lorn": Lorentzian broadening.
                - None: no broadening (uses torch.linalg.eigh).

        factor: The degree of broadening (broadening factor). [Default=1E-12]
        sort_out: If True; eigen-vector/value tensors are reordered so that
            any "ghost" entries are moved to the end. "Ghost" are values which
            emerge as a result of zero-padding. [DEFAULT=True]
        aux: Converts zero-padding to identity-padding. This this can improve
            the stability of backwards propagation. [DEFAULT=True]

    Keyword Args:
        direct_inv (bool): If True then the matrix inversion will be computed
            directly rather than via a call to torch.linalg..solve. Only
            relevant to the cholesky scheme. [DEFAULT=False]

    Returns:
        w: The eigenvalues, in ascending order.
        v: The eigenvectors.

    Notes:
        Results from backward passes through eigen-decomposition operations
        tend to suffer from numerical stability [*]_  issues when operating
        on systems with degenerate eigenvalues. Fortunately,  the stability
        of such operations can be increased through the application of eigen
        value broadening. However, such methods will induce small errors in
        the returned gradients as they effectively mutate  the eigen-values
        in the backwards pass. Thus, it is important to be aware that while
        increasing the extent of  broadening will help to improve stability
        it will also increase the error in the gradients.

        Two different broadening methods have been  implemented within this
        class. Conditional broadening as described by Seeger [MS2019]_, and
        Lorentzian as detailed by Liao [LH2019]_. During the forward pass the
        `torch.linalg.eigh` function is used to calculate both the eigenvalues &
        the eigenvectors (U & :math:`\lambda` respectively). The gradient
        is then calculated following:

        .. math:: \bar{A} = U (\bar{\Lambda} + sym(F \circ (U^t \bar{U}))) U^T

        Where bar indicates a value's gradient, passed in from the previous
        layer, :math:`\Lambda` is the diagonal matrix associated with the
        :math:`\bar{\lambda}` values, :math:`\circ`  is the so called Hadamard
        product, :math:`sym` is the symmetrisation operator and F is:

        .. math:: F_{i, j} = \frac{I_{i \ne j}}{h(\lambda_i - \lambda_j)}

        Where, for conditional broadening, h is:

        .. math:: h(t) = max(|t|, \epsilon)sgn(t)

        and for, Lorentzian broadening:

        .. math:: h(t) = \frac{t^2 + \epsilon}{t}

        The advantage of conditional broadening is that is is only  applied
        when it is needed, thus the errors induced in the gradients will be
        restricted to systems whose gradients would be nan's otherwise. The
        Lorentzian method, on the other hand, will apply broadening to all
        systems, irrespective of whether or not it is necessary. Note that if
        the h function is a unity operator then this is identical to a
        standard backwards pass through an eigen-solver.

        Mathematical discussions regarding the Cholesky decomposition are
        made with reference to the  "Generalized Symmetric Definite
        Eigenproblems" chapter of Lapack. [Lapack]_

        When operating in batch mode the zero valued padding columns and rows
        will result in the generation of "ghost" eigen-values/vectors. These
        are mostly harmless, but make it more difficult to extract the actual
        eigen-values/vectors. This function will move the "ghost" entities to
        the ends of their respective lists, making it easy to clip them out.

        .. [*] Where stability is defined as the propensity of a function to
               return nan values or some raise an error.

    Warnings:
        If operating upon zero-padded packed tensors then degenerate and  zero
        valued eigen values will be encountered. This will **always** cause an
        error during the backwards pass unless broadening is enacted.

        As ``torch.linalg.eigh`` sorts its results prior to returning them, it is
        likely that any "ghost" eigen-values/vectors, which result from zero-
        padded packing, will be located in the middle of the returned arrays.
        This makes down-stream processing more challenging. Thus, the sort_out
        option is enabled by default. This results in the "ghost" values being
        moved to the end. **However**, this method identifies any entry with a
        zero-valued eigenvalue and an eigenvector which can be interpreted as
        a column of an identity matrix as a ghost.

    References:
        .. [MS2019] Seeger, M., Hetzel, A., Dai, Z., & Meissner, E. Auto-
                    Differentiating Linear Algebra. ArXiv:1710.08717 [Cs,
                    Stat], Aug. 2019. arXiv.org,
                    http://arxiv.org/abs/1710.08717.
        .. [LH2019] Liao, H.-J., Liu, J.-G., Wang, L., & Xiang, T. (2019).
                    Differentiable Programming Tensor Networks. Physical
                    Review X, 9(3).
        .. [Lapack] www.netlib.org/lapack/lug/node54.html (Accessed 10/08/2020)

    """

    # Initial setup to make function calls easier to deal with
    # If smearing use _SymEigB otherwise use the internal torch.syeig function
    func = _SymEigB.apply if broadening_method else torch.linalg.eigh
    # Set up for the arguments
    args = (broadening_method, factor) if broadening_method else (True,)

    if aux:
        # Convert from zero-padding to identity padding
        is_zero = torch.eq(a, 0)
        mask = torch.all(is_zero, dim=-1) & torch.all(is_zero, dim=-2)
        a = a + torch.diag_embed(mask.type(a.dtype))

    if b is None:  # For standard eigenvalue problem
        w, v = func(a, *args)  # Call the required eigen-solver

    else:  # Otherwise it will be a general eigenvalue problem

        # Cholesky decomposition can only act on positive definite matrices;
        # which is problematic for zero-padded tensors. Similar issues are
        # encountered in the Löwdin scheme. To ensure positive definiteness
        # the diagonals of padding columns/rows are therefore set to 1.

        # Create a mask which is True wherever a column/row pair is 0-valued
        is_zero = torch.eq(b, 0)
        mask = torch.all(is_zero, dim=-1) & torch.all(is_zero, dim=-2)

        # Set the diagonals at these locations to 1
        b = b + torch.diag_embed(mask.type(a.dtype))

        # For Cholesky decomposition scheme
        if scheme == "chol":

            # Perform Cholesky factorization (A = LL^{T}) of B to attain L
            l = torch.linalg.cholesky(b)

            # Compute the inverse of L:
            if kwargs.get("direct_inv", False):
                # Via the direct method if specifically requested
                l_inv = torch.inverse(l)
            else:
                # Otherwise compute via an indirect method (default)
                # l_inv = torch.solve(torch.eye(a.shape[-1], dtype=a.dtype,
                #                               device=b.device), l)[0]
                l_inv = torch.linalg.solve(
                    l, torch.eye(a.shape[-1], dtype=a.dtype, device=b.device)
                )
            # Transpose of l_inv: improves speed in batch mode
            l_inv_t = torch.transpose(l_inv, -1, -2)

            # To obtain C, perform the reduction operation C = L^{-1}AL^{-T}
            if l_inv_t.dtype in (
                torch.complex32,
                torch.complex64,
                torch.complex128,
            ):
                l_inv_t = torch.conj(l_inv_t)

            c = l_inv @ a @ l_inv_t

            # The eigenvalues of Az = λBz are the same as Cy = λy; hence:
            w, v_ = func(c, *args)

            # Eigenvectors, however, are not, so they must be recovered:
            #   z = L^{-T}y
            v = l_inv_t @ v_

        elif scheme == "lowd":  # For Löwdin Orthogonalisation scheme

            # Perform the BV = WV eigen decomposition.
            w, v = func(b, *args)

            # Embed w to construct "small b"; inverse power is also done here
            # tvoid inf values later on.
            b_small = torch.diag_embed(w**-0.5).to(v.dtype)

            # Construct symmetric orthogonalisation matrix via:
            #   B^{-1/2} = V b^{-1/2} V^{T}
            b_so = v @ b_small @ v.transpose(-1, -2)

            # A' (a_prime) can then be constructed as: A' = B^{-1/2} A B^{-1/2}
            a_prime = b_so @ a @ b_so

            # Decompose the now orthogonalised A' matrix
            w, v_prime = func(a_prime, *args)

            # the correct eigenvector is then recovered via
            #   V = B^{-1/2} V'
            v = b_so @ v_prime

        else:  # If an unknown scheme was specified
            raise ValueError("Unknown scheme selected.")

    # If sort_out is enabled, then move ghosts to the end.
    if sort_out:
        w, v = _eig_sort_out(w, v, not aux)

    # Return the eigenvalues and eigenvectors
    return w, v


def sym(x: Tensor, dim0: int = -1, dim1: int = -2) -> Tensor:
    """Symmetries the specified tensor.

    Arguments:
        x: The tensor to be symmetrised.
        dim0: First dimension to be transposed. [DEFAULT=-1]
        dim1: Second dimension to be transposed [DEFAULT=-2]

    Returns:
        x_sym: The symmetrised tensor.

    """
    return (x + x.transpose(dim0, dim1)) / 2


def triangular_root(x: Union[Tensor, Real]) -> Union[Tensor, Real]:
    r"""Triangular root of ``x``.

    Calculates the triangular root of a given input:

    .. math:: n = \frac{\sqrt(8x + 1) - 1)}{2}

    Arguments:
        x: Value whose triangular root is to be calculated.

    Returns:
        n: Triangular root of ``x``.
    """
    return ((8 * x + 1) ** 0.5 - 1) / 2


def tetrahedral_root(x: Union[Tensor, Real]) -> Union[Tensor, Real]:
    r"""Tetrahedral root of ``x``.

    Calculates the tetrahedral root of a given input:

    .. math:: n = \sqrt[3]{3x+\sqrt{9{x^2}-\frac{1}{27}}} + \sqrt[3]{3x-\sqrt{9{x^2}-\frac{1}{27}}} -1

    Arguments:
        x: Value whose tetrahedral root is to be calculated.

    Returns:
        n: Tetrahedral root of ``x``.
    """
    a = (9 * x**2 - (1 / 27)) ** 0.5
    return (3 * x + a) ** (1 / 3) + (3 * x - a) ** (1 / 3) - 1
