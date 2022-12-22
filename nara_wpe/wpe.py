import functools
import operator

import click
import numpy as np


def get_working_shape(shape):
    "Flattens all but the last two dimension."
    product = functools.reduce(operator.mul, [1] + list(shape[:-2]))
    return [product] + list(shape[-2:])


def segment_axis(
        x,
        length,
        shift,
        axis=-1,
        end='cut',  # in ['pad', 'cut', None]
        pad_mode='constant',
        pad_value=0,
):

    """Generate a new array that chops the given array along the given axis
     into overlapping frames.

    Note: if end='pad' the return is maybe a copy

    Args:
        x: The array to segment
        length: The length of each frame
        shift: The number of array elements by which to step forward
               Negative values are also allowed.
        axis: The axis to operate on; if None, act on the flattened array
        end: What to do with the last frame, if the array is not evenly
                divisible into pieces. Options are:
                * 'cut'   Simply discard the extra values
                * None    No end treatment. Only works when fits perfectly.
                * 'pad'   Pad with a constant value
                * 'conv_pad' Special padding for convolution, assumes
                             shift == 1, see example below
        pad_mode: see numpy.pad
        pad_value: The value to use for end='pad'

    Examples:
        >>> # import cupy as np
        >>> segment_axis(np.arange(10), 4, 2)  # simple example
        array([[0, 1, 2, 3],
               [2, 3, 4, 5],
               [4, 5, 6, 7],
               [6, 7, 8, 9]])
        >>> segment_axis(np.arange(10), 4, -2)  # negative shift
        array([[6, 7, 8, 9],
               [4, 5, 6, 7],
               [2, 3, 4, 5],
               [0, 1, 2, 3]])
        >>> segment_axis(np.arange(5).reshape(5), 4, 1, axis=0)
        array([[0, 1, 2, 3],
               [1, 2, 3, 4]])
        >>> segment_axis(np.arange(5).reshape(5), 4, 2, axis=0, end='cut')
        array([[0, 1, 2, 3]])
        >>> segment_axis(np.arange(5).reshape(5), 4, 2, axis=0, end='pad')
        array([[0, 1, 2, 3],
               [2, 3, 4, 0]])
        >>> segment_axis(np.arange(5).reshape(5), 4, 1, axis=0, end='conv_pad')
        array([[0, 0, 0, 0],
               [0, 0, 0, 1],
               [0, 0, 1, 2],
               [0, 1, 2, 3],
               [1, 2, 3, 4],
               [2, 3, 4, 0],
               [3, 4, 0, 0],
               [4, 0, 0, 0]])
        >>> segment_axis(np.arange(6).reshape(6), 4, 2, axis=0, end='pad')
        array([[0, 1, 2, 3],
               [2, 3, 4, 5]])
        >>> segment_axis(np.arange(10).reshape(2, 5), 4, 1, axis=-1)
        array([[[0, 1, 2, 3],
                [1, 2, 3, 4]],
        <BLANKLINE>
               [[5, 6, 7, 8],
                [6, 7, 8, 9]]])
        >>> segment_axis(np.arange(10).reshape(5, 2).T, 4, 1, axis=1)
        array([[[0, 2, 4, 6],
                [2, 4, 6, 8]],
        <BLANKLINE>
               [[1, 3, 5, 7],
                [3, 5, 7, 9]]])
        >>> segment_axis(np.asfortranarray(np.arange(10).reshape(2, 5)),
        ...                 4, 1, axis=1)
        array([[[0, 1, 2, 3],
                [1, 2, 3, 4]],
        <BLANKLINE>
               [[5, 6, 7, 8],
                [6, 7, 8, 9]]])
        >>> segment_axis(np.arange(8).reshape(2, 2, 2).transpose(1, 2, 0),
        ...                 2, 1, axis=0, end='cut')
        array([[[[0, 4],
                 [1, 5]],
        <BLANKLINE>
                [[2, 6],
                 [3, 7]]]])
        >>> a = np.arange(7).reshape(7)
        >>> b = segment_axis(a, 4, -2, axis=0, end='cut')
        >>> a += 1  # a and b point to the same memory
        >>> b
        array([[3, 4, 5, 6],
               [1, 2, 3, 4]])

        >>> segment_axis(np.arange(7), 8, 1, axis=0, end='pad').shape
        (1, 8)
        >>> segment_axis(np.arange(8), 8, 1, axis=0, end='pad').shape
        (1, 8)
        >>> segment_axis(np.arange(9), 8, 1, axis=0, end='pad').shape
        (2, 8)
        >>> segment_axis(np.arange(7), 8, 2, axis=0, end='cut').shape
        (0, 8)
        >>> segment_axis(np.arange(8), 8, 2, axis=0, end='cut').shape
        (1, 8)
        >>> segment_axis(np.arange(9), 8, 2, axis=0, end='cut').shape
        (1, 8)

        >>> x = np.arange(1, 10)
        >>> filter_ = np.array([1, 2, 3])
        >>> np.convolve(x, filter_)
        array([ 1,  4, 10, 16, 22, 28, 34, 40, 46, 42, 27])
        >>> x_ = segment_axis(x, len(filter_), 1, end='conv_pad')
        >>> x_
        array([[0, 0, 1],
               [0, 1, 2],
               [1, 2, 3],
               [2, 3, 4],
               [3, 4, 5],
               [4, 5, 6],
               [5, 6, 7],
               [6, 7, 8],
               [7, 8, 9],
               [8, 9, 0],
               [9, 0, 0]])
        >>> x_ @ filter_[::-1]  # Equal to convolution
        array([ 1,  4, 10, 16, 22, 28, 34, 40, 46, 42, 27])

        >>> segment_axis(np.arange(19), 16, 4, axis=-1, end='pad')
        array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
               [ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,  0]])

        >>> import torch
        >>> segment_axis(torch.tensor(np.arange(10)), 4, 2)  # simple example
        tensor([[0, 1, 2, 3],
                [2, 3, 4, 5],
                [4, 5, 6, 7],
                [6, 7, 8, 9]])
        >>> segment_axis(torch.tensor(np.arange(10) + 1j), 4, 2)  # simple example
        tensor([[0.+1.j, 1.+1.j, 2.+1.j, 3.+1.j],
                [2.+1.j, 3.+1.j, 4.+1.j, 5.+1.j],
                [4.+1.j, 5.+1.j, 6.+1.j, 7.+1.j],
                [6.+1.j, 7.+1.j, 8.+1.j, 9.+1.j]], dtype=torch.complex128)
    """
    backend = {
        'numpy': 'numpy',
        'cupy.core.core': 'cupy',
        'torch': 'torch',
    }[x.__class__.__module__]

    if backend == 'numpy':
        xp = np
    elif backend == 'cupy':
        import cupy
        xp = cupy
    elif backend == 'torch':
        import torch
        xp = torch
    else:
        raise Exception('Can not happen')

    try:
        ndim = x.ndim
    except AttributeError:
        # For Pytorch 1.2 and below
        ndim = x.dim()

    axis = axis % ndim

    # Implement negative shift with a positive shift and a flip
    # stride_tricks does not work correct with negative stride
    if shift > 0:
        do_flip = False
    elif shift < 0:
        do_flip = True
        shift = abs(shift)
    else:
        raise ValueError(shift)

    if pad_mode == 'constant':
        pad_kwargs = {'constant_values': pad_value}
    else:
        pad_kwargs = {}

    # Pad
    if end == 'pad':
        if x.shape[axis] < length:
            npad = np.zeros([ndim, 2], dtype=int)
            npad[axis, 1] = length - x.shape[axis]
            x = xp.pad(x, pad_width=npad, mode=pad_mode, **pad_kwargs)
        elif shift != 1 and (x.shape[axis] + shift - length) % shift != 0:
            npad = np.zeros([ndim, 2], dtype=int)
            npad[axis, 1] = shift - ((x.shape[axis] + shift - length) % shift)
            x = xp.pad(x, pad_width=npad, mode=pad_mode, **pad_kwargs)

    elif end == 'conv_pad':
        assert shift == 1, shift
        npad = np.zeros([ndim, 2], dtype=int)
        npad[axis, :] = length - shift
        x = xp.pad(x, pad_width=npad, mode=pad_mode, **pad_kwargs)
    elif end is None:
        assert (x.shape[axis] + shift - length) % shift == 0, \
            '{} = x.shape[axis]({}) + shift({}) - length({})) % shift({})' \
            ''.format((x.shape[axis] + shift - length) % shift,
                      x.shape[axis], shift, length, shift)
    elif end == 'cut':
        pass
    else:
        raise ValueError(end)

    # Calculate desired shape and strides
    shape = list(x.shape)
    # assert shape[axis] >= length, shape
    del shape[axis]
    shape.insert(axis, (x.shape[axis] + shift - length) // shift)
    shape.insert(axis + 1, length)

    def get_strides(array):
        try:
            return list(array.strides)
        except AttributeError:
            # fallback for torch
            return list(array.stride())

    strides = get_strides(x)
    strides.insert(axis, shift * strides[axis])

    # Alternative to np.ndarray.__new__
    # I am not sure if np.lib.stride_tricks.as_strided is better.
    # return np.lib.stride_tricks.as_strided(
    #     x, shape=shape, strides=strides)
    try:
        if backend == 'numpy':
            x = np.lib.stride_tricks.as_strided(x, strides=strides, shape=shape)
        elif backend == 'cupy':
            x = x.view()
            x._set_shape_and_strides(strides=strides, shape=shape)
        elif backend == 'torch':
            import torch
            x = torch.as_strided(x, size=shape, stride=strides)
        else:
            raise Exception('Can not happen')

        # return np.ndarray.__new__(np.ndarray, strides=strides,
        #                           shape=shape, buffer=x, dtype=x.dtype)
    except Exception:
        print('strides:', get_strides(x), ' -> ', strides)
        print('shape:', x.shape, ' -> ', shape)
        try:
            print('flags:', x.flags)
        except AttributeError:
            pass  # for pytorch
        print('Parameters:')
        print('shift:', shift, 'Note: negative shift is implemented with a '
                               'following flip')
        print('length:', length, '<- Has to be positive.')
        raise
    if do_flip:
        return xp.flip(x, axis=axis)
    else:
        return x

def _lstsq(A, B):
    assert A.shape == B.shape, (A.shape, B.shape)
    shape = A.shape

    working_shape = get_working_shape(shape)

    A = A.reshape(working_shape)
    B = B.reshape(working_shape)

    C = np.zeros_like(A)
    for i in range(working_shape[0]):
        C[i] = np.linalg.lstsq(A[i], B[i])[0]
    return C.reshape(*shape)


def _stable_solve(A, B):
    """
    Use np.linalg.solve with fallback to np.linalg.lstsq.
    Equal to np.linalg.lstsq but faster.

    Note: limited currently by A.shape == B.shape

    This function try's np.linalg.solve with independent dimensions,
    when this is not working the function fall back to np.linalg.solve
    for each matrix. If one matrix does not work it fall back to
    np.linalg.lstsq.

    The reason for not using np.linalg.lstsq directly is the execution time.
    Examples:
    A and B have the shape (500, 6, 6), than a loop over lstsq takes
    108 ms and this function 28 ms for the case that one matrix is singular
    else 1 ms.

    >>> def normal(shape):
    ...     return np.random.normal(size=shape) + 1j * np.random.normal(size=shape)

    >>> A = normal((6, 6))
    >>> B = normal((6, 6))
    >>> C1 = np.linalg.solve(A, B)
    >>> C2, *_ = np.linalg.lstsq(A, B)
    >>> C3 = _stable_solve(A, B)
    >>> C4 = _lstsq(A, B)
    >>> np.testing.assert_allclose(C1, C2)
    >>> np.testing.assert_allclose(C1, C3)
    >>> np.testing.assert_allclose(C1, C4)

    >>> A = np.zeros((6, 6), dtype=np.complex128)
    >>> B = np.zeros((6, 6), dtype=np.complex128)
    >>> C1 = np.linalg.solve(A, B)
    Traceback (most recent call last):
    ...
    numpy.linalg.LinAlgError: Singular matrix
    >>> C2, *_ = np.linalg.lstsq(A, B)
    >>> C3 = _stable_solve(A, B)
    >>> C4 = _lstsq(A, B)
    >>> np.testing.assert_allclose(C2, C3)
    >>> np.testing.assert_allclose(C2, C4)

    >>> A = normal((3, 6, 6))
    >>> B = normal((3, 6, 6))
    >>> C1 = np.linalg.solve(A, B)
    >>> C2, *_ = np.linalg.lstsq(A, B)
    Traceback (most recent call last):
    ...
    numpy.linalg.LinAlgError: 3-dimensional array given. Array must be two-dimensional
    >>> C3 = _stable_solve(A, B)
    >>> C4 = _lstsq(A, B)
    >>> np.testing.assert_allclose(C1, C3)
    >>> np.testing.assert_allclose(C1, C4)


    >>> A[2, 3, :] = 0
    >>> C1 = np.linalg.solve(A, B)
    Traceback (most recent call last):
    ...
    numpy.linalg.LinAlgError: Singular matrix
    >>> C2, *_ = np.linalg.lstsq(A, B)
    Traceback (most recent call last):
    ...
    numpy.linalg.LinAlgError: 3-dimensional array given. Array must be two-dimensional
    >>> C3 = _stable_solve(A, B)
    >>> C4 = _lstsq(A, B)
    >>> np.testing.assert_allclose(C3, C4)


    """
    assert A.shape[:-2] == B.shape[:-2], (A.shape, B.shape)
    assert A.shape[-1] == B.shape[-2], (A.shape, B.shape)
    try:
        return np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        shape_A, shape_B = A.shape, B.shape
        assert shape_A[:-2] == shape_A[:-2]
        working_shape_A = get_working_shape(shape_A)
        working_shape_B = get_working_shape(shape_B)
        A = A.reshape(working_shape_A)
        B = B.reshape(working_shape_B)

        C = np.zeros_like(B)
        for i in range(working_shape_A[0]):
            # lstsq is much slower, use it only when necessary
            try:
                C[i] = np.linalg.solve(A[i], B[i])
            except np.linalg.LinAlgError:
                C[i] = np.linalg.lstsq(A[i], B[i])[0]
        return C.reshape(*shape_B)


def build_y_tilde(Y, taps, delay):
    """

    Note: The returned y_tilde consumes a similar amount of memory as Y, because
        of tricks with strides. Usually the memory consumprion is K times
        smaller than the memory consumprion of a contignous array,

    >>> T, D = 20, 2
    >>> Y = np.arange(start=1, stop=T * D + 1).reshape([T, D]).T
    >>> print(Y)
    [[ 1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39]
     [ 2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40]]
    >>> taps, delay = 4, 2
    >>> Y_tilde = build_y_tilde(Y, taps, delay)
    >>> print(Y_tilde.shape, (taps*D, T))
    (8, 20) (8, 20)
    >>> print(Y_tilde)
    [[ 0  0  1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33 35]
     [ 0  0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34 36]
     [ 0  0  0  1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33]
     [ 0  0  0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34]
     [ 0  0  0  0  1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31]
     [ 0  0  0  0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32]
     [ 0  0  0  0  0  1  3  5  7  9 11 13 15 17 19 21 23 25 27 29]
     [ 0  0  0  0  0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30]]
    >>> Y_tilde = build_y_tilde(Y, taps, 0)
    >>> print(Y_tilde.shape, (taps*D, T), Y_tilde.strides)
    (8, 20) (8, 20) (-8, 16)
    >>> print('Pseudo size:', Y_tilde.nbytes)
    Pseudo size: 1280
    >>> print('Reak size:', Y_tilde.base.base.base.base.nbytes)
    Reak size: 368
    >>> print(Y_tilde)
    [[ 1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39]
     [ 2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40]
     [ 0  1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33 35 37]
     [ 0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38]
     [ 0  0  1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33 35]
     [ 0  0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34 36]
     [ 0  0  0  1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33]
     [ 0  0  0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34]]

    The first columns are zero because of the delay.

    """
    S = Y.shape[:-2]
    D = Y.shape[-2]
    T = Y.shape[-1]

    def pad(x, axis=-1, pad_width=taps + delay - 1):
        npad = np.zeros([x.ndim, 2], dtype=int)
        npad[axis, 0] = pad_width
        x = np.pad(x,
                   pad_width=npad,
                   mode='constant',
                   constant_values=0)
        return x

    # Y_ = segment_axis(pad(Y), K, 1, axis=-1)
    # Y_ = np.flip(Y_, axis=-1)
    # if delay > 0:
    #     Y_ = Y_[..., :-delay, :]
    # # Y_: ... x D x T x K
    # Y_ = np.moveaxis(Y_, -1, -3)
    # # Y_: ... x K x D x T
    # Y_ = np.reshape(Y_, [*S, K * D, T])
    # # Y_: ... x KD x T

    # ToDo: write the shape
    Y_ = pad(Y)
    Y_ = np.moveaxis(Y_, -1, -2)
    Y_ = np.flip(Y_, axis=-1)
    Y_ = np.ascontiguousarray(Y_)
    Y_ = np.flip(Y_, axis=-1)
    Y_ = segment_axis(Y_, taps, 1, axis=-2)
    Y_ = np.flip(Y_, axis=-2)
    if delay > 0:
        Y_ = Y_[..., :-delay, :, :]
    Y_ = np.reshape(Y_, list(S) + [T, taps * D])
    Y_ = np.moveaxis(Y_, -2, -1)

    return Y_


def hermite(x):
    return x.swapaxes(-2, -1).conj()


def wpe_v0(Y, taps=10, delay=3, iterations=3, psd_context=0, statistics_mode='full'):
    """
    Closest implementation to
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6255769 but rather
    slow.
    Args:
        Y: Complex valued STFT signal with shape (F, D, T) or (D, T).
        taps: Filter order
        delay: Delay as a guard interval, such that X does not become zero.
        iterations:
        psd_context: Defines the number of elements in the time window
            to improve the power estimation. Total number of elements will
            be (psd_context + 1 + psd_context).
        statistics_mode: Either 'full' or 'valid'.
            'full': Pad the observation with zeros on the left for the
            estimation of the correlation matrix and vector.
            'valid': Only calculate correlation matrix and vector on valid
            slices of the observation.


    Returns:
        Estimated signal with the same shape as Y

    """
    if statistics_mode == 'full':
        s = Ellipsis
    elif statistics_mode == 'valid':
        s = (Ellipsis, slice(delay + taps - 1, None))
    else:
        raise ValueError(statistics_mode)

    X = np.copy(Y)
    if Y.ndim == 2:
        for iteration in range(iterations):
            inverse_power = get_power_inverse(X, psd_context=psd_context)
            filter_matrix_conj = get_filter_matrix_conj_v5(
                Y[s], inverse_power[s], taps, delay
            )
            X = perform_filter_operation_v4(Y, filter_matrix_conj, taps, delay)
    elif Y.ndim == 3:
        F = Y.shape[0]
        for f in range(F):
            X[f, :, :] = wpe_v0(
                Y[f, :, :],
                taps=taps,
                delay=delay,
                iterations=iterations,
                psd_context=psd_context
            )
    else:
        raise NotImplementedError('Input shape is to be (F, D, T) or (D, T).')
    return X


def wpe_v6(Y, taps=10, delay=3, iterations=3, psd_context=0, statistics_mode='full'):
    """
    Batched WPE implementation.

    Short of wpe_v7 with no extern references.
    Applicable in for-loops.

    Args:
        Y: Complex valued STFT signal with shape (..., D, T).
        taps: Filter order
        delay: Delay as a guard interval, such that X does not become zero.
        iterations:
        psd_context: Defines the number of elements in the time window
            to improve the power estimation. Total number of elements will
            be (psd_context + 1 + psd_context).
        statistics_mode: Either 'full' or 'valid'.
            'full': Pad the observation with zeros on the left for the
            estimation of the correlation matrix and vector.
            'valid': Only calculate correlation matrix and vector on valid
            slices of the observation.

    Returns:
        Estimated signal with the same shape as Y

    """

    if statistics_mode == 'full':
        s = Ellipsis
    elif statistics_mode == 'valid':
        s = (Ellipsis, slice(delay + taps - 1, None))
    else:
        raise ValueError(statistics_mode)

    X = np.copy(Y)
    Y_tilde = build_y_tilde(Y, taps, delay)
    for iteration in range(iterations):
        inverse_power = get_power_inverse(X, psd_context=psd_context)
        Y_tilde_inverse_power = Y_tilde * inverse_power[..., None, :]
        R = np.matmul(Y_tilde_inverse_power[s], hermite(Y_tilde[s]))
        P = np.matmul(Y_tilde_inverse_power[s], hermite(Y[s]))
        G = _stable_solve(R, P)
        X = Y - np.matmul(hermite(G), Y_tilde)

    return X


def wpe_v7(Y, taps=10, delay=3, iterations=3, psd_context=0, statistics_mode='full'):
    """
    Batched and modular WPE version.

    Args:
        Y: Complex valued STFT signal with shape (..., D, T).
        taps: Filter order
        delay: Delay as a guard interval, such that X does not become zero.
        iterations:
        psd_context: Defines the number of elements in the time window
            to improve the power estimation. Total number of elements will
            be (psd_context + 1 + psd_context).
        statistics_mode: Either 'full' or 'valid'.
            'full': Pad the observation with zeros on the left for the
            estimation of the correlation matrix and vector.
            'valid': Only calculate correlation matrix and vector on valid
            slices of the observation.

    Returns:
        Estimated signal with the same shape as Y
    """
    X = Y
    Y_tilde = build_y_tilde(Y, taps, delay)

    if statistics_mode == 'full':
        s = Ellipsis
    elif statistics_mode == 'valid':
        s = (Ellipsis, slice(delay + taps - 1, None))
    else:
        raise ValueError(statistics_mode)

    for iteration in range(iterations):
        inverse_power = get_power_inverse(X, psd_context=psd_context)
        G = get_filter_matrix_v7(Y=Y[s], Y_tilde=Y_tilde[s], inverse_power=inverse_power[s])
        X = perform_filter_operation_v5(Y=Y, Y_tilde=Y_tilde, filter_matrix=G)
    return X


def wpe_v8(
        Y,
        taps=10,
        delay=3,
        iterations=3,
        psd_context=0,
        statistics_mode='full',
        inplace=False
):
    """
    Loopy Multiple Input Multiple Output Weighted Prediction Error [1, 2] implementation
    
    In many cases this implementation is the fastest Numpy implementation.
    It loops over the independent axes. This reduces the memory footprint
    and in experiments it turned out that this is faster than a batched 
    implementation (i.e. `wpe_v6` or `wpe_v7`) due to memory locality.

    Args:
        Y: Complex valued STFT signal with shape (..., D, T).
        taps: Filter order
        delay: Delay as a guard interval, such that X does not become zero.
        iterations:
        psd_context: Defines the number of elements in the time window
            to improve the power estimation. Total number of elements will
            be (psd_context + 1 + psd_context).
        statistics_mode: Either 'full' or 'valid'.
            'full': Pad the observation with zeros on the left for the
            estimation of the correlation matrix and vector.
            'valid': Only calculate correlation matrix and vector on valid
            slices of the observation.
        inplace: Whether to change Y inplace. Has only advantages, when Y has
            independent axes, because the core WPE algorithm does not support
            an inplace modification of the observation.
            This option may be relevant, when Y is so large, that you do not
            want to double the memory consumption (i.e. save Y and the
            dereverberated signal in the memory).

    Returns:
        Estimated signal with the same shape as Y

    [1] "Generalization of multi-channel linear prediction methods for blind MIMO
        impulse response shortening", Yoshioka, Takuya and Nakatani, Tomohiro, 2012
    [2] NARA-WPE: A Python package for weighted prediction error dereverberation in
        Numpy and Tensorflow for online and offline processing, Drude, Lukas and
        Heymann, Jahn and Boeddeker, Christoph and Haeb-Umbach, Reinhold, 2018

    """
    ndim = Y.ndim
    if ndim == 2:
        out = wpe_v6(
            Y,
            taps=taps,
            delay=delay,
            iterations=iterations,
            psd_context=psd_context,
            statistics_mode=statistics_mode
        )
        if inplace:
            Y[...] = out
        return out
    elif ndim >= 3:
        if inplace:
            out = Y
        else:
            out = np.empty_like(Y)

        for index in np.ndindex(Y.shape[:-2]):
            out[index] = wpe_v6(
                Y=Y[index],
                taps=taps,
                delay=delay,
                iterations=iterations,
                psd_context=psd_context,
                statistics_mode=statistics_mode,
            )
        return out
    else:
        raise NotImplementedError(
            'Input shape has to be (..., D, T) and not {}.'.format(Y.shape)
        )


wpe = wpe_v7


def online_wpe_step(
        input_buffer, power_estimate, inv_cov, filter_taps,
        alpha, taps, delay
    ):
    """
    One step of online dereverberation.

    Args:
        input_buffer: Buffer of shape (taps+delay+1, F, D)
        power_estimate: Estimate for the current PSD
        inv_cov: Current estimate of R^-1
        filter_taps: Current estimate of filter taps (F, taps*D, D)
        alpha (float): Smoothing factor
        taps (int): Number of filter taps
        delay (int): Delay in frames

    Returns:
        Dereverberated frame of shape (F, D)
        Updated estimate of R^-1
        Updated estimate of the filter taps
    """

    F, D = input_buffer.shape[-2:]
    window = input_buffer[:-delay - 1][::-1]
    window = window.transpose(1, 2, 0).reshape((F, taps * D))
    pred = (
        input_buffer[-1] -
        np.einsum('fid,fi->fd', np.conjugate(filter_taps), window)
    )
    nominator = np.einsum('fij,fj->fi', inv_cov, window)
    denominator = (alpha * power_estimate).astype(window.dtype)
    denominator += np.einsum('fi,fi->f', np.conjugate(window), nominator)
    kalman_gain = nominator * _stable_positive_inverse(denominator)[:, None]

    inv_cov_k = inv_cov - np.einsum(
        'fj,fjm,fi->fim',
        np.conjugate(window),
        inv_cov,
        kalman_gain,
        optimize='optimal'
    )
    inv_cov_k /= alpha

    filter_taps_k = (
        filter_taps +
        np.einsum('fi,fm->fim', kalman_gain, np.conjugate(pred))
    )

    return pred, inv_cov_k, filter_taps_k


class OnlineWPE:
    """
    A recursive approach which carries the covariance matrices
    as well as the filter taps and the power estimate.
    The online step is a special case for online framewise
    dereverberation.

    Args:
        taps (int): Number of filter taps
        delay (int): Delay in frames
        alpha (float): Smoothing factor, 0.9999
        power_estimate: Estimate of power as an initialization
                        (frequency_bins,)

    Returns:
        Dereverberated frame of shape (F, D)
    """

    def __init__(self, taps, delay, alpha, power_estimate=None, channel=8,
                 frequency_bins=257):
        self.alpha = alpha
        self.taps = taps
        self.delay = delay

        self.inv_cov = np.stack(
            [np.eye(channel * taps) for _ in range(frequency_bins)]
        )
        self.filter_taps = np.zeros((frequency_bins, channel * taps, channel))
        if power_estimate is not None:
            assert frequency_bins == power_estimate.shape[0], \
                "({},) =! {}".format(frequency_bins, power_estimate.shape)
            self.power = np.ones(frequency_bins) * power_estimate
        self.buffer = np.zeros(
            (self.taps + self.delay + 1, frequency_bins, channel),
            dtype=np.complex128
        )

    def step_frame(self, frame):
        """
        Online WPE in framewise fashion.
        Args:
            frame: (F, D)
        Returns:
            prediction: (F, D)
        """
        assert self.buffer.shape[-2:] == frame.shape[-2:],\
            "Set channel and frequency bins."

        prediction, window = self._get_prediction(frame)

        self._update_buffer(frame)
        self._update_power_block()
        self._update_kalman_gain(window)
        self._update_inv_cov(window)
        self._update_taps(prediction)

        return prediction

    def step_block(self, block, block_shift=1):
        """
        Online WPE in blockwise fashion.
        Args:
            block: (taps+delay+1, F, D)
            block_shift:
        Returns:
            prediction: (F, D)
        """
        assert self.buffer.shape[-2:] == block.shape[-2:],\
            "Set channel and frequency bins."
        assert self.buffer.shape[0] == block.shape[0],\
            "Check block length. ({}+{}+1, F, D)".format(self.taps, self.delay)

        prediction, window = self._get_prediction(block, block_shift)

        self._update_buffer(block)
        self._update_power_block()
        self._update_kalman_gain(window)
        self._update_inv_cov(window)
        self._update_taps(prediction)

        return prediction

    def _get_prediction(self, observation, block_shift=1):
        #TODO: Only block shift of 1 works.
        F, D = observation.shape[-2:]
        window = self.buffer[:-self.delay - 1]
        window = window.transpose(1, 2, 0).reshape((F, self.taps * D))
        if observation.ndim == 2:
            observation = observation[None, ...]
        prediction = (
            observation[-block_shift] -
            np.einsum('fid,fi->fd', np.conjugate(self.filter_taps), window)
        )
        return prediction, window

    def _update_taps(self, prediction):
        self.filter_taps = (
            self.filter_taps +
            np.einsum('fi,fm->fim', self.kalman_gain, np.conjugate(prediction))
        )

    def _update_inv_cov(self, window):
        self.inv_cov = self.inv_cov - np.einsum(
            'fj,fjm,fi->fim',
            np.conjugate(window),
            self.inv_cov,
            self.kalman_gain,
            optimize='optimal'
        )
        self.inv_cov /= self.alpha

    def _update_kalman_gain(self, window):
        nominator = np.einsum('fij,fj->fi', self.inv_cov, window)
        denominator = (self.alpha * self.power).astype(window.dtype)
        denominator += np.einsum('fi,fi->f', np.conjugate(window), nominator)
        self.kalman_gain = nominator * _stable_positive_inverse(denominator)[:, None]

    def _update_power_block(self):
        self.power = np.mean(
            get_power(self.buffer.transpose(1, 2, 0)),
            -1
        )

    def _update_buffer(self, update):
        if update.ndim == 2:
            self.buffer = np.roll(self.buffer, -1, axis=0)
            assert update.shape[-2:] == self.buffer.shape[-2:]
            self.buffer[-1] = update
        elif update.ndim == 3:
            assert self.buffer.shape == update.shape, 'Shape inconsistent.'
            self.buffer = update


def abs_square(x):
    """

    Params:
        x: np.ndarray

    https://github.com/numpy/numpy/issues/9679

    Bug in numpy 1.13.1
    >> np.ones(32768).imag ** 2
    Traceback (most recent call last):
    ...
    ValueError: output array is read-only
    >> np.ones(32767).imag ** 2
    array([ 0.,  0.,  0., ...,  0.,  0.,  0.])

    >>> abs_square(np.ones(32768)).shape
    (32768,)
    >>> abs_square(np.ones(32768, dtype=np.complex64)).shape
    (32768,)
    """

    if np.iscomplexobj(x):
        return x.real ** 2 + x.imag ** 2
    else:
        return x ** 2


def window_mean_slow(x, lr_context):
    """
    Does not support axis because of np.convolve.

    >>> window_mean_slow([1, 1, 1, 1, 1], 1)
    array([1., 1., 1., 1., 1.])
    >>> window_mean_slow([1, 2, 3, 4, 5], 1)
    array([1.5, 2. , 3. , 4. , 4.5])
    >>> x = [1, 1, 13, 1, 1]
    >>> np.testing.assert_equal(window_mean_slow(x, (0, 1)), [1, 7, 7, 1, 1])
    >>> np.testing.assert_equal(window_mean_slow(x, (1, 0)), [1, 1, 7, 7, 1])
    >>> np.testing.assert_equal(window_mean_slow(x, (0, 2)), [5, 5, 5, 1, 1])
    >>> np.testing.assert_equal(window_mean_slow(x, (2, 0)), [1, 1, 5, 5, 5])
    >>> np.testing.assert_equal(window_mean_slow(x, (1, 2)), [5, 4, 4, 5, 1])
    >>> np.testing.assert_equal(window_mean_slow(x, (2, 1)), [1, 5, 4, 4, 5])
    """
    # axis = -1
    if isinstance(lr_context, int):
        lr_context = [lr_context, lr_context]
    assert len(lr_context) == 2, lr_context
    l_context, r_context = lr_context

    x = np.asarray(x)
    filter = np.ones(np.sum(lr_context) + 1)
    conv = np.convolve(x, filter, mode='full')  # ToDo: axis
    count = np.convolve(np.ones(x.shape, np.int64), filter, mode='full')  # ToDo: axis

    s = slice(r_context, conv.shape[-1] - l_context)  # ToDo: axis
    return conv[s] / count[s]


def window_mean(x, lr_context, axis=-1):
    """
    Take the mean of x at each index with a left and right context.
    Pseudo code for lr_context == (1, 1):
        y = np.zeros(...)
        for i in range(...):
            if not edge_case(i):
                y[i] = (x[i - 1] + x[i] + x[i + 1]) / 3
            elif i == 0:
                y[i] = (x[i] + x[i + 1]) / 2
            else:
                y[i] = (x[i - 1] + x[i]) / 2
        return y

    >>> window_mean([1, 1, 1, 1, 1], 1)
    array([1., 1., 1., 1., 1.])
    >>> window_mean([1, 2, 3, 4, 5], 1)
    array([1.5, 2. , 3. , 4. , 4.5])
    >>> x = [1, 1, 13, 1, 1]
    >>> np.testing.assert_equal(window_mean(x, (0, 1)), [1, 7, 7, 1, 1])
    >>> np.testing.assert_equal(window_mean(x, (1, 0)), [1, 1, 7, 7, 1])
    >>> np.testing.assert_equal(window_mean(x, (0, 2)), [5, 5, 5, 1, 1])
    >>> np.testing.assert_equal(window_mean(x, (2, 0)), [1, 1, 5, 5, 5])
    >>> np.testing.assert_equal(window_mean(x, (1, 2)), [5, 4, 4, 5, 1])
    >>> np.testing.assert_equal(window_mean(x, (2, 1)), [1, 5, 4, 4, 5])
    >>> np.testing.assert_equal(window_mean(x, (9, 9)), [3.4] * 5)

    >>> x = np.random.normal(size=(20, 50))
    >>> lr_context = np.random.randint(0, 5, size=2)
    >>> a = window_mean(x, lr_context, axis=1)
    >>> b = window_mean(x, lr_context, axis=-1)
    >>> c = window_mean(x.T, lr_context, axis=0).T
    >>> d = [window_mean_slow(s, lr_context) for s in x]
    >>> np.testing.assert_equal(a, b)
    >>> np.testing.assert_equal(a, c)
    >>> np.testing.assert_almost_equal(a, d)

    >>> import bottleneck as bn
    >>> a = window_mean(x, [lr_context[0], 0], axis=-1)
    >>> b = bn.move_mean(x, lr_context[0] + 1, min_count=1)
    >>> np.testing.assert_almost_equal(a, b)

    >>> a = window_mean(x, [lr_context[0], 0], axis=0)
    >>> b = bn.move_mean(x, lr_context[0] + 1, min_count=1, axis=0)
    >>> np.testing.assert_almost_equal(a, b)

    """
    if isinstance(lr_context, int):
        lr_context = [lr_context + 1, lr_context]
    else:
        assert len(lr_context) == 2, lr_context
        tmp_l_context, tmp_r_context = lr_context
        lr_context = tmp_l_context + 1, tmp_r_context

    x = np.asarray(x)

    window_length = sum(lr_context)
    if window_length == 0:
        return x

    pad_width = np.zeros((x.ndim, 2), dtype=np.int64)
    pad_width[axis] = lr_context

    first_slice = [slice(None)] * x.ndim
    first_slice[axis] = slice(sum(lr_context), None)
    second_slice = [slice(None)] * x.ndim
    second_slice[axis] = slice(None, -sum(lr_context))

    def foo(x):
        cumsum = np.cumsum(np.pad(x, pad_width, mode='constant'), axis=axis)
        return cumsum[tuple(first_slice)] - cumsum[tuple(second_slice)]

    ones_shape = [1] * x.ndim
    ones_shape[axis] = x.shape[axis]

    return foo(x) / foo(np.ones(ones_shape, np.int64))


def get_power_online(signal):
    """

    Args:
        signal : Signal with shape (F, D, T).
    Returns:
        Inverse power with shape (F,)

    """
    power_estimate = get_power(signal)
    power_estimate = np.mean(power_estimate, -1)
    return power_estimate


def get_power(signal, psd_context=0):
    """

    In case psd_context is an tuple with length 2,
    the two values describe the left and right hand context.

    Args:
        signal: (F, D, T) or (D, T)
        psd_context: tuple or int

    """
    if len(signal.shape) == 2:
        signal = signal[None, ...]

    power = np.mean(abs_square(signal), axis=-2)

    if psd_context != 0:
        if isinstance(psd_context, tuple):
            context = psd_context[0] + 1 + psd_context[1]
        else:
            assert int(psd_context) == psd_context, psd_context
            context = int(2 * psd_context + 1)
            psd_context = (psd_context, psd_context)

        power = np.apply_along_axis(
            np.correlate,
            0,
            power,
            np.ones(context),
            mode='full'
        )[psd_context[1]:-psd_context[0]]

        denom = np.apply_along_axis(
            np.correlate,
            0,
            np.zeros_like(power) + 1,
            np.ones(context),
            mode='full'
        )[psd_context[1]:-psd_context[0]]

        power /= denom

    elif psd_context == 0:
        pass
    else:
        raise ValueError(psd_context)
    return np.squeeze(power)


def _stable_positive_inverse(power):
    """
    Calculate the inverse of a positive value.
    """
    eps = 1e-10 * np.max(power)
    if eps == 0:
        # Special case when signal is zero.
        # Does not happen on real data.
        # This only happens in artificial cases, e.g. redacted signal parts,
        # where the signal is set to be zero from a human.
        #
        # The scale of the power does not matter, so take 1.
        inverse_power = np.ones_like(power)
    else:
        inverse_power = 1 / np.maximum(power, eps)
    return inverse_power


def get_power_inverse(signal, psd_context=0):
    """
    Assumes single frequency bin with shape (D, T).

    >>> s = 1 / np.array([np.arange(1, 6)]*3)
    >>> get_power_inverse(s)
    array([ 1.,  4.,  9., 16., 25.])
    >>> get_power_inverse(s * 0 + 1, 1)
    array([1., 1., 1., 1., 1.])
    >>> get_power_inverse(s, 1)
    array([ 1.6       ,  2.20408163,  7.08196721, 14.04421326, 19.51219512])
    >>> get_power_inverse(s, np.inf)
    array([3.41620801, 3.41620801, 3.41620801, 3.41620801, 3.41620801])
    >>> get_power_inverse(s * 0.)
    array([1., 1., 1., 1., 1.])
    """
    power = np.mean(abs_square(signal), axis=-2)

    if np.isposinf(psd_context):
        power = np.broadcast_to(np.mean(power, axis=-1, keepdims=True), power.shape)
    elif psd_context > 0:
        assert int(psd_context) == psd_context, psd_context
        psd_context = int(psd_context)
        # import bottleneck as bn
        # Handle the corner case correctly (i.e. sum() / count)
        # Use bottleneck when only left context is requested
        # power = bn.move_mean(power, psd_context*2+1, min_count=1)
        power = window_mean(power, (psd_context, psd_context))
    elif psd_context == 0:
        pass
    else:
        raise ValueError(psd_context)
    return _stable_positive_inverse(power)


def get_Psi(Y, t, taps):
    """
    Psi from https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6255769
    equation 31
    """
    D = Y.shape[0]

    def get_Y_tilde(t_):
        return np.kron(np.eye(D), Y[:, t_]).T

    assert t - taps + 1 >= 0
    return np.concatenate([get_Y_tilde(t_) for t_ in range(t, t - taps, -1)])


def get_correlations(Y, inverse_power, taps, delay):
    D, T = Y.shape

    correlation_matrix = np.zeros((D * D * taps, D * D * taps), dtype=Y.dtype)
    correlation_vector = np.zeros((D * D * taps, 1), dtype=Y.dtype)
    for t in range(delay + taps - 1, T):
        Psi = get_Psi(Y, t - delay, taps)
        correlation_matrix += inverse_power[t] * np.matmul(Psi.conj(), Psi.T)
        correlation_vector \
            += inverse_power[t] * np.matmul(Psi.conj(), Y[:, t])[:, None]

    return correlation_matrix, correlation_vector


def get_Psi_narrow(Y, t, taps):
    assert t - taps + 1 >= 0
    selector = slice(t, t - taps if t - taps >= 0 else None, -1)
    return Y[:, selector]


def get_correlations_narrow(Y, inverse_power, taps, delay):
    D, T = Y.shape

    correlation_matrix = np.zeros((taps, D, taps, D), dtype=Y.dtype)
    correlation_vector = np.zeros((taps, D, D), dtype=Y.dtype)

    for t in range(delay + taps - 1, T):
        Psi = get_Psi_narrow(Y, t - delay, taps)
        Psi_conj_norm = inverse_power[t] * Psi.conj()
        correlation_matrix += np.einsum('dk,el->kdle', Psi_conj_norm, Psi)
        correlation_vector += np.einsum('dk,e->ked', Psi_conj_norm, Y[:, t])

    correlation_matrix = np.reshape(correlation_matrix, (taps * D, taps * D))
    return correlation_matrix, correlation_vector


def get_correlations_narrow_v5(Y, inverse_power, taps, delay):
    D, T = Y.shape

    # TODO: Large gains also expected when precalculating Psi.
    # TODO: Small gains expected, when views are pre-calculated in main.
    # TODO: Larger gains expected with scipy.signal.signaltools.fftconvolve().
    # Code without fft will be easier to port to Chainer.
    # Shape (D, T - taps + 1, taps)
    Psi = segment_axis(Y, taps, 1, axis=-1)[:, :T - delay - taps + 1, ::-1]
    Psi_conj_norm = inverse_power[None, delay + taps - 1:, None] * Psi.conj()

    correlation_matrix = np.einsum('dtk,etl->kdle', Psi_conj_norm, Psi)
    correlation_vector = np.einsum(
        'dtk,et->ked', Psi_conj_norm, Y[:, delay + taps - 1:]
    )

    correlation_matrix = np.reshape(correlation_matrix, (taps * D, taps * D))
    return correlation_matrix, correlation_vector


def get_correlations_v2(Y, inverse_power, taps, delay):
    """
    Later, this version of the correlation matrix can be used without the
    additional column reordering. For now, it needs to be compatible to v1.
    """
    D, T = Y.shape

    correlation_matrix, correlation_vector = get_correlations_narrow(
        Y, inverse_power, taps, delay
    )
    correlation_matrix = np.kron(np.eye(D), correlation_matrix)
    correlation_vector = np.reshape(correlation_vector, (taps * D * D, 1))

    selector = np.transpose(np.reshape(
        np.arange(D * D * taps), (-1, taps, D)
    ), (1, 0, 2)).flatten()
    correlation_matrix = correlation_matrix[selector, :]
    correlation_matrix = correlation_matrix[:, selector]

    return correlation_matrix, correlation_vector


def get_correlations_v6(Y, Y_tilde, inverse_power):
    Y_tilde_inverse_power = Y_tilde * inverse_power[..., None, :]
    R = np.matmul(Y_tilde_inverse_power, hermite(Y_tilde))
    P = np.matmul(Y_tilde_inverse_power, hermite(Y))
    return R, P


def get_filter_matrix_conj(correlation_matrix, correlation_vector, taps, D):
    """
    Args:
        correlation_matrix: Shape (taps * D * D, taps * D * D)
        correlation_vector: Shape (taps * D * D,)
        taps:
        D:
    """
    stacked_filter_conj = np.linalg.solve(
        correlation_matrix, correlation_vector
    )
    filter_matrix_conj = np.transpose(
        np.reshape(stacked_filter_conj, (taps, D, D)), (0, 2, 1)
    )
    return filter_matrix_conj


def get_filter_matrix_conj_v5(Y, inverse_power, taps, delay):
    D, T = Y.shape

    correlation_matrix, correlation_vector = get_correlations_narrow_v5(
        Y, inverse_power, taps, delay
    )

    correlation_vector = np.reshape(correlation_vector, (D * D * taps, 1))
    selector = np.transpose(np.reshape(
        np.arange(D * D * taps), (-1, taps, D)
    ), (1, 0, 2)).flatten()
    inv_selector = np.argsort(selector)
    correlation_vector = correlation_vector[inv_selector, :]

    # Idea is to solve matrix inversion independently for each block matrix.
    # This should still be faster and more stable than np.linalg.inv().
    # print(np.linalg.cond(correlation_matrix))
    stacked_filter_conj = np.reshape(
        np.linalg.solve(
            correlation_matrix[None, :, :],
            np.reshape(correlation_vector, (D, D * taps, 1))
        ),
        (D * D * taps, 1)
    )
    stacked_filter_conj = stacked_filter_conj[selector, :]

    filter_matrix_conj = np.transpose(
        np.reshape(stacked_filter_conj, (taps, D, D)),
        (0, 2, 1)
    )
    return filter_matrix_conj


def get_filter_matrix_conj_v6(Y, Psi, inverse_power, taps, delay):
    D, T = Y.shape

    correlation_matrix, correlation_vector = get_correlations_narrow_v6(
        Y, Psi, inverse_power, taps, delay
    )

    correlation_vector = np.reshape(correlation_vector, (D * D * taps, 1))
    selector = np.transpose(np.reshape(
        np.arange(D * D * taps), (-1, taps, D)
    ), (1, 0, 2)).flatten()
    correlation_vector = correlation_vector[np.argsort(selector), :]

    # Idea is to solve matrix inversion independently for each block matrix.
    # This should still be faster and more stable than np.linalg.inv().
    # print(np.linalg.cond(correlation_matrix))
    stacked_filter_conj = np.reshape(
        np.linalg.solve(
            correlation_matrix[None, :, :],
            np.reshape(correlation_vector, (D, D * taps, 1))
        ),
        (D * D * taps, 1)
    )
    stacked_filter_conj = stacked_filter_conj[selector, :]

    filter_matrix_conj = np.transpose(
        np.reshape(stacked_filter_conj, (taps, D, D)),
        (0, 2, 1)
    )
    return filter_matrix_conj


def get_filter_matrix_v7(Y, Y_tilde, inverse_power):
    R, P = get_correlations_v6(Y, Y_tilde, inverse_power)
    G = _stable_solve(R, P)
    return G


def perform_filter_operation(Y, filter_matrix_conj, taps, delay):
    """
    >>> D, T, taps, delay = 1, 10, 2, 1
    >>> Y = np.ones([D, T])
    >>> filter_matrix_conj = np.ones([taps, D, D])
    >>> X = perform_filter_operation(Y, filter_matrix_conj, taps, delay)
    >>> X.shape
    (1, 10)
    >>> X
    array([[ 1.,  0., -1., -1., -1., -1., -1., -1., -1., -1.]])
    >>> filter = np.array(np.squeeze([1,*(-np.squeeze(filter_matrix_conj))]))
    >>> filter
    array([ 1., -1., -1.])
    >>> np.convolve(np.squeeze(Y), filter)[:T]
    array([ 1.,  0., -1., -1., -1., -1., -1., -1., -1., -1.])


    """
    _, T = Y.shape
    X = np.copy(Y)  # Can be avoided by providing X from outside.
    for t in range(0, T):  # Changed, since t - tau was negative.
        for tau in range(delay, delay + taps - 1 + 1):
            if t - tau >= 0:
                # assert t - tau >= 0, (t, tau)
                assert tau - delay >= 0, (tau, delay)
                X[:, t] -= np.matmul(
                    filter_matrix_conj[tau - delay, :, :].T,
                    Y[:, t - tau]
                )
    return X


def perform_filter_operation_v4(Y, filter_matrix_conj, taps, delay):
    """

    Args:
        Y: D x T
        filter_matrix_conj: taps x D x D
        taps: scalar
        delay: scalar

    Returns: D x T

    >>> def arange(*shape, dtype, start=0):
    ...     _map_to_real_dtype = {np.dtype(np.complex128): np.float64}
    ...     dtype = np.dtype(dtype)
    ...     if dtype.kind in 'if':
    ...         return np.arange(start, start+np.prod(shape)).reshape(shape).astype(dtype)
    ...     elif dtype.kind == 'c':
    ...         shape = list(shape)
    ...         shape[-1] *= 2
    ...         return arange(*shape, dtype=_map_to_real_dtype[dtype],
    ...                       start=start).view(dtype)
    ...     else:
    ...         raise TypeError(dtype, dtype.kind)

    >>> D, T, taps = 2, 5, 3
    >>> delay = 1
    >>> Y = arange(D, T, dtype=np.complex128)
    >>> Y
    array([[ 0. +1.j,  2. +3.j,  4. +5.j,  6. +7.j,  8. +9.j],
           [10.+11.j, 12.+13.j, 14.+15.j, 16.+17.j, 18.+19.j]])
    >>> filter_matrix_conj = arange(taps, D, D, dtype=np.complex128)
    >>> Y
    array([[ 0. +1.j,  2. +3.j,  4. +5.j,  6. +7.j,  8. +9.j],
           [10.+11.j, 12.+13.j, 14.+15.j, 16.+17.j, 18.+19.j]])
    >>> filter_matrix_conj
    array([[[ 0. +1.j,  2. +3.j],
            [ 4. +5.j,  6. +7.j]],
    <BLANKLINE>
           [[ 8. +9.j, 10.+11.j],
            [12.+13.j, 14.+15.j]],
    <BLANKLINE>
           [[16.+17.j, 18.+19.j],
            [20.+21.j, 22.+23.j]]])
    >>> perform_filter_operation_v4(Y, filter_matrix_conj, taps, delay).shape
    (2, 5)
    >>> perform_filter_operation_v4(Y, filter_matrix_conj, taps, delay)
    array([[  0.+1.000e+00j,  18.-9.100e+01j,  56.-3.790e+02j,
            114.-9.270e+02j, 128.-1.177e+03j],
           [ 10.+1.100e+01j,  32.-1.250e+02j,  74.-4.730e+02j,
            136.-1.097e+03j, 150.-1.395e+03j]])

    Fallback test to conventional convolution

    >>> D, T, taps = 1, 5, 2
    >>> delay = 1
    >>> Y = arange(D, T, dtype=np.complex128)
    >>> Y = arange(D, T, dtype=np.float64) + 1
    >>> filter_matrix_conj = arange(taps, D, D, dtype=np.complex128)
    >>> filter_matrix_conj = arange(taps, D, D, dtype=np.float64) + 1
    >>> Y
    array([[1., 2., 3., 4., 5.]])
    >>> filter_matrix_conj
    array([[[1.]],
    <BLANKLINE>
           [[2.]]])
    >>> perform_filter_operation_v4(Y, filter_matrix_conj, taps, delay).shape
    (1, 5)
    >>> o, = perform_filter_operation_v4(Y, filter_matrix_conj, taps, delay)
    >>> o
    array([ 1.,  1., -1., -3., -5.])
    >>> np.convolve(Y[0], [1, *(-np.squeeze(filter_matrix_conj))])
    array([  1.,   1.,  -1.,  -3.,  -5., -13., -10.])

    """
    _, T = Y.shape
    X = np.copy(Y)  # Can be avoided by providing X from outside.

    # TODO: Second loop can be removed with using segment_axis. No large gain.
    for tau_minus_delay in range(0, taps):
        X[:, (delay + tau_minus_delay):] -= np.einsum(
            'de,dt',
            filter_matrix_conj[tau_minus_delay, :, :],
            Y[:, :(T - delay - tau_minus_delay)]
        )
    return X


def perform_filter_operation_v5(Y, Y_tilde, filter_matrix):
    X = Y - np.matmul(hermite(filter_matrix), Y_tilde)
    return X


@click.command()
@click.option(
    '--channels',
    default=8,
    help='Audio Channels D'
)
@click.option(
    '--sampling_rate',
    default=16000,
    help='Sampling rate of audio'
)
@click.option(
    '--file_template',
    help='Audio example. Full path required. Included example: AMI_WSJ20-Array1-{}_T10c0201.wav'
)
@click.option(
    '--taps_frequency_dependent',
    is_flag=True,
    help='Whether taps are frequency dependent or not'
)
@click.option(
    '--delay',
    default=3,
    help='Delay'
)
@click.option(
    '--iterations',
    default=5,
    help='Iterations of WPE'
)
def main(channels, sampling_rate, file_template, taps_frequency_dependent,
         delay, iterations):
    """
    User interface for WPE. The defaults of the command line interface are
    suited for example audio files of nara_wpe.

     'Yoshioka2012GeneralWPE'
        sampling_rate = 8000
        delay = 2
        iterations = 2

    """
    from nara_wpe import project_root
    import soundfile as sf
    from nara_wpe.utils import stft
    from nara_wpe.utils import istft
    from nara_wpe.utils import get_stft_center_frequencies
    from tqdm import tqdm
    from librosa.core.audio import resample

    stft_options = dict(
        size=512,
        shift=128,
        window_length=None,
        fading=True,
        pad=True,
        symmetric_window=False
    )

    def get_taps(f, mode=taps_frequency_dependent):
        if mode:
            if center_frequencies[f] < 800:
                taps = 18
            elif center_frequencies[f] < 1500:
                taps = 15
            else:
                taps = 12
        else:
            taps = 10
        return taps

    if file_template == 'AMI_WSJ20-Array1-{}_T10c0201.wav':
        signal_list = [
            sf.read(str(project_root / 'data' / file_template.format(d + 1)))[0]
            for d in range(channels)
            ]
    else:
        signal = sf.read(file_template)[0].transpose(1, 0)
        signal_list = list(signal)
    signal_list = [resample(x_, 16000, sampling_rate) for x_ in signal_list]
    y = np.stack(signal_list, axis=0)

    center_frequencies = get_stft_center_frequencies(
        stft_options['size'],
        sampling_rate
    )

    Y = stft(y, **stft_options)

    X = np.copy(Y)
    D, T, F = Y.shape
    for f in tqdm(range(F), total=F):
        taps = get_taps(f)
        X[:, :, f] = wpe_v7(
            Y[:, :, f],
            taps=taps,
            delay=delay,
            iterations=iterations
        )

    x = istft(X, size=stft_options['size'], shift=stft_options['shift'])

    sf.write(
        str(project_root / 'data' / 'wpe_out.wav'),
        x[0], samplerate=sampling_rate
    )
    print('Output in {}'.format(str(project_root / 'data' / 'wpe_out.wav')))


if __name__ == '__main__':
    main()
