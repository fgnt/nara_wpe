import functools
import operator

import click
import numpy as np


def segment_axis(
        x,
        length,
        shift,
        axis=-1,
        end:  "in ['pad', 'cut', None]"='cut',
        pad_mode='constant',
        pad_value=0,
):

    """Generate a new array that chops the given array along the given axis
     into overlapping frames.

    Args:
        x: The array to segment
        length: The length of each frame
        shift: The number of array elements by which to step forward
        axis: The axis to operate on; if None, act on the flattened array
        end: What to do with the last frame, if the array is not evenly
                divisible into pieces. Options are:
                * 'cut'   Simply discard the extra values
                * None    No end treatment. Only works when fits perfectly.
                * 'pad'   Pad with a constant value
        pad_mode:
        pad_value: The value to use for end='full'

    Examples:
        >>> segment_axis(np.arange(10), 4, 2)
        array([[0, 1, 2, 3],
               [2, 3, 4, 5],
               [4, 5, 6, 7],
               [6, 7, 8, 9]])
        >>> segment_axis(np.arange(5).reshape(5), 4, 1, axis=0)
        array([[0, 1, 2, 3],
               [1, 2, 3, 4]])
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
        >>> a = np.arange(5).reshape(5)
        >>> b = segment_axis(a, 4, 2, axis=0)
        >>> a += 1  # a and b point to the same memory
        >>> b
        array([[1, 2, 3, 4]])

    """
    axis = axis % x.ndim
    elements = x.shape[axis]

    if shift <= 0:
        raise ValueError('Can not shift forward by less than 1 element.')

    # full
    if end == 'pad':
        npad = np.zeros([x.ndim, 2], dtype=np.int)
        pad_fn = functools.partial(
            np.pad, pad_width=npad, mode=pad_mode, constant_values=pad_value
        )
        if elements < length:
            npad[axis, 1] = length - elements
            x = pad_fn(x)
        elif not shift == 1 and not (elements + shift - length) % shift == 0:
            npad[axis, 1] = shift - ((elements + shift - length) % shift)
            x = pad_fn(x)
    elif end is None:
        assert (elements + shift - length) % shift == 0, \
            '{} = elements({}) + shift({}) - length({})) % shift({})' \
            ''.format((elements + shift - length) % shift,
                      elements, shift, length, shift)
    elif end == 'cut':
        pass
    else:
        raise ValueError(end)

    shape = list(x.shape)
    del shape[axis]
    shape.insert(axis, (elements + shift - length) // shift)
    shape.insert(axis + 1, length)

    strides = list(x.strides)
    strides.insert(axis, shift * strides[axis])

    return np.lib.stride_tricks.as_strided(x, strides=strides, shape=shape)


def _lstsq(A, B):
    assert A.shape == B.shape, (A.shape, B.shape)
    shape = A.shape
    working_shape = [functools.reduce(operator.mul, [1, *shape[:-2]]), *shape[-2:]]
    A = A.reshape(working_shape)
    B = B.reshape(working_shape)

    C = np.zeros_like(A)
    for i in range(working_shape[0]):
        C[i], *_ = np.linalg.lstsq(A[i], B[i])
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
    numpy.linalg.linalg.LinAlgError: Singular matrix
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
    numpy.linalg.linalg.LinAlgError: 3-dimensional array given. Array must be two-dimensional
    >>> C3 = _stable_solve(A, B)
    >>> C4 = _lstsq(A, B)
    >>> np.testing.assert_allclose(C1, C3)
    >>> np.testing.assert_allclose(C1, C4)


    >>> A[2, 3, :] = 0
    >>> C1 = np.linalg.solve(A, B)
    Traceback (most recent call last):
    ...
    numpy.linalg.linalg.LinAlgError: Singular matrix
    >>> C2, *_ = np.linalg.lstsq(A, B)
    Traceback (most recent call last):
    ...
    numpy.linalg.linalg.LinAlgError: 3-dimensional array given. Array must be two-dimensional
    >>> C3 = _stable_solve(A, B)
    >>> C4 = _lstsq(A, B)
    >>> np.testing.assert_allclose(C3, C4)


    """
    assert A.shape[:-2] == B.shape[:-2], (A.shape, B.shape)
    assert A.shape[-1] == B.shape[-2], (A.shape, B.shape)
    try:
        return np.linalg.solve(A, B)
    except np.linalg.linalg.LinAlgError:
        shape_A, shape_B = A.shape, B.shape
        assert shape_A[:-2] == shape_A[:-2]
        working_shape_A = [functools.reduce(operator.mul, [1, *shape_A[:-2]]),
                           *shape_A[-2:]]
        working_shape_B = [functools.reduce(operator.mul, [1, *shape_B[:-2]]),
                           *shape_B[-2:]]
        A = A.reshape(working_shape_A)
        B = B.reshape(working_shape_B)

        C = np.zeros_like(B)
        for i in range(working_shape_A[0]):
            # lstsq is much slower, use it only when necessary
            try:
                C[i] = np.linalg.solve(A[i], B[i])
            except np.linalg.linalg.LinAlgError:
                C[i], *_ = np.linalg.lstsq(A[i], B[i])
        return C.reshape(*shape_B)


def build_y_tilde(Y, taps, delay):
    """
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
    >>> print(Y_tilde.shape, (taps*D, T))
    (8, 20) (8, 20)
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
    *S, D, T = Y.shape

    def pad(x, axis=-1, pad_width=taps + delay - 1):
        npad = np.zeros([x.ndim, 2], dtype=np.int)
        npad[axis, 0] = pad_width
        x = np.pad(x,
                   pad_width=npad,
                   mode='constant',
                   constant_values=0)
        return x

    Y_ = segment_axis(pad(Y), taps, 1, axis=-1)

    Y_ = np.flip(Y_, axis=-1)

    if delay > 0:
        Y_ = Y_[..., :-delay, :]
    # Y_: ... x D x T x taps
    Y_ = np.moveaxis(Y_, -1, -3)
    # Y_: ... x taps x D x T
    Y_ = np.reshape(Y_, [*S, taps * D, T])
    # Y_: ... x taps*D x T
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
        s = [Ellipsis, slice(delay + taps - 1, None)]
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
    Short of wpe_v7 with no extern references.
    Applicable in for-loops.
    """

    if statistics_mode == 'full':
        s = Ellipsis
    elif statistics_mode == 'valid':
        s = [Ellipsis, slice(delay + taps - 1, None)]
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
        X = Y - (hermite(G) @ Y_tilde)

    return X


def wpe_v7(Y, taps=10, delay=3, iterations=3, psd_context=0, statistics_mode='full'):
    """
    Modular wpe version.
    """
    X = Y
    Y_tilde = build_y_tilde(Y, taps, delay)

    if statistics_mode == 'full':
        s = Ellipsis
    elif statistics_mode == 'valid':
        s = [Ellipsis, slice(delay + taps - 1, None)]
    else:
        raise ValueError(statistics_mode)

    for iteration in range(iterations):
        inverse_power = get_power_inverse(X, psd_context=psd_context)
        G = get_filter_matrix_v7(Y=Y[s], Y_tilde=Y_tilde[s], inverse_power=inverse_power[s])
        X = perform_filter_operation_v5(Y=Y, Y_tilde=Y_tilde, filter_matrix=G)
    return X


def wpe_v8(Y, taps=10, delay=3, iterations=3, psd_context=0, statistics_mode='full'):
    """
    v8 is faster than v7 and offers an optional batch mode.
    """
    if Y.ndim == 2:
        return wpe_v6(
            Y,
            taps=taps,
            delay=delay,
            iterations=iterations,
            psd_context=psd_context,
            statistics_mode=statistics_mode
        )
    elif Y.ndim == 3:
        batch_axis = 0
        F = Y.shape[batch_axis]
        index = [slice(None)] * Y.ndim

        out = []
        for f in range(F):
            index[batch_axis] = f
            out.append(wpe_v6(
                Y=Y[index],
                taps=taps,
                delay=delay,
                iterations=iterations,
                psd_context=psd_context,
                statistics_mode=statistics_mode
            ))
        return np.stack(out, axis=batch_axis)
    else:
        raise NotImplementedError('Input shape is to be (F, D, T) or (D, T).')


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
        filter_taps: Current estimate of filter taps (F, taps*D, taps)
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
    kalman_gain = nominator / denominator[:, None]
    
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


def abs_square(x: np.ndarray):
    """

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

    if psd_context is not 0:
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


def get_power_inverse(signal, psd_context=0):
    """
    Assumes single frequency bin with shape (D, T).

    Args:
        signal: (D, T)
        psd_context: tuple or scalar
    """

    power = get_power(signal, psd_context)

    eps = 1e-10 * np.max(power)
    inverse_power = 1 / np.maximum(power, eps)
    return inverse_power


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
        correlation_matrix += inverse_power[t] * np.dot(Psi.conj(), Psi.T)
        correlation_vector \
            += inverse_power[t] * np.dot(Psi.conj(), Y[:, t])[:, None]

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
                X[:, t] -= filter_matrix_conj[tau - delay, :, :].T @ Y[:, t - tau]
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
    X = Y - (hermite(filter_matrix) @ Y_tilde)
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
