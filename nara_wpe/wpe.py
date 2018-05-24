import functools
import operator

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
    axis = axis % x.ndim

    # Pad
    if end == 'pad':
        if x.shape[axis] < length:
            npad = np.zeros([x.ndim, 2], dtype=np.int)
            npad[axis, 1] = length - x.shape[axis]
            x = np.pad(x, pad_width=npad, mode=pad_mode,
                       constant_values=pad_value)
        elif shift != 1 and (x.shape[axis] + shift - length) % shift != 0:
            npad = np.zeros([x.ndim, 2], dtype=np.int)
            npad[axis, 1] = shift - ((x.shape[axis] + shift - length) % shift)
            x = np.pad(x, pad_width=npad, mode=pad_mode,
                       constant_values=pad_value)
    elif end is None:
        assert (x.shape[axis] + shift - length) % shift == 0, \
            '{} = x.shape[axis]({}) + shift({}) - length({})) % shift({})' \
            ''.format((x.shape[axis] + shift - length) % shift,
                      x.shape[axis], shift, length, shift)
    elif end == 'cut':
        pass
    else:
        raise ValueError(end)

    shape = list(x.shape)
    del shape[axis]
    shape.insert(axis, (x.shape[axis] + shift - length) // shift)
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


def build_y_tilde(Y, K, delay):
    """
    >>> T, D = 20, 2
    >>> Y = np.arange(start=1, stop=T * D + 1).reshape([T, D]).T
    >>> print(Y)
    [[ 1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39]
     [ 2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40]]
    >>> K, delay = 4, 2
    >>> Y_tilde = build_y_tilde(Y, K, delay)
    >>> print(Y_tilde.shape, (K*D, T))
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
    >>> Y_tilde = build_y_tilde(Y, K, 0)
    >>> print(Y_tilde.shape, (K*D, T))
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

    def pad(x, axis=-1, pad_width=K + delay - 1):
        npad = np.zeros([x.ndim, 2], dtype=np.int)
        npad[axis, 0] = pad_width
        x = np.pad(x,
                   pad_width=npad,
                   mode='constant',
                   constant_values=0)
        return x

    Y_ = segment_axis(pad(Y), K, 1, axis=-1)

    Y_ = np.flip(Y_, axis=-1)

    if delay > 0:
        Y_ = Y_[..., :-delay, :]
    # Y_: ... x D x T x K
    Y_ = np.moveaxis(Y_, -1, -3)
    # Y_: ... x K x D x T
    Y_ = np.reshape(Y_, [*S, K * D, T])
    # Y_: ... x KD x T
    return Y_


def hermite(x):
    return x.swapaxes(-2, -1).conj()


def print_matrix(matrix):
    max_length = max([len(str(x)) for x in matrix.ravel()])
    for row in matrix:
        for column in row:
            print(('{:' + str(max_length) + '.0f}').format(column), end='')
        print()


def wpe_v0(Y, K=10, delay=3, iterations=3, psd_context=0, statistics_mode='full'):
    """

    Args:
        Y: Complex valued STFT signal with shape (F, D, T) or (D, T).
        K: Filter order
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
        Estimated signal with same shape as Y.

    """
    X = np.copy(Y)
    if Y.ndim == 2:
        for iteration in range(iterations):
            inverse_power = get_power_inverse(X, neighborhood=psd_context)
            filter_matrix_conj = get_filter_matrix_conj_v5(
                Y, inverse_power, K, delay
            )
            X = perform_filter_operation_v4(Y, filter_matrix_conj, K, delay)
    elif Y.ndim == 3:
        F = Y.shape[0]
        for f in range(F):
            X[f, :, :] = wpe(
                Y[f, :, :],
                K=K, delay=delay, iterations=iterations,
                psd_context=psd_context, statistics_mode=statistics_mode
            )
    else:
        raise NotImplementedError('Input shape is to be (F, D, T) or (D, T).')
    return X


def wpe_v1(Y, K=10, delay=3, iterations=3):
    D, T = Y.shape
    X = np.copy(Y)
    for iteration in range(iterations):
        inverse_power = get_power_inverse(X)
        correlation_matrix, correlation_vector = get_correlations(
            Y, inverse_power, K, delay
        )
        filter_matrix_conj = get_filter_matrix_conj(
            correlation_matrix, correlation_vector, K, D
        )
        X = perform_filter_operation(Y, filter_matrix_conj, K, delay)
    return X


def wpe_v2(Y, K=10, delay=3, iterations=3):
    D, T = Y.shape
    X = np.copy(Y)
    for iteration in range(iterations):
        inverse_power = get_power_inverse(X)
        correlation_matrix, correlation_vector = get_correlations_v2(
            Y, inverse_power, K, delay
        )
        filter_matrix_conj = get_filter_matrix_conj(
            correlation_matrix, correlation_vector, K, D
        )
        X = perform_filter_operation(Y, filter_matrix_conj, K, delay)
    return X


def wpe_v4(Y, K=10, delay=3, iterations=3):
    D, T = Y.shape
    X = np.copy(Y)
    for iteration in range(iterations):
        inverse_power = get_power_inverse(X)
        correlation_matrix, correlation_vector = get_correlations_v2(
            Y, inverse_power, K, delay
        )
        filter_matrix_conj = get_filter_matrix_conj(
            correlation_matrix, correlation_vector, K, D
        )
        X = perform_filter_operation_v4(Y, filter_matrix_conj, K, delay)
    return X


def wpe_v5(Y, K=10, delay=3, iterations=3):
    X = np.copy(Y)
    for iteration in range(iterations):
        inverse_power = get_power_inverse(X)
        filter_matrix_conj = get_filter_matrix_conj_v5(
            Y, inverse_power, K, delay
        )
        X = perform_filter_operation_v4(Y, filter_matrix_conj, K, delay)
    return X


def wpe_v6(Y, K=10, delay=3, iterations=3, neighborhood=0):
    """
    Short of wpe_v7.

    """
    X = np.copy(Y)
    Y_tilde = build_y_tilde(Y, K, delay)
    for iteration in range(iterations):
        inverse_power = get_power_inverse(X, neighborhood=neighborhood)
        Y_tilde_inverse_power = Y_tilde * inverse_power[..., None, :]
        R = np.matmul(Y_tilde_inverse_power, hermite(Y_tilde))
        P = np.matmul(Y_tilde_inverse_power, hermite(Y))
        G = _stable_solve(R, P)
        X = Y - (hermite(G) @ Y_tilde)

    return X


def wpe_v7(Y, K=10, delay=3, iterations=3, neighborhood=0, mode='cut'):
    """

    mode=='pad':
        - Pad Y with zeros on the left for the estimation of the correlation
          matrix and vector. This value optimizes the cost function of wpe.
    mode=='cut':
        - Consider only valid slices of observations for the estimation of the
          correlation and matrix and vector.

    consider_inital_sampels_in_est
    """
    X = Y
    Y_tilde = build_y_tilde(Y, K, delay)

    if mode == 'pad':
        s = Ellipsis
    elif mode == 'cut':
        s = [Ellipsis, slice(delay + K - 1, None)]
    else:
        raise ValueError(mode)

    for iteration in range(iterations):
        inverse_power = get_power_inverse(X, neighborhood=neighborhood)
        G = get_filter_matrix_v7(Y=Y[s], Y_tilde=Y_tilde[s], inverse_power=inverse_power[s])
        X = perform_filter_operation_v5(Y=Y, Y_tilde=Y_tilde, filter_matrix=G)
    return X


def wpe_v8(Y, K=10, delay=3, iterations=3, neighborhood=0, batch_axis=0):
    # if Y.ndim == 2:
    #     return wpe_v6(
    #         Y,
    #         K=K,
    #         delay=delay,
    #         iterations=iterations,
    #         neighborhood=neighborhood,
    #     )
    assert Y.ndim == 3, Y.shape

    F = Y.shape[batch_axis]
    index = [slice(None)] * Y.ndim

    out = []
    for f in range(F):
        index[batch_axis] = f
        out.append(wpe_v6(
            Y=Y[index],
            K=K,
            delay=delay,
            iterations=iterations,
            neighborhood=neighborhood,
        ))
    return np.stack(out, axis=batch_axis)


wpe = wpe_v7

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


def get_power_inverse(signal, neighborhood=0):
    """
    Assumes single frequency bin with shape (D, T).

    >>> s = 1 / np.array([np.arange(1, 6)]*3)
    >>> get_power_inverse(s)
    array([ 1.,  4.,  9., 16., 25.])
    >>> get_power_inverse(s * 0 + 1, 1)
    array([1., 1., 1., 1., 1.])
    >>> get_power_inverse(s, 1)
    array([ 1.        ,  1.6       ,  2.20408163,  7.08196721, 14.04421326])
    >>> get_power_inverse(s, np.inf)
    array([3.41620801, 3.41620801, 3.41620801, 3.41620801, 3.41620801])
    """
    power = np.mean(abs_square(signal), axis=-2)

    if np.isposinf(neighborhood):
        power = np.broadcast_to(np.mean(power, axis=-1, keepdims=True), power.shape)
    elif neighborhood > 0:
        assert int(neighborhood) == neighborhood, neighborhood
        neighborhood = int(neighborhood)
        import bottleneck as bn
        # Handle the corner case correctly (i.e. sum() / count)
        power = bn.move_mean(power, neighborhood*2+1, min_count=1)
    elif neighborhood == 0:
        pass
    else:
        raise ValueError(neighborhood)
    eps = 1e-10 * np.max(power)
    inverse_power = 1 / np.maximum(power, eps)
    return inverse_power


def get_Psi(Y, t, K):
    D = Y.shape[0]

    def get_Y_tilde(t_):
        return np.kron(np.eye(D), Y[:, t_]).T

    assert t - K + 1 >= 0
    return np.concatenate([get_Y_tilde(t_) for t_ in range(t, t - K, -1)])


def get_correlations(Y, inverse_power, K, delay):
    D, T = Y.shape

    correlation_matrix = np.zeros((D * D * K, D * D * K), dtype=Y.dtype)
    correlation_vector = np.zeros((D * D * K, 1), dtype=Y.dtype)
    for t in range(delay + K - 1, T):
        Psi = get_Psi(Y, t - delay, K)
        correlation_matrix += inverse_power[t] * np.dot(Psi.conj(), Psi.T)
        correlation_vector \
            += inverse_power[t] * np.dot(Psi.conj(), Y[:, t])[:, None]

    return correlation_matrix, correlation_vector


def get_Psi_narrow(Y, t, K):
    assert t - K + 1 >= 0
    selector = slice(t, t - K if t - K >= 0 else None, -1)
    return Y[:, selector]


def get_correlations_narrow(Y, inverse_power, K, delay):
    D, T = Y.shape

    correlation_matrix = np.zeros((K, D, K, D), dtype=Y.dtype)
    correlation_vector = np.zeros((K, D, D), dtype=Y.dtype)

    for t in range(delay + K - 1, T):
        Psi = get_Psi_narrow(Y, t - delay, K)
        Psi_conj_norm = inverse_power[t] * Psi.conj()
        correlation_matrix += np.einsum('dk,el->kdle', Psi_conj_norm, Psi)
        correlation_vector += np.einsum('dk,e->ked', Psi_conj_norm, Y[:, t])

    correlation_matrix = np.reshape(correlation_matrix, (K * D, K * D))
    return correlation_matrix, correlation_vector


def get_correlations_narrow_v5(Y, inverse_power, K, delay):
    D, T = Y.shape

    # TODO: Large gains also expected when precalculating Psi.
    # TODO: Small gains expected, when views are pre-calculated in main.
    # TODO: Larger gains expected with scipy.signal.signaltools.fftconvolve().
    # Code without fft will be easier to port to Chainer.
    # Shape (D, T - K + 1, K)
    Psi = segment_axis(Y, K, 1, axis=-1)[:, :T - delay - K + 1, ::-1]
    Psi_conj_norm = inverse_power[None, delay + K - 1:, None] * Psi.conj()

    correlation_matrix = np.einsum('dtk,etl->kdle', Psi_conj_norm, Psi)
    correlation_vector = np.einsum(
        'dtk,et->ked', Psi_conj_norm, Y[:, delay + K - 1:]
    )

    correlation_matrix = np.reshape(correlation_matrix, (K * D, K * D))
    return correlation_matrix, correlation_vector


def get_correlations_v2(Y, inverse_power, K, delay):
    """
    Later, this version of the correlation matrix can be used without the
    additional column reordering. For now, it needs to be compatible to v1.
    """
    D, T = Y.shape

    correlation_matrix, correlation_vector = get_correlations_narrow(
        Y, inverse_power, K, delay
    )
    correlation_matrix = np.kron(np.eye(D), correlation_matrix)
    correlation_vector = np.reshape(correlation_vector, (K * D * D, 1))

    selector = np.transpose(np.reshape(
        np.arange(D * D * K), (-1, K, D)
    ), (1, 0, 2)).flatten()
    correlation_matrix = correlation_matrix[selector, :]
    correlation_matrix = correlation_matrix[:, selector]

    return correlation_matrix, correlation_vector


def get_correlations_v6(Y, Y_tilde, inverse_power):
    Y_tilde_inverse_power = Y_tilde * inverse_power[..., None, :]
    R = np.matmul(Y_tilde_inverse_power, hermite(Y_tilde))
    P = np.matmul(Y_tilde_inverse_power, hermite(Y))
    return R, P


def get_filter_matrix_conj(correlation_matrix, correlation_vector, K, D):
    """

    :param correlation_matrix: Shape (K * D * D, K * D * D)
    :param correlation_vector: Shape (K * D * D,)
    :param K:
    :param D:
    :return:
    """
    stacked_filter_conj = np.linalg.solve(
        correlation_matrix, correlation_vector
    )
    filter_matrix_conj = np.transpose(
        np.reshape(stacked_filter_conj, (K, D, D)), (0, 2, 1)
    )
    return filter_matrix_conj


def get_filter_matrix_conj_v5(Y, inverse_power, K, delay):
    D, T = Y.shape

    correlation_matrix, correlation_vector = get_correlations_narrow_v5(
        Y, inverse_power, K, delay
    )

    correlation_vector = np.reshape(correlation_vector, (D * D * K, 1))
    selector = np.transpose(np.reshape(
        np.arange(D * D * K), (-1, K, D)
    ), (1, 0, 2)).flatten()
    inv_selector = np.argsort(selector)
    correlation_vector = correlation_vector[inv_selector, :]

    # Idea is to solve matrix inversion independently for each block matrix.
    # This should still be faster and more stable than np.linalg.inv().
    # print(np.linalg.cond(correlation_matrix))
    stacked_filter_conj = np.reshape(
        np.linalg.solve(
            correlation_matrix[None, :, :],
            np.reshape(correlation_vector, (D, D * K, 1))
        ),
        (D * D * K, 1)
    )
    stacked_filter_conj = stacked_filter_conj[selector, :]

    filter_matrix_conj = np.transpose(
        np.reshape(stacked_filter_conj, (K, D, D)),
        (0, 2, 1)
    )
    return filter_matrix_conj


def get_filter_matrix_conj_v6(Y, Psi, inverse_power, K, delay):
    D, T = Y.shape

    correlation_matrix, correlation_vector = get_correlations_narrow_v6(
        Y, Psi, inverse_power, K, delay
    )

    correlation_vector = np.reshape(correlation_vector, (D * D * K, 1))
    selector = np.transpose(np.reshape(
        np.arange(D * D * K), (-1, K, D)
    ), (1, 0, 2)).flatten()
    correlation_vector = correlation_vector[np.argsort(selector), :]

    # Idea is to solve matrix inversion independently for each block matrix.
    # This should still be faster and more stable than np.linalg.inv().
    # print(np.linalg.cond(correlation_matrix))
    stacked_filter_conj = np.reshape(
        np.linalg.solve(
            correlation_matrix[None, :, :],
            np.reshape(correlation_vector, (D, D * K, 1))
        ),
        (D * D * K, 1)
    )
    stacked_filter_conj = stacked_filter_conj[selector, :]

    filter_matrix_conj = np.transpose(
        np.reshape(stacked_filter_conj, (K, D, D)),
        (0, 2, 1)
    )
    return filter_matrix_conj


def get_filter_matrix_v7(Y, Y_tilde, inverse_power):
    R, P = get_correlations_v6(Y, Y_tilde, inverse_power)
    G = _stable_solve(R, P)
    return G


def perform_filter_operation(Y, filter_matrix_conj, K, delay):
    """
    >>> D, T, K, delay = 1, 10, 2, 1
    >>> Y = np.ones([D, T])
    >>> filter_matrix_conj = np.ones([K, D, D])
    >>> X = perform_filter_operation(Y, filter_matrix_conj, K, delay)
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
        for tau in range(delay, delay + K - 1 + 1):
            if t - tau >= 0:
                # assert t - tau >= 0, (t, tau)
                assert tau - delay >= 0, (tau, delay)
                X[:, t] -= filter_matrix_conj[tau - delay, :, :].T @ Y[:, t - tau]
    return X


def perform_filter_operation_v4(Y, filter_matrix_conj, K, delay):
    """

    Args:
        Y: D x T
        filter_matrix_conj: K x D x D
        K: scalar
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

    >>> D, T, K = 2, 5, 3
    >>> delay = 1
    >>> Y = arange(D, T, dtype=np.complex128)
    >>> Y
    array([[ 0. +1.j,  2. +3.j,  4. +5.j,  6. +7.j,  8. +9.j],
           [10.+11.j, 12.+13.j, 14.+15.j, 16.+17.j, 18.+19.j]])
    >>> filter_matrix_conj = arange(K, D, D, dtype=np.complex128)
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
    >>> perform_filter_operation_v4(Y, filter_matrix_conj, K, delay).shape
    (2, 5)
    >>> perform_filter_operation_v4(Y, filter_matrix_conj, K, delay)
    array([[  0.+1.000e+00j,  18.-9.100e+01j,  56.-3.790e+02j,
            114.-9.270e+02j, 128.-1.177e+03j],
           [ 10.+1.100e+01j,  32.-1.250e+02j,  74.-4.730e+02j,
            136.-1.097e+03j, 150.-1.395e+03j]])

    Fallback test to conventional convolution

    >>> D, T, K = 1, 5, 2
    >>> delay = 1
    >>> Y = arange(D, T, dtype=np.complex128)
    >>> Y = arange(D, T, dtype=np.float64) + 1
    >>> filter_matrix_conj = arange(K, D, D, dtype=np.complex128)
    >>> filter_matrix_conj = arange(K, D, D, dtype=np.float64) + 1
    >>> Y
    array([[1., 2., 3., 4., 5.]])
    >>> filter_matrix_conj
    array([[[1.]],
    <BLANKLINE>
           [[2.]]])
    >>> perform_filter_operation_v4(Y, filter_matrix_conj, K, delay).shape
    (1, 5)
    >>> o, = perform_filter_operation_v4(Y, filter_matrix_conj, K, delay)
    >>> o
    array([ 1.,  1., -1., -3., -5.])
    >>> np.convolve(Y[0], [1, *(-np.squeeze(filter_matrix_conj))])
    array([  1.,   1.,  -1.,  -3.,  -5., -13., -10.])

    """
    _, T = Y.shape
    X = np.copy(Y)  # Can be avoided by providing X from outside.

    # TODO: Second loop can be removed with using segment_axis. No large gain.
    for tau_minus_delay in range(0, K):
        X[:, (delay + tau_minus_delay):] -= np.einsum(
            'de,dt',
            filter_matrix_conj[tau_minus_delay, :, :],
            Y[:, :(T - delay - tau_minus_delay)]
        )
    return X


def perform_filter_operation_v5(Y, Y_tilde, filter_matrix):
    X = Y - (hermite(filter_matrix) @ Y_tilde)
    return X


def main():
    from nara_wpe import project_root
    import soundfile as sf
    from nara_wpe.utils import stft
    from nara_wpe.utils import istft_loop as istft
    from nara_wpe.utils import get_stft_center_frequencies
    from tqdm import tqdm
    from librosa.core.audio import resample

    channels = 8

    parameter_set = 'Katka'

    if parameter_set == 'Katka':
        sampling_rate = 16000
        stft_size, stft_shift = 512, 128
        delay = 3
        iterations = 5

        def get_K(f):
            return 10

    elif parameter_set == 'Yoshioka2012GeneralWPE':
        sampling_rate = 8000
        stft_size, stft_shift = 128, 64
        delay = 2
        iterations = 2

        def get_K(f):
            if center_frequencies[f] < 800:
                K = 18
            elif center_frequencies[f] < 1500:
                K = 15
            else:
                K = 12
            return K

    else:
        raise ValueError

    file_template = 'AMI_WSJ20-Array1-{}_T10c0201.wav'
    signal_list = [
        sf.read(str(project_root / 'data' / file_template.format(d + 1)))[0]
        for d in range(channels)
        ]
    signal_list = [resample(x_, 16000, sampling_rate) for x_ in signal_list]
    y = np.stack(signal_list, axis=0)

    center_frequencies = get_stft_center_frequencies(stft_size, sampling_rate)

    Y = stft(y, size=stft_size, shift=stft_shift)

    X = np.copy(Y)
    D, T, F = Y.shape
    for f in tqdm(range(F), total=F):
        K = get_K(f)
        X[:, :, f] = wpe_v5(Y[:, :, f], K=K, delay=delay, iterations=iterations)

    x = istft(X, size=stft_size, shift=stft_shift)

    sf.write(
        str(project_root / 'data' / 'wpe_out.wav'),
        x[0], samplerate=sampling_rate
    )


if __name__ == '__main__':
    main()
