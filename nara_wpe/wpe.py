import numpy as np


def segment_axis(x, length, shift, axis=-1):
    axis = axis % x.ndim

    shape = list(x.shape)
    del shape[axis]
    shape.insert(axis, (x.shape[axis] + shift - length) // shift)
    shape.insert(axis + 1, length)

    strides = list(x.strides)
    strides.insert(axis, shift * strides[axis])

    return np.lib.stride_tricks.as_strided(x, strides=strides, shape=shape)


def print_matrix(matrix):
    max_length = max([len(str(x)) for x in matrix.ravel()])
    for row in matrix:
        for column in row:
            print(('{:' + str(max_length) + '.0f}').format(column), end='')
        print()


def wpe(Y, K=10, delay=3, iterations=3):
    """

    Args:
        Y: Complex valued STFT signal with shape (D, T)
        K: Filter order
        delay: Delay as a guard interval, such that X does not become zero.
        iterations:

    Returns:

    """
    X = np.copy(Y)
    for iteration in range(iterations):
        inverse_power = get_power_inverse(X)
        filter_matrix_conj = get_filter_matrix_conj_v5(
            Y, inverse_power, K, delay
        )
        X = perform_filter_operation_v4(Y, filter_matrix_conj, K, delay)
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


def get_power_inverse(signal):
    """Assumes single frequency bin with shape (D, T)."""
    power = np.mean(signal.real ** 2 + signal.imag ** 2, axis=0)
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


def get_filter_matrix_conj(correlation_matrix, correlation_vector, K, D):
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


def perform_filter_operation(Y, filter_matrix_conj, K, delay):
    _, T = Y.shape
    X = np.copy(Y)  # Can be avoided by providing X from outside.
    for t in range(delay + K - 1, T):  # Changed, since t - tau was negative.
        for tau in range(delay, delay + K - 1 + 1):
            assert t - tau >= 0, (t, tau)
            assert tau - delay >= 0, (tau, delay)
            X[:, t] -= filter_matrix_conj[tau - delay, :, :].T @ Y[:, t - tau]
    return X


def perform_filter_operation_v4(Y, filter_matrix_conj, K, delay):
    _, T = Y.shape
    X = np.copy(Y)  # Can be avoided by providing X from outside.

    # TODO: Second loop can be removed with using segment_axis. No large gain.
    for tau_minus_delay in range(0, K):
        X[:, (delay + K - 1):] -= np.einsum(
            'de,dt',
            filter_matrix_conj[tau_minus_delay, :, :],
            Y[:, (K - 1 - tau_minus_delay):(T - delay - tau_minus_delay)]
        )
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
