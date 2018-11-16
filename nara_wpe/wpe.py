from pathlib import Path

import click
import numpy as np
import soundfile as sf

from nara_wpe.utils import stable_solve, hermite
from nara_wpe.utils import segment_axis
from nara_wpe.utils import stft, istft


def wpe_v0(
        Y,
        taps=10,
        delay=3,
        iterations=3,
        psd_context=0,
        statistics_mode='full'
):
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
            filter_matrix_conj = get_filter_matrix_conj_v0(
                Y[s], inverse_power[s], taps, delay
            )
            X = perform_filter_operation_v0(Y, filter_matrix_conj, taps, delay)
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


def wpe(
        Y,
        taps=10,
        delay=3,
        iterations=3,
        psd_context=0,
        statistics_mode='full'
):
    """
    Modular wpe version which is faster than wpe_v0.
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
        G = get_filter_matrix(
            Y=Y[s], Y_tilde=Y_tilde[s], inverse_power=inverse_power[s]
        )
        X = perform_filter_operation(Y=Y, Y_tilde=Y_tilde, filter_matrix=G)
    return X


def wpe_batch(
        Y,
        taps=10,
        delay=3,
        iterations=3,
        psd_context=0,
        statistics_mode='full'
):
    """
    Batch wrapper for wpe.
    """
    ndim = Y.ndim
    if ndim == 2:
        return wpe(
            Y,
            taps=taps,
            delay=delay,
            iterations=iterations,
            psd_context=psd_context,
            statistics_mode=statistics_mode
        )
    elif ndim >= 3:
        shape = Y.shape
        if ndim > 3:
            Y = Y.reshape(np.prod(shape[:-2]), *shape[-2:])

        batch_axis = 0
        F = Y.shape[batch_axis]
        index = [slice(None)] * Y.ndim

        out = []
        for f in range(F):
            index[batch_axis] = f
            out.append(wpe_v6(
                Y=Y[tuple(index)],
                taps=taps,
                delay=delay,
                iterations=iterations,
                psd_context=psd_context,
                statistics_mode=statistics_mode
            ))
        if ndim > 3:
            return np.stack(out, axis=batch_axis).reshape(shape)
        else:
            return np.stack(out, axis=batch_axis)
    else:
        raise NotImplementedError('Input shape has to be (F, D, T) or (D, T).')


wpe_v6 = wpe
wpe_v7 = wpe
wpe_v8 = wpe_batch


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
        self.kalman_gain = nominator / denominator[:, None]

    def _update_power(self, beta=0.95):
        current_frame = self.buffer[-1]
        current_power = current_frame.real ** 2 + current_frame.imag ** 2
        self.power = self.power * beta + (1 - beta) * np.mean(
                                                        current_power, axis=-1)

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


def recursive_wpe(
        Y,
        taps=10,
        delay=3,
        alpha=0.9999,
):
    """Applies WPE in a framewise recursive fashion.

        Args:
            Y : Observed signal of shape (F, D, T)
            power_estimate: Estimate for the clean signal PSD of shape (F, T)
            alpha (float): Smoothing factor for the recursion
            taps (int, optional): Number of filter taps.
            delay (int, optional): Delay
        Returns:
            Enhanced signal (F, D, T)
        """
    F, D, T = Y.shape[:]
    wpe = OnlineWPE(
        taps,
        delay,
        alpha,
        channel=D,
        frequency_bins=F
    )
    Y = Y.transpose(2, 0, 1)
    prediction = []
    for frame in Y:
        prediction.append(wpe.step_frame(frame))

    return np.stack(prediction).transpose(1, 2, 0)


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
        npad = np.zeros([x.ndim, 2], dtype=np.int)
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
        return cumsum[first_slice] - cumsum[second_slice]

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

    >>> s = 1 / np.array([np.arange(1, 6)]*3)
    >>> get_power_inverse(s)
    array([ 1.,  4.,  9., 16., 25.])
    >>> get_power_inverse(s * 0 + 1, 1)
    array([1., 1., 1., 1., 1.])
    >>> get_power_inverse(s, 1)
    array([ 1.6       ,  2.20408163,  7.08196721, 14.04421326, 19.51219512])
    >>> get_power_inverse(s, np.inf)
    array([3.41620801, 3.41620801, 3.41620801, 3.41620801, 3.41620801])
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


def get_filter_matrix(Y, Y_tilde, inverse_power):
    Y_tilde_inverse_power = Y_tilde * inverse_power[..., None, :]
    R = np.matmul(Y_tilde_inverse_power, hermite(Y_tilde))
    P = np.matmul(Y_tilde_inverse_power, hermite(Y))
    return stable_solve(R, P)


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


def get_filter_matrix_conj_v0(Y, inverse_power, taps, delay):
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


def perform_filter_operation_v0(Y, filter_matrix_conj, taps, delay):
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



def perform_filter_operation(Y, Y_tilde, filter_matrix):
    return Y - np.matmul(hermite(filter_matrix), Y_tilde)


@click.command()
@click.argument(
    'files', nargs=-1,
    type=click.Path(exists=True),
)
@click.option(
    '--output_dir',
    default=None,
    help='Output directory.'
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
@click.option(
    '--taps',
    default=10,
    help='Number of filter taps of WPE'
)
@click.option(
    '--psd_context',
    default=0,
    help='Left and right hand context'
)
def main(files, output_dir, delay, iterations, taps, psd_context):
    """
    User interface for WPE. The defaults of the command line interface are
    suited for example audio files of nara_wpe.

     'Yoshioka2012GeneralWPE'
        sampling_rate = 8000
        delay = 2
        iterations = 2

    """
    stft_options = dict(
        size=512,
        shift=128,
        window_length=None,
        fading=True,
        pad=True,
        symmetric_window=False
    )

    if len(files) > 1:
        signal_list = [
            sf.read(str(file))[0]
            for file in files
            ]
        y = np.stack(signal_list, axis=0)
        sampling_rate = sf.read(str(files[0]))[1]
    else:
        y, sampling_rate = sf.read(files)
        y = y.transpose(1, 0)

    Y = stft(y, **stft_options).transpose(2, 0, 1)
    Z = wpe(Y, taps, delay, iterations, psd_context).transpose(1, 2, 0)
    x = istft(Z, size=stft_options['size'], shift=stft_options['shift'])

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        if len(files) > 1:
            for i, file in enumerate(files):
                sf.write(
                    str(output_dir / Path(file).name),
                    x[i],
                    samplerate=sampling_rate
                )
        else:
            sf.write(
                str(output_dir / Path(files).name),
                x,
                samplerate=sampling_rate
            )
    else:
        # TODO: this does not work, yet. Idea: Usage of pipelining
        import sys
        sys.stdout.write(str(x.tobytes()))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
