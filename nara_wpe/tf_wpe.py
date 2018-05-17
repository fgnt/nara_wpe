import tensorflow as tf
from tensorflow.contrib import signal as tf_signal
from tensorflow.contrib.compiler.jit import experimental_jit_scope as jit_scope
from collections import namedtuple

_MaskTypes = namedtuple(
    'MaskTypes', ['DIRECT', 'RATIO', 'DIRECT_INV', 'RATIO_INV'])

MASK_TYPES = _MaskTypes('direct', 'ratio', 'direct_inverse', 'ratio_inverse')


def get_power_inverse(signal, channel_axis=0):
    """Assumes single frequency bin with shape (D, T)."""
    power = tf.reduce_mean(
        tf.real(signal) ** 2 + tf.imag(signal) ** 2, axis=channel_axis)
    eps = 1e-10 * tf.reduce_max(power)
    inverse_power = tf.reciprocal(tf.maximum(power, eps))
    return inverse_power


def get_correlations_narrow(Y, inverse_power, K, delay):
    """

    :param Y: [D, T] `Tensor`
    :param inverse_power: [T] `Tensor`
    :param K: Number of taps
    :param delay: delay
    :return:
    """
    dyn_shape = tf.shape(Y)
    T = dyn_shape[-1]
    D = dyn_shape[0]

    # TODO: Large gains also expected when precalculating Psi.
    # TODO: Small gains expected, when views are pre-calculated in main.
    # TODO: Larger gains expected with scipy.signal.signaltools.fftconvolve().
    # Code without fft will be easier to port to Chainer.
    Psi = tf_signal.frame(Y, K, 1, axis=-1)[:, :T - delay - K + 1, ::-1]
    Psi_conj_norm = (
        tf.cast(inverse_power[None, delay + K - 1:, None], Psi.dtype)
        * tf.conj(Psi)
    )

    correlation_matrix = tf.einsum('dtk,etl->kdle', Psi_conj_norm, Psi)
    correlation_vector = tf.einsum(
        'dtk,et->ked', Psi_conj_norm, Y[:, delay + K - 1:]
    )

    correlation_matrix = tf.reshape(correlation_matrix, (K * D, K * D))
    return correlation_matrix, correlation_vector


def get_correlations(
        Y, inverse_power, K, delay, mask_logits=None,
        mask_type=MASK_TYPES.DIRECT
):
    """

    :param Y: [F, D, T] `Tensor`
    :param inverse_power: [F, T] `Tensor`
    :param K: Number of taps
    :param delay: delay
    :return:
    """
    dyn_shape = tf.shape(Y)
    F = dyn_shape[0]
    D = dyn_shape[1]
    T = dyn_shape[2]

    # TODO: Large gains also expected when precalculating Psi.
    # TODO: Small gains expected, when views are pre-calculated in main.
    # TODO: Larger gains expected with scipy.signal.signaltools.fftconvolve().
    # Code without fft will be easier to port to Chainer.
    # Shape (D, T - K + 1, K)
    # Y = Y / tf.norm(Y, axis=(-2, -1), keep_dims=True)
    Psi = tf_signal.frame(Y, K, 1, axis=-1)[..., :T - delay - K + 1, ::-1]
    Psi_conj_norm = (
        tf.cast(inverse_power[:, None, delay + K - 1:, None], Psi.dtype)
        * tf.conj(Psi)
    )

    if mask_logits is not None:
        # Using logits instead of a 'normal' mask is numerical more stable.
        # There are a few ways to apply the mask:
        # DIRECT: Mask is limited to values between 0 and 1
        # RATIO: Mask values are positive and unlimited
        # *_INV: Use 1-Mask to mask only the reverberation (may be easier
        # to interpret)
        logits = tf.cast(mask_logits[:, None, delay + K - 1:, None], Y.dtype)
        if mask_type == MASK_TYPES.DIRECT or mask_type == MASK_TYPES.DIRECT_INV:
            scale = -1. if mask_type == MASK_TYPES.DIRECT else 1.
            Psi_conj_norm += Psi_conj_norm * tf.exp(scale * logits)
        elif mask_type == MASK_TYPES.RATIO or mask_type == MASK_TYPES.RATIO_INV:
            scale = -1. if mask_type == MASK_TYPES.DIRECT else 1.
            Psi_conj_norm *= tf.exp(scale * logits)

    correlation_matrix = tf.einsum('fdtk,fetl->fkdle', Psi_conj_norm, Psi)
    correlation_vector = tf.einsum(
        'fdtk,fet->fked', Psi_conj_norm, Y[..., delay + K - 1:]
    )

    correlation_matrix = tf.reshape(correlation_matrix, (F, K * D, K * D))
    return correlation_matrix, correlation_vector


def get_filter_matrix_conj(
        Y, inverse_power, correlation_matrix, correlation_vector,
        K, delay, mode='solve'):
    dyn_shape = tf.shape(Y)
    D = dyn_shape[0]

    correlation_vector = tf.reshape(correlation_vector, (D * D * K, 1))
    selector = tf.reshape(tf.transpose(tf.reshape(
        tf.range(D * D * K), (D, K, D)
    ), (1, 0, 2)), (-1,))
    inv_selector = tf.reshape(tf.transpose(tf.reshape(
        tf.range(D * D * K), (K, D, D)
    ), (1, 0, 2)), (-1,))

    correlation_vector = tf.gather(correlation_vector, inv_selector)

    # Idea is to solve matrix inversion independently for each block matrix.
    # This should still be faster and more stable than np.linalg.inv().
    # print(np.linalg.cond(correlation_matrix))

    if mode == 'inv':
        with tf.device('/cpu:0'):
            inv_correlation_matrix = tf.matrix_inverse(correlation_matrix)
        stacked_filter_conj = tf.einsum(
            'ab,cb->ca',
            inv_correlation_matrix, tf.reshape(correlation_vector, (D, D * K))
        )
        stacked_filter_conj = tf.reshape(stacked_filter_conj, (D * D * K, 1))
    elif mode == 'solve':
        with tf.device('/cpu:0'):
            stacked_filter_conj = tf.reshape(
                tf.matrix_solve(
                    tf.tile(correlation_matrix[None, ...], [D, 1, 1]),
                    tf.reshape(correlation_vector, (D, D * K, 1))
                ),
                (D * D * K, 1)
            )
    elif mode == 'solve_ls':
        g = tf.get_default_graph()
        with tf.device('/cpu:0'), g.gradient_override_map(
                {"MatrixSolveLs": "CustomMatrixSolveLs"}):
            stacked_filter_conj = tf.reshape(
                tf.matrix_solve_ls(
                    tf.tile(correlation_matrix[None, ...], [D, 1, 1]),
                    tf.reshape(correlation_vector, (D, D * K, 1)), fast=False
                ),
                (D * D * K, 1)
            )
    else:
        raise ValueError
    stacked_filter_conj = tf.gather(stacked_filter_conj, selector)

    filter_matrix_conj = tf.transpose(
        tf.reshape(stacked_filter_conj, (K, D, D)),
        (0, 2, 1)
    )
    return filter_matrix_conj


def perform_filter_operation(Y, filter_matrix_conj, K, delay):
    """

    :param Y:
    :param filter_matrix_conj: Shape (K, D, D)
    :param K:
    :param delay:
    :return:

    >>> D, T, K, delay = 1, 10, 2, 1
    >>> tf.enable_eager_execution()
    >>> Y = tf.ones([D, T])
    >>> filter_matrix_conj = tf.ones([K, D, D])
    >>> X = perform_filter_operation(Y, filter_matrix_conj, K, delay)
    >>> X.shape
    TensorShape([Dimension(1), Dimension(10)])
    >>> X.numpy()  # Note: The second value should be 0.
    array([[ 1.,  1., -1., -1., -1., -1., -1., -1., -1., -1.]], dtype=float32)
    """
    dyn_shape = tf.shape(Y)
    T = dyn_shape[1]

    # TODO: Second loop can be removed with using segment_axis. No large gain.
    reverb_tail = 0

    def add_tap(accumulated, tau_minus_delay):
        return accumulated + tf.einsum(
            'de,dt',
            filter_matrix_conj[tau_minus_delay, :, :],
            Y[:, (K - 1 - tau_minus_delay):(T - delay - tau_minus_delay)]
        )
    reverb_tail = tf.foldl(
        add_tap, tf.range(0, K),
        initializer=tf.zeros_like(Y[:, (delay + K - 1):])
    )
    return tf.concat(
        [Y[:, :(delay + K - 1)],
         Y[:, (delay + K - 1):] - reverb_tail], axis=-1)


def perform_filter_operation_v2(Y, filter_matrix_conj, K, delay):
    """

    >>> D, T, K, delay = 1, 10, 2, 1
    >>> tf.enable_eager_execution()
    >>> Y = tf.ones([D, T])
    >>> filter_matrix_conj = tf.ones([K, D, D])
    >>> X = perform_filter_operation_v2(Y, filter_matrix_conj, K, delay)
    >>> X.shape
    TensorShape([Dimension(1), Dimension(10)])
    >>> X.numpy()
    array([[ 1.,  0., -1., -1., -1., -1., -1., -1., -1., -1.]], dtype=float32)
    """
    dyn_shape = tf.shape(Y)
    T = dyn_shape[1]

    def add_tap(accumulated, tau_minus_delay):
        new = tf.einsum(
            'de,dt',
            filter_matrix_conj[tau_minus_delay, :, :],
            Y[:, :(T - delay - tau_minus_delay)]
        )
        paddings = tf.convert_to_tensor([[0, 0], [delay + tau_minus_delay, 0]])
        new = tf.pad(new, paddings, "CONSTANT")
        return accumulated + new

    reverb_tail = tf.foldl(
        add_tap, tf.range(0, K),
        initializer=tf.zeros_like(Y)
    )
    return Y - reverb_tail


def single_frequency_wpe(Y, K=10, delay=3, iterations=3, mode='inv'):
    """

    Args:
        Y: Complex valued STFT signal with shape (D, T)
        K: Filter order
        delay: Delay as a guard interval, such that X does not become zero.
        iterations:
        mode: Different implementations for inverse in R^-1 r.

    Returns:

    """

    for iteration in range(iterations):
        with jit_scope():
            inverse_power = get_power_inverse(Y)
        correlation_matrix, correlation_vector = get_correlations_narrow(
            Y, inverse_power, K, delay
        )
        filter_matrix_conj = get_filter_matrix_conj(
            Y, inverse_power, correlation_matrix, correlation_vector,
            K, delay, mode=mode
        )
        with jit_scope():
            Y = perform_filter_operation(Y, filter_matrix_conj, K, delay)
    return Y, inverse_power


def wpe(
        Y, K=10, delay=3, iterations=3, mode='inv'
):
    """WPE for all frequencies at once. Use this for regular processing.

    :param Y:
    :param K:
    :param delay:
    :param iterations:
    :param mode:
    :return:
    """
    def iteration(y):
        enhanced, inverse_power = single_frequency_wpe(
            y, K, delay, iterations, mode=mode)
        return (enhanced, inverse_power)

    enhanced, inverse_power = tf.map_fn(
        iteration, Y, dtype=(Y.dtype, Y.dtype.real_dtype)
    )

    return enhanced, inverse_power


def wpe_step(
        Y, inverse_power, K=10, delay=3, mode='inv', mask_logits=None,
        mask_type='direct', learnable=False, Y_stats=None
):
    """Single WPE step. More suited for backpropagation.

    Args:
        Y: Complex valued STFT signal with shape (F, D, T)
        inverse_power: Power signal with shape (F, T)
        K: Filter order
        delay: Delay as a guard interval, such that X does not become zero.
        mode: Different implementations for inverse in R^-1 r.
        mask_logits: If provided, is used to modify the observed inverse power
            based on the mask_type selector provided.
        mask_type: Different ways how to apply the mask_logits to the
            inverse power.

    Returns:
    """
    F = Y.shape.as_list()[0]
    initial_f = tf.constant(0)

    with tf.name_scope('WPE'):
        with tf.name_scope('correlations'):
            if Y_stats is None:
                Y_stats = Y
            correlation_matrix, correlation_vector = get_correlations(
                Y_stats, inverse_power, K, delay, mask_logits,
                mask_type=mask_type
            )

        def step(inp):
            (Y_f, inverse_power_f,
                correlation_matrix_f, correlation_vector_f) = inp
            with tf.name_scope('filter_matrix'):
                filter_matrix_conj = get_filter_matrix_conj(
                    Y_f, inverse_power_f,
                    correlation_matrix_f, correlation_vector_f,
                    K, delay, mode=mode
                )
            with tf.name_scope('apply_filter'):
                enhanced = perform_filter_operation(
                    Y_f, filter_matrix_conj, K, delay)
            return enhanced, filter_matrix_conj

        enhanced, filter_matrix_conj = tf.map_fn(
            step,
            (Y, inverse_power, correlation_matrix, correlation_vector),
            dtype=(Y.dtype, Y.dtype),
            parallel_iterations=100
        )

        return enhanced, correlation_matrix, correlation_vector


def block_wpe_step(
    Y, inverse_power, K=10, delay=3, mode='inv',
    block_length_in_seconds=2., forgetting_factor=0.7,
    fft_shift=256, sampling_rate=16000
):
    """Applies wpe in a block-wise fashion.

    Args:
        Y (tf.Tensor): Complex valued STFT signal with shape (F, D, T)
        inverse_power ([type]): Power signal with shape (F, T)
        K (int, optional): Defaults to 10. [description]
        delay (int, optional): Defaults to 3. [description]
        mode (str, optional): Defaults to 'inv'. [description]
        mask_logits ([type], optional): Defaults to None. [description]
        mask_type (str, optional): Defaults to 'direct'. [description]
    """
    frames_per_block = block_length_in_seconds * sampling_rate // fft_shift
    frames_per_block = tf.cast(frames_per_block, tf.int32)
    framed_Y = tf_signal.frame(
        Y, frames_per_block, frames_per_block, pad_end=True)
    framed_inverse_power = tf_signal.frame(
        inverse_power, frames_per_block, frames_per_block, pad_end=True)
    num_blocks = tf.shape(framed_Y)[-2]

    enhanced_arr = tf.TensorArray(
        framed_Y.dtype, size=num_blocks, clear_after_read=True)
    start_block = tf.constant(0)
    correlation_matrix, correlation_vector = get_correlations(
        framed_Y[..., start_block, :], framed_inverse_power[..., start_block, :],
        K, delay
    )
    num_bins = Y.shape[0]
    num_channels = Y.shape[1].value
    if num_channels is None:
        num_channels = tf.shape(Y)[1]
    num_frames = tf.shape(Y)[-1]

    def cond(k, *_):
        return k < num_blocks

    with tf.name_scope('block_WPE'):
        def block_step(
                k, enhanced, correlation_matrix_tm1, correlation_vector_tm1):

            def _init_step():
                return correlation_matrix_tm1, correlation_vector_tm1

            def _update_step():
                correlation_matrix, correlation_vector = get_correlations(
                    framed_Y[..., k, :], framed_inverse_power[..., k, :],
                    K, delay
                )
                return (
                    (1. - forgetting_factor) * correlation_matrix_tm1
                    + forgetting_factor * correlation_matrix,
                    (1. - forgetting_factor) * correlation_vector_tm1
                    + forgetting_factor * correlation_vector
                )

            correlation_matrix, correlation_vector = tf.case(
                ((tf.equal(k, 0), _init_step),), default=_update_step
            )

            def step(inp):
                (Y_f, inverse_power_f,
                    correlation_matrix_f, correlation_vector_f) = inp
                with tf.name_scope('filter_matrix'):
                    filter_matrix_conj = get_filter_matrix_conj(
                        Y_f, inverse_power_f,
                        correlation_matrix_f, correlation_vector_f,
                        K, delay, mode=mode
                    )
                with tf.name_scope('apply_filter'):
                    enhanced_f = perform_filter_operation(
                        Y_f, filter_matrix_conj, K, delay)
                return enhanced_f

            enhanced_block = tf.map_fn(
                step,
                (framed_Y[..., k, :], framed_inverse_power[..., k, :],
                 correlation_matrix, correlation_vector),
                dtype=framed_Y.dtype,
                parallel_iterations=100
            )

            enhanced = enhanced.write(k, enhanced_block)
            return k + 1, enhanced, correlation_matrix, correlation_vector

        _, enhanced_arr, _, _ = tf.while_loop(
            cond, block_step,
            (start_block, enhanced_arr, correlation_matrix, correlation_vector)
        )

        enhanced = enhanced_arr.stack()
        enhanced = tf.transpose(enhanced, (1, 2, 0, 3))
        enhanced = tf.reshape(enhanced, (num_bins, num_channels, -1))

        return enhanced[..., :num_frames]


def batched_apply_filter(Y, filter_matrix_conj, num_frames, K=10, delay=3):
    """Applies WPE filter for a batch of examples.

    Args:
        Y (tf.Tensor): shape (B, F, D, T)
        filter_matrix_conj (tf.Tensor): shape (B, F, K, D, D)
        num_frames (tf.Tensor): shape (B,)
        K (int, optional): Defaults to 10.
        delay (int, optional): Defaults to 3.

    Returns:
        tf.Tensor: dereveberated signal
    """

    max_frames = tf.reduce_max(num_frames)

    def batch_step(inp):
        Y_b, frames, filter_matrix_conj_b = inp

        def _pad(x):
            padding = max_frames - frames
            zeros = tf.cast(tf.zeros(()), Y_b.dtype)
            paddings = (len(x.shape) - 1) * ((0, 0),) + ((0, padding),)
            return tf.pad(x, paddings, constant_values=zeros)

        def freq_step(inp):
            Y_b_f, filter_matrix_conj_b_f = inp
            return perform_filter_operation(
                Y_b_f, filter_matrix_conj_b_f, K, delay)

        enhanced_b = tf.map_fn(
            freq_step, (Y_b[..., :frames], filter_matrix_conj_b), dtype=Y_b.dtype)

        return _pad(enhanced_b)

    return tf.map_fn(
        batch_step, (Y, num_frames, filter_matrix_conj),
        dtype=Y.dtype
    )


def single_example_online_dereverb(
        observation, power_estimate, alpha, taps=10, delay=2,
        only_use_final_filters=False, update_power_estimate=False):
    """
    Args:
        observation: (frames, bins, channels)
        power_estimate: (frames, bins)
        alpha: scalar

    """
    num_frames = tf.shape(observation)[0]
    num_bins = observation.shape[1]
    num_ch = tf.shape(observation)[-1]
    dtype = observation.dtype
    k = tf.constant(0)

    inv_cov_tm1 = tf.eye(num_ch * taps, batch_shape=[num_bins], dtype=dtype)
    filter_taps_tm1 = tf.zeros((num_bins, num_ch * taps, num_ch), dtype=dtype)
    enhanced_arr = tf.TensorArray(
        dtype, size=num_frames, clear_after_read=False)

    def dereverb_step(k_, inv_cov_tm1, filter_taps_tm1, enhanced):
        def copy_through():
            enhanced_k = enhanced.write(k_, observation[k_])
            return k_ + 1, inv_cov_tm1, filter_taps_tm1, enhanced_k

        def update():
            window = observation[k_ - delay - taps:k_ - delay][::-1]
            window = tf.reshape(
                tf.transpose(window, (1, 2, 0)), (-1, taps * num_ch)
            )
            window_conj = tf.conj(window)
            pred = (
                observation[k_] -
                tf.einsum('lim,li->lm', tf.conj(filter_taps_tm1), window)
            )

            if update_power_estimate:
                cur_power_estimate = tf.reduce_mean(
                    tf.stack([enhanced.read(k_ - 1), pred], axis=-1),
                    axis=(-2, -1)
                )
            else:
                cur_power_estimate = power_estimate[k_]

            nominator = tf.einsum('lij,lj->li', inv_cov_tm1, window)
            denominator = tf.cast(alpha * cur_power_estimate, window.dtype)
            denominator += tf.einsum('li,li->l', window_conj, nominator)
            kalman_gain = nominator / denominator[:, None]

            _gain_window = tf.einsum('li,lj->lij', kalman_gain, window_conj)
            inv_cov_k = 1. / alpha * (
                inv_cov_tm1 - tf.einsum(
                    'lij,ljm->lim', _gain_window, inv_cov_tm1)
            )

            filter_taps_k = (
                filter_taps_tm1 +
                tf.einsum('li,lm->lim', kalman_gain, tf.conj(pred))
            )
            enhanced_k = enhanced.write(k_, pred)
            return k_ + 1, inv_cov_k, filter_taps_k, enhanced_k

        return tf.case(
            [(tf.less(k_, taps + delay), copy_through)], default=update
        )

    def cond(k, *_):
        return tf.less(k, num_frames)

    _, _, final_filter_taps, enhanced = tf.while_loop(
        cond, dereverb_step, (k, inv_cov_tm1, filter_taps_tm1, enhanced_arr))

    # Only for testing / oracle purposes
    def dereverb_with_filters(k_, filter_taps, enhanced):
        def copy_through():
            enhanced_k = enhanced.write(k_, observation[k_])
            return k_ + 1, filter_taps, enhanced_k

        def update():
            window = observation[k_ - delay - taps:k_ - delay][::-1]
            window = tf.reshape(
                tf.transpose(window, (1, 2, 0)), (-1, taps * num_ch)
            )
            pred = (
                observation[k_] -
                tf.einsum('lim,li->lm', tf.conj(filter_taps), window)
            )
            enhanced_k = enhanced.write(k_, pred)
            return k_ + 1, filter_taps, enhanced_k

        return tf.case(
            [(tf.less(k_, taps + delay), copy_through)], default=update
        )

    if only_use_final_filters:
        k = tf.constant(0)
        enhanced_arr = tf.TensorArray(
            dtype, size=num_frames, clear_after_read=False)
        _, _, enhanced = tf.while_loop(
            cond, dereverb_with_filters, (k, final_filter_taps, enhanced_arr))

    return enhanced.stack()
