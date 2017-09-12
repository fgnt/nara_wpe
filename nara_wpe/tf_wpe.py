import tensorflow as tf
from tensorflow.contrib import signal as tf_signal
from tensorflow.contrib.compiler.jit import experimental_jit_scope as jit_scope
import nara_wpe.gradient_overrides
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
    # Shape (D, T - K + 1, K)
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
    Y = Y / tf.norm(Y, axis=(-2, -1), keep_dims=True)
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
    """
    dyn_shape = tf.shape(Y)
    T = dyn_shape[1]

    # TODO: Second loop can be removed with using segment_axis. No large gain.
    reverb_tail = list()
    for tau_minus_delay in range(0, K):
        reverb_tail.append(tf.einsum(
            'de,dt',
            filter_matrix_conj[tau_minus_delay, :, :],
            Y[:, (K - 1 - tau_minus_delay):(T - delay - tau_minus_delay)]
        ))
    reverb_tail = tf.add_n(reverb_tail)
    return tf.concat(
        [Y[:, :(delay + K - 1)],
         Y[:, (delay + K - 1):] - reverb_tail], axis=-1)


def single_frequency_wpe(Y, K=10, delay=3, iterations=3, use_inv=False):
    """

    Args:
        Y: Complex valued STFT signal with shape (D, T)
        K: Filter order
        delay: Delay as a guard interval, such that X does not become zero.
        iterations:

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
            K, delay, use_inv=use_inv
        )
        with jit_scope():
            Y = perform_filter_operation(Y, filter_matrix_conj, K, delay)
    return Y


def wpe(
        Y, K=10, delay=3, iterations=3, mode='solve'
):
    F = Y.shape.as_list()[0]
    outputs = tf.TensorArray(Y.dtype, size=F)
    initial_f = tf.constant(0)

    def should_continue(f, *args):
        return f < F

    def iteration(f, outputs_):
        enhanced = single_frequency_wpe(
            Y[f], K, delay, iterations, mode=mode)
        outputs_ = outputs_.write(f, enhanced)
        return f + 1, outputs_

    _, enhanced_array = tf.while_loop(
        should_continue, iteration, [initial_f, outputs]
    )

    return enhanced_array.stack()


def wpe_step(
        Y, inverse_power, K=10, delay=3, use_inv=False, mask_logits=None,
        mask_type='direct'
):
    """

    Args:
        Y: Complex valued STFT signal with shape (F, D, T)
        inverse_power: Power signal with shape (F, T)
        K: Filter order
        delay: Delay as a guard interval, such that X does not become zero.
        iterations:

    Returns:

    """
    F = Y.shape.as_list()[0]
    outputs = tf.TensorArray(Y.dtype, size=F)
    initial_f = tf.constant(0)

    with tf.name_scope('WPE'):
        with tf.name_scope('correlations'):
            correlation_matrix, correlation_vector = get_correlations(
                Y, inverse_power, K, delay, mask_logits, mask_type=mask_type
            )

        def should_continue(f, *args):
            return f < F

        def iteration(f, outputs_):
            with tf.name_scope('filter_matrix'):
                filter_matrix_conj = get_filter_matrix_conj(
                    Y[f], inverse_power[f],
                    correlation_matrix[f], correlation_vector[f],
                    K, delay, mode=mode
                )
            with tf.name_scope('apply_filter'):
                enhanced = perform_filter_operation(
                    Y[f], filter_matrix_conj, K, delay)
            outputs_ = outputs_.write(f, enhanced)
            return f + 1, outputs_

        _, enhanced_array = tf.while_loop(
            should_continue, iteration,
            [initial_f, outputs]
        )

        return enhanced_array.stack(), \
               correlation_matrix, correlation_vector
