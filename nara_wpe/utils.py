"""
This file contains the STFT function and related helper functions.
"""
import numpy as np
from math import ceil
import scipy
import functools
import operator

from scipy import signal
from numpy.fft import rfft, irfft

import string


def hermite(x):
    return x.swapaxes(-2, -1).conj()


def stable_solve(A, B):
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
    >>> C3 = stable_solve(A, B)
    >>> C4 = lstsq(A, B)
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
    >>> C3 = stable_solve(A, B)
    >>> C4 = lstsq(A, B)
    >>> np.testing.assert_allclose(C2, C3)
    >>> np.testing.assert_allclose(C2, C4)

    >>> A = normal((3, 6, 6))
    >>> B = normal((3, 6, 6))
    >>> C1 = np.linalg.solve(A, B)
    >>> C2, *_ = np.linalg.lstsq(A, B)
    Traceback (most recent call last):
    ...
    numpy.linalg.linalg.LinAlgError: 3-dimensional array given. Array must be two-dimensional
    >>> C3 = stable_solve(A, B)
    >>> C4 = lstsq(A, B)
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
    >>> C3 = stable_solve(A, B)
    >>> C4 = lstsq(A, B)
    >>> np.testing.assert_allclose(C3, C4)


    """
    assert A.shape[:-2] == B.shape[:-2], (A.shape, B.shape)
    assert A.shape[-1] == B.shape[-2], (A.shape, B.shape)
    try:
        return np.linalg.solve(A, B)
    except np.linalg.linalg.LinAlgError:
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
            except np.linalg.linalg.LinAlgError:
                C[i] = np.linalg.lstsq(A[i], B[i])[0]
        return C.reshape(*shape_B)


def get_working_shape(shape):
    "Flattens all but the last two dimension."
    product = functools.reduce(operator.mul, [1] + list(shape[:-2]))
    return [product] + list(shape[-2:])


def lstsq(A, B):
    assert A.shape == B.shape, (A.shape, B.shape)
    shape = A.shape

    working_shape = get_working_shape(shape)

    A = A.reshape(working_shape)
    B = B.reshape(working_shape)

    C = np.zeros_like(A)
    for i in range(working_shape[0]):
        C[i] = np.linalg.lstsq(A[i], B[i])[0]
    return C.reshape(*shape)


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
        pad_value: The value to use for end='pad'

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

    if x.__class__.__module__ == 'cupy.core.core':
        import cupy
        xp = cupy
    else:
        xp = np

    axis = axis % x.ndim
    elements = x.shape[axis]

    if shift <= 0:
        raise ValueError('Can not shift forward by less than 1 element.')

    # Pad
    if end == 'pad':
        npad = np.zeros([x.ndim, 2], dtype=np.int)
        pad_fn = functools.partial(
            xp.pad, pad_width=npad, mode=pad_mode, constant_values=pad_value
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

    if xp == np:
        return np.lib.stride_tricks.as_strided(x, strides=strides, shape=shape)
    else:
        x = x.view()
        x._set_shape_and_strides(strides=strides, shape=shape)
        return x


def segment_axis_v2(
        x,
        length,
        shift,
        axis=-1,
        end='cut',
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


# http://stackoverflow.com/a/3153267
def roll_zeropad(a, shift, axis=None):
    """
    Roll array elements along a given axis.

    Elements off the end of the array are treated as zeros.

   Args:
        a: array_like
            Input array.
        shift: int
            The number of places by which elements are shifted.
        axis (int): optional,
            The axis along which elements are shifted.  By default, the array
            is flattened before shifting, after which the original
            shape is restored.

    Returns:
        ndarray: Output array, with the same shape as `a`.

    Note:
        roll     : Elements that roll off one end come back on the other.
        rollaxis : Roll the specified axis backwards, until it lies in a
                   given position.

    Examples:
        >>> x = np.arange(10)
        >>> roll_zeropad(x, 2)
        array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
        >>> roll_zeropad(x, -2)
        array([2, 3, 4, 5, 6, 7, 8, 9, 0, 0])

        >>> x2 = np.reshape(x, (2,5))
        >>> x2
        array([[0, 1, 2, 3, 4],
               [5, 6, 7, 8, 9]])
        >>> roll_zeropad(x2, 1)
        array([[0, 0, 1, 2, 3],
               [4, 5, 6, 7, 8]])
        >>> roll_zeropad(x2, -2)
        array([[2, 3, 4, 5, 6],
               [7, 8, 9, 0, 0]])
        >>> roll_zeropad(x2, 1, axis=0)
        array([[0, 0, 0, 0, 0],
               [0, 1, 2, 3, 4]])
        >>> roll_zeropad(x2, -1, axis=0)
        array([[5, 6, 7, 8, 9],
               [0, 0, 0, 0, 0]])
        >>> roll_zeropad(x2, 1, axis=1)
        array([[0, 0, 1, 2, 3],
               [0, 5, 6, 7, 8]])
        >>> roll_zeropad(x2, -2, axis=1)
        array([[2, 3, 4, 0, 0],
               [7, 8, 9, 0, 0]])

        >>> roll_zeropad(x2, 50)
        array([[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]])
        >>> roll_zeropad(x2, -50)
        array([[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]])
        >>> roll_zeropad(x2, 0)
        array([[0, 1, 2, 3, 4],
               [5, 6, 7, 8, 9]])

    """
    a = np.asanyarray(a)
    if shift == 0:
        return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n - shift), axis))
        res = np.concatenate((a.take(np.arange(n - shift, n), axis), zeros),
                             axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n - shift, n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n - shift), axis)), axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res


def stft(
        time_signal,
        size,
        shift,
        axis=-1,
        window=signal.blackman,
        window_length=None,
        fading=True,
        pad=True,
        symmetric_window=False,
):
    """
    ToDo: Open points:
     - sym_window need literature
     - fading why it is better?
     - should pad have more degrees of freedom?

    Calculates the short time Fourier transformation of a multi channel multi
    speaker time signal. It is able to add additional zeros for fade-in and
    fade out and should yield an STFT signal which allows perfect
    reconstruction.

    Args:
        time_signal: Multi channel time signal with dimensions
            AA x ... x AZ x T x BA x ... x BZ.
        size: Scalar FFT-size.
        shift: Scalar FFT-shift, the step between successive frames in
            samples. Typically shift is a fraction of size.
        axis: Scalar axis of time.
            Default: None means the biggest dimension.
        window: Window function handle. Default is blackman window.
        fading: Pads the signal with zeros for better reconstruction.
        window_length: Sometimes one desires to use a shorter window than
            the fft size. In that case, the window is padded with zeros.
            The default is to use the fft-size as a window size.
        pad: If true zero pad the signal to match the shape, else cut
        symmetric_window: symmetric or periodic window. Assume window is
            periodic. Since the implementation of the windows in scipy.signal have a
            curious behaviour for odd window_length. Use window(len+1)[:-1]. Since
            is equal to the behaviour of MATLAB.

    Returns:
        Single channel complex STFT signal with dimensions
            AA x ... x AZ x T' times size/2+1 times BA x ... x BZ.
    """
    time_signal = np.array(time_signal)

    axis = axis % time_signal.ndim

    if window_length is None:
        window_length = size

    # Pad with zeros to have enough samples for the window function to fade.
    if fading:
        pad_width = np.zeros((time_signal.ndim, 2), dtype=np.int)
        pad_width[axis, :] = window_length - shift
        time_signal = np.pad(time_signal, pad_width, mode='constant')

    if symmetric_window:
        window = window(window_length)
    else:
        # https://github.com/scipy/scipy/issues/4551
        window = window(window_length + 1)[:-1]

    time_signal_seg = segment_axis_v2(
        time_signal,
        window_length,
        shift=shift,
        axis=axis,
        end='pad' if pad else 'cut'
    )

    letters = string.ascii_lowercase[:time_signal_seg.ndim]
    mapping = letters + ',' + letters[axis + 1] + '->' + letters

    try:
        # ToDo: Implement this more memory efficient
        return rfft(
            np.einsum(mapping, time_signal_seg, window),
            n=size,
            axis=axis + 1
        )
    except ValueError as e:
        raise ValueError(
            'Could not calculate the stft, something does not match.\n' +
            'mapping: {}, '.format(mapping) +
            'time_signal_seg.shape: {}, '.format(time_signal_seg.shape) +
            'window.shape: {}, '.format(window.shape) +
            'size: {}'.format(size) +
            'axis+1: {axis+1}'
        )


def _samples_to_stft_frames(samples, size, shift):
    """
    Calculates STFT frames from samples in time domain.

    Args:
        samples: Number of samples in time domain.
        size: FFT size.
        shift: Hop in samples.

    Returns:
        Number of STFT frames.
    """
    # I changed this from np.ceil to math.ceil, to yield an integer result.
    return ceil((samples - size + shift) / shift)


def _stft_frames_to_samples(frames, size, shift):
    """
    Calculates samples in time domain from STFT frames

    Args:
        frames: Number of STFT frames.
        size: FFT size.
        shift: Hop in samples.

    Returns:
        Number of samples in time domain.
    """
    return frames * shift + size - shift


def _biorthogonal_window_brute_force(analysis_window, shift,
                                     use_amplitude=False):
    """
    The biorthogonal window (synthesis_window) must verify the criterion:
        synthesis_window * analysis_window plus it's shifts must be one.
        1 == sum m from -inf to inf over (synthesis_window(n - mB) * analysis_window(n - mB))
        B ... shift
        n ... time index
        m ... shift index

    Args:
        analysis_window:
        shift:

    """
    size = len(analysis_window)

    influence_width = (size - 1) // shift

    denominator = np.zeros_like(analysis_window)

    if use_amplitude:
        analysis_window_square = analysis_window
    else:
        analysis_window_square = analysis_window ** 2
    for i in range(-influence_width, influence_width + 1):
        denominator += roll_zeropad(analysis_window_square, shift * i)

    if use_amplitude:
        synthesis_window = 1 / denominator
    else:
        synthesis_window = analysis_window / denominator
    return synthesis_window


_biorthogonal_window_fastest = _biorthogonal_window_brute_force


def istft(
        stft_signal,
        size=1024,
        shift=256,
        window=signal.blackman,
        fading=True,
        window_length=None,
        symmetric_window=False,
):
    """
    Calculated the inverse short time Fourier transform to exactly reconstruct
    the time signal.

    Notes:
        Be careful if you make modifications in the frequency domain (e.g.
        beamforming) because the synthesis window is calculated according to
        the unmodified! analysis window.

    Args:
        stft_signal: Single channel complex STFT signal
            with dimensions (..., frames, size/2+1).
        size: Scalar FFT-size.
        shift: Scalar FFT-shift. Typically shift is a fraction of size.
        window: Window function handle.
        fading: Removes the additional padding, if done during STFT.
        window_length: Sometimes one desires to use a shorter window than
            the fft size. In that case, the window is padded with zeros.
            The default is to use the fft-size as a window size.
        symmetric_window: symmetric or periodic window. Assume window is
            periodic. Since the implementation of the windows in scipy.signal have a
            curious behaviour for odd window_length. Use window(len+1)[:-1]. Since
            is equal to the behaviour of MATLAB.

    Returns:
        Single channel complex STFT signal
        Single channel time signal.
    """
    # Note: frame_axis and frequency_axis would make this function much more
    #       complicated
    stft_signal = np.array(stft_signal)

    assert stft_signal.shape[-1] == size // 2 + 1, str(stft_signal.shape)

    if window_length is None:
        window_length = size

    if symmetric_window:
        window = window(window_length)
    else:
        window = window(window_length + 1)[:-1]

    window = _biorthogonal_window_fastest(window, shift)

    # window = _biorthogonal_window_fastest(
    #     window, shift, use_amplitude_for_biorthogonal_window)
    # if disable_sythesis_window:
    #     window = np.ones_like(window)

    time_signal = np.zeros(
        list(stft_signal.shape[:-2]) +
        [stft_signal.shape[-2] * shift + window_length - shift]
    )

    # Get the correct view to time_signal
    time_signal_seg = segment_axis_v2(
        time_signal, window_length, shift, end=None
    )

    # Unbuffered inplace add
    np.add.at(
        time_signal_seg,
        Ellipsis,
        window * np.real(irfft(stft_signal))[..., :window_length]
    )
    # The [..., :window_length] is the inverse of the window padding in rfft.

    # Compensate fade-in and fade-out
    if fading:
        time_signal = time_signal[
            ..., window_length - shift:time_signal.shape[-1] - (window_length - shift)]

    return time_signal


def istft_single_channel(stft_signal, size=1024, shift=256,
          window=signal.blackman, fading=True, window_length=None,
          use_amplitude_for_biorthogonal_window=False,
          disable_sythesis_window=False):
    """
    Calculated the inverse short time Fourier transform to exactly reconstruct
    the time signal.

    Notes:
        Be careful if you make modifications in the frequency domain (e.g.
        beamforming) because the synthesis window is calculated according to the
        unmodified! analysis window.

    Args:
        stft_signal: Single channel complex STFT signal
            with dimensions frames times size/2+1.
        size: Scalar FFT-size.
        shift: Scalar FFT-shift. Typically shift is a fraction of size.
        window: Window function handle.
        fading: Removes the additional padding, if done during STFT.
        window_length: Sometimes one desires to use a shorter window than
            the fft size. In that case, the window is padded with zeros.
            The default is to use the fft-size as a window size.

    Returns:
        Single channel complex STFT signal
        Single channel time signal.
    """
    assert stft_signal.shape[1] == size // 2 + 1, str(stft_signal.shape)

    if window_length is None:
        window = window(size)
    else:
        window = window(window_length)
        window = np.pad(window, (0, size-window_length), mode='constant')
    window = _biorthogonal_window_fastest(window, shift,
                                          use_amplitude_for_biorthogonal_window)
    if disable_sythesis_window:
        window = np.ones_like(window)

    time_signal = scipy.zeros(stft_signal.shape[0] * shift + size - shift)

    for j, i in enumerate(range(0, len(time_signal) - size + shift, shift)):
        time_signal[i:i + size] += window * np.real(irfft(stft_signal[j]))

    # Compensate fade-in and fade-out
    if fading:
        time_signal = time_signal[size-shift:len(time_signal)-(size-shift)]

    return time_signal


def stft_to_spectrogram(stft_signal):
    """
    Calculates the power spectrum (spectrogram) of an stft signal.
    The output is guaranteed to be real.

    Args:
        stft_signal: Complex STFT signal with dimensions
            #time_frames times #frequency_bins.

    Returns:
        Real spectrogram with same dimensions as input.
    """
    spectrogram = stft_signal.real**2 + stft_signal.imag**2
    return spectrogram


def spectrogram(time_signal, *args, **kwargs):
    """
    Thin wrapper of stft with power spectrum calculation.

    Args:
        time_signal:
        *args:
        **kwargs:

    Returns:

    """
    return stft_to_spectrogram(stft(time_signal, *args, **kwargs))


def spectrogram_to_energy_per_frame(spectrogram):
    """
    The energy per frame is sometimes used as an additional feature to the MFCC
    features. Here, it is calculated from the power spectrum.

    Args:
        spectrogram: Real valued power spectrum.

    Returns:
        Real valued energy per frame.
    """
    energy = np.sum(spectrogram, 1)

    # If energy is zero, we get problems with log
    energy = np.where(energy == 0, np.finfo(float).eps, energy)
    return energy


def get_stft_center_frequencies(size=1024, sample_rate=16000):
    """
    It is often necessary to know, which center frequency is
    represented by each frequency bin index.

    Args:
        size: Scalar FFT-size.
        sample_rate: Scalar sample frequency in Hertz.

    Returns:
        Array of all relevant center frequencies
    """
    frequency_index = np.arange(0, size/2 + 1)
    return frequency_index * sample_rate / size
