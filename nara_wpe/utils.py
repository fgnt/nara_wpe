"""
This file contains the STFT function and related helper functions.
"""
import numpy as np
from math import ceil
import scipy

from scipy import signal
from numpy.fft import rfft, irfft

import string

from nara_wpe.wpe import segment_axis as segment_axis_v2


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
        pad_width = np.zeros((time_signal.ndim, 2), dtype=int)
        pad_width[axis, :] = window_length - shift
        time_signal = np.pad(time_signal, pad_width, mode='constant')

    if isinstance(window, str):
        window = getattr(signal.windows, window)

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


def _samples_to_stft_frames(
        samples,
        size,
        shift,
        *,
        pad=True,
        fading=False,
):
    """
    Calculates number of STFT frames from number of samples in time domain.

    Args:
        samples: Number of samples in time domain.
        size: FFT size.
            window_length often equal to FFT size. The name size should be
            marked as deprecated and replaced with window_length.
        shift: Hop in samples.
        pad: See stft.
        fading: See stft. Note to keep old behavior, default value is False.

    Returns:
        Number of STFT frames.

    >>> _samples_to_stft_frames(19, 16, 4)
    2
    >>> _samples_to_stft_frames(20, 16, 4)
    2
    >>> _samples_to_stft_frames(21, 16, 4)
    3

    >>> stft(np.zeros(19), 16, 4, fading=False).shape
    (2, 9)
    >>> stft(np.zeros(20), 16, 4, fading=False).shape
    (2, 9)
    >>> stft(np.zeros(21), 16, 4, fading=False).shape
    (3, 9)

    >>> _samples_to_stft_frames(19, 16, 4, fading=True)
    8
    >>> _samples_to_stft_frames(20, 16, 4, fading=True)
    8
    >>> _samples_to_stft_frames(21, 16, 4, fading=True)
    9

    >>> stft(np.zeros(19), 16, 4).shape
    (8, 9)
    >>> stft(np.zeros(20), 16, 4).shape
    (8, 9)
    >>> stft(np.zeros(21), 16, 4).shape
    (9, 9)

    >>> _samples_to_stft_frames(21, 16, 3, fading=True)
    12
    >>> stft(np.zeros(21), 16, 3).shape
    (12, 9)
    >>> _samples_to_stft_frames(21, 16, 3)
    3
    >>> stft(np.zeros(21), 16, 3, fading=False).shape
    (3, 9)
    """
    if fading:
        samples = samples + 2 * (size - shift)

    # I changed this from np.ceil to math.ceil, to yield an integer result.
    frames = (samples - size + shift) / shift
    if pad:
        return ceil(frames)
    return int(frames)


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

    if isinstance(window, str):
        window = getattr(signal.windows, window)

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
