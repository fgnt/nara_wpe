import numpy as np
import torch


def torch_segment_axis(
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
        >>> import torch
        >>> torch_segment_axis(torch.arange(10), 4, 2)
        tensor([[0, 1, 2, 3],
                [2, 3, 4, 5],
                [4, 5, 6, 7],
                [6, 7, 8, 9]])
        >>> torch_segment_axis(torch.arange(5).reshape(5), 4, 1, axis=0)
        tensor([[0, 1, 2, 3],
                [1, 2, 3, 4]])
        >>> torch_segment_axis(torch.arange(10).reshape(2, 5), 4, 1, axis=-1)
        tensor([[[0, 1, 2, 3],
                 [1, 2, 3, 4]],
        <BLANKLINE>
                [[5, 6, 7, 8],
                 [6, 7, 8, 9]]])
        >>> torch_segment_axis(torch.arange(10).reshape(5, 2).t(), 4, 1, axis=1)
        tensor([[[0, 2, 4, 6],
                 [2, 4, 6, 8]],
        <BLANKLINE>
                [[1, 3, 5, 7],
                 [3, 5, 7, 9]]])
        >>> torch_segment_axis(torch.flip(torch.arange(10), [0]), 4, 2)
        tensor([[9, 8, 7, 6],
                [7, 6, 5, 4],
                [5, 4, 3, 2],
                [3, 2, 1, 0]])
        >>> a = torch.arange(5).reshape(5)
        >>> b = torch_segment_axis(a, 4, 2, axis=0)
        >>> a += 1  # a and b point to the same memory
        >>> b
        tensor([[1, 2, 3, 4]])

    """
    x: torch.tensor

    axis = axis % x.ndimension()
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

    strides = list(x.stride())
    strides.insert(axis, shift * strides[axis])

    # raise AssertionError(strides, shape, x.shape, elements, shift, length, shift)

    return x.clone().set_(x.storage(), x.storage_offset(), stride=strides, size=shape)


def torch_moveaxis(x: torch.tensor, source, destination):
    """

    >>> torch_moveaxis(torch.ones(2, 25), 1, 0).shape
    torch.Size([25, 2])
    >>> torch_moveaxis(torch.ones(2, 25), -1, -2).shape
    torch.Size([25, 2])
    >>> torch_moveaxis(torch.ones(2, 25), 0, 1).shape
    torch.Size([25, 2])
    >>> torch_moveaxis(torch.ones(2, 25), -2, -1).shape
    torch.Size([25, 2])
    """
    ndim = x.ndimension()
    permutation = list(range(ndim))
    source = permutation.pop(source)
    permutation.insert(destination % ndim, source)
    return x.permute(*permutation)


def build_y_tilde(Y, taps, delay):
    """

    Note: The returned y_tilde consumes a similar amount of memory as Y, because
        of tricks with strides. Usually the memory consumprion is K times
        smaller than the memory consumprion of a contignous array,

    >>> T, D = 20, 2
    >>> Y = torch.arange(start=1, end=T * D + 1).to(dtype=torch.complex128).reshape([T, D]).t()
    >>> print(Y.numpy())
    [[ 1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39]
     [ 2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40]]
    >>> taps, delay = 4, 2
    >>> Y_tilde = build_y_tilde(Y, taps, delay)
    >>> print(Y_tilde.shape, (taps*D, T))
    torch.Size([8, 20]) (8, 20)
    >>> print(Y_tilde.numpy())
    [[ 0  0  0  0  0  1  3  5  7  9 11 13 15 17 19 21 23 25 27 29]
     [ 0  0  0  0  0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30]
     [ 0  0  0  0  1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31]
     [ 0  0  0  0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32]
     [ 0  0  0  1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33]
     [ 0  0  0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34]
     [ 0  0  1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33 35]
     [ 0  0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34 36]]
    >>> Y_tilde = build_y_tilde(Y, taps, 0)
    >>> print(Y_tilde.shape, (taps*D, T), Y_tilde.stride())
    torch.Size([8, 20]) (8, 20) (1, 2)
    >>> print('Pseudo size:', np.prod(Y_tilde.size()) * Y_tilde.element_size())
    Pseudo size: 1280
    >>> print('Reak size:', Y_tilde.storage().size() * Y_tilde.storage().element_size())
    Reak size: 368
    >>> print(Y_tilde.numpy())
    [[ 0  0  0  1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33]
     [ 0  0  0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34]
     [ 0  0  1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33 35]
     [ 0  0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34 36]
     [ 0  1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33 35 37]
     [ 0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38]
     [ 1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39]
     [ 2  4  6  8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40]]

    The first columns are zero because of the delay.

    """
    S = Y.shape[:-2]
    D = Y.shape[-2]
    T = Y.shape[-1]

    def pad(x, axis=-1, pad_width=taps + delay - 1):
        npad = np.zeros([x.ndimension(), 2], dtype=np.int)
        npad[axis, 0] = pad_width
        # x_np = (np.pad(x.numpy(),
        #            pad_width=npad,
        #            mode='constant',
        #            constant_values=0))
        x = torch.nn.functional.pad(
            x,
            pad=npad[::-1].ravel().tolist(),
            mode='constant',
            value=0,
        )
        # assert x_np.shape == x.shape, (x_np.shape, x.shape)
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
    Y_ = torch_moveaxis(Y_, -1, -2)
    Y_ = torch.flip(Y_, dims=[-1 % Y_.ndimension()])
    Y_ = Y_.contiguous()  # Y_ = np.ascontiguousarray(Y_)
    Y_ = torch.flip(Y_, dims=[-1 % Y_.ndimension()])
    Y_ = torch_segment_axis(Y_, taps, 1, axis=-2)

    # Pytorch does not support negative strides.
    # Without this flip, the output of this function does not match the
    # analytical form, but the output of WPE will be equal.
    # Y_ = torch.flip(Y_, dims=[-2 % Y_.ndimension()])

    if delay > 0:
        Y_ = Y_[..., :-delay, :, :]
    Y_ = torch.reshape(Y_, list(S) + [T, taps * D])
    Y_ = torch_moveaxis(Y_, -2, -1)

    return Y_


def get_power_inverse(signal, psd_context=0):
    """
    Assumes single frequency bin with shape (D, T).

    >>> s = 1 / torch.tensor([np.arange(1, 6).astype(np.complex128)]*3)
    >>> get_power_inverse(s).numpy()
    array([ 1.,  4.,  9., 16., 25.])
    >>> get_power_inverse(s * 0 + 1, 1).numpy()
    array([1., 1., 1., 1., 1.])
    >>> get_power_inverse(s, 1).numpy()
    array([ 1.6       ,  2.20408163,  7.08196721, 14.04421326, 19.51219512])
    >>> get_power_inverse(s, np.inf).numpy()
    array([3.41620801, 3.41620801, 3.41620801, 3.41620801, 3.41620801])
    """
    power = torch.mean(torch.abs(signal)**2, dim=-2)

    if np.isposinf(psd_context):
        raise NotImplementedError(psd_context)
        # power = torch.broadcast_to(torch.mean(power, dim=-1, keepdims=True), power.shape)
    elif psd_context > 0:
        raise NotImplementedError(psd_context)
        #     assert int(psd_context) == psd_context, psd_context
        #     psd_context = int(psd_context)
        #     # import bottleneck as bn
        #     # Handle the corner case correctly (i.e. sum() / count)
        #     # Use bottleneck when only left context is requested
        #     # power = bn.move_mean(power, psd_context*2+1, min_count=1)
        #     power = window_mean(power, (psd_context, psd_context))
    elif psd_context == 0:
        pass
    else:
        raise ValueError(psd_context)
    eps = 1e-10 * torch.max(power)
    inverse_power = 1 / torch.max(power, eps)
    return inverse_power


def hermite(x):
    return x.transpose(-2, -1)  #.conj()


def wpe_v6(Y, taps=10, delay=3, iterations=3, psd_context=0, statistics_mode='full'):
    """
    Short of wpe_v7 with no extern references.
    Applicable in for-loops.

    >>> T = np.random.randint(100, 120)
    >>> D = np.random.randint(2, 8)
    >>> K = np.random.randint(3, 5)
    >>> delay = np.random.randint(0, 2)
    
    # Real test:
    >>> Y = np.random.normal(size=(D, T))
    >>> from nara_wpe import wpe as np_wpe
    >>> desired = np_wpe.wpe_v6(Y, K, delay, statistics_mode='full')
    >>> actual = wpe_v6(torch.tensor(Y), K, delay, statistics_mode='full').numpy()
    >>> np.testing.assert_allclose(actual, desired, atol=1e-6)

    # Complex test:
    >>> Y = np.random.normal(size=(D, T)) + 1j * np.random.normal(size=(D, T))
    >>> from nara_wpe import wpe as np_wpe
    >>> desired = np_wpe.wpe_v6(Y, K, delay, statistics_mode='full')
    >>> actual = wpe_v6(torch.tensor(Y), K, delay, statistics_mode='full').numpy()
    >>> np.testing.assert_allclose(actual, desired, atol=1e-6)
    """

    if statistics_mode == 'full':
        s = Ellipsis
    elif statistics_mode == 'valid':
        s = (Ellipsis, slice(delay + taps - 1, None))
    else:
        raise ValueError(statistics_mode)

    X = torch.clone(Y)
    Y_tilde = build_y_tilde(Y, taps, delay)
    for iteration in range(iterations):
        inverse_power = get_power_inverse(X, psd_context=psd_context)
        Y_tilde_inverse_power = Y_tilde * inverse_power[..., None, :]
        R = torch.matmul(Y_tilde_inverse_power[s], hermite(Y_tilde[s]))
        P = torch.matmul(Y_tilde_inverse_power[s], hermite(Y[s]))
        # G = _stable_solve(R, P)
        G, _ = torch.solve(P, R)
        X = Y - torch.matmul(hermite(G), Y_tilde)

    return X
