import numpy as np
from numpy import (arange, array, pi, cos, exp, log10, ones_like, sqrt, zeros)

_ERB_L = 24.7
_ERB_Q = 9.265


def hertz_to_erbscale(frequency):
    """Returns ERB-frequency from frequency in Hz.
    Implements Equation 16 in [Hohmann2002]_.

    Parameters
    ----------
    frequency : scalar
        The Frequency in Hertz.

    Returns
    -------
    erb : scalar
        The corresponding value on the ERB-Scale.

    """
    return _ERB_Q * np.log(1 + frequency / (_ERB_L * _ERB_Q))


def erbscale_to_hertz(erb):
    """Returns frequency in Hertz from ERB value.
    Implements Equation 17 in [Hohmann2002]_.

    Parameters
    ----------
    erb : scalar
        The corresponding value on the ERB-Scale.

    Returns
    -------
    frequency : scalar
        The Frequency in Hertz.

    """
    return (exp(erb / _ERB_Q) - 1) * _ERB_L * _ERB_Q


def band_index_to_hertz(band_index, norm_freq=2000):
    erb_norm = _ERB_Q * np.log(1 + norm_freq / (_ERB_Q * _ERB_L))  # norm_freq = 2000Hz → ERB 尺度
    erb_values = erb_norm + band_index  # shift: band index → ERB scale
    freqs = (np.exp(erb_values / _ERB_Q) - 1) * _ERB_Q * _ERB_L  # ERB scale → Hz
    return freqs


if __name__ == '__main__':
    erb_locations = np.arange(-5, 18, step=0.72)
    # 得到相对于参考频率 norm_freq 2000HZ 的 band index
    # [-5. - 4.28 - 3.56 - 2.84 - 2.12 - 1.4 - 0.68  0.04  0.76  1.48  2.2   2.92
    #  3.64  4.36  5.08  5.8   6.52  7.24  7.96  8.68  9.4  10.12 10.84 11.56
    #  12.28 13.   13.72 14.44 15.16 15.88 16.6  17.32]
    center_freqs = band_index_to_hertz(erb_locations)

    print(center_freqs)
    # [1070.44498013  1175.44213048  1288.92422067  1411.57692843
    #  1544.14134183  1687.41843716  1842.27391851  2009.64344848
    #  2190.53830168  2386.05147494  2597.36429144  2825.75353842
    #  3072.59918177  3339.39270401  3627.74611603  3939.40169722
    #  4276.24252252  4640.30384032  5033.78536977  5459.06459183
    #  5918.71111444  6415.50219846  6952.4395384   7532.76739915
    #  8159.99221833  8837.90379282  9570.59817731 10362.50243335
    #  11218.40137842 12143.46649655 13143.28718534 14223.90452797]
    print("数量:", len(center_freqs))
