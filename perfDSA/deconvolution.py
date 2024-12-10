import matplotlib.pyplot as plt
from numpy.linalg import solve
import numpy as np
import math
import random
from scipy import signal
from numpy.fft import fft, ifft, ifftshift
from scipy.optimize import curve_fit
# Assumes ydata = f(xdata, *params) + eps
from math import exp
from scipy.stats import gamma
from scipy.linalg import circulant
from sklearn.decomposition import TruncatedSVD


def convmatrix(h, N):
    """
    Fills a convolution matrix, similar to the matlab convmtx()

    From matlab docs
    A = convmtx(h,n) returns the convolution matrix, A,
    such that the product of A and a vector, x, is the convolution of h and x.

    If h is a column vector of length m, A is (m+n-1)-by-n and the
    product of A and a column vector, x, of length n is the convolution
    of h and x.

    If h is a row vector of length m, A is n-by-(m+n-1) and
    the product of a row vector, x, of length n with A is the
    convolution of h and x.

    convmtx handles edge conditions by zero padding.

    Implementation: two arrays: x, y, with: y = h * x
    and we want to know h;

      y[0] = x[0]*h[0] + x[-1]*h[1] ... + x[-n]*h[n]

    thus

      y[i] = x[i]*h[i] + x[i-1}*h[1] + ... + x[i-n]*h[n]

    and as matrix

      y = H*x

    with
          [ h[0]   0    0    0 .... 0     ]
          [ h[1]  h[0]  0    0 .... 0     ]
          [ ....                          ]
     H =  [ h[i]  h[i-1] ......    h[i-n] ]
          [ ....                          ]
          [ 0       0   0    0 .... h[n]  ]

    """
    H = np.zeros((len(h) + N - 1, N))

    # for all rows
    for i in range(len(h) + N - 1):
        start_j = max(0, i - len(h) + 1)
        end_j = min(N, i + 1)
        for j in range(start_j, end_j):
            H[i, j] = h[i - j]
    return H.T


def convmtx(v, n):
    """Generates a convolution matrix

    Usage: X = convm(v,n)
    Given a vector v of length N, an N+n-1 by n convolution matrix is
    generated of the following form:
              |  v(0)  0      0     ...      0    |
              |  v(1) v(0)    0     ...      0    |
              |  v(2) v(1)   v(0)   ...      0    |
         X =  |   .    .      .              .    |
              |   .    .      .              .    |
              |   .    .      .              .    |
              |  v(N) v(N-1) v(N-2) ...  v(N-n+1) |
              |   0   v(N)   v(N-1) ...  v(N-n+2) |
              |   .    .      .              .    |
              |   .    .      .              .    |
              |   0    0      0     ...    v(N)   |
    And then it's trasposed to fit the MATLAB return value.
    That is, v is assumed to be causal, and zero-valued after N.
    """
    N = len(v) + 2 * n - 2
    xpad = np.concatenate([np.zeros(n - 1), v[:], np.zeros(n - 1)])
    X = np.zeros((len(v) + n - 1, n))
    # Construct X column by column
    for i in range(n):
        X[:, i] = xpad[n - i - 1:N - i]

    return X.transpose()


# spdiags(B,d,m,n) --> m-by-n sparse matrix by taking columns of B and
# placing them along the diagonals specified by d
def mydiags(diag, start, rows, cols):
    A = np.zeros((rows, cols))
    for i in range(rows):
        for (j, v) in enumerate(diag):
            c = i + start[j]
            if c >= 0 and c < cols:
                A[i, c] = v
    return A


def gamma_variate(t, t_0=2, t_max=6, y_max=1, alpha=2, C=0):
    """
    Gamma variate function, that is commonly used to describe time intensity curves
    """
    # formulation according to Madsen (PMB, 1992)
    # normal definition only valid for t >= t_0
    if t < t_0:
        return C
    # linear mapping
    t_prime = (t - t_0) / (t_max - t_0)
    return y_max * t_prime ** alpha * exp(alpha * (1 - t_prime)) + C


def get_gamma_param_estimates(x, y):
    """
    Determines appropriate estimates of the gamma curve parameters
    """
    C = y[0]
    y_max = max(y) - C
    t_max = max(zip(y, x))[1]
    t_0 = 0.5 * t_max
    alpha = 3
    return (t_0, t_max, y_max, alpha, C)


vgamma = np.vectorize(gamma_variate)


def my_gamma_fit(x, y):
    """
    Performs a gamma curve fit for a signal
    """
    initial_params = get_gamma_param_estimates(x, y)
    popt, pcov = curve_fit(vgamma, x, y, p0=initial_params, bounds=(0.0001, np.inf))
    return_dict = {"t_0": popt[0], "t_max": popt[1], "y_max": popt[2], "alpha": popt[3], "C": popt[4]}
    return return_dict


# functions for fitting that does not contain C?

def gamma_variate_no_C(t, t_0=2, t_max=6, y_max=1, alpha=2):
    """
    Gamma variate function, that is commonly used to describe time intensity curves
    """
    # formulation according to Madsen (PMB, 1992)
    # normal definition only valid for t >= t_0
    if t < t_0:
        return 0
    # linear mapping
    t_prime = (t - t_0) / (t_max - t_0)
    return y_max * t_prime ** alpha * exp(alpha * (1 - t_prime))


def get_gamma_param_estimates_no_C(x, y):
    """
    Determines appropriate estimates of the gamma curve parameters
    """
    # C = y[0]
    y_max = max(y)  # - C
    t_max = max(zip(y, x))[1]
    t_0 = 0.5 * t_max
    alpha = 3
    return (t_0, t_max, y_max, alpha)


vgamma_no_C = np.vectorize(gamma_variate_no_C)


def my_gamma_fit_no_C(x, y):
    """
    Performs a gamma curve fit for a signal
    """
    initial_params = get_gamma_param_estimates_no_C(x, y)
    popt, pcov = curve_fit(vgamma_no_C, x, y, p0=initial_params, bounds=(0.0001, np.inf))
    return_dict = {"t_0": popt[0], "t_max": popt[1], "y_max": popt[2], "alpha": popt[3]}
    return return_dict


# try: make artery and brain longer, brainnew_cub_offset_long based on gamma fit,
# artery based on adding zeros. And then do the deconvolution again.

def extend_curve_with_gamma_fit(x, y, s):
    x_ext = np.linspace(0, s * max(x), num=int(len(x) * s), endpoint=True)
    y_gamma = vgamma(x_ext, **my_gamma_fit(x, y))
    y_gamma[:len(y)] = y[:]
    return x_ext, y_gamma


def extend_curve_with_constant(x, y, s, c=0.0):
    x_ext = np.linspace(0, s * max(x), num=int(len(x) * s), endpoint=True)
    y_gamma = x_ext * 0.0 + c
    y_gamma[:len(y)] = y[:]
    return x_ext, y_gamma


def getFlowParameters(times, values):
    """
    Assumes a TIC that is similar to a gamma-variate (i.e. baseline is at zero).
    Will determine the following parameters:
    * integral (area under the curve)
    * t_max, y_max
    * t_max_up_slope, max_up_slope
    * peak_start
    * peak_end
    * peak_width (FWHM)
    * t_arrival (= t_max - 2*(t_max-t_fwhm_start))
    """
    # Sanity check of input values
    if len(values) == 0:
        return dict()
    if len(times) == 0:
        times = np.array(range(len(values)), dtype="float")

    # Get max value
    (max_elem, y_max) = max(enumerate(values), key=(lambda x: x[1]))
    t_max = times[max_elem]

    # Half max
    hm = 0.5 * y_max

    # Initialize
    prevx = times[0]
    prevy = values[0]
    peak_start = None
    peak_end = None
    integral = 0.0
    max_upslope = -np.inf
    t_max_upslope = None

    # For all segments from prevx --x
    for (x, y) in zip(times[1:], values[1:]):
        # Check whether this is the first upslope peak crossing
        if prevy < hm and y >= hm and not peak_start:
            peak_start = prevx + (hm - prevy) / (y - prevy) * (x - prevx)
        # And keep track of the (last) downslope peak crossing
        if prevy > hm and y <= hm:
            peak_end = prevx + (hm - prevy) / (y - prevy) * (x - prevx)

        # Update the integral
        integral += 0.5 * (prevy + y) * (x - prevx)

        # Compute slope
        slope = (y - prevy) / (x - prevx)
        if slope > max_upslope:
            max_upslope = slope
            t_max_upslope = 0.5 * (x + prevx)

        # Prepare for next segment
        prevy = y
        prevx = x

    # Compute t_arrival as the "mirror" location of t_max w.r.t. peak_start
    if peak_start:
        t_arrival = t_max - 2.0 * (t_max - peak_start)
    else:
        t_arrival = None

    # If either peak_start or peak_end is missing
    if peak_start or peak_end:
        # it is filled with the first/last value
        if not peak_start:
            peak_start = times[0]
        if not peak_end:
            peak_end = times[-1]
        # and peak_width is computed
        peak_width = peak_end - peak_start
    else:
        # otherwise can not compute peak_width
        peak_width = None

    # Return all values in a dict
    return_dict = {
        "t_max": t_max,
        "y_max": y_max,
        "t_arrival": t_arrival,
        "max_upslope": max_upslope,
        "t_max_upslope": t_max_upslope,
        "peak_width": peak_width,
        "peak_start": peak_start,
        "peak_end": peak_end,
        "integral": integral}
    return return_dict


# source: https://gist.github.com/danstowell/f2d81a897df9e23cc1da
def wiener_deconvolution(signal, kernel, lambd):
    "lambd is the SNR"
    kernel = np.hstack((kernel, np.zeros(len(signal) - len(kernel))))  # zero pad the kernel to same length
    H = fft(kernel)
    deconvolved = np.real(ifft(fft(signal) * np.conj(H) / (H * np.conj(H) + lambd ** 2)))
    return deconvolved


def svd_deconvolve(y, x, r_alpha, block_circulant=True):
    """
    :param y: vector of a time density curve
    :param x: vector of the arterial input function
    :param r_alpha: number of components to keep for svd
    :param block_circulant: whether to use circulant svd
    Note: x and y should be of the same length
    """
    epsilon = 0
    if block_circulant:
        new_len = 2 * max(len(x), len(y))
        x = np.concatenate([x, np.zeros(new_len - len(x))])
        y = np.concatenate([y, np.zeros(new_len - len(y))])
        X = circulant(x)
    else:
        X = convmtx(x, len(y))
        # X = X[:, len(x)//2: len(x)//2 + len(y)]

    [U, svals, VH] = np.linalg.svd(X)
    Dp_alpha = np.zeros(X.shape)
    for i in range(r_alpha):
        if svals[i] > epsilon:
            Dp_alpha[i, i] = 1 / svals[i]
    return VH.T.dot(Dp_alpha.T).dot(U.T).dot(y)


def svd_deconvolve_3d(y, x, r_alpha, block_circulant=True):
    """
    :param y: 3D matrix, assuming axis 0 is the time axis
    :param x: vector of the arterial input function
    :param r_alpha: number of components to keep for svd
    :param block_circulant: whether to use circulant svd
    :param axis: deconvolution is performed along this axis
    Note: x and y should be of the same length
    """
    epsilon = 0
    if block_circulant:
        new_len = 2 * max(len(x), y.shape[0])
        x = np.concatenate([x, np.zeros(new_len - len(x))])
        y = np.concatenate([y, np.zeros((new_len - y.shape[0], *y.shape[1:]))], axis=0)
        X = circulant(x)
    else:
        X = convmtx(x, len(y))
        # X = X[:, len(x)//2: len(x)//2 + len(y)]

    [U, svals, VH] = np.linalg.svd(X)
    Dp_alpha = np.zeros(X.shape)
    for i in range(r_alpha):
        if svals[i] > epsilon:
            Dp_alpha[i, i] = 1 / svals[i]
    return np.swapaxes(VH.T @ Dp_alpha.T @ U.T @ np.swapaxes(y, 0, 1), 0, 1)


def svd_deconvolve_with_OI(y, x, oi_threshold, block_circulant=True):
    for r_alpha in reversed(range(1, len(x) + 1)):
        impulse_residue_function = svd_deconvolve(y, x, r_alpha, block_circulant=block_circulant)
        oi = 0
        for k in range(1, len(impulse_residue_function) - 1):
            oi += abs(
                impulse_residue_function[k - 1] - 2 * impulse_residue_function[k] + impulse_residue_function[k + 1])
        if oi != 0:
            oi = oi / (len(impulse_residue_function) * max(impulse_residue_function))
        if oi <= oi_threshold:
            # print("Best R alpha: {}".format(r_alpha))
            return impulse_residue_function, r_alpha
    raise RuntimeError("OI threshold not reached!")


def svd_deconvolve_3D_with_OI(y, x, oi_threshold, block_circulant=True):
    oi_satisfied = np.zeros([*y.shape[1:]], dtype=bool)
    final_irf = svd_deconvolve_3d(y, x, len(x), block_circulant=block_circulant)
    r_alpha = np.ones([*y.shape[1:]]) * len(x)
    for r in reversed(range(1, len(x) + 1)):
        irf = svd_deconvolve_3d(y, x, r, block_circulant=block_circulant)
        oi = np.sum([np.abs(irf[k - 1] - 2 * irf[k] + irf[k + 1]) for k in range(1, irf.shape[0] - 1)], axis=0)
        irf_max_map = np.max(irf, axis=0)
        oi[irf_max_map != 0] = oi[irf_max_map != 0] / (irf.shape[0] * irf_max_map[irf_max_map != 0])

        newly_oi_satisfied_indices = (oi <= oi_threshold) & (~oi_satisfied)
        final_irf[:, newly_oi_satisfied_indices] = irf[:, newly_oi_satisfied_indices]
        r_alpha[newly_oi_satisfied_indices] = r
        oi_satisfied[oi <= oi_threshold] = True
        if np.all(oi_satisfied):
            # print("Best R alpha: {}".format(r_alpha))
            return final_irf, r_alpha
    raise RuntimeError("OI threshold not reached!")


if __name__ == '__main__':
    '''
    N = 300
    n = np.array(range(N)).T
    w = 5
    n1 = 70
    n2 = 130
    x = np.fromiter((2.1 * np.exp(-0.5 * ((elem - n1) / w) * ((elem - n1) / w)) -
                     0.5 * np.exp(-0.5 * ((elem - n2) / w) * ((elem - n2) / w)) * (n2 - elem) for elem in n),
                    dtype="float")
    h = np.array([elem * (pow(0.9, elem)) * math.sin(0.2 * math.pi * elem) for elem in n])
    # Convolve
    y = signal.convolve(h, x)
    # Keep only first part
    y = y[0:N]
    random.seed("myseed")
    yn = y + [0.2 * (random.random() * 2.0 - 1.0) for i in y]
    D = mydiags([1, -2, 1], range(3), N - 2, N)

    # And now the reverse: find the convolution kernel, given the noisy signal and the noisy input
    random.seed("seed")
    xn = x + [0.1 * (random.random() * 2.0 - 1.0) for i in x]
    Xn = convmatrix(xn, N)
    Xn = Xn[0:N, :]
    e = y - Xn.dot(h)
    np.max(e)

    lambdas = [0.1, 1, 2, 5]
    legends = ["orig"]
    plt.plot(h)
    # inv_xn = np.fft.ifft(1/np.fft.fft(xn))
    # h_deconv= signal.convolve(yn, inv_xn, mode='full')
    # h_deconv = h_deconv[-len(h):]  # signal trimmed to the required length
    # plt.plot(h_deconv)
    # legends.append("deconvolved")
    for lam in lambdas:
        h_deconv = wiener_deconvolution(y, xn, lam)
        plt.plot(h_deconv)
        legends.append("lam = {}".format(lam))
        plt.title(
            "Original h and deconvolution from noisy input and signal\n"
            "derivative regularization, various lambdas".format(lam))
    # for lam in lambdas:
    #     gh = solve(Xn.T.dot(Xn) + lam * (D.T.dot(D)), Xn.T.dot(yn))
    #     plt.plot(gh)
    #     legends.append("lam = {}".format(lam))
    # plt.title(
    #     "Original h and deconvolution from noisy input and signal\nderivative regularization, various lambdas".format(
    #         lam))
    plt.legend(legends)
    plt.ylim(-25, 25)
    plt.show()
    plt.close()

    # and what happens if we run it on real data?

    from scipy.interpolate import interp1d

    # get data from csv file
    a = np.genfromtxt('./data/timing-artery-brain.txt', delimiter=' ')

    # get timings, and brain and arterial curves
    timings = a[:, 0]
    brain_orig = a[:, 2]
    artery_orig = a[:, 1]

    # cubic interpolation (to get regular time sampling)
    brain_cubic = interp1d(timings, brain_orig, kind='cubic')
    artery_cubic = interp1d(timings, artery_orig, kind='cubic')

    xnew = np.linspace(0, max(timings), num=len(timings) * 4, endpoint=True)

    # determines baseline
    baseline = sum(brain_orig[:6] / 6.0)

    # create normalized (w.r.t. offset and height) curves
    brain_baseline = brain_cubic(xnew) - baseline
    artery_baseline = artery_cubic(xnew) - baseline
    brain_norm = brain_baseline / np.max(brain_baseline)
    artery_norm = artery_baseline / np.max(artery_baseline)

    plt.plot(timings, brain_orig, timings, artery_orig, xnew, brain_cubic(xnew), xnew, artery_cubic(xnew))
    plt.legend(["brain_orig", "artery_orig", "brain_cubic", "artery_cubic"])
    plt.title("Original and interpolated TIC for brain and artery")
    plt.show()
    plt.close()

    plt.plot(xnew, brain_norm, xnew, artery_norm)
    plt.legend(["brain_norm", "artery_norm"])
    plt.title("Normalized TIC for brain and artery")
    plt.show()
    plt.close()

    plt.plot(timings, brain_orig, timings, artery_orig)
    plt.legend(["brain_orig", "artery_orig"])
    plt.title("Original TIC for brain and artery")
    plt.show()
    plt.close()

    # lam = 5
    # gh = solve(X.T.dot(X) + lam*(D.T.dot(D)), X.T.dot(yn))
    lambdas = [0.1, 1, 2, 5]
    N = len(artery_norm)
    A = convmatrix(artery_norm, N)
    A = A[0:N, :]
    D = mydiags([1, -2, 1], range(3), N - 2, N)
    legends = []
    for lam in lambdas:
        response = wiener_deconvolution(brain_norm, artery_norm, lam)
        plt.plot(response)
        legends.append("response for lam = {}".format(lam))
    plt.title("Response curve for various lambdas")
    plt.legend(legends)
    plt.show()
    plt.close()

    popt = my_gamma_fit(xnew, artery_norm)
    artery_gamma = vgamma(xnew, **popt)
    plt.plot(xnew, artery_norm, xnew, artery_gamma)
    plt.legend(["artery_norm", "artery_gamma"])
    plt.title("Gamma fit of artery TIC")
    plt.show()
    print(popt)

    popt = my_gamma_fit(xnew, brain_norm)
    brain_gamma = vgamma(xnew, **popt)
    plt.plot(xnew, brain_norm, xnew, brain_gamma)
    plt.legend(["brain_norm", "brain_gamma"])
    plt.title("Gamma fit of brain TIC")
    plt.show()
    print(popt)

    # Conclusion: t_0 is not very reliable as a start of the curve, as it can be pretty small (brain curve).
    # May it it would be better to use t_max - 2*(t_max - t_half_max).
    # Also, check the response curves at the end of the doc. The peak of the response curve does not match the
    # difference in t_max. May be that makes sense?
    #

    # try: make artery and brain longer, brainnew_cub_offset_long based on gamma fit,
    # artery based on adding zeros. And then do the deconvolution again.
    #
    # Reasoning: the tail of the response curve is incorrect because of the too short
    # timing. By using a gamma-fit curve, we can make the tail correct. The arterial
    # input curve tail already approached zero.
    # xnew_long = np.linspace(0,2.0*max(timings), num=len(timings)*4, endpoint=True)
    # brain_gamma_long = vgamma(xnew_long, *popt)

    # arterynew_long = np.array(list(range(4*len(timings))), dtype="float")
    # for i in range(4*len(timings)):
    #    if i < 2*len(timings):
    #        arterynew_long[i] = arterynew_cub_offset[i]
    #    else:
    #        arterynew_long[i] = 0.0#

    # print(arterynew_long)
    # print(arterynew_cub_offset)

    x_long, brain_long = extend_curve_with_gamma_fit(xnew, brain_norm, 3.0)
    x_long, artery_long = extend_curve_with_gamma_fit(xnew, artery_norm, 3.0)
    brain_gamma_long = vgamma(x_long, **my_gamma_fit(xnew, brain_norm))
    artery_gamma_long = vgamma(x_long, **my_gamma_fit(xnew, artery_norm))

    plt.plot(xnew, brain_norm, x_long, brain_long)
    plt.plot(xnew, artery_norm, x_long, artery_long)
    plt.legend(["brain_norm", "brain_long", "artery_norm", "artery_long"])
    plt.show()

    plt.plot(xnew, brain_norm)
    plt.plot(xnew, artery_norm)
    plt.legend(["brain_norm", "artery_norm"])
    plt.show()

    # lam = 5
    # gh = solve(X.T.dot(X) + lam*(D.T.dot(D)), X.T.dot(yn))

    lambdas = [0.1, 0.5, 1, 2, 5]
    N = len(artery_long)
    A = convmatrix(artery_long, N)
    A = A[0:N, :]
    D = mydiags([1, -2, 1], range(3), N - 2, N)
    legends = []
    for lam in lambdas:
        response = wiener_deconvolution(brain_long, artery_long, lam)
        # response = solve(A.T.dot(A)+lam*(D.T.dot(D)), A.T.dot(brain_long))
        plt.plot(x_long[:len(xnew)], response[:len(xnew)])
        legends.append("response for lam = {}".format(lam))
    plt.title("Long response curve for various lambdas when extending signal")
    plt.legend(legends)
    plt.show()
    # extension with original signal does not work to get better fit...

    lambdas = [0.1, 0.5, 1, 2, 5]
    N = len(artery_norm)
    A = convmatrix(artery_norm, N)
    A = A[0:N, :]
    D = mydiags([1, -2, 1], range(3), N - 2, N)
    legends = []
    for lam in lambdas:
        response = wiener_deconvolution(brain_gamma, artery_norm, lam)
        # response = solve(A.T.dot(A) + lam * (D.T.dot(D)), A.T.dot(brain_gamma))
        plt.plot(xnew, response)
        legends.append("response for lam = {}".format(lam))
    plt.title("Response curve for various lambdas using brain gamma fit")
    plt.legend(legends)
    plt.show()

    lambdas = [0.1, 0.5, 1, 2, 5]
    N = len(artery_gamma)
    A = convmatrix(artery_gamma, N)
    A = A[0:N, :]
    D = mydiags([1, -2, 1], range(3), N - 2, N)
    legends = []
    for lam in lambdas:
        response = wiener_deconvolution(brain_norm, artery_gamma, lam)
        # response = solve(A.T.dot(A) + lam * (D.T.dot(D)), A.T.dot(brain_norm))
        plt.plot(xnew, response)
        legends.append("response for lam = {}".format(lam))
    plt.title("Response curve for various lambdas using artery gamma fit")
    plt.legend(legends)
    plt.show()

    lambdas = [0.1, 0.5, 1, 2, 5]
    N = len(artery_gamma)
    A = convmatrix(artery_gamma, N)
    A = A[0:N, :]
    D = mydiags([1, -2, 1], range(3), N - 2, N)
    legends = []
    for lam in lambdas:
        response = wiener_deconvolution(brain_gamma, artery_gamma, lam)
        # response = solve(A.T.dot(A) + lam * (D.T.dot(D)), A.T.dot(brain_gamma))
        plt.plot(xnew, response)
        legends.append("response for lam = {}".format(lam))
    plt.title("Response curve for various lambdas using artery and brain gamma fit")
    plt.legend(legends)
    plt.show()

    # Conclusion: in this case, a response curve can be computed accurately when the brain (tissue) curve is a gamma fit.
    # Observation: time point of max of response curve does not accurately match delta t_max (see above, 3.3 and 6.0).
    # That probably makes sense?
    # Second conclusion: based on this (single) set of curves, I would conclude that direct deconvolution may not work well.
    # And if we need to deconvolve with gamma curves, why not directly use the gamma curves for our purpose?

    print(getFlowParameters(xnew, brain_norm))
    '''
    """----------------------------Debugging-------------------------------"""
    # lambdas = [0.1, 0.5, 1, 2, 5]
    # legends = []
    # aif = np.array([21.20714704, 20.8395171, 20.5528231, 20.48052082, 21.42349496, 27.08881067, 44.8627971,
    #                 76.43629428, 109.50640699, 132.82374855, 144.97448343, 146.65207811, 138.17788373, 121.37375709,
    #                 100.68948835, 81.41946377, 66.15133496, 53.79174294, 44.09003964, 39.7])
    # aif = np.array([21.20714704, 20.8395171, 20.5528231, 20.48052082, 21.42349496, 27.08881067, 44.8627971,
    #                 76.43629428, 109.50640699, 132.82374855, 144.97448343, 146.65207811, 138.17788373, 121.37375709,
    #                 100.68948835, 81.41946377, 66.15133496, 53.79174294, 44.09003964, 39.78978823, 39.39874966])
    # # tdc = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])  # [0,0]
    # tdc = np.array([0., 0.76286316, 1.0844879, 0.70492554, 0.4186554, 1.6290588, 3.225357, 5.264679,
    #                 12.687271, 29.146805, 49.858353, 66.48479, 74.6608, 71.81746, 63.653458, 53.57007,
    #                 44.78015, 41.817078, 42.70256, 42.31566, 41.318726])  # [512, 512]

    # aif = np.array([0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0])
    # tdc = np.array([0, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0])
    # print("tdc: {}".format(tdc))
    # print("aif: {}".format(aif))
    # for lam in lambdas:
    #     response = wiener_deconvolution(tdc, aif, lam)
    #     print("lamda: {}; tmax: {}; irf: {}".format(lam, np.argmax(response), response))
    #     # response = solve(A.T.dot(A) + lam * (D.T.dot(D)), A.T.dot(brain_norm))
    #     plt.plot(response)
    #     legends.append("response for lam = {}".format(lam))
    # plt.title("DSA debugging")
    # plt.legend(legends)
    # plt.show()

    # lambdas = [0.1, 0.5, 1, 2, 5]
    # legends = []
    # # a1 = np.array([0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0])
    # a1 = np.array([0, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # b1 = np.array([0, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0])
    # print("a1: {}".format(a1))
    # print("b1: {}".format(b1))
    # for lam in lambdas:
    #     in1 = (a1, b1)[len(b1) > len(a1)]
    #     in2 = (a1, b1)[len(b1) <= len(a1)]
    #     # response = wiener_deconvolution(in1, in2, lam)
    #     response = svd_deconvolve(in1, in2, r_alpha=12)
    #     print("lamda: {}; tmax: {}; irf: {}".format(lam, np.argmax(response), response))
    #     if np.argmax(response) <= (len(response) - np.argmax(response)):
    #         in1 = in1[np.argmax(response):]
    #     else:
    #         in2 = in2[(len(response) - np.argmax(response)):]
    #     a2 = (in1, in2)[len(a1) < len(b1)]
    #     b2 = (in1, in2)[len(a1) >= len(b1)]
    #     print("a2: {}".format(a2))
    #     print("b2: {}".format(b2))
    #     # response = solve(A.T.dot(A) + lam * (D.T.dot(D)), A.T.dot(brain_norm))
    #     plt.plot(response)
    #     legends.append("response for lam = {}".format(lam))
    # plt.title("DSA debugging")
    # plt.legend(legends)
    # plt.show()

    '''Deconvolution using truncated singular value decomposition'''
    fsize = 16
    lwidth = 2

    M = 17
    # psf = np.ones((1, 2 * M + 1))  # point spread function
    psf = np.ones(2 * M + 1)  # point spread function
    psf = psf / np.sum(psf)  # Normalization of the psf
    # psf = np.array([0, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])/21

    'Construct unknown signal f'
    N = 400
    x = np.linspace(0, 1, N)
    a = 1.99323054838
    x = np.linspace(gamma.ppf(0.01, a), gamma.ppf(0.99, a), N)
    # % f = zeros(N,1);
    # % f(1:(end/2)) = 1;
    # f = np.sin(2 * np.pi * x)
    f = gamma.pdf(x, a)
    psf = signal.unit_impulse(400, 50) * 0.5
    # f = f[:, None]
    # f = np.vstack(f)  # Force vector f to be vertical

    'Construct the convolution matrix'
    A = convmtx(psf, N)
    A = A[:, M: (A.shape[1] - M)]

    # Simulate the measurement
    # m = np.dot(A, f)
    m = np.convolve(f, psf, mode='full')
    # Simulate noisy measurement
    sigma = .1
    # mn = m + sigma * np.random.rand(N)

    # Compute truncated SVD solution
    # Determine the SVD of matrix A
    [U, svals, VH] = np.linalg.svd(A)
    # [U, svals, V] = svd.fit(A)
    D = np.diag(svals)

    # Compute reconstruction
    r_alpha = 15
    Dp_alpha = np.zeros(A.shape)
    for iii in range(r_alpha):
        Dp_alpha[iii, iii] = 1 / svals[iii]

    # f0 = np.transpose(VH).dot(Dp_alpha).dot(np.transpose(U)).dot(m)
    f1, r_alpha = svd_deconvolve_with_OI(m, psf, oi_threshold=0.07)
    # f1 = wiener_deconvolution(m, psf, lambd=1)
    plt.plot(f)
    plt.plot(psf)
    plt.plot(m)
    plt.plot(f1)
    # plt.legend(['f', 'f1'])
    plt.legend(['f', 'psf', 'm', 'f1'])
    plt.show()
    # fn = np.transpose(VH).dot(Dp_alpha).dot(np.transpose(U)).dot(mn)
