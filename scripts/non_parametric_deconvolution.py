#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import pinv


def modelfree_deconv(data, aif, dt, hct=0.45, epsilon=1e-12, dtype=np.float32):
    """ non parametric (model free) deconvolution.

    Keyword arguments:
        data: numpy array with shape t, x, y
        aif:  arterial input function (1D array)
        dt:   time between frames in milliseconds
        hct: hematocrit
        n_offset :  number of baseline frames
        epsilon:    used for numeric stability
    """

    # data = np.transpose(data, axes=[2, 0, 1]).astype(dtype)
    data = data.astype(dtype)
    aif = aif.astype(dtype)
    # aif = aif / (1 - hct)
    dt /= 1000.0

    # remove baseline bias
    # minuend_data = np.mean(data[:n_offset], axis=0)
    # minuend_aif  = np.mean(aif[:n_offset], axis=0)
    # for i in range(data.shape[0]):
    #     data[i] = data[i] - minuend_data
    #     aif[i]  = aif[i]  - minuend_aif
    # data = data[n_offset:]
    # aif  = aif[n_offset:]

    # convolution matrix of the aif (volterra interpolation)
    A = np.zeros((len(aif), len(aif)), dtype=float)
    for i in range(1, len(aif)):
        A[i, 0] = (2 * aif[i] + aif[i - 1]) / 6
        A[i, i] = (2 * aif[0] + aif[1]) / 6
    for i in range(2, len(aif)):
        for j in range(1, i):
            A[i, j] = (2 * aif[i - j] + aif[i - j - 1]) / 6 \
                      + (2 * aif[i - j] + aif[i - j + 1]) / 6

    # inverse of the convolution matrix
    A_inv = pinv(A, rcond=0.15)

    # impulse response
    I = np.dot(1 / dt * A_inv,
               data.reshape(len(aif), np.prod(data.shape) // len(aif)))

    # compute flow in ml/100ml/min
    CBF = np.max(I, axis=0).reshape(data.shape[1:])
    CBF[CBF < 0] = 0
    # F = f * (100 * 60)

    # compute Tmax
    Tmax = dt * np.argmax(I, axis=0).reshape(data.shape[1:])

    # compute blood volume in ml/100ml
    CBV = dt * np.sum(I, axis=0).reshape(data.shape[1:])
    CBV[CBV < 0] = 0
    # V = v * 100

    # mean transit time in s
    MTT = CBV / (CBF + epsilon)

    return CBV, CBF, MTT, Tmax
