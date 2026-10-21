import numpy as np
import time
import warnings
from typing import Callable, Optional, Union

import torch
from numpy.linalg import pinv

from skimage.measure import block_reduce

from pathlib import Path
import torch
from PIL import Image
import torch.nn.functional as F


def modelfree_deconv(data:np.ndarray, aif:np.ndarray, dt, hct:float=0, rcond:float=0.15) -> np.ndarray:
    """ Non-parametric (model free) deconvolution.

    :param data: np.ndarray with shape (N,W,H), containing the Tissue Concentration Time Curve.
    :param aif: np.ndarray with shape (N), containing the Arterial Input Function.
    :param dt: time between frames in seconds.
    :param hct: hematocrit.
    :param rcond: relative singular-value threshold of the truncated SVD.
    :return: an (N,W,H) numpy array containing the per-pixel result of deconv(data,aif).
    """

    aif = aif / (1 - hct)

    # convolution matrix of the aif (volterra interpolation)
    a = np.zeros((len(aif), len(aif)), dtype=float)
    for i in range(1, len(aif)):
        a[i, 0] = (2 * aif[i] + aif[i - 1]) / 6
        a[i, i] = (2 * aif[0] + aif[1]) / 6
    for i in range(2, len(aif)):
        for j in range(1, i):
            a[i, j] = (2 * aif[i - j] + aif[i - j - 1]) / 6 \
                      + (2 * aif[i - j] + aif[i - j + 1]) / 6

    # inverse of the convolution matrix
    a_inv = pinv(a, rcond=rcond)

    # impulse response
    i = np.dot(1 / dt * a_inv,
               data.reshape(len(aif), np.prod(data.shape) // len(aif)))

    i = i.reshape(data.shape)
    return i


LEGACY_BACKGROUND = 170.0  # background intensity hard-coded in perfDSA <= v1.0
DECONVOLUTION_METHODS = ("ssvd", "csvd", "osvd")
OSVD_THRESHOLDS = tuple(float(t) for t in np.geomspace(0.005, 0.6, 24))


def oscillation_index(trf: np.ndarray) -> np.ndarray:
    """ Oscillation index of response functions (L, P) (Wu et al. 2003): mean |second difference| / maximum. """
    fmax = trf.max(axis=0)
    d2 = np.abs(trf[2:] - 2 * trf[1:-1] + trf[:-2]).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(fmax > 0, d2 / (len(trf) * fmax), np.inf)


def circulant_deconv(data: np.ndarray, aif: np.ndarray, dt, method: str = "osvd", rcond: float = 0.15,
                     oi_target: float = 0.035, hct: float = 0, chunk: int = 65536) -> np.ndarray:
    """ Block-circulant SVD deconvolution (Wu et al. 2003), insensitive to tracer delay between AIF and tissue.

    The AIF and the curves are zero-padded to L = 2N samples. The singular values of the resulting circulant matrix
    are the magnitudes of the DFT of the AIF, so the truncated SVD is computed with the FFT.

    :param data: (N, ...) tissue concentration time curves.
    :param aif: (N) arterial input function.
    :param dt: time between frames in seconds.
    :param method: "csvd" (fixed relative threshold `rcond`) or "osvd" (per pixel, the smallest threshold whose
        response has an oscillation index <= `oi_target`).
    :param hct: hematocrit.
    :return: (2N, ...) response functions; samples >= N represent negative delays (tissue before AIF).
    """
    aif = np.asarray(aif, dtype=float) / (1 - hct)
    n = len(aif)
    length = 2 * n
    spectrum = np.fft.rfft(aif, length)
    mag = np.abs(spectrum)
    relative = mag / mag.max() if mag.max() > 0 else mag
    inverse = np.where(mag > 0, 1 / np.where(mag > 0, spectrum, 1), 0)
    thresholds = (rcond,) if method == "csvd" else OSVD_THRESHOLDS

    flat = data.reshape(n, -1)
    out = np.empty((length, flat.shape[1]))
    for start in range(0, flat.shape[1], chunk):
        sl = slice(start, min(start + chunk, flat.shape[1]))
        spectrum_c = np.fft.rfft(flat[:, sl].astype(np.float64), length, axis=0)
        done = np.zeros(sl.stop - sl.start, dtype=bool)
        for threshold in thresholds:  # ascending: least regularisation first
            trf = np.fft.irfft(spectrum_c * (inverse * (relative >= threshold))[:, None], length, axis=0) / dt
            if method == "csvd":
                out[:, sl] = trf
                break
            ok = ~done & (oscillation_index(trf) <= oi_target)
            out[:, sl][:, ok] = trf[:, ok]
            done |= ok
        else:
            out[:, sl][:, ~done] = trf[:, ~done]  # never smooth enough: strongest regularisation
    return out.reshape((length,) + data.shape[1:])


def estimate_background(images: np.ndarray, arterial_fraction: float = 0.005, arrival_threshold: float = 0.1,
                        fov_min: float = 10.0) -> tuple[float, np.ndarray]:
    """ Estimates the zero-contrast (background) intensity of a subtracted DSA run from its own frames.

    Contrast arrival is detected on the darkest pixels of the minimum intensity projection (the large arteries):
    the first frame in which their mean intensity has dropped by more than `arrival_threshold` of its total drop.
    The background is the median intensity, within the field of view, of the frames before arrival.

    :param images: (N, W, H) uniformly sampled DSA frames (vessels dark).
    :param arterial_fraction: fraction of field-of-view pixels (darkest in the MinIP) used to detect arrival.
    :param arrival_threshold: fraction of the arterial intensity drop that defines contrast arrival.
    :param fov_min: pixels whose mean intensity is below this value are outside the collimated field of view.
    :return: (background intensity, indices of the frames used).
    """
    fov = images.mean(axis=0) > fov_min
    if not fov.any():
        fov = np.ones(images.shape[1:], dtype=bool)
    pixels = images[:, fov]
    n_art = max(1, int(arterial_fraction * pixels.shape[1]))
    arteries = np.argsort(pixels.min(axis=0))[:n_art]
    curve = pixels[:, arteries].mean(axis=1)
    drop = curve[0] - curve
    if drop.max() <= 0:
        pre = np.arange(len(images))  # no contrast arrival detected: every frame is background
    else:
        arrival = int(np.argmax(drop >= arrival_threshold * drop.max()))
        pre = np.arange(arrival)
    if len(pre) == 0:
        warnings.warn("No frame before contrast arrival: background estimated from the first frame, which already "
                      "contains contrast. Perfusion values may be biased.")
        pre = np.array([0])
    return float(np.median(pixels[pre])), pre


def segment_ica_top(net, img, out_img_path=None, device='cuda'):
    net.eval()
    img = torch.as_tensor(img.copy()).float().contiguous().to(device=device, dtype=torch.float32)
    img = torch.unsqueeze(img, 0)
    with torch.no_grad():
        masks_pred = net(img)
        # masks_pred = (F.sigmoid(masks_pred) > 0.5).float()
        mask_pred = Image.fromarray((255*(F.sigmoid(masks_pred) > 0.5)).squeeze().cpu().detach().numpy().astype(np.uint8))
    if out_img_path is not None:
        Path(out_img_path).parent.mkdir(parents=True, exist_ok=True)
        mask_pred.save(out_img_path)
    return mask_pred


class DSASequence:
    _n_images = 0
    def DSA_Length(self) -> int:
        """ Get the DSA length, in frames.

        :return: an integer giving the length of the DSA sequence.
        """
        return self._n_images

    _image_size = (0, 0)
    def Image_size(self) -> tuple[int,int]:
        """ Get the DSA image width and height.

        :return: a tuple (width, height).
        """
        return self._image_size

    _dsa_images_mod: np.ndarray = None  # dye concentration map with base level 0 and blood vessels lighter.
    def MIP(self) -> np.ndarray:
        """ Returns the Maximum Intensity Map of the dye concentration.

        :return: the Maximum Intensity Map as a (W,H) numpy array.
        """
        return np.max(self._dsa_images_mod, axis=0)

    _dsa_images_raw: np.ndarray = None  # standard DSA with base level at the background and blood vessels darker.
    def Frames(self, frames: int | slice | np.ndarray | list[int]) -> np.ndarray:
        """ Get a specified frame or frames from the DSA sequence.

        :param frames: the index or indices of the frame(s) to return.
        :return: a (W,H) numpy array if frames was a single value, or (N,W,H) otherwise.
        """
        return self._dsa_images_raw[frames, :, :]

    def MINIP(self) -> np.ndarray:
        """ Returns the Minimum Intensity Map of the DSA frames.

        :return: the Minimum Intensity Map as a (W,H) numpy array.
        """
        return np.min(self._dsa_images_raw, axis=0)

    def TCTC(self, pos: tuple = ()) -> np.ndarray:
        """ Returns the Tissue Concentration Time Curve at a certain position, if specified, or the entire TCTC map otherwise.

        :pos: a tuple (x,y) of the position to sample, if specified.
        :return: a length (N) numpy array if pos was specified, or (N,W,H) otherwise.
        """
        if pos == ():
            return self._dsa_images_mod
        return self._dsa_images_mod[:, pos[0], pos[1]]

    _timestep: float = np.nan
    def Timestep(self) -> float:
        """ Time between frames, in seconds. """
        return self._timestep

    _time: np.ndarray = None
    def Time(self) -> np.ndarray:
        """ Frame times, in seconds. """
        return self._time

    _background: float = np.nan
    def Background(self) -> float:
        """ Zero-contrast intensity of the subtracted DSA, used to convert intensities to concentration. """
        return self._background

    _aif: np.ndarray = None

    def AIF(self) -> np.ndarray:
        """ Returns the Arterial Input Function.

        :return: a length (N) numpy array.
        """
        if self._aif is None:
            raise Exception("Calculate AIF first by calling sample_AIF_from_position(pos, sampleDistance) or "
                            "sample_AIF_from_mask(mask)")
        return self._aif

    _trf_map: np.ndarray = None

    def TRF(self, pos: tuple = ()) -> np.ndarray:
        """ Returns the Tissue Response Function at a certain position.

        :pos: a tuple (x,y) of the position to sample, if specified.
        :return: a length (N) numpy array if pos was specified, or (N,W,H) otherwise.
        """
        if self._trf_map is None:
            self._trf_map = np.empty(1)
            self.calculate_TRF_map()
        if pos == ():
            return self._trf_map
        return self._trf_map[:, pos[0], pos[1]]

    _cbf: np.ndarray = None

    def CBF(self) -> np.ndarray:
        """ Returns the Cerebral Blood Flow map.

        :return: a (W,H) numpy array.
        """
        if self._cbf is None:
            self._cbf = np.empty(1)
            self.get_perfusion_parameters()
        return self._cbf

    _cbv: np.ndarray = None

    def CBV(self) -> np.ndarray:
        """ Returns the Cerebral Blood Volume map.

        :return: a (W,H) numpy array.
        """
        if self._cbv is None:
            self._cbv = np.empty(1)
            self.get_perfusion_parameters()
        return self._cbv

    _mtt: np.ndarray = None

    def MTT(self) -> np.ndarray:
        """ Returns the Mean Transit Time map.

        :return: a (W,H) numpy array.
        """
        if self._mtt is None:
            self._mtt = np.empty(1)
            self.get_perfusion_parameters()
        return self._mtt

    _tmax: np.ndarray = None

    def Tmax(self) -> np.ndarray:
        """ Returns the Time of Maximum Flow map.

        :return: a (W,H) numpy array.
        """
        if self._tmax is None:
            self._tmax = np.empty(1)
            self.get_perfusion_parameters()
        return self._tmax

    def __init__(self, images: np.ndarray, frame_interval: float, background: Union[str, float] = "auto",
                 deconvolution: str = "ssvd", rcond: float = 0.15, oi_target: float = 0.035):
        """ Creates a DSAFile object.

        :param images: an (W, H, N) numpy array of uniformly sampled DSA images.
        :param frame_interval: the time between two consecutive frames, in milliseconds.
        :param background: zero-contrast intensity of the subtracted images. "auto" (default) estimates it from the
            frames before contrast arrival (see estimate_background); a number fixes it. perfDSA <= v1.0 used a
            fixed value of 170 (LEGACY_BACKGROUND).
        :param deconvolution: "ssvd" (default; truncated SVD of the Volterra convolution matrix, as in v1.0),
            "csvd" (block-circulant SVD with fixed threshold) or "osvd" (block-circulant SVD with per-pixel
            oscillation-index threshold). Block-circulant SVD is insensitive to tracer delay.
        :param rcond: relative singular-value threshold for "ssvd" and "csvd".
        :param oi_target: oscillation-index target for "osvd".
        """
        if deconvolution not in DECONVOLUTION_METHODS:
            raise ValueError("deconvolution must be one of {}".format(DECONVOLUTION_METHODS))
        self._deconvolution, self._rcond, self._oi_target = deconvolution, rcond, oi_target
        (w, h, n) = images.shape
        self._timestep = frame_interval / 1000  # seconds
        self._time = np.arange(n) * self._timestep
        self._n_images = n
        self._image_size = (w, h)
        self._dsa_images_raw = np.moveaxis(images, 2, 0)
        if background == "auto":
            self._background, _ = estimate_background(self._dsa_images_raw)
        else:
            self._background = float(background)
        self._create_from_raw_images()

    def _create_from_raw_images(self):
        self._preprocess_dsa()
        self._aif = None
        self._trf_map = None
        self._cbf = None
        self._cbv = None
        self._mtt = None
        self._tmax = None
        self._vesselmap = None

    def get_downsampled_parameters(self, scale: int, mask: Optional[np.ndarray], func: Callable = np.nanmean) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """ Obtain downsampled versions of the perfusion maps.

        :param scale: Downsample by this amount; e.g. a 1024x1024 map with scale=4 becomes 256x256.
        :param mask: Boolean mask to discard values from the downsampling process, e.g. to filter out vessels.
        :param func: Function to apply to the downsampling process.
        :return: Tuple containing (CBV, CBF, MTT, T_max)
        """
        # Copies: masking must not overwrite the cached full-resolution maps.
        cbv = self.CBV().copy()
        cbf = self.CBF().copy()
        mtt = self.MTT().copy()
        tmax = self.Tmax().astype(np.float32)  # int cannot hold nan values
        if mask is not None:
            cbv[mask] = np.nan
            cbf[mask] = np.nan
            mtt[mask] = np.nan
            tmax[mask] = np.nan
        cbv = block_reduce(cbv, block_size=scale, func=func)
        cbf = block_reduce(cbf, block_size=scale, func=func)
        mtt = block_reduce(mtt, block_size=scale, func=func)
        tmax = block_reduce(tmax, block_size=scale, func=func)
        return cbv, cbf, mtt, tmax

    def sample_AIF_from_perfDSA(self, model_path:str, device: torch.device="cuda") -> tuple[np.ndarray,np.ndarray]:
        '''Load the perfDSA segmentation model and create a mask'''
        if device == "cuda": 
            if not torch.cuda.is_available():
                print("CUDA requested but not available, using CPU instead.")
                device = "cpu"
            else:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
        model = torch.load(model_path, map_location=device, weights_only=False)
        print(f'Model loaded from {model_path}')
        mask = segment_ica_top(model, np.expand_dims(self.MINIP(), 0), None, device)  # PIL.Image
        mask = np.greater(mask, 128)  # boolean np.ndarray
        return self.sample_AIF_from_mask(mask), mask

    def sample_AIF_from_mask(self, mask: np.ndarray) -> np.ndarray:
        """ Samples and stores the Arterial Input Function from a mask image.

        :param mask: a size (W,H) boolean numpy array containing the mask of pixels to extract the AIF from.
        :return: a size (N) numpy array containing the sampled Arterial Input Function.
        """
        masked_ica_vessel = np.ma.masked_array(self._dsa_images_mod, mask=~np.repeat(mask[np.newaxis, ...].astype(bool),self._dsa_images_mod.shape[0],
                                                                    axis=0))
        self._aif = masked_ica_vessel.mean(axis=(1, 2))
        return self._aif

    def sample_AIF_from_position(self, pos: tuple, dist: int) -> np.ndarray:
        """ Samples and stores the Arterial Input Function from a square area around user-defined position pos.

        :param pos: tuple (x,y) defining the center position of the sample area.
        :param dist: distance in pixels between the edges to the center of the box. I.e. the sampling area will be from pos-dist:pos+dist.
        :return: a size (N) numpy array containing the sampled Arterial Input Function.
        """
        x = pos[0]
        y = pos[1]
        # Handle image borders
        minx = max(x - dist, 0)
        maxx = min(x + dist, self._image_size[0])
        miny = max(y - dist, 0)
        maxy = min(y + dist, self._image_size[1])
        region = self.TCTC()[:, minx:maxx, miny:maxy]
        self._aif = np.max(region, axis=(1, 2))
        return self._aif

    def _preprocess_dsa(self):
        mod:np.ndarray = self._dsa_images_raw.astype(np.float32)
        mod = mod / self._background  # background -> 1
        mod = 1 - mod  # inverse dsa image to have positive TICs when contrasts arrive (background -> 0)
        self._dsa_images_mod = mod

    def calculate_TRF_map(self) -> np.ndarray:
        """ Calculates and returns the Tissue Response Function map. Will be called automatically when requesting any of the perfusion maps.

        :return: a size (N,W,H) numpy array containing the per-pixel Tissue Response Functions.
        """
        time_start = time.time()
        if self._deconvolution == "ssvd":
            self._trf_map = modelfree_deconv(self.TCTC(), self.AIF(), self._timestep, rcond=self._rcond)
        else:
            self._trf_map = circulant_deconv(self.TCTC(), self.AIF(), self._timestep, method=self._deconvolution,
                                             rcond=self._rcond, oi_target=self._oi_target)
        print("Calculated TRF map in %.3fs" % (time.time() - time_start))
        return self.TRF()

    def get_perfusion_parameters(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Calculates, stores and returns the perfusion maps. Will be called automatically when requesting any of the perfusion maps. Adapted from https://github.com/RuishengSu/perfDSA.

        :return: a tuple (CBV,CBF,MTT,T_max) containing each of the perfusion maps as (W,H) numpy arrays.
        """
        time_start = time.time()
        trf = self.TRF()
        # Units: the concentration scale is relative (AIF and tissue share it), so CBF and CBV are relative indices.
        # flow in 1/s (relative)
        self._cbf = np.max(trf, axis=0)
        self._cbf[self._cbf < 0] = 0

        # Tmax in s
        idx = np.argmax(trf, axis=0)
        if trf.shape[0] > self._n_images:  # block-circulant: samples >= N are negative delays
            idx = np.where(idx >= self._n_images, idx - trf.shape[0], idx)
        self._tmax = self._timestep * idx

        # blood volume (relative, dimensionless)
        self._cbv = self._timestep * np.sum(trf, axis=0)
        self._cbv[self._cbv < 0] = 0

        epsilon = 1e-12
        # mean transit time in s
        self._mtt = self._cbv / (self._cbf + epsilon)

        print("Calculated perfusion maps in %.3fs" % (time.time() - time_start))
        return self._cbv, self._cbf, self._mtt, self._tmax

