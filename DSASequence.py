import numpy as np
import time
from typing import Callable, Optional

import torch
from numpy.linalg import pinv

from skimage.measure import block_reduce

from pathlib import Path
import torch
from PIL import Image
import torch.nn.functional as F


def modelfree_deconv(data:np.ndarray, aif:np.ndarray, dt, hct:float=0) -> np.ndarray:
    """ Non-parametric (model free) deconvolution.

    :param data: np.ndarray with shape (N,W,H), containing the Tissue Concentration Time Curve.
    :param aif: np.ndarray with shape (N), containing the Arterial Input Function.
    :param dt: time between frames in seconds.
    :param hct: hematocrit.
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
    a_inv = pinv(a, rcond=0.15)

    # impulse response
    i = np.dot(1 / dt * a_inv,
               data.reshape(len(aif), np.prod(data.shape) // len(aif)))

    i = i.reshape(data.shape)
    return i


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

    _dsa_images_raw: np.ndarray = None  # standard DSA with base level at 170 and blood vessels darker.
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
        return self._timestep

    _time: np.ndarray = None
    def Time(self) -> np.ndarray:
        return self._time

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

    def __init__(self, images: np.ndarray, frame_interval: float):
        """ Creates a DSAFile object.

        :param images: an (W, H, N) numpy array of uniformly sampled DSA images.
        :param timestep: the time between two consecutive frames, in seconds.
        """
        (w, h, n) = images.shape
        self._timestep = frame_interval
        self._time = np.arange(n) * frame_interval
        self._n_images = n
        self._image_size = (w, h)
        self._dsa_images_raw = np.moveaxis(images, 2, 0)
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
        cbv = self.CBV()
        cbf = self.CBF()
        mtt = self.MTT()
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
        mod = mod / 170  # Adjust range from [0..1]
        mod = 1 - mod  # inverse dsa image to have positive TICs when contrasts arrive
        self._dsa_images_mod = mod

    def calculate_TRF_map(self) -> np.ndarray:
        """ Calculates and returns the Tissue Response Function map. Will be called automatically when requesting any of the perfusion maps.

        :return: a size (N,W,H) numpy array containing the per-pixel Tissue Response Functions.
        """
        time_start = time.time()
        self._trf_map = modelfree_deconv(self.TCTC(), self.AIF(), self._timestep)
        print("Calculated TRF map in %.3fs" % (time.time() - time_start))
        return self.TRF()

    def get_perfusion_parameters(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ Calculates, stores and returns the perfusion maps. Will be called automatically when requesting any of the perfusion maps. Adapted from https://github.com/RuishengSu/perfDSA.

        :return: a tuple (CBV,CBF,MTT,T_max) containing each of the perfusion maps as (W,H) numpy arrays.
        """
        time_start = time.time()
        trf = self.TRF()
        # compute flow in ml/100ml/min
        self._cbf = np.max(trf, axis=0)
        self._cbf[self._cbf < 0] = 0
        # F = f * (100 * 60)

        # compute Tmax
        self._tmax = self._timestep * np.argmax(trf, axis=0)

        # compute blood volume in ml/100ml
        self._cbv = self._timestep * np.sum(trf, axis=0)
        self._cbv[self._cbv < 0] = 0
        # V = v * 100

        epsilon = 1e-12
        # mean transit time in s
        self._mtt = self._cbv / (self._cbf + epsilon)

        print("Calculated perfusion maps in %.3fs" % (time.time() - time_start))
        return self._cbv, self._cbf, self._mtt, self._tmax

