"""Tests of the perfusion computation on a small synthetic DSA run with known perfusion (no model weights needed).

Run from the repository root:  python -m pytest tests
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from DSASequence import LEGACY_BACKGROUND, DSASequence, estimate_background  # noqa: E402

DT_MS = 250.0
MTT_S, CBV_TRUE = 3.0, 0.1


def synthetic_run(background=170.0, n=60, size=48, t0_s=1.5, noise=0.0, seed=0):
    """Uniformly sampled subtracted DSA (vessels dark) with an AIF vessel and tissue of known CBV and MTT.

    Returns images (W, H, N) on the perfDSA intensity scale, the AIF mask and the tissue mask.
    """
    t = np.arange(n) * DT_MS / 1000
    x = np.clip((t - t0_s) / 1.5, 0, None)
    aif = 0.5 * x ** 3 * np.exp(3 * (1 - x))                       # gamma-variate AIF, peak 0.5
    fine = np.arange(0, t[-1] + 1e-9, 0.01)
    aif_f = np.interp(fine, t, aif)
    residue = np.exp(-fine / MTT_S)
    tissue = np.interp(t, fine, np.convolve(aif_f, residue)[:len(fine)] * 0.01 * CBV_TRUE / MTT_S)

    conc = np.zeros((n, size, size))
    yy, xx = np.mgrid[:size, :size]
    aif_mask = (yy - 8) ** 2 + (xx - size // 2) ** 2 <= 9
    tissue_mask = (yy > 16) & (yy < size - 4) & (xx > 4) & (xx < size - 4)
    conc[:, aif_mask] = aif[:, None]
    conc[:, tissue_mask] = tissue[:, None]
    conc += np.random.default_rng(seed).standard_normal(conc.shape) * noise
    images = background * (1 - conc)
    return np.moveaxis(images, 0, 2), aif_mask, tissue_mask


def perfusion(images, aif_mask, background):
    dsa = DSASequence(images, frame_interval=DT_MS, background=background)
    dsa.sample_AIF_from_mask(aif_mask)
    return dsa


@pytest.mark.parametrize("background", [160.0, 170.0, 178.0])
def test_background_is_estimated_from_the_run(background):
    images, _, _ = synthetic_run(background=background, noise=0.002)
    estimate, frames = estimate_background(np.moveaxis(images, 2, 0))
    assert estimate == pytest.approx(background, abs=0.2)
    assert len(frames) >= 3  # frames before contrast arrival (arrival at 1.5 s, 250 ms frames)


@pytest.mark.parametrize("background", [166.0, 170.0, 174.0])
def test_maps_do_not_depend_on_the_background_level(background):
    ref_images, aif_mask, tissue = synthetic_run(background=170.0)
    ref = perfusion(ref_images, aif_mask, "auto")
    images, _, _ = synthetic_run(background=background)
    dsa = perfusion(images, aif_mask, "auto")
    for get in (DSASequence.CBV, DSASequence.CBF, DSASequence.MTT, DSASequence.Tmax):
        np.testing.assert_allclose(get(dsa)[tissue], get(ref)[tissue], rtol=1e-4, atol=1e-9)


def test_hard_coded_background_biases_cbv():
    """Documents the perfDSA <= v1.0 behaviour that motivated the change: a 2-grey-level offset biases CBV."""
    images, aif_mask, tissue = synthetic_run(background=168.0)
    legacy = perfusion(images, aif_mask, LEGACY_BACKGROUND)
    auto = perfusion(images, aif_mask, "auto")
    assert np.median(auto.CBV()[tissue]) == pytest.approx(CBV_TRUE, rel=0.05)
    assert np.median(legacy.CBV()[tissue]) > 1.3 * CBV_TRUE


def test_legacy_background_reproduces_v1_concentration():
    images, aif_mask, _ = synthetic_run(background=170.0)
    dsa = perfusion(images, aif_mask, LEGACY_BACKGROUND)
    np.testing.assert_allclose(dsa.TCTC(), 1 - np.moveaxis(images, 2, 0).astype(np.float32) / 170, rtol=1e-6)


def test_units_are_seconds():
    images, aif_mask, tissue = synthetic_run()
    dsa = perfusion(images, aif_mask, "auto")
    assert dsa.Timestep() == pytest.approx(DT_MS / 1000)
    assert dsa.Time()[-1] == pytest.approx((images.shape[2] - 1) * DT_MS / 1000)
    assert np.median(dsa.MTT()[tissue]) == pytest.approx(MTT_S, rel=0.1)
    assert np.median(dsa.CBF()[tissue]) == pytest.approx(CBV_TRUE / MTT_S, rel=0.1)   # 1/s
    assert np.median(dsa.Tmax()[tissue]) <= 2 * DT_MS / 1000                          # true Tmax is 0 s


def test_downsampling_does_not_modify_cached_maps():
    images, aif_mask, _ = synthetic_run()
    dsa = perfusion(images, aif_mask, "auto")
    before = {f: getattr(dsa, f)().copy() for f in ("CBV", "CBF", "MTT", "Tmax")}
    mask = np.zeros(dsa.Image_size(), dtype=bool)
    mask[:10] = True
    dsa.get_downsampled_parameters(4, mask)
    for f, v in before.items():
        np.testing.assert_array_equal(getattr(dsa, f)(), v)


def test_dicom_loading_keeps_the_last_frame(tmp_path):
    pydicom = pytest.importorskip("pydicom")
    perfdsa = pytest.importorskip("perfDSA")
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    n, h, w = 13, 16, 16
    pixels = np.full((n, h, w), 200, dtype=np.uint8)
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.12.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = Dataset()
    ds.file_meta = meta
    ds.NumberOfFrames, ds.Rows, ds.Columns = n, h, w
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    ds.FrameTimeVector = [0.0] + [500.0] * (n - 1)          # 0, 500, ..., 6000 ms
    ds.PixelData = pixels.tobytes()
    path = tmp_path / "run.dcm"
    try:
        ds.save_as(str(path), enforce_file_format=True)    # pydicom >= 3
    except TypeError:
        ds.save_as(str(path), write_like_original=False)   # pydicom 2.x

    seq = perfdsa.load_and_preprocess_dicom(str(path), desired_frame_interval=250)
    # First frame is dropped by design (500 ms .. 6000 ms remain): 500, 750, ..., 6000 ms = 23 frames.
    assert seq.shape[2] == 23


def synthetic_delayed(delay_s, n=72, size=32):
    t = np.arange(n) * DT_MS / 1000
    x = np.clip((t - 1.5) / 1.5, 0, None)
    aif = 0.5 * x ** 3 * np.exp(3 * (1 - x))
    fine = np.arange(0, t[-1] + 1e-9, 0.01)
    residue = np.where(fine >= delay_s, np.exp(-(fine - delay_s) / MTT_S), 0.0)
    tissue = np.interp(t, fine, np.convolve(np.interp(fine, t, aif), residue)[:len(fine)] * 0.01 * CBV_TRUE / MTT_S)
    conc = np.zeros((n, size, size))
    aif_mask = np.zeros((size, size), bool)
    aif_mask[2:5, 2:5] = True
    tissue_mask = np.zeros((size, size), bool)
    tissue_mask[10:, :] = True
    conc[:, aif_mask] = aif[:, None]
    conc[:, tissue_mask] = tissue[:, None]
    return np.moveaxis(170.0 * (1 - conc), 0, 2), aif_mask, tissue_mask


@pytest.mark.parametrize("method", ["csvd", "osvd"])
def test_block_circulant_svd_is_delay_insensitive(method):
    cbf = {}
    for delay in (0.0, 2.0):
        images, aif_mask, tissue = synthetic_delayed(delay)
        dsa = DSASequence(images, frame_interval=DT_MS, deconvolution=method)
        dsa.sample_AIF_from_mask(aif_mask)
        cbf[delay] = np.median(dsa.CBF()[tissue])
        assert dsa.TRF().shape[0] == 2 * images.shape[2]
        assert abs(np.median(dsa.Tmax()[tissue]) - delay) <= 1.0
    legacy = {}
    for delay in (0.0, 2.0):
        images, aif_mask, tissue = synthetic_delayed(delay)
        dsa = DSASequence(images, frame_interval=DT_MS, deconvolution="ssvd")
        dsa.sample_AIF_from_mask(aif_mask)
        legacy[delay] = np.median(dsa.CBF()[tissue])
    assert abs(cbf[2.0] / cbf[0.0] - 1) < 0.05            # block-circulant: delay-insensitive
    assert legacy[2.0] / legacy[0.0] < 0.9                  # standard SVD loses CBF with delay


def test_default_deconvolution_unchanged():
    images, aif_mask, _ = synthetic_run()
    a = perfusion(images, aif_mask, "auto")
    b = DSASequence(images, frame_interval=DT_MS, background="auto", deconvolution="ssvd", rcond=0.15)
    b.sample_AIF_from_mask(aif_mask)
    np.testing.assert_array_equal(a.CBF(), b.CBF())
