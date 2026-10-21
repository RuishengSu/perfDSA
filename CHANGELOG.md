# Changelog

## v1.1 (unreleased)

Corrections found during the Robust-perfDSA work, with tests (`tests/`).

### Changed: the background intensity is estimated per run (previously a fixed 170)

Concentration is computed as C = 1 − I / background.

- **v1.0:** the background was hard-coded to 170. Any deviation of a run's subtracted background from 170 is
  integrated by the deconvolution as if it were contrast. On a digital phantom, an offset of −2 grey levels raised
  CBV by about 100%; +2 grey levels drove CBV to zero.
- **Now:** `DSASequence(..., background="auto")` estimates the background from the frames before contrast
  arrival (`estimate_background`). Arrival is detected on the darkest MinIP pixels, i.e. the arteries.
- **Legacy behaviour:** `background=170` (also `perfDSA.py -b 170`) reproduces v1.0.

### Fixed: units

- The frame interval passed to `DSASequence` is in ms, but it was used as if in seconds.
- The labels were wrong: the AIF plot axis said "Time (s)" while showing ms, and the comments said mL/100 mL/min
  and s.
- Time is now handled in seconds:

  | Output | Unit |
  |---|---|
  | `Time()`, `Timestep()` | s |
  | CBF | 1/s (relative) |
  | CBV | relative, dimensionless |
  | MTT, Tmax | s |

- Compared with v1.0: CBF × 1000, MTT and Tmax ÷ 1000, CBV unchanged.

### Fixed: `get_downsampled_parameters` modified the cached maps

Masking wrote NaNs into the internal CBV, CBF and MTT maps, affecting every later use. The function now works on
copies.

### Fixed: last acquired frame dropped

`load_and_preprocess_dicom` resampled to `np.arange(t0, t_end, dt)`, which excludes `t_end`. The last frame on the
uniform grid is now kept.

### Fixed: missing requirements

`nibabel`, `pandas`, `scipy` and `scikit-image` are imported but were not listed in `requirements.txt`.

### Known limitations (not changed; see the Robust-perfDSA audit)

- Tmax is overestimated by about one frame interval, because of the half weight of the first response sample in
  the Volterra scheme.
- Standard (non-circulant) truncated SVD is delay-sensitive: CBF −17% to −20% for 1–3 s tracer delay in phantom
  tests.
