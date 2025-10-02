from __future__ import annotations

import numpy as np

from ..utils.resample import rebin_linear

SPEED_OF_LIGHT_KMS = 299792.458


def estimate_shift_nm(
    ref_wavelength: np.ndarray,
    ref_flux: np.ndarray,
    wavelength: np.ndarray,
    flux: np.ndarray,
) -> float:
    """Estimate wavelength offset using cross-correlation.

    The spectra are resampled onto a common linear grid before computing the
    discrete cross-correlation. The location of the highest peak is returned in
    nanometres as a relative shift that aligns ``flux`` with ``ref_flux``.
    """

    ref_wavelength = np.asarray(ref_wavelength, dtype=float)
    ref_flux = np.asarray(ref_flux, dtype=float)
    wavelength = np.asarray(wavelength, dtype=float)
    flux = np.asarray(flux, dtype=float)

    sort_idx = np.argsort(ref_wavelength)
    ref_wavelength = ref_wavelength[sort_idx]
    ref_flux = ref_flux[sort_idx]

    grid = ref_wavelength
    span = grid[-1] - grid[0] if grid.size > 1 else 0.0
    if span <= 0:
        return 0.0

    ref_resampled = np.asarray(ref_flux, dtype=float)
    ref_resampled -= np.nanmedian(ref_resampled)

    if np.allclose(ref_resampled, 0):
        return 0.0

    def score(shift_nm: float) -> float:
        shifted = rebin_linear(wavelength - shift_nm, flux, grid)
        shifted -= np.nanmedian(shifted)
        return float(np.dot(ref_resampled, shifted))

    window = max(0.5, span * 0.05)
    coarse_shifts = np.linspace(-window, window, 81)
    coarse_scores = [score(s) for s in coarse_shifts]
    best_idx = int(np.argmax(coarse_scores))
    best_shift = coarse_shifts[best_idx]

    fine_window = window / 4
    fine_shifts = np.linspace(best_shift - fine_window, best_shift + fine_window, 41)
    fine_scores = [score(s) for s in fine_shifts]
    fine_best = fine_shifts[int(np.argmax(fine_scores))]

    return float(fine_best)


def estimate_radial_velocity_kms(
    ref_wavelength: np.ndarray,
    ref_flux: np.ndarray,
    wavelength: np.ndarray,
    flux: np.ndarray,
) -> float:
    """Infer a radial velocity offset using the wavelength shift estimator."""

    shift_nm = estimate_shift_nm(ref_wavelength, ref_flux, wavelength, flux)
    pivot = np.nanmedian(ref_wavelength)
    if pivot == 0 or not np.isfinite(pivot):
        return 0.0
    return SPEED_OF_LIGHT_KMS * (shift_nm / pivot)
