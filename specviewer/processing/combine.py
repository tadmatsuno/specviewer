from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from ..utils.resample import rebin_linear
from .correlation import estimate_shift_nm


@dataclass
class CombinationResult:
    wavelength: np.ndarray
    flux: np.ndarray
    shifts_applied_nm: List[float]


def combine_spectra(
    spectra: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    reference_index: int = 0,
    rest_frame: bool = False,
) -> CombinationResult:
    """Average spectra onto the reference wavelength grid.

    ``spectra`` should be an iterable of ``(wavelength, flux)`` pairs. When
    ``rest_frame`` is ``True`` each spectrum is shifted to match the reference
    using the cross-correlation estimator before averaging.
    """

    spectra = list(spectra)
    if not spectra:
        raise ValueError("No spectra provided")

    ref_wave, ref_flux = spectra[reference_index]
    ref_wave = np.asarray(ref_wave, dtype=float)
    if ref_wave.ndim != 1:
        raise ValueError("Wavelength arrays must be one-dimensional")

    accum = np.zeros_like(ref_wave, dtype=float)
    weights = np.zeros_like(ref_wave, dtype=float)
    shifts: List[float] = []

    for wave, flux in spectra:
        wave = np.asarray(wave, dtype=float)
        flux = np.asarray(flux, dtype=float)
        if wave.ndim != 1 or flux.ndim != 1:
            raise ValueError("Wavelength and flux arrays must be one-dimensional")
        if wave.size != flux.size:
            raise ValueError("Wavelength and flux arrays must match in length")

        shift = 0.0
        if rest_frame:
            shift = estimate_shift_nm(ref_wave, ref_flux, wave, flux)
            wave = wave - shift
        rebinned = rebin_linear(wave, flux, ref_wave)
        valid = np.isfinite(rebinned)
        accum[valid] += rebinned[valid]
        weights[valid] += 1.0
        shifts.append(shift)

    weights = np.where(weights == 0, 1.0, weights)
    combined = accum / weights
    return CombinationResult(wavelength=ref_wave, flux=combined, shifts_applied_nm=shifts)

