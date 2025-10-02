from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class SearchParameters:
    target: str
    wavelength_nm: Optional[float] = None
    min_resolution: Optional[float] = None
    search_radius_arcsec: float = 30.0
    download_dir: Path = field(default_factory=lambda: Path.cwd() / "downloaded_spectra")
    snr_center_nm: Optional[float] = None
    snr_half_range_nm: Optional[float] = None

    @property
    def snr_window_nm(self) -> Optional[tuple[float, float]]:
        if self.snr_center_nm is None or self.snr_half_range_nm is None:
            return None
        return (
            self.snr_center_nm - self.snr_half_range_nm,
            self.snr_center_nm + self.snr_half_range_nm,
        )


@dataclass
class SpectrumMetadata:
    obs_id: str
    target: str
    instrument: str
    wvl_min_nm: Optional[float]
    wvl_max_nm: Optional[float]
    resolution: Optional[float]
    snr_reported: Optional[float]
    release_date: Optional[str]
    access_url: Optional[str] = None
    product_type: Optional[str] = None
    selected: bool = True
    local_path: Optional[Path] = None
    snr_measured: Optional[float] = None
    used_in_combination: bool = False
    is_combined_product: bool = False
    is_public: bool = True
    dp_id: Optional[str] = None
    proposal_id: Optional[str] = None

    def display_tuple(self) -> tuple[str, ...]:
        wvl_range = "-"
        if self.wvl_min_nm is not None and self.wvl_max_nm is not None:
            wvl_range = f"{self.wvl_min_nm:.1f} - {self.wvl_max_nm:.1f}"
        res = f"{self.resolution:.0f}" if self.resolution is not None else "-"
        snr = f"{self.snr_reported:.1f}" if self.snr_reported is not None else "-"
        snr_meas = f"{self.snr_measured:.1f}" if self.snr_measured is not None else ""
        selected = "x" if self.selected else ""
        return (
            self.target,
            self.instrument,
            wvl_range,
            res,
            snr,
            self.release_date or "-",
            selected,
            snr_meas,
            self.dp_id or "-",
            self.proposal_id or "-",
        )


@dataclass
class SpectrumData:
    metadata: SpectrumMetadata
    wavelength: np.ndarray
    flux: np.ndarray
    flux_err: Optional[np.ndarray] = None

    def measure_snr(self, window_nm: tuple[float, float]) -> Optional[float]:
        if self.wavelength.size == 0 or self.flux_err is None:
            return None
        mask = (self.wavelength >= window_nm[0]) & (self.wavelength <= window_nm[1])
        if not np.any(mask):
            return None
        flux = self.flux[mask]
        err = self.flux_err[mask]
        valid = np.isfinite(flux) & np.isfinite(err) & (err > 0) & (flux != 0.0)
        if not np.any(valid):
            return None
        ratio = np.abs(flux[valid] / err[valid])
        if ratio.size == 0:
            return None
        value = np.nanmedian(ratio)
        if not np.isfinite(value):
            return None
        return float(value)
