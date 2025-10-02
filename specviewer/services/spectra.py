from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

import numpy as np
import requests
from astropy.io import fits

from ..processing.combine import combine_spectra

from ..models import SearchParameters, SpectrumData, SpectrumMetadata

logger = logging.getLogger(__name__)


class SpectrumRepository:
    def fetch(self, metadata: SpectrumMetadata, params: SearchParameters) -> SpectrumData:
        raise NotImplementedError



class ESOArchiveRepository(SpectrumRepository):
    """Download spectra from the ESO archive anonymously and parse FITS payloads."""

    def __init__(
        self,
        session: requests.Session | None = None,
        chunk_size: int = 65536,
    ) -> None:
        self.session = session or requests.Session()
        self.chunk_size = chunk_size

    def _filename(self, metadata: SpectrumMetadata) -> str:
        if metadata.local_path is not None:
            return metadata.local_path.name
        if metadata.dp_id:
            return f"{metadata.dp_id}.fits"
        return f"{metadata.obs_id.replace('/', '_')}.fits"

    def _parse_fits(self, path: Path, metadata: SpectrumMetadata) -> SpectrumData:
        with fits.open(path) as hdul:
            for hdu in hdul:
                data = getattr(hdu, "data", None)
                columns = getattr(getattr(hdu, "columns", None), "names", None)
                if data is None or columns is None:
                    continue
                upper_map = {name.upper(): name for name in columns}
                wave_key = next((upper_map[key] for key in ("WAVE", "WAVELENGTH", "LAMBDA") if key in upper_map), None)
                flux_key = next((upper_map[key] for key in upper_map if "FLUX" in key), None)
                err_key = next((upper_map[key] for key in upper_map if "ERR" in key or "ERROR" in key), None)
                if wave_key and flux_key:
                    wavelength = np.asarray(data[wave_key], dtype=float)
                    flux = np.asarray(data[flux_key], dtype=float)
                    flux_err = None
                    if err_key is not None:
                        flux_err = np.asarray(data[err_key], dtype=float)
                    if wavelength.ndim > 1:
                        wavelength = wavelength[0]
                    if flux.ndim > 1:
                        flux = flux[0]
                    if flux_err is not None and flux_err.ndim > 1:
                        flux_err = flux_err[0]
                    if wavelength.size and flux.size:
                        return SpectrumData(
                            metadata=metadata,
                            wavelength=wavelength,
                            flux=flux,
                            flux_err=flux_err,
                        )
                # Fall back to generic table interpretation
                if columns and len(columns) >= 2:
                    wave_col = next((name for name in columns if "WAVE" in name.upper()), columns[0])
                    flux_col = next((name for name in columns if "FLUX" in name.upper()), columns[1])
                    err_col = next((name for name in columns if "ERR" in name.upper() or "ERROR" in name.upper()), None)
                    wave = np.asarray(data[wave_col], dtype=float)
                    flx = np.asarray(data[flux_col], dtype=float)
                    err = None
                    if err_col is not None:
                        err = np.asarray(data[err_col], dtype=float)
                    if wave.ndim > 1:
                        wave = wave[0]
                    if flx.ndim > 1:
                        flx = flx[0]
                    if err is not None and err.ndim > 1:
                        err = err[0]
                    if wave.size and flx.size:
                        return SpectrumData(metadata=metadata, wavelength=wave, flux=flx, flux_err=err)
        raise RuntimeError("Unable to locate wavelength/flux columns in FITS file")

    def fetch(self, metadata: SpectrumMetadata, params: SearchParameters) -> SpectrumData:
        if not metadata.dp_id:
            raise RuntimeError("No dp_id available for spectrum")
        download_dir = params.download_dir
        download_dir.mkdir(parents=True, exist_ok=True)

        filename = self._filename(metadata)
        destination = download_dir / filename

        url = self._dataportal_url(metadata.dp_id)
        self._stream_to_file(url, destination)
        metadata.local_path = destination
        return self._parse_fits(destination, metadata)

    # ------------------------------------------------------------------ internal helpers
    def _stream_to_file(self, url: str, destination: Path) -> None:
        response = self.session.get(url, stream=True, timeout=120)
        response.raise_for_status()
        tmp_path = destination.with_suffix(destination.suffix + ".part")
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(self.chunk_size):
                if chunk:
                    handle.write(chunk)
        tmp_path.replace(destination)

    @staticmethod
    def _dataportal_url(dp_id: str) -> str:
        dp_id = dp_id.strip()
        return f"https://dataportal.eso.org/dataportal_new/file/{dp_id}"


class SpectrumStore:
    """Caches downloaded spectra in memory."""

    def __init__(self, repository: SpectrumRepository | None = None) -> None:
        if repository is None:
            repository = ESOArchiveRepository()
        self.repository = repository
        self._cache: dict[str, SpectrumData] = {}

    def get(self, metadata: SpectrumMetadata, params: SearchParameters) -> SpectrumData:
        if metadata.obs_id not in self._cache:
            self._cache[metadata.obs_id] = self.repository.fetch(metadata, params)
        return self._cache[metadata.obs_id]

    def bulk_get(self, records: Iterable[SpectrumMetadata], params: SearchParameters) -> List[SpectrumData]:
        return [self.get(record, params) for record in records]

    def store(self, spectrum: SpectrumData) -> None:
        self._cache[spectrum.metadata.obs_id] = spectrum

    def cached(self, metadata: SpectrumMetadata) -> SpectrumData | None:
        return self._cache.get(metadata.obs_id)


class SpectrumCombiner:
    def combine(self, spectra: Iterable[SpectrumData], rest_frame: bool = False) -> SpectrumData:
        spectra = list(spectra)
        if not spectra:
            raise ValueError("No spectra provided for combination")

        ref_spec = spectra[0]
        combo = combine_spectra(
            [(spec.wavelength, spec.flux) for spec in spectra],
            rest_frame=rest_frame,
        )
        combined_flux = combo.flux
        ref_wave = combo.wavelength

        wmins = [spec.metadata.wvl_min_nm for spec in spectra if spec.metadata.wvl_min_nm is not None]
        wmaxs = [spec.metadata.wvl_max_nm for spec in spectra if spec.metadata.wvl_max_nm is not None]
        resolutions = [spec.metadata.resolution for spec in spectra if spec.metadata.resolution is not None]

        combined_metadata = SpectrumMetadata(
            obs_id="COMBINED-" + "+".join(spec.metadata.obs_id for spec in spectra),
            target=ref_spec.metadata.target,
            instrument="Combined",
            wvl_min_nm=min(wmins) if wmins else ref_spec.metadata.wvl_min_nm,
            wvl_max_nm=max(wmaxs) if wmaxs else ref_spec.metadata.wvl_max_nm,
            resolution=float(np.mean(resolutions)) if resolutions else ref_spec.metadata.resolution,
            snr_reported=None,
            release_date="-",
            access_url=None,
            product_type="COMBINED",
            selected=True,
            is_combined_product=True,
        )

        combined_spectrum = SpectrumData(
            metadata=combined_metadata,
            wavelength=ref_wave,
            flux=np.asarray(combined_flux),
        )
        return combined_spectrum
