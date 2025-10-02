from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from pyvo.dal import tap

from ..models import SearchParameters, SpectrumMetadata

logger = logging.getLogger(__name__)


class ESOQueryError(RuntimeError):
    """Raised when an ESO query cannot be completed."""


@dataclass
class ESOQueryResult:
    records: List[SpectrumMetadata]
    is_mock: bool = False
    message: str | None = None


def _resolve_target(target: str) -> SkyCoord:
    """Best-effort coordinate parser for the user supplied target string."""
    target = target.strip()
    if not target:
        raise ESOQueryError("Target or coordinate must be provided")

    # Try to interpret as sexagesimal coordinates first
    try:
        return SkyCoord(target, unit=(u.hourangle, u.deg))
    except Exception:
        pass

    # Try to interpret as two decimal degrees separated by whitespace / comma
    parts = [p for p in target.replace(",", " ").split() if p]
    if len(parts) >= 2:
        try:
            lon, lat = float(parts[0]), float(parts[1])
            return SkyCoord(lon, lat, unit="deg")
        except Exception:
            pass

    # Fall back to name resolution (Simbad). This requires network access and
    # therefore may raise an exception which we forward to the caller.
    try:
        return SkyCoord.from_name(target)
    except Exception as exc:
        raise ESOQueryError(f"Unable to interpret target '{target}': {exc}") from exc


def _to_metadata_row(row) -> SpectrumMetadata:
    def safe_get(name: str):
        try:
            return row[name]
        except Exception:
            return None

    def to_float(value, scale: float = 1.0):
        if value is None:
            return None
        try:
            return float(value) * scale
        except Exception:
            return None

    wmin_nm = to_float(safe_get("em_min"), scale=1e9)
    wmax_nm = to_float(safe_get("em_max"), scale=1e9)
    res_power = to_float(safe_get("em_res_power"))

    snr = to_float(safe_get("o_snr"))
    if snr is None:
        snr = to_float(safe_get("signal_to_noise"))
    if snr is None:
        snr = to_float(safe_get("snr"))

    release_date = safe_get("t_release") or safe_get("obs_release_date")
    if release_date is not None:
        release_date = str(release_date)

    obs_id = safe_get("obs_publisher_did") or safe_get("obs_id") or safe_get("dp_id")
    obs_id = str(obs_id) if obs_id is not None else "unknown"

    instrument = safe_get("instrument_name") or safe_get("instrument_short_name")
    target = safe_get("target_name") or safe_get("s_ra")

    product_type = safe_get("dataproduct_subtype") or safe_get("dataproduct_type")
    rights = safe_get("data_rights") or safe_get("access_rights")
    dp_id = safe_get("dp_id")
    proposal_id = safe_get("proposal_id") 
    is_public = True
    if rights is not None:
        try:
            is_public = str(rights).lower() == "public"
        except Exception:
            is_public = True

    download_url = None
    if dp_id:
        download_url = f"https://dataportal.eso.org/dataportal_new/file/{dp_id}"

    return SpectrumMetadata(
        obs_id=obs_id,
        target=str(target) if target is not None else "",
        instrument=str(instrument) if instrument is not None else "",
        wvl_min_nm=wmin_nm,
        wvl_max_nm=wmax_nm,
        resolution=res_power,
        snr_reported=snr,
        release_date=release_date,
        access_url=download_url,
        product_type=str(product_type) if product_type is not None else None,
        is_public=is_public,
        dp_id=str(dp_id) if dp_id is not None else None,
        proposal_id=str(proposal_id) if proposal_id is not None else None,
    )


def _mock_results(params: SearchParameters) -> ESOQueryResult:
    rng = np.random.default_rng(42)
    centre = params.wavelength_nm or rng.uniform(350, 900)
    widths = rng.uniform(5, 80, size=3)
    records = []
    for idx, width in enumerate(widths, start=1):
        wmin = centre - width / 2
        wmax = centre + width / 2
        records.append(
            SpectrumMetadata(
                obs_id=f"MOCK-{idx:03d}",
                target=f"{params.target or 'Mock Target'}",
                instrument=f"MockInst-{idx}",
                wvl_min_nm=wmin,
                wvl_max_nm=wmax,
                resolution=float(rng.uniform(1000, 80000)),
                snr_reported=float(rng.uniform(5, 100)),
                release_date="2020-01-01",
                access_url=None,
                product_type="MOCK",
                is_public=bool(idx % 2),
            )
        )
    return ESOQueryResult(records, is_mock=True, message="Using mock results; ESO query failed.")


class ESOClient:
    """High level interface around the provided TAP query helper."""

    def __init__(self) -> None:
        self._tap_request = None

    def search(self, params: SearchParameters) -> ESOQueryResult:
        logger.info("Running ESO query for %s", params.target)
        try:
            coord = _resolve_target(params.target)
        except ESOQueryError as exc:
            logger.exception("Failed to resolve target")
            raise

        try:
            query = _build_query(
                coord,
                search_radius_arcsec=params.search_radius_arcsec,
                wvl_include_nm=params.wavelength_nm,
                min_resolution=params.min_resolution,
            )
            results = _run_query(query)
        except Exception as exc:
            logger.warning("ESO query failed: %s", exc, exc_info=True)
            return _mock_results(params)

        table = getattr(results, "table", None)
        if table is None or len(table) == 0:
            return ESOQueryResult(records=[], message="Query returned no rows.")

        records = [_to_metadata_row(row) for row in table]
        return ESOQueryResult(records=records)
ESO_TAP_URL = "http://archive.eso.org/tap_obs"


def _build_query(
    coord: SkyCoord,
    search_radius_arcsec: float,
    wvl_include_nm: float | None,
    min_resolution: float | None,
) -> str:
    query = [
        "SELECT *",
        "FROM ivoa.ObsCore",
        "WHERE intersects(s_region, circle('', {ra:.6f}, {dec:.6f}, {radius:.6f})) = 1".format(
            ra=coord.ra.degree,
            dec=coord.dec.degree,
            radius=search_radius_arcsec / 3600,
        ),
        "AND dataproduct_type = 'spectrum'",
    ]
    if wvl_include_nm is not None:
        centre_m = wvl_include_nm * 1e-9
        query.append(f"AND em_min < {centre_m:.6e} AND em_max > {centre_m:.6e}")
    if min_resolution is not None:
        query.append(f"AND em_res_power > {min_resolution:.0f}")
    return "\n".join(query)


def _run_query(query: str):
    service = tap.TAPService(ESO_TAP_URL)
    return service.search(query=query, maxrec=1000)
