from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Tuple

from ..models import SearchParameters, SpectrumData, SpectrumMetadata
from .spectra import SpectrumStore


class SpectrumDownloader:
    """Ensures spectra are materialised locally via the configured store."""

    def __init__(self, store: SpectrumStore | None = None) -> None:
        self.store = store or SpectrumStore()

    def download(
        self,
        records: Iterable[SpectrumMetadata],
        params: SearchParameters,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        failure_callback: Optional[Callable[[SpectrumMetadata, Exception], None]] = None,
    ) -> List[SpectrumData]:
        items = list(records)
        total = len(items)
        downloaded: List[SpectrumData] = []
        for idx, record in enumerate(items, start=1):
            try:
                downloaded.append(self.store.get(record, params))
            except Exception as exc:
                if failure_callback is not None:
                    failure_callback(record, exc)
                raise
            if progress_callback is not None:
                progress_callback(idx, total)
        return downloaded
