from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
)

from ..models import SpectrumData
from ..services.spectra import SpectrumCombiner


@dataclass
class ViewerResult:
    combined: SpectrumData
    sources: list[SpectrumData]


class SpectrumViewerDialog(QDialog):
    """Dialog that renders spectra and allows optional combination."""

    def __init__(
        self,
        spectra: Iterable[SpectrumData],
        combiner: SpectrumCombiner,
        parent=None,
        on_combine: Callable[[SpectrumData, list[SpectrumData]], None] | None = None,
        snr_window: tuple[float, float] | None = None,
    ) -> None:
        print(snr_window)
        super().__init__(parent)
        self.setWindowTitle("Spectrum Viewer")
        self.resize(900, 600)
        self._spectra = list(spectra)
        self._combiner = combiner
        self._result: Optional[ViewerResult] = None
        self._on_combine = on_combine
        self._snr_window = snr_window

        self._rest_frame_box = QCheckBox("Align to rest frame")
        self._rest_frame_box.stateChanged.connect(self._redraw)

        self._combine_button = QPushButton("Combine spectra")
        self._combine_button.clicked.connect(self._combine)

        controls = QHBoxLayout()
        controls.addWidget(self._rest_frame_box)
        controls.addStretch(1)
        controls.addWidget(self._combine_button)

        self._figure = Figure(figsize=(8, 5), dpi=100)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._axes = self._figure.add_subplot(111)
        self._axes.set_xlabel("Wavelength (nm)")
        self._axes.set_ylabel("Flux (arbitrary units)")

        toolbar = NavigationToolbar2QT(self._canvas, self)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self._canvas, 1)
        layout.addWidget(toolbar)

        self._redraw()
        self.setModal(False)
        self.setWindowModality(Qt.NonModal)

    # ------------------------------------------------------------------ public API
    @property
    def result(self) -> Optional[ViewerResult]:
        return self._result

    # ------------------------------------------------------------------ helpers
    def _redraw(self) -> None:
        self._axes.clear()
        self._axes.set_xlabel("Wavelength")
        self._axes.set_ylabel("Normalized flux")

        if not self._spectra:
            self._axes.text(0.5, 0.5, "No spectra", ha="center", transform=self._axes.transAxes)
            self._canvas.draw()
            return

        rest_frame = self._rest_frame_box.isChecked()
        reference_peak = None
        if rest_frame:
            reference_peak = self._peak_position(self._spectra[0])

        xmins: list[float] = []
        xmaxs: list[float] = []
        ymin: float | None = None
        ymax: float | None = None

        for spectrum in self._spectra:
            wavelength = spectrum.wavelength
            flux = spectrum.flux
            if rest_frame and reference_peak is not None:
                peak = self._peak_position(spectrum)
                shift = peak - reference_peak
                wavelength = wavelength - shift
            if wavelength.size:
                xmins.append(float(np.nanmin(wavelength)))
                xmaxs.append(float(np.nanmax(wavelength)))

            if self._snr_window is not None:
                mask = (wavelength >= self._snr_window[0]) & (wavelength <= self._snr_window[1])
                if not np.any(mask):
                    mask = slice(None)
            else:
                mask = slice(None)

            plot_wave = wavelength[mask]
            plot_flux = flux[mask]

            median = np.nanmedian(plot_flux[plot_flux > 0])
            if not np.isfinite(median) or median == 0:
                fallback = np.nanmedian(flux)
                median = fallback if np.isfinite(fallback) and fallback != 0 else 1.0
            normalised_flux = plot_flux / median
            normalised_flux_all = flux / median

            label = spectrum.metadata.dp_id or spectrum.metadata.obs_id
            self._axes.plot(wavelength, normalised_flux_all, label=label)

            if normalised_flux.size:
                local_min = float(np.nanmin(normalised_flux))
                local_max = float(np.nanmax(normalised_flux))
                if np.isfinite(local_min):
                    ymin = local_min if ymin is None else min(ymin, local_min)
                if np.isfinite(local_max):
                    ymax = local_max if ymax is None else max(ymax, local_max)

        self._axes.legend(loc="upper right", fontsize="small")
        if self._snr_window is not None:
            self._axes.set_xlim(self._snr_window[0], self._snr_window[1])
        elif xmins and xmaxs:
            min_wave, max_wave = min(xmins), max(xmaxs)
            span = max_wave - min_wave
            if span > 0:
                desired = min(span, max(span * 0.4, 10.0))
                centre = (min_wave + max_wave) / 2
                half = desired / 2
                self._axes.set_xlim(centre - half, centre + half)

        if ymin is not None and ymax is not None and ymin < ymax:
            padding = 0.15 * (ymax - ymin)
#            self._axes.set_ylim(ymin - padding, ymax + padding)
            self._axes.set_ylim(-0.15, 1.15)
        self._canvas.draw()

    @staticmethod
    def _peak_position(spectrum: SpectrumData) -> float:
        flux = spectrum.flux
        wavelength = spectrum.wavelength
        if flux.size == 0:
            return 0.0
        return float(wavelength[np.argmax(flux)])

    def _combine(self) -> None:
        if not self._spectra:
            return
        rest_frame = self._rest_frame_box.isChecked()
        sources = list(self._spectra)
        combined = self._combiner.combine(sources, rest_frame=rest_frame)
        self._spectra.append(combined)
        self._result = ViewerResult(combined=combined, sources=sources)
        if self._on_combine is not None:
            self._on_combine(combined, sources)
        self._redraw()
