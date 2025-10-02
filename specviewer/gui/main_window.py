from __future__ import annotations

import sys
from typing import Callable, Iterable, List, Optional

from PyQt5.QtCore import Qt, QThread, QModelIndex, pyqtSignal, QAbstractTableModel
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QTableView,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
)

from ..models import SearchParameters, SpectrumData, SpectrumMetadata
from ..services.downloader import SpectrumDownloader
from ..services.eso import ESOClient, ESOQueryError, ESOQueryResult
from ..services.spectra import SpectrumCombiner, SpectrumStore
from .spectrum_viewer import SpectrumViewerDialog


class SpectrumTableModel(QAbstractTableModel):
    headers = (
        "Target",
        "Instrument",
        "Wavelength (nm)",
        "Resolution",
        "Reported SNR",
        "Release",
        "Selected",
        "Measured SNR",
        "dp_id",
        "proposal_id"
    )

    def __init__(self) -> None:
        super().__init__()
        self.records: List[SpectrumMetadata] = []
        self._sort_column: int = 0
        self._sort_desc: bool = False

    # Qt model API ------------------------------------------------------
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.records)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.headers)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # noqa: N802
        if not index.isValid():
            return None
        metadata = self.records[index.row()]
        column = index.column()

        if role == Qt.DisplayRole:
            return metadata.display_tuple()[column]
        if role == Qt.CheckStateRole and column == 6:
            return Qt.Checked if metadata.selected else Qt.Unchecked
        if role == Qt.ForegroundRole:
            if metadata.is_combined_product:
                return QBrush(QColor("#1a73e8"))
            if metadata.used_in_combination:
                return QBrush(QColor("#6c757d"))
            if not metadata.is_public:
                return QBrush(QColor("#b91c1c"))
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):  # noqa: N802
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.headers[section]
        return super().headerData(section, orientation, role)

    def flags(self, index: QModelIndex):  # noqa: N802
        if not index.isValid():
            return Qt.ItemIsEnabled
        base = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        if index.column() == 6:
            base |= Qt.ItemIsUserCheckable
        return base

    def setData(self, index: QModelIndex, value, role: int = Qt.EditRole):  # noqa: N802
        if not index.isValid():
            return False
        if index.column() == 6 and role == Qt.CheckStateRole:
            metadata = self.records[index.row()]
            if metadata.used_in_combination:
                return False
            metadata.selected = value == Qt.Checked
            self.dataChanged.emit(index, index, [Qt.CheckStateRole, Qt.DisplayRole])
            return True
        return False

    def sort(self, column: int, order: Qt.SortOrder = Qt.AscendingOrder) -> None:  # noqa: N802
        descending = order == Qt.DescendingOrder
        key = self._sort_key(column)
        self.layoutAboutToBeChanged.emit()
        self.records.sort(key=key, reverse=descending)
        self._sort_column = column
        self._sort_desc = descending
        self.layoutChanged.emit()

    # Helpers -----------------------------------------------------------
    def set_records(self, records: List[SpectrumMetadata]) -> None:
        self.beginResetModel()
        self.records = list(records)
        self.endResetModel()
        self.sort(self._sort_column, Qt.DescendingOrder if self._sort_desc else Qt.AscendingOrder)

    def record_at(self, row: int) -> SpectrumMetadata:
        return self.records[row]

    def add_record(self, metadata: SpectrumMetadata) -> None:
        self.beginInsertRows(QModelIndex(), len(self.records), len(self.records))
        self.records.append(metadata)
        self.endInsertRows()
        self.sort(self._sort_column, Qt.DescendingOrder if self._sort_desc else Qt.AscendingOrder)

    @staticmethod
    def _sort_key(column: int):
        def safe_lower(value: Optional[str]) -> str:
            return value.lower() if isinstance(value, str) else ""

        mapping = {
            0: lambda m: safe_lower(m.target),
            1: lambda m: safe_lower(m.instrument),
            2: lambda m: (
                m.wvl_min_nm if m.wvl_min_nm is not None else float("inf"),
                m.wvl_max_nm if m.wvl_max_nm is not None else float("inf"),
            ),
            3: lambda m: m.resolution if m.resolution is not None else float("-inf"),
            4: lambda m: m.snr_reported if m.snr_reported is not None else float("-inf"),
            5: lambda m: safe_lower(m.release_date or ""),
            6: lambda m: not m.selected,
            7: lambda m: m.snr_measured if m.snr_measured is not None else float("-inf"),
            8: lambda m: safe_lower(m.dp_id or ""),
            9: lambda m: safe_lower(m.proposal_id or ""),
        }
        return mapping.get(column, lambda m: safe_lower(m.target))


class SearchWorker(QThread):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, client: ESOClient, params: SearchParameters) -> None:
        super().__init__()
        self.client = client
        self.params = params

    def run(self) -> None:
        try:
            result = self.client.search(self.params)
        except ESOQueryError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # pragma: no cover - network issues
            self.failed.emit(str(exc))
        else:
            self.finished.emit(result)


class DownloadWorker(QThread):
    finished = pyqtSignal(list)
    failed = pyqtSignal(object, str)
    progress = pyqtSignal(int, int)

    def __init__(
        self,
        downloader: SpectrumDownloader,
        records: List[SpectrumMetadata],
        params: SearchParameters,
    ) -> None:
        super().__init__()
        self.downloader = downloader
        self.records = records
        self.params = params

    def run(self) -> None:
        def callback(done: int, total: int) -> None:
            self.progress.emit(done, total)

        failed_record = {"metadata": None, "error": None}

        def failure(record, exc):
            failed_record["metadata"] = record
            failed_record["error"] = exc

        try:
            spectra = self.downloader.download(
                self.records,
                self.params,
                progress_callback=callback,
                failure_callback=failure,
            )
        except Exception as exc:  # pragma: no cover - network issues
            meta = failed_record["metadata"]
            self.failed.emit(meta, str(exc))
            return
        self.finished.emit(spectra)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ESO Spectrum Viewer")
        self.resize(1200, 760)

        self.eso_client = ESOClient()
        self.store = SpectrumStore()
        self.downloader = SpectrumDownloader(self.store)
        self.combiner = SpectrumCombiner()

        self.current_params: Optional[SearchParameters] = None
        self._viewer_dialogs: list[SpectrumViewerDialog] = []

        self._build_ui()

    # UI ----------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        central.setLayout(root_layout)

        form_grid = QGridLayout()
        root_layout.addLayout(form_grid)

        self.target_edit = QLineEdit()
        self.wavelength_edit = QLineEdit()
        self.resolution_edit = QLineEdit()
        self.radius_edit = QLineEdit("30")
        self.snr_centre_edit = QLineEdit()
        self.snr_range_edit = QLineEdit("1.0")

        form_grid.addWidget(QLabel("Target / Coordinates"), 0, 0)
        form_grid.addWidget(self.target_edit, 0, 1)
        form_grid.addWidget(QLabel("Wavelength (nm)"), 0, 2)
        form_grid.addWidget(self.wavelength_edit, 0, 3)
        form_grid.addWidget(QLabel("Min resolution"), 0, 4)
        form_grid.addWidget(self.resolution_edit, 0, 5)
        form_grid.addWidget(QLabel("Radius (arcsec)"), 0, 6)
        form_grid.addWidget(self.radius_edit, 0, 7)
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self._on_search)
        form_grid.addWidget(self.search_button, 0, 8)

        form_grid.addWidget(QLabel("Wavelength centre"), 1, 0)
        form_grid.addWidget(self.snr_centre_edit, 1, 1)
        form_grid.addWidget(QLabel("Range (± )"), 1, 2)
        form_grid.addWidget(self.snr_range_edit, 1, 3)
        self.measure_button = QPushButton("Measure SNR")
        self.measure_button.clicked.connect(self._on_measure_snr)
        form_grid.addWidget(self.measure_button, 1, 4)

        actions = QHBoxLayout()
        root_layout.addLayout(actions)
        self.select_all_button = QPushButton("Select All")
        self.select_all_button.clicked.connect(lambda: self._set_selection(True))
        actions.addWidget(self.select_all_button)
        self.deselect_all_button = QPushButton("Deselect All")
        self.deselect_all_button.clicked.connect(lambda: self._set_selection(False))
        actions.addWidget(self.deselect_all_button)
        self.download_button = QPushButton("Download Selected")
        self.download_button.clicked.connect(self._on_download)
        actions.addWidget(self.download_button)
        self.view_button = QPushButton("View Selected")
        self.view_button.clicked.connect(self._on_view)
        actions.addWidget(self.view_button)
        self.combine_button = QPushButton("Combine Selected")
        self.combine_button.clicked.connect(self._on_combine)
        actions.addWidget(self.combine_button)
        actions.addStretch(1)

        self.table_model = SpectrumTableModel()
        self.table = QTableView()
        self.table.setModel(self.table_model)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSelectionMode(QTableView.SingleSelection)
        self.table.doubleClicked.connect(self._on_table_double_click)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(8, QHeaderView.ResizeToContents)
        self.table.setColumnWidth(8, 220)
        header.sectionClicked.connect(self._on_header_clicked)
        root_layout.addWidget(self.table, 1)

        status = self.statusBar()
        self.status_label = QLabel("Ready")
        status.addPermanentWidget(self.status_label, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        self.progress.setFixedWidth(200)
        status.addPermanentWidget(self.progress)

    # Helpers -----------------------------------------------------------
    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _parse_float(self, value: str) -> Optional[float]:
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"Cannot parse '{value}' as a number") from exc

    def _collect_params(self) -> SearchParameters:
        wavelength = self._parse_float(self.wavelength_edit.text()) if self.wavelength_edit.text().strip() else None
        min_res = self._parse_float(self.resolution_edit.text()) if self.resolution_edit.text().strip() else None
        radius = self._parse_float(self.radius_edit.text()) if self.radius_edit.text().strip() else None
        snr_centre = self._parse_float(self.snr_centre_edit.text()) if self.snr_centre_edit.text().strip() else None
        snr_range = self._parse_float(self.snr_range_edit.text()) if self.snr_range_edit.text().strip() else None
        params = SearchParameters(
            target=self.target_edit.text(),
            wavelength_nm=wavelength,
            min_resolution=min_res,
            search_radius_arcsec=radius if radius is not None else 30.0,
            snr_center_nm=snr_centre,
            snr_half_range_nm=abs(snr_range) if snr_range is not None else None,
        )
        if self.current_params is not None:
            params.download_dir = self.current_params.download_dir
        self.current_params = params
        return params

    def _selected_records(self) -> List[SpectrumMetadata]:
        return [record for record in self.table_model.records if record.selected]

    def _set_selection(self, value: bool) -> None:
        for metadata in self.table_model.records:
            if metadata.used_in_combination and value:
                continue
            metadata.selected = value and not metadata.used_in_combination
        if self.table_model.records:
            self.table_model.dataChanged.emit(
                self.table_model.index(0, 6),
                self.table_model.index(self.table_model.rowCount() - 1, 6),
                [Qt.CheckStateRole, Qt.DisplayRole],
            )

    # Event handlers ----------------------------------------------------
    def _on_search(self) -> None:
        try:
            params = self._collect_params()
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid input", str(exc))
            return

        if not params.target.strip():
            QMessageBox.warning(self, "Missing target", "Please provide a target name or coordinates")
            return

        self._set_status("Querying ESO archive...")
        self.search_button.setEnabled(False)
        self.worker = SearchWorker(self.eso_client, params)
        self.worker.finished.connect(self._on_search_finished)
        self.worker.failed.connect(self._on_search_failed)
        self.worker.finished.connect(lambda _: self.worker.deleteLater())
        self.worker.failed.connect(lambda _: self.worker.deleteLater())
        self.worker.start()

    def _on_search_finished(self, result: ESOQueryResult) -> None:
        self.table_model.set_records(result.records)
        if result.is_mock:
            self._set_status(result.message or "Using mock results")
        else:
            self._set_status(result.message or f"Loaded {len(result.records)} spectra")
        self.search_button.setEnabled(True)

    def _on_search_failed(self, message: str) -> None:
        QMessageBox.critical(self, "ESO query failed", message)
        self._set_status("Query failed")
        self.search_button.setEnabled(True)

    def _on_header_clicked(self, section: int) -> None:
        order = Qt.DescendingOrder if (self.table_model._sort_column == section and not self.table_model._sort_desc) else Qt.AscendingOrder
        self.table_model.sort(section, order)

    def _on_table_double_click(self, index: QModelIndex) -> None:
        if index.column() == 6:
            self.table_model.setData(index, Qt.Unchecked if self.table_model.data(index, Qt.CheckStateRole) == Qt.Checked else Qt.Checked, Qt.CheckStateRole)

    def _on_download(self) -> None:
        try:
            params = self._collect_params()
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid input", str(exc))
            return

        selected = self._selected_records()
        if not selected:
            QMessageBox.information(self, "No selection", "Select at least one spectrum to download")
            return

        self.download_thread = DownloadWorker(self.downloader, selected, params)
        self.download_thread.progress.connect(self._on_download_progress)
        self.download_thread.finished.connect(lambda spectra: self._on_download_finished(spectra, params))
        self.download_thread.failed.connect(lambda record, message: self._on_download_failed(record, message))
        self.download_thread.finished.connect(lambda _: self.download_thread.deleteLater())
        self.download_thread.failed.connect(lambda _: self.download_thread.deleteLater())
        self.progress.setVisible(True)
        self.progress.setRange(0, max(1, len(selected)))
        self.progress.setValue(0)
        self._set_status("Downloading spectra...")
        self.download_thread.start()

    def _on_download_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)

    def _on_download_finished(self, spectra: List[SpectrumData], params: SearchParameters) -> None:
        for spectrum in spectra:
            if params.snr_window_nm is not None:
                spectrum.metadata.snr_measured = spectrum.measure_snr(params.snr_window_nm)
        self.progress.setVisible(False)
        self.table_model.layoutChanged.emit()
        self._set_status(f"Downloaded {len(spectra)} spectra to {params.download_dir}")

    def _on_download_failed(self, record: SpectrumMetadata | None, message: str) -> None:
        QMessageBox.critical(
            self,
            "Download failed",
            (
                "Unable to download one or more spectra.\n"
                "Ensure the product is public or check your network connection.\n"
                + (
                    f"\nURL: {record.access_url if record is not None else 'Unknown'}"
                    if record is not None and record.access_url
                    else ""
                )
                + f"\n\nDetails: {message}"
            ),
        )
        self.progress.setVisible(False)
        self._set_status("Download failed")

    def _ensure_params(self) -> Optional[SearchParameters]:
        try:
            return self._collect_params()
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid input", str(exc))
            return None

    def _on_view(self) -> None:
        params = self._ensure_params()
        if params is None:
            return
        selected = self._selected_records()
        if not selected:
            QMessageBox.information(self, "No selection", "Select spectra to view")
            return
        spectra = self.store.bulk_get(selected, params)
        dialog = SpectrumViewerDialog(
            spectra,
            combiner=self.combiner,
            parent=self,
            on_combine=self._on_viewer_combined,
            snr_window = params.snr_window_nm
        )
        dialog.setAttribute(Qt.WA_DeleteOnClose, True)
        dialog.finished.connect(lambda _: self._remove_viewer_dialog(dialog))
        self._viewer_dialogs.append(dialog)
        dialog.show()

    def _on_combine(self) -> None:
        params = self._ensure_params()
        if params is None:
            return
        selected = self._selected_records()
        if len(selected) < 2:
            QMessageBox.information(self, "Need more spectra", "Select at least two spectra to combine")
            return
        spectra = self.store.bulk_get(selected, params)
        combined = self.combiner.combine(spectra, rest_frame=False)
        self._register_combination(combined, spectra)
        self._set_status("Created combined spectrum")

    def _register_combination(self, spectrum: SpectrumData, sources: Iterable[SpectrumData]) -> None:
        params = self.current_params
        if params and params.snr_window_nm is not None:
            spectrum.metadata.snr_measured = spectrum.measure_snr(params.snr_window_nm)
        for src in sources:
            meta = src.metadata
            if meta.is_combined_product:
                continue
            meta.used_in_combination = True
            meta.selected = False
        self.store.store(spectrum)
        self.table_model.add_record(spectrum.metadata)
        self.table_model.layoutChanged.emit()

    def _on_measure_snr(self) -> None:
        params = self._ensure_params()
        if params is None:
            return
        window = params.snr_window_nm
        if window is None:
            QMessageBox.information(self, "Missing window", "Specify both centre and range to measure SNR")
            return
        updated = 0
        updated_rows: list[int] = []
        missing_uncertainty = 0
        for row, metadata in enumerate(self.table_model.records):
            cached = self.store.cached(metadata)
            if cached is None:
                continue
            snr_value = cached.measure_snr(window)
            if snr_value is None:
                metadata.snr_measured = None
                missing_uncertainty += 1
                continue
            metadata.snr_measured = snr_value
            updated += 1
            updated_rows.append(row)
        if updated == 0:
            if missing_uncertainty > 0:
                QMessageBox.information(
                    self,
                    "No spectra",
                    "Flux uncertainties not available in the downloaded files; unable to measure SNR.",
                )
            else:
                QMessageBox.information(self, "No spectra", "Download spectra before measuring SNR")
        else:
            if updated_rows:
                top_row = min(updated_rows)
                bottom_row = max(updated_rows)
                top_index = self.table_model.index(top_row, 7)
                bottom_index = self.table_model.index(bottom_row, 7)
                self.table_model.dataChanged.emit(top_index, bottom_index, [Qt.DisplayRole])
            self._set_status(f"Measured SNR for {updated} spectra")

    def _remove_viewer_dialog(self, dialog: SpectrumViewerDialog) -> None:
        try:
            self._viewer_dialogs.remove(dialog)
        except ValueError:  # dialog already removed
            pass

    def _on_viewer_combined(self, combined: SpectrumData, sources: list[SpectrumData]) -> None:
        self._register_combination(combined, sources)


class SpectrumViewerApp:
    def __init__(self, argv: list[str] | None = None) -> None:
        self.app = QApplication(argv or sys.argv)
        self.window = MainWindow()

    def run(self) -> int:
        self.window.show()
        return self.app.exec_()


def run_app() -> int:
    app = SpectrumViewerApp()
    return app.run()
