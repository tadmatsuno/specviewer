from __future__ import annotations

from .gui.main_window import SpectrumViewerApp, run_app


class SpecViewerApp(SpectrumViewerApp):
    """Backward compatible alias exposing the PyQt-based application."""

    def __init__(self, argv: list[str] | None = None) -> None:
        super().__init__(argv)


def main() -> int:
    return run_app()


if __name__ == "__main__":
    raise SystemExit(main())
