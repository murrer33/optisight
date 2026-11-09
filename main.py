"""
OptiSight: AI Marketing, Packaging & Shelf Attention Analyzer
Unified Entrypoint & Compatibility Facade.

Modular Architecture:
  - core/saliency.py      : 5-Layer Biomimetic Saliency Engine, Itti-Koch, Center-Bias, WCAG & OCR
  - core/metrics.py       : NSS, CC, SIM, AUC-Judd, Baselines, Ablation & Grid-Search Optimizer
  - reports/generator.py  : 1920x1080 Executive Presentation Slide Generator
  - cli/runner.py         : Headless Batch, Benchmark & Dataset Evaluation CLI
  - ui/app.py             : PyQt6 Desktop Overlay, HUD, A/B Battle Modal & System Tray
"""

import importlib
from core.saliency import SaliencyEngine, Quadrant, QUADRANT_LABELS_TR, QUADRANT_LABELS_EN
from core.metrics import SaliencyMetrics
from reports.generator import ReportGenerator
from cli.runner import run_batch_analysis, handle_cli_args, build_arg_parser


def __getattr__(name: str):
    """Lazily expose GUI classes from `ui.app` only when explicitly requested."""
    gui_exports = {
        "SnippingOverlay",
        "LiveRegionOverlay",
        "FloatingControlHUD",
        "ABBattleModal",
        "MiniLauncherPill",
        "HotkeyBridge",
        "OptiSightApp",
        "launch_gui",
    }
    if name in gui_exports:
        _ui_app = importlib.import_module("ui.app")

        return getattr(_ui_app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main() -> None:
    """Entrypoint for both `python main.py` and the `optisight` console command."""
    if handle_cli_args():
        return
    _ui_app = importlib.import_module("ui.app")

    _ui_app.launch_gui()


__all__ = [
    "SaliencyEngine",
    "Quadrant",
    "QUADRANT_LABELS_TR",
    "QUADRANT_LABELS_EN",
    "SaliencyMetrics",
    "ReportGenerator",
    "run_batch_analysis",
    "handle_cli_args",
    "build_arg_parser",
    "main",
]


if __name__ == "__main__":
    main()
