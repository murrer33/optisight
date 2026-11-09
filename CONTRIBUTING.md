# Contributing to OptiSight

Thank you for your interest in contributing to **OptiSight**! This document outlines our modular architecture, development setup, code quality checks, and benchmarking requirements.

---

## 1. Development Environment Setup

Clone the repository and install the package in editable mode with development and optional OCR dependencies:

```bash
git clone https://github.com/murrer33/optisight.git
cd optisight
python -m pip install -e ".[dev,ocr]"
```

Verify that the `optisight` CLI command is registered and operational:

```bash
optisight --benchmark
```

---

## 2. Modular Package Architecture

Keep headless computer-vision and reporting modules strictly decoupled from `PyQt6` GUI widgets:

- **`core/saliency.py`**: 5-Layer Saliency Engine, Itti-Koch pyramid baseline, `validate_image`, physical viewing-distance PSF blur, CVD simulation, WCAG 2.1 contrast, and OCR/MSER text detection. **Must never import `PyQt6`.**
- **`core/metrics.py`**: Psychophysical evaluation metrics (`NSS`, `CC`, `SIM`, `AUC-Judd`), multi-baseline runner, 6-part layer ablation, and simplex grid-search weight optimizer. **Must never import `PyQt6`.**
- **`reports/generator.py`**: `1920x1080` Executive Presentation Slide Generator built purely on `Pillow` (`PIL`) and `OpenCV`. **Must never import `PyQt6`.**
- **`cli/runner.py`**: Headless command-line interface (`--batch`, `--benchmark`, `--eval-dataset`, `--ablation`, `--optimize-weights`).
- **`ui/app.py`**: Desktop `PyQt6` overlays (`SnippingOverlay`, `LiveRegionOverlay`, `FloatingControlHUD`, `ABBattleModal`, `MiniLauncherPill`, and `OptiSightApp`).

---

## 3. Running Tests, Linting & Type Checks

Before opening a Pull Request, ensure all 4 quality gates pass locally:

### 3.1 Unit, Regression & CLI Smoke Tests
```bash
python -m unittest discover -s tests -v
```

### 3.2 Code Formatting (`black`)
```bash
python -m black --check core reports cli ui main.py saliency_engine.py saliency_metrics.py tests
```
*(Run without `--check` to auto-format.)*

### 3.3 Linting (`ruff`)
```bash
python -m ruff check core reports cli ui main.py saliency_engine.py saliency_metrics.py tests
```

### 3.4 Static Type Checking (`mypy`)
```bash
python -m mypy core reports cli main.py saliency_engine.py saliency_metrics.py
```

---

## 4. Evaluating Saliency Changes (`--eval-dataset`)

If your contribution modifies any saliency layer, filter kernel, or fusion weight in `core/saliency.py`, you **must** evaluate its impact on `NSS`, `CC`, `SIM`, and `AUC-Judd` against the benchmark suites:

```bash
# Download real benchmark samples (UEyes, MIT1003, SALICON) if not already present
python benchmarks/download_samples.py

# Run dataset evaluation with baselines and layer ablation
optisight --eval-dataset benchmarks/datasets/ueyes/images benchmarks/datasets/ueyes/density \
  --fixations-dir benchmarks/datasets/ueyes/fixations --ablation
```

Include the before/after `NSS` and `CC` numbers in your Pull Request description.
