# Changelog

All notable changes to **OptiSight** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.3.0] - 2026-11-09

### Added
- **Automatic Full-Shelf Product Detection & Planogram Efficiency Audit (`core/saliency.py`, `ui/app.py`, `cli/runner.py`)**:
  - Added `SaliencyEngine.detect_shelf_products` and `SaliencyEngine.analyze_shelf_planogram` to automatically segment individual product facings (SKUs) across multi-row retail shelves and compare each product's **Attention Share (`%`)** against its physical **Shelf Area Share (`%`)** (`efficiency_index = SoA / SoS`).
  - Added `📦 Raf Ürünleri` button in the GUI HUD and `--shelf-audit` flag in the CLI.
- **Bundled Lightweight ONNX Saliency Model (`MSI-Net Lite`, `197.8 KB`) & Hybrid Engine Mode**:
  - Bundled `assets/models/msinet_saliency_lite.onnx` (Multi-Scale Information Network with ASPP dilated convolutions rates `1, 2, 4, 8` and CoordConv spatial priors).
  - Added `engine_mode` (`"CLASSIC"`, `"ONNX"`, `"HYBRID"`) to `SaliencyEngine.compute_saliency_map`, GUI HUD (`Klasik 5-K` / `ONNX MSI` / `Hibrit`), and CLI (`--engine {classic,onnx,hybrid}`).
- **Empirical Comparison of Classical vs. ONNX vs. Hybrid Engines (`benchmarks/EVALUATION_REPORT.md`)**:
  - Evaluated all three engine modes across **UEyes**, **MIT1003**, and **SALICON** (`HYBRID` improves UEyes NSS from `0.595` to `1.021`, MIT1003 NSS from `0.924` to `0.993`, and SALICON NSS from `0.931` to `0.952`).
- **Right-to-Left (`RTL`) Reading Direction Support**:
  - Added `reading_direction="ltr" | "rtl"` across `_cached_center_bias`, `compute_saliency_map`, `calculate_metrics`, GUI (`⬅ RTL Okuma`), and CLI (`--rtl`), mirroring the Golden Zone prior to Top-Right (`cx=0.54`).
- **Enhanced Rule-Based Design Recommendations (`generate_fix_recommendations`)**:
  - Added actionable rules for Shelf Planogram Efficiency deficits (`SoA/SoS < 0.80`), RTL/LTR reading entry mismatches, and concrete numeric design remedies.

---

## [2.2.0] - 2026-11-08

### Added
- **0–100 Overall Attention Score (`overall_attention_score`)**:
  - Defined `SaliencyEngine.compute_overall_attention_score` combining Focal Concentration (`45%`), Visual Clarity (`100 - clutter_score`, `30%`), Spatial Hierarchy & Dead-Zone Balance (`25%`), and target AOI contrast/capture adjustments.
- **1-Page Executive PDF Report Generator (`reports/generator.py`)**:
  - Added `ReportGenerator.save_pdf_report`, `generate_single_pdf_report`, and `generate_ab_pdf_report` (300 DPI single-page PDF export in both GUI and CLI `--batch`).
- **Rule-Based Actionable 'Fix' List (`fix_recommendations`)**:
  - Added `SaliencyEngine.generate_fix_recommendations` citing concrete numeric thresholds (e.g., `[Kontrast] 'Satın Al Butonu' kontrastı 1.54:1, hedef ≥4.5:1`).
- **Dedicated Semantic AOI Tagging (`LOGO`, `PRICE`, `CTA`)**:
  - Added one-click `+Logo Seç`, `+Fiyat Seç`, and `+CTA Seç` buttons in the HUD (`ui/app.py`) and semantic role classification (`_infer_aoi_type`) with fixation rank (`#1..#4`) and attention share (`%`).
- **4 Quantitative Before/After Product Case Studies (`benchmarks/CASE_STUDIES.md`)**:
  - Added `benchmarks/generate_case_studies.py` and `benchmarks/CASE_STUDIES.md` covering Cold Brew Packaging (`6 -> 80/100`), Sneaker E-Commerce Ad (`26 -> 74/100`), Mobile Fintech UI (`44 -> 76/100`), and Pharmacy Vitamin Box (`27 -> 69/100`).

---

## [2.1.0] - 2026-11-08

### Added
- **Real Eye-Tracking Benchmark Evaluation (`benchmarks/`)**:
  - Added `benchmarks/download_samples.py` and evaluated `OptiSight` across **UEyes** (UI, Posters, Ads), **MIT1003** (Natural & Urban Photography), and **SALICON** (MS-COCO General Scenes).
  - Saved full dataset evaluation, ablation, and grid-search results to [`benchmarks/benchmark_results.json`](benchmarks/benchmark_results.json) and [`benchmarks/EVALUATION_REPORT.md`](benchmarks/EVALUATION_REPORT.md).
- **Classical Saliency Baselines (`core/saliency.py`, `core/metrics.py`)**:
  - Implemented **Itti, Koch & Niebur (TPAMI 1998)** multi-scale center-surround pyramid baseline (`SaliencyEngine.compute_itti_koch`), **Anisotropic Center-Bias** baseline, and raw **Hou & Zhang (CVPR 2007) Spectral Residual** baseline.
- **6-Part Layer Ablation & Simplex Grid-Search Optimizer**:
  - Added `--ablation` and `--optimize-weights` CLI flags (`core/metrics.py`, `cli/runner.py`) to disable each layer (`No_Spectral`, `No_LAB`, `No_Edge`, `No_CenterBias`, `No_Face`, `No_Objectness`) and calibrate domain weight presets (`SaliencyEngine.PRESET_WEIGHTS`).
- **Modular Package Architecture**:
  - Split monolithic `main.py` and `saliency_engine.py` into `core/` (`saliency.py`, `metrics.py`), `reports/` (`generator.py`), `cli/` (`runner.py`), and `ui/` (`app.py`) while keeping top-level backward-compatible facades.
- **Robust Error Handling & Input Sanitization (`SaliencyEngine.validate_image`)**:
  - Rejects `None`, corrupted, or `<8x8` images with descriptive `ValueError`, converts 2D grayscale/4-channel BGRA automatically, caps oversized images (`>2560px`) proportionally, and falls back gracefully to MSER when `rapidocr_onnxruntime` is absent.
- **Code Quality, Typing & Developer Documentation**:
  - Added `ruff`, `black`, and `mypy` checks to `pyproject.toml` and GitHub Actions CI (`.github/workflows/ci.yml`).
  - Added `CHANGELOG.md` and `CONTRIBUTING.md`.

### Changed
- Rewrote `README.md` in clean UTF-8 with English documentation first, followed by Turkish, real UEyes/MIT1003/SALICON benchmark tables, category success/failure notes, and a scientific **Limitations** section.

---

## [2.0.1] - 2026-11-08

### Fixed
- Fixed NumPy 2.x compatibility in `compute_auc_judd` by using `np.trapezoid` (with fallback to `np.trapz`).
- Added headless Linux X11/Wayland guard around `pynput` imports and installed `libegl1` in Ubuntu CI runners.
- Published automated Windows standalone executable (`OptiSight.exe`) via GitHub Actions Releases.

---

## [2.0.0] - 2026-11-08

### Added
- Replaced heuristic skin-color face gate with bundled **OpenCV YuNet ONNX (`FaceDetectorYN`, 227 KB)** with `<=640px` inference downscaling.
- Added physically grounded viewing distance simulation (`simulate_shelf_distance`) using visual angle, pixels-per-degree (`PPD`), and 1-arcminute retinal Point Spread Function (`PSF`).
- Added W3C **WCAG 2.1 Relative Luminance Contrast** auditing (`compute_wcag_contrast`) restricted strictly to Text/CTA AOIs and single-line effective cap-height estimation (`_estimate_char_height_px`).
- Added Viénot (1999) Color Vision Deficiency (Protanopia, Deuteranopia, Tritanopia, Monochrome) filters.
- Made A/B Battle attention share size-invariant (`np.mean`) and `#1` fixation winner based on local foveal patch energy.

---

## [1.0.0] - 2026-11-07

### Added
- Initial release of OptiSight Desktop Overlay (`Alt+H` single scan, `Alt+B` A/B comparison, `1920x1080` executive report generator).
