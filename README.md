# OptiSight: Predictive Visual Attention, Saliency & WCAG Legibility Analyzer

[![CI & Automated Release](https://github.com/murrer33/optisight/actions/workflows/ci.yml/badge.svg)](https://github.com/murrer33/optisight/actions)
[![Latest Release](https://img.shields.io/github/v/release/murrer33/optisight?label=Windows%20.exe%20Release)](https://github.com/murrer33/optisight/releases)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)

> **100% Offline, Real-Time (<40ms) Biomimetic Visual Saliency, Gaze Scanpath Simulation, Physical Viewing-Distance Acuity & W3C WCAG 2.1 Contrast Auditor for Retail Packaging, Ad Creatives & UI Layouts.**
>
> *For Turkish documentation, scroll down to the [Türkçe Dokümantasyon](#-optisight-tahmini-görsel-dikkat-saliency-ve-wcag-okunabilirlik-analizörü-türkçe) section below.*

---

## Overview (English)

**OptiSight** is a desktop overlay and headless CLI engine that predicts early human visual attention (the first **0.5 to 3.0 seconds** of bottom-up + face-guided visual inspection) on **FMCG retail packaging**, **digital ad creatives**, **e-commerce banners**, and **UI layouts** without requiring eye-tracking hardware or cloud APIs.

### Key Capabilities

1. **5-Layer Biomimetic Saliency Fusion (`core/saliency.py`)**
   - **Spectral Residual FFT (`28%`)**: Hou & Zhang (CVPR 2007) log-spectrum statistical rarity detector.
   - **CIE-LAB Chromatic & Luminance Pop-Out (`14%`)**: Perceptually uniform Euclidean distance from mean canvas color.
   - **Multi-Scale Edge & Typography Density (`25%`)**: Sobel gradient magnitude + morphological stroke closure for headlines, prices, and UI borders.
   - **Anisotropic Center & Golden-Zone Prior (`18%`)**: LRU-cached read-only Gaussian bias modeling top-left F-pattern and central fixation tendencies with foveal spatial-prior calibration.
   - **YuNet Deep Face Detection (`15%`)**: Bundled `227 KB` OpenCV `FaceDetectorYN` ONNX model (`assets/models/face_detection_yunet_2023mar.onnx`) with automatic `<=640px` inference scaling and Haar+Eye fallback.
   - **Center-Surround DoG Objectness Gate**: Suppresses tiny `<0.8%` area micro-icons while boosting coherent foreground objects (`>=3.5%` area).

2. **Interactive Desktop Overlay & A/B Battle Modal (`ui/app.py`)**
   - Global hotkeys: **`Alt + H`** (Single Region Scan), **`Alt + B`** (Side-by-Side A/B Packaging/Ad Battle), **`Alt + Q`** (Quit).
   - **Size-Invariant A/B Comparison**: Compares mean normalized saliency density (`np.mean`) and `#1` fixation foveal patch energy so differently cropped snips are never biased by pixel area.
   - **3-Second Animated Scanpath Replay**: Smooth 60 FPS foveal spotlight across fixations `#1 -> #4` computed via Inhibition-of-Return (IoR).

3. **Physically Grounded Shelf Distance & Accessibility Simulation**
   - **Angular Viewing Distance Blur (`0.5m` to `3.0m`)**: Computes visual angle $\theta = 2 \arctan(W_{\text{cm}} / 2 D_{\text{cm}})$, derives pixels-per-degree ($\text{PPD} = W_{\text{px}} / \theta$), and applies a 1-arcminute retinal Point Spread Function ($\sigma = (\text{PPD}/60) \times 0.42 \times \text{acuity\_factor}$).
   - **Color Vision Deficiency (CVD) Filters**: Real-time **Protanopia**, **Deuteranopia**, **Tritanopia** (Viénot 1999 sRGB matrices), and **Monochrome** simulation.

4. **Automatic Text/CTA Detection & WCAG 2.1 Contrast Audit**
   - Detects up to 5 prominent Headlines, Prices, and CTA Buttons (via cached `RapidOCR` ONNX or MSER + Sobel morphological fallback) sorted by visual prominence.
   - Evaluates **W3C WCAG 2.1 Relative Luminance Contrast (`1.0:1` – `21.0:1`)** strictly on text/CTA regions (`AAA`, `AA`, `AA-Large`, `LOW`) and verifies whether single-line cap-height subtends $\ge 0.18^\circ$ at the selected viewing distance.

5. **1920x1080 Executive Report & 1-Page PDF Generator (`reports/`, `cli/`)**
   - Unified **0–100 Overall Attention Score (`overall_attention_score`)**, rule-based **"Fix List" (`fix_recommendations`)** with concrete numeric targets (e.g. `CTA contrast 1.54:1, target >= 4.5:1`), and dedicated **Logo / Price / CTA** AOI tagging (`fixation_rank` & `share %`).
   - One-click **1-Page PDF** (`300 DPI`) and PNG slide export (`1920x1080`) or batch folder processing (`optisight --batch`) exporting `summary.json`, `summary.csv`, `.png`, and `.pdf` executive reports. See [`benchmarks/CASE_STUDIES.md`](benchmarks/CASE_STUDIES.md) for **4 Before/After product case studies**.

6. **Distinctive Retail & Multi-Engine Capabilities (`v2.3.0`)**
   - **Full-Shelf Automatic Product Detection & Planogram Audit (`📦 Raf Ürünleri` / `--shelf-audit`)**: Automatically segments individual SKUs across multi-row retail shelves (`Precision: 71.5%`, `Recall: 83.1%`, `F1: 76.9%`, `Mean IoU: 0.909` @ `IoU >= 0.50`) and compares each product's **Attention Share (`%`)** against its physical **Shelf Area Share (`%`)** (`Efficiency Index = SoA / SoS`, flagging `<0.85x` underperformers and `>=1.15x` Hero SKUs).
   - **Bundled Analytic ONNX Saliency Model (`DoG-ASPP Saliency Prior`, `197.8 KB`) & Hybrid Engine (`--engine {classic,onnx,hybrid}`)**: An untrained, analytically constructed multi-scale ASPP (`dilation = 1, 2, 4, 8`) + CoordConv ONNX computational graph (`assets/models/msinet_saliency_lite.onnx`; zero learned weights, zero dataset leakage). When blended with the 5-Layer classical engine (`65% Classic + 35% ONNX`), it boosts contextual coherence on **UEyes UI/Ad layouts (`Held-Out Test NSS = 0.717 ± 0.381`, `AUC-Judd = 0.702 ± 0.089`)**.
   - **Right-to-Left (`RTL`) Reading Direction (`⬅ RTL Okuma` / `--rtl`)**: Mirrors the horizontal F-pattern Golden Zone prior from Top-Left (`cx=0.46`) to Top-Right (`cx=0.54`) for Arabic, Hebrew, and Persian layouts.

---

## Real Eye-Tracking Benchmark & Out-of-Sample Validation (`N = 360` Total, `60 Train / 60 Test` per Dataset)

OptiSight (`v2.3.0`) is evaluated against **three real human eye-tracking datasets** (**UEyes**, **MIT1003**, and **SALICON** — **$N=120$ pairs per dataset, $360$ total**) using a strict **Out-of-Sample Train/Test split ($N_{\text{train}}=60$ for weight calibration, $N_{\text{test}}=60$ held-out test set for unbiased evaluation)**. All metrics below are reported on the **unseen $N_{\text{test}}=60$ test partitions** as **$\text{Mean} \pm \text{Std}$ (`95% CI`)**. Full details, ONNX model card, and reproduction commands are in [`benchmarks/EVALUATION_REPORT.md`](benchmarks/EVALUATION_REPORT.md) and [`benchmarks/benchmark_results.json`](benchmarks/benchmark_results.json).

> **Note on `--benchmark` (Synthetic Smoke Test) vs. `--eval-dataset` (Real Eye-Tracking):** Running `optisight --benchmark` executes a fast (`<1s`) 4-image synthetic psychophysical smoke test for CI regression checking (`OptiSight_5Layer NSS = 2.218, AUC = 0.918` vs. `CenterBias NSS = 1.427`). Because synthetic stimuli have zero background clutter and idealized single-target geometry, their NSS values (`~2.2`) are naturally higher than complex real-world scenes (`~0.70–1.35`). The table below presents the **primary scientific benchmark on real human eye-tracking data**.

### Held-Out Test Set Benchmark Comparison ($N_{\text{test}}=60$ unseen pairs per dataset, $\mu \pm \sigma$)

| Dataset (Visual Domain) | Model / Baseline | NSS $\uparrow$ ($\mu \pm \sigma$, 95% CI) | CC $\uparrow$ ($\mu \pm \sigma$) | SIM $\uparrow$ ($\mu \pm \sigma$) | AUC-Judd $\uparrow$ ($\mu \pm \sigma$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **UEyes** *(UI, Posters, Ads, Web)* | **`OptiSight_Hybrid` (65% Classic + 35% ONNX)** | **$0.717 \pm 0.381$** ($\pm 0.096$) | **$0.322 \pm 0.167$** | **$0.378 \pm 0.082$** | **$0.702 \pm 0.089$** |
| **UEyes** *(UI, Posters, Ads, Web)* | **`OptiSight_Optimized` (Train-Tuned `30/5/20/25/20`)** | **$0.705 \pm 0.406$** ($\pm 0.103$) | $0.308 \pm 0.164$ | $0.373 \pm 0.078$ | $0.694 \pm 0.093$ |
| **UEyes** *(UI, Posters, Ads, Web)* | **`OptiSight_5Layer` (Default `28/14/25/18/15`)** | **$0.704 \pm 0.394$** ($\pm 0.100$) | **$0.310 \pm 0.156$** | **$0.373 \pm 0.077$** | **$0.695 \pm 0.089$** |
| **UEyes** *(UI, Posters, Ads, Web)* | `CenterBias_Baseline` (Gaussian Prior) | $0.546 \pm 0.434$ ($\pm 0.110$) | $0.236 \pm 0.187$ | $0.354 \pm 0.080$ | $0.657 \pm 0.119$ |
| **UEyes** *(UI, Posters, Ads, Web)* | `SpectralResidual_Baseline` (Hou & Zhang 2007) | $0.531 \pm 0.354$ ($\pm 0.090$) | $0.225 \pm 0.120$ | $0.343 \pm 0.061$ | $0.685 \pm 0.079$ |
| **UEyes** *(UI, Posters, Ads, Web)* | `IttiKoch_Baseline` (TPAMI 1998 Pyramid) | $0.414 \pm 0.370$ ($\pm 0.094$) | $0.181 \pm 0.161$ | $0.348 \pm 0.080$ | $0.637 \pm 0.107$ |
| **MIT1003** *(Natural & Urban Photos)* | **`OptiSight_Optimized` (Train-Tuned `10/5/10/20/55`)** | **$1.347 \pm 0.924$** ($\pm 0.234$) | **$0.397 \pm 0.116$** | **$0.293 \pm 0.050$** | **$0.823 \pm 0.133$** |
| **MIT1003** *(Natural & Urban Photos)* | **`OptiSight_5Layer` (Default `28/14/25/18/15`)** | **$1.285 \pm 0.769$** ($\pm 0.195$) | **$0.375 \pm 0.113$** | **$0.288 \pm 0.050$** | **$0.822 \pm 0.130$** |
| **MIT1003** *(Natural & Urban Photos)* | `CenterBias_Baseline` (Photographic Framing Prior) | $1.151 \pm 0.491$ ($\pm 0.124$) | $0.353 \pm 0.114$ | $0.275 \pm 0.056$ | $0.820 \pm 0.127$ |
| **MIT1003** *(Natural & Urban Photos)* | `IttiKoch_Baseline` (TPAMI 1998 Pyramid) | $0.787 \pm 0.875$ ($\pm 0.222$) | $0.215 \pm 0.168$ | $0.262 \pm 0.065$ | $0.700 \pm 0.200$ |
| **MIT1003** *(Natural & Urban Photos)* | `SpectralResidual_Baseline` (Hou & Zhang 2007) | $0.738 \pm 1.098$ ($\pm 0.278$) | $0.164 \pm 0.150$ | $0.255 \pm 0.072$ | $0.706 \pm 0.181$ |
| **SALICON** *(MS-COCO General Scenes)* | **`OptiSight_Optimized` (Train-Tuned `25/5/5/15/50`)** | **$1.335 \pm 0.420$** ($\pm 0.106$) | **$0.590 \pm 0.104$** | **$0.485 \pm 0.130$** | **$0.853 \pm 0.078$** |
| **SALICON** *(MS-COCO General Scenes)* | **`OptiSight_5Layer` (Default `28/14/25/18/15`)** | **$1.249 \pm 0.405$** ($\pm 0.102$) | **$0.556 \pm 0.119$** | **$0.478 \pm 0.132$** | **$0.845 \pm 0.082$** |
| **SALICON** *(MS-COCO General Scenes)* | `CenterBias_Baseline` (Gaussian Prior) | $1.014 \pm 0.345$ ($\pm 0.087$) | $0.493 \pm 0.146$ | $0.458 \pm 0.140$ | $0.809 \pm 0.093$ |
| **SALICON** *(MS-COCO General Scenes)* | `IttiKoch_Baseline` (TPAMI 1998 Pyramid) | $0.741 \pm 0.451$ ($\pm 0.114$) | $0.317 \pm 0.163$ | $0.426 \pm 0.117$ | $0.717 \pm 0.123$ |
| **SALICON** *(MS-COCO General Scenes)* | `SpectralResidual_Baseline` (Hou & Zhang 2007) | $0.641 \pm 0.443$ ($\pm 0.112$) | $0.239 \pm 0.145$ | $0.385 \pm 0.089$ | $0.714 \pm 0.110$ |

### Layer Ablation Summary on Held-Out Test Set ($N_{\text{test}}=60$, Impact on NSS $\uparrow$)

| Disabled Component | UEyes Test NSS ($\Delta$) | MIT1003 Test NSS ($\Delta$) | SALICON Test NSS ($\Delta$) | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Full `OptiSight_5Layer`** | **$0.704 \pm 0.394$** | **$1.285 \pm 0.769$** | **$1.249 \pm 0.405$** | Calibrated `28/14/25/18/15` default reference |
| `- Spectral Residual` | $0.668$ ($-0.036$) | $1.258$ ($-0.027$) | $1.226$ ($-0.023$) | Essential for statistical rarity & foreground pop-out |
| `- CIE-LAB Distance` | $0.731$ ($+0.027$) | $1.295$ ($+0.010$) | $1.266$ ($+0.017$) | Moderate LAB weight (`0.14`) balances single-color packaging pop-out against busy scene backgrounds |
| `- Edge/Typography` | **$0.621$ ($-0.083$)** | $1.260$ ($-0.025$) | $1.195$ ($-0.054$) | **Most critical feature layer on UI, packaging & posters** |
| `- Center/Golden Bias` | **$0.515$ ($-0.189$)** | **$0.801$ ($-0.484$)** | **$0.876$ ($-0.373$)** | Strongest spatial prior across all human eye-tracking datasets |
| `- YuNet Face Layer` | $0.702$ ($-0.002$) | **$1.247$ ($-0.038$)** | **$1.208$ ($-0.041$)** | Boosts portrait and lifestyle scenes; neutral on text-only UI |
| `- DoG Objectness Gate` | $0.727$ ($+0.023$) | $1.361$ ($+0.076$) | $1.240$ ($-0.009$) | Suppresses micro-icon noise on dense retail cards and general scenes |

### Detector Precision / Recall & `0–100` Score Validation
- **Shelf Planogram Product Detector (`detect_shelf_products`)**: Evaluated across 15 multi-row retail planograms (`124` ground-truth SKUs @ $\text{IoU} \ge 0.50$): **`Precision = 71.5%`**, **`Recall = 83.1%`**, **`F1 = 76.9%`**, **`Mean Matched IoU = 0.909`**.
- **Built-In Morphological Text & CTA Detector (`detect_text_and_cta_regions`)**: Evaluated across 15 retail/ad banners (`45` ground-truth Headline/Price/CTA boxes @ $\text{IoU} \ge 0.45$): **`Precision = 100.0%`**, **`Recall = 82.2%`**, **`F1 = 90.2%`**, **`Mean Matched IoU = 0.826`**.
- **`0–100` Overall Attention Score Correlation**: Across the held-out test sets ($N_{\text{test}}=60$ per dataset), the single-image `overall_attention_score` correlates positively with human gaze concentration (`r = +0.344` on UEyes) and with achieved prediction accuracy (`r = +0.483` with `NSS` on UEyes, `r = +0.571` on MIT1003, `r = +0.453` on SALICON).

---

## Visual Domain Notes & Scientific Limitations

### Performance by Visual Category
- **Retail Packaging (FMCG)**: Strong on isolated brand logos, promotional badges (`"%50 OFF"`), and physical shelf distance legibility (`0.5m`–`3.0m`). On all-over illustrative packaging (e.g., busy cereal boxes), use the `UEYES_UI` weight preset (`0.15, 0.05, 0.40, 0.35, 0.05`) so background color blocks do not dilute typography peaks.
- **UI, Web & Mobile Screens**: Strong on top-left F-pattern hierarchy, CTA button contrast, and headline capture. Dense navigation bars with medium-sized bordered cards can still compete with minimalist text-only hero titles.
- **Natural Photography**: Strong on human portraits (via YuNet ONNX) and distinct foreground objects.

### Known Scientific Limitations
1. **Bottom-Up + Face Heuristic vs. Task-Driven Semantic Gaze**: OptiSight models early, bottom-up visual salience (`0.5s`–`3.0s` free viewing) plus face detection. It does **not** use a heavy deep semantic transformer (such as DeepGaze IIE or TranSalNet) and cannot model top-down user intent (e.g., *"find the nutritional calories table"*).
2. **No 3D Gaze-Following Vector**: While YuNet accurately detects human faces, OptiSight places attention on the face itself rather than projecting where the person's eyes are looking (gaze-following toward an adjacent product).
3. **Otsu Bimodal WCAG Approximation**: `compute_wcag_contrast` separates foreground text strokes from backgrounds using Otsu thresholding and `85th/15th` luminance percentiles. On multi-color gradient buttons or semi-transparent glassmorphism overlays, contrast ratios are approximate ($\pm 0.4$).
4. **Heuristic TTFF & Gaze Order**: Fixations `#1 -> #4` are generated deterministically via local maxima + Inhibition-of-Return (IoR), representing spatial priority ranks rather than stochastic saccade durations across a population.

---

## Installation & Quick Start

### Option A: Standalone Windows Executable (`.exe`)
Download **`OptiSight.exe`** directly from [**GitHub Releases**](https://github.com/murrer33/optisight/releases) — no Python installation required.

### Option B: Python Package (`pip install`)

```bash
git clone https://github.com/murrer33/optisight.git
cd optisight
pip install -e ".[dev,ocr]"
```

### Usage (GUI & Headless CLI)

```bash
# 1. Launch Interactive Desktop Overlay GUI (System Tray + Alt+H / Alt+B Hotkeys)
optisight
# or:
python main.py

# 2. Headless Batch Analysis on a folder of designs (at 1.5m shelf distance, 18cm width)
optisight --batch ./my_designs --output ./audit_reports --distance 1.5 --width-cm 18.0 --lang en

# 3. Run Synthetic Psychophysical Benchmark (NSS, CC, SIM, AUC-Judd)
optisight --benchmark

# 4. Evaluate on UEyes / MIT1003 / SALICON with Baselines, Ablation & Weight Optimization
optisight --eval-dataset benchmarks/datasets/ueyes/images benchmarks/datasets/ueyes/density \
  --fixations-dir benchmarks/datasets/ueyes/fixations --ablation --optimize-weights
```

### Building `OptiSight.exe` Locally with PyInstaller

```bash
pyinstaller --noconfirm --clean --onefile --windowed --name "OptiSight" \
  --add-data "assets/models/face_detection_yunet_2023mar.onnx;assets/models" \
  main.py
```

---

## Project Structure

```text
optisight/
├── core/
│   ├── saliency.py          # 5-Layer Saliency Engine, Itti-Koch, validate_image, WCAG & OCR
│   └── metrics.py           # NSS, CC, SIM, AUC-Judd, Baselines, Ablation & Grid-Search
├── reports/
│   └── generator.py         # 1920x1080 Executive Presentation Slide Generator (PIL)
├── cli/
│   └── runner.py            # Headless CLI (--batch, --benchmark, --eval-dataset, --ablation)
├── ui/
│   └── app.py               # PyQt6 Desktop Overlay, Floating HUD, A/B Modal & System Tray
├── assets/models/           # Bundled YuNet ONNX face detector (227 KB)
├── benchmarks/              # Real UEyes/MIT1003/SALICON samples, JSON results & Evaluation Report
├── tests/                   # Deterministic regression, unit & CLI smoke tests
├── main.py                  # Unified CLI/GUI entrypoint
├── pyproject.toml           # Packaging, console_scripts, ruff, black & mypy config
├── CHANGELOG.md             # Version history
└── CONTRIBUTING.md          # Developer setup, linting, typing & PR guidelines
```

---
---

# 👁️ OptiSight: Tahmini Görsel Dikkat, Saliency ve WCAG Okunabilirlik Analizörü (Türkçe)

**OptiSight**, perakende ürün ambalajları (FMCG), dijital reklam kreatifleri, e-ticaret bannerları ve kullanıcı arayüzü (UI) tasarımlarının insan gözü tarafından **ilk 0.5 ile 3.0 saniye** içinde nasıl tarandığını tamamen **çevrimdışı (offline)** ve gerçek zamanlı (**<40ms**) simüle eden açık kaynaklı bir masaüstü ve CLI analiz aracıdır.

### Temel Özellikler

- **5 Katmanlı Biyomimetik Dikkat Motoru (`core/saliency.py`)**:
  1. **Spectral Residual FFT (`%30`)**: Frekans uzayında istatistiksel olarak ayrışan görsel yenilik bölgelerini yakalar.
  2. **CIE-LAB Renk ve Kontrast Pop-Out (`%25`)**: İnsan renk algısına uyumlu LAB uzayında sıcak/zıt renk sıçramalarını tespit eder.
  3. **Çok Ölçekli Kenar ve Tipografi Yoğunluğu (`%25`)**: Başlıklar, fiyat etiketleri ve CTA buton çerçevelerini yakalar.
  4. **Anizotropik Merkez ve Altın Bölge Önceliği (`%10`)**: F-tipi okuma ve ekran merkezi sabitlenme eğilimini modeller.
  5. **YuNet Derin Öğrenme Yüz Tespiti (`%10`)**: Cilt rengi filtresine ihtiyaç duymayan gömülü `227 KB` ONNX yüz modeli.
- **Fiziksel Açısal Raf Mesafesi Simülasyonu (`0.5m – 3.0m`)**: Ambalajın gerçek genişliği (`cm`) ve bakış mesafesine (`m`) göre piksel/derece (`PPD`) ve 1 ark-dakika (`1 arcmin`) insan gözü çözünürlük sınırını hesaplayarak gerçekçi raf bulanıklığı uygular.
- **Otomatik Metin/CTA Tespiti ve W3C WCAG 2.1 Kontrast Denetimi**: Yalnızca metin, fiyat ve buton (CTA) kutularında kontrast oranını (`1.0:1` – `21.0:1`) ve seçilen mesafedeki karakter yüksekliği okunabilirliğini ($\ge 0.18^\circ$) denetler.
- **Boyut-Bağımsız A/B Rekabet Mücadelesi (`Alt + B`)**: İki farklı tasarım farklı boyutlarda kırpılsa bile piksel toplamı yerine normalize yoğunluk ortalaması (`np.mean`) ve `#1` numaralı ilk odak enerjisi ile adil karşılaştırma yapar.

### Hızlı Kurulum ve Kullanım

1. **Hazır Windows `.exe` İndir**: Python kurmadan doğrudan [**GitHub Releases**](https://github.com/murrer33/optisight/releases) sayfasından `OptiSight.exe` dosyasını indirip çalıştırabilirsiniz.
2. **Geliştirici / CLI Kurulumu**:
   ```bash
   pip install -e ".[dev,ocr]"
   optisight --benchmark
   optisight --batch ./ornek_tasarimlar --output ./raporlar --distance 1.5 --width-cm 15.0 --lang tr
   ```

### Bilimsel Sınırlılıklar (Limitations)
- OptiSight, **ilk 0.5 – 3.0 saniyelik erken görsel dikkat (bottom-up saliency + yüz önceliği)** aşamasını simüle eder; kullanıcının spesifik bir görevi aradığı (ör. *"içindekiler tablosundaki kaloriyi bul"*) üst-ten-aşağıya (top-down) bilişsel süreçleri veya insan yüzünün baktığı 3B bakış yönü hedefini (gaze-following) modellemez.
- Gerçek insan göz izleme cihazlarının (120Hz+ kızılötesi kornea yansıması) yerini tamamen almaz; tasarım aşamasında hızlı ön-eleme, kontrast/okunabilirlik kontrolü ve A/B varyant karşılaştırması için tasarlanmıştır.
