# OptiSight Before/After Product & Marketing Case Studies (`v2.2.0`)

This document presents **4 quantitative Before/After case studies** evaluating how rule-based fixes (`fix_recommendations`), **Logo / Price / CTA** AOI hierarchy, and W3C WCAG 2.1 contrast improvements impact the **0–100 Overall Attention Score (`overall_attention_score`)**, **Fixation Order (`#1..#4`)**, and **Share of Attention (`%`)**.

All 1-page PDF executive reports, PNG slides, and raw JSON metrics are generated reproducibly into `benchmarks/case_studies/output/` (git-ignored to keep the repository lightweight) or can be downloaded directly from [**GitHub Releases (`OptiSight-Case-Studies-Reports.zip`, 4.5 MB)**](https://github.com/murrer33/optisight/releases/download/v2.3.0/OptiSight-Case-Studies-Reports.zip):

```bash
python benchmarks/generate_case_studies.py
```

---

## Executive Summary Table (Before vs. After)

| Case Study | Viewing Distance & Width | Before Score (`0-100`) | After Score (`0-100`) | Score Gain ($\Delta$) | Clutter Reduction | Primary CTA Contrast (Before $\rightarrow$ After) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Cold Brew Coffee Packaging** *(FMCG Retail)* | `1.5m` • `14 cm` | `6 / 100` | **`80 / 100`** | **`+74 pts`** | `96` $\rightarrow$ `58` (`-38`) | `1.37:1` (FAIL) $\rightarrow$ **`10.42:1` (AAA)** |
| **2. Sneaker E-Commerce Hero Ad** *(Digital Banner)* | `0.6m` • `22 cm` | `26 / 100` | **`74 / 100`** | **`+48 pts`** | `77` $\rightarrow$ `54` (`-23`) | `1.54:1` (FAIL) $\rightarrow$ **`7.88:1` (AAA)** |
| **3. Mobile Fintech Subscription UI** *(App Screen)* | `0.5m` • `12 cm` | `44 / 100` | **`76 / 100`** | **`+32 pts`** | `49` $\rightarrow$ `53` (`+4`) | `1.53:1` (FAIL) $\rightarrow$ **`10.69:1` (AAA)** |
| **4. Pharmacy Vitamin C Box** *(Aisle Shelf)* | `1.8m` • `12 cm` | `27 / 100` | **`69 / 100`** | **`+42 pts`** | `84` $\rightarrow$ `63` (`-21`) | `1.22:1` (FAIL) $\rightarrow$ **`5.32:1` (AA)** |

---

## Case Study 1: Artisanal Cold Brew Coffee Packaging (`1.5m` Shelf Distance)

### Context & Audit Setup
- **Physical Parameters**: `14 cm` package width viewed from `1.5m` supermarket aisle distance.
- **Target AOIs Tagged**:
  - `LOGO`: `Marka Logosu (ROAST CO)` (`Top-Left`)
  - `PRICE`: `Fiyat Etiketi (89 TL)` (`Top-Right`)
  - `CTA`: `Kampanya / CTA (%30 İNDİRİM)` (`Bottom-Center`)

### Before Design Diagnosis (`Overall Score: 6 / 100`)
- **Rule-Based Fix List Triggered**:
  - `[Marka Algısı] 'Marka Logosu (ROAST CO)' dikkat payı düşük (%1.2, hedef ≥%5); Sol-Üst Altın Bölge'ye yaklaştırın veya kontrastını güçlendirin.`
  - `[Kontrast] 'Fiyat Etiketi (89 TL)' kontrastı 1.37:1, hedef ≥4.5:1 (Kritik <3.0:1) (zemin/yazı ton farkını artırın).`
  - `[Odak Yakalama] 'Kampanya / CTA (%30 İNDİRİM)' ilk 4 sabitlenmeye giremedi (%2.3 pay, hedef ≥%8); sıcak renk kontrastı veya boşluk (whitespace) ekleyin.`
  - `[Görsel Karmaşa] Karmaşa endeksi 96/100, hedef ≤45/100 (arka plan dokularını ve ikincil çerçeveleri sadeleştirin).`

### After Design Results (`Overall Score: 80 / 100`, `+74 pts`)

| Target AOI | Before Share (%) | After Share (%) | Before Fixation Rank | After Fixation Rank | Before WCAG | After WCAG |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logo (`ROAST CO`)** | `1.2%` | **`19.5%`** | Unfixated | **`#1. Fixation`** | — | — |
| **Price (`89 TL`)** | `3.4%` | **`13.2%`** | Unfixated | **`#4. Fixation`** | `1.37:1` (LOW) | **`10.84:1` (AAA)** |
| **CTA (`HEMEN AL`)** | `2.3%` | **`17.1%`** | Unfixated | **`#3. Fixation`** | `1.37:1` (LOW) | **`10.42:1` (AAA)** |

---

## Case Study 2: Sneaker E-Commerce Hero Banner (`0.6m` Desktop/Mobile)

### Context & Audit Setup
- **Physical Parameters**: `22 cm` banner width viewed at `0.6m` desktop/laptop distance.
- **Target AOIs Tagged**:
  - `LOGO`: `Marka Logosu (AEROX)`
  - `PRICE`: `Fiyat Etiketi (1.499 TL)`
  - `CTA`: `Satın Al Butonu (SEPETE EKLE)`

### Before vs. After AOI Metrics (`Overall Score: 26/100 -> 74/100`, `+48 pts`)

| Target AOI | Before Share (%) | After Share (%) | Before Fixation Rank | After Fixation Rank | Before WCAG | After WCAG |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logo (`AEROX`)** | `2.0%` | **`12.6%`** | Unfixated | **`#2. Fixation`** | — | — |
| **Price (`1.499 TL`)** | `3.4%` | **`13.2%`** | Unfixated | **`#4. Fixation`** | `1.54:1` (LOW) | **`10.61:1` (AAA)** |
| **CTA (`SEPETE EKLE`)** | `3.7%` | **`11.9%`** | Unfixated | High Foveal Share | `1.54:1` (LOW) | **`7.88:1` (AAA)** |

---

## Case Study 3: Mobile Fintech Subscription Screen (`0.5m` Handheld UI)

### Context & Audit Setup
- **Physical Parameters**: `12 cm` mobile viewport width at `0.5m` handheld viewing distance.
- **Target AOIs Tagged**:
  - `LOGO`: `Uygulama Logosu (PAYNOVA)`
  - `PRICE`: `Üyelik Fiyatı (49 TL/Ay)`
  - `CTA`: `Ana CTA Butonu (ÜCRETSİZ BAŞLA)`

### Before vs. After AOI Metrics (`Overall Score: 44/100 -> 76/100`, `+32 pts`)

| Target AOI | Before Share (%) | After Share (%) | Before Fixation Rank | After Fixation Rank | Before WCAG | After WCAG |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logo (`PAYNOVA`)** | `1.8%` | **`14.3%`** | Unfixated | **`#1. Fixation`** | — | — |
| **Price (`49 TL/Ay`)** | `4.4%` | **`13.8%`** | Unfixated | High Foveal Share | `1.55:1` (LOW) | **`5.88:1` (AA)** |
| **CTA (`ÜCRETSİZ BAŞLA`)** | `3.2%` | **`24.9%`** | Unfixated | **`#3. Fixation`** | `1.53:1` (LOW) | **`10.69:1` (AAA)** |

---

## Case Study 4: Pharmacy Vitamin C Supplement Box (`1.8m` Distant Shelf)

### Context & Audit Setup
- **Physical Parameters**: `12 cm` supplement box viewed from `1.8m` behind the pharmacy counter.
- **Target AOIs Tagged**:
  - `LOGO`: `Marka Logosu (VITACARE)`
  - `PRICE`: `Fiyat Etiketi (195 TL)`
  - `CTA`: `Ana Fayda / CTA (1000 MG GÜÇ)`

### Before vs. After AOI Metrics (`Overall Score: 27/100 -> 69/100`, `+42 pts`)

| Target AOI | Before Share (%) | After Share (%) | Before Fixation Rank | After Fixation Rank | Before WCAG | After WCAG |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logo (`VITACARE`)** | `2.1%` | **`19.0%`** | Unfixated | **`#1. Fixation`** | — | — |
| **Price (`195 TL`)** | `1.7%` | **`8.5%`** | Unfixated | High Foveal Share | `1.21:1` (LOW) | **`6.19:1` (AA)** |
| **CTA (`1000 MG GÜÇ`)** | `3.0%` | **`22.7%`** | Unfixated | **`#2. Fixation`** | `1.22:1` (LOW) | **`5.32:1` (AA)** |
