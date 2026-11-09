"""
OptiSight Phase 3 Before/After Case Study Generator & Auditor.
Generates 4 realistic Before/After product & marketing designs, evaluates explicit
Logo, Price, and CTA AOIs (fixation rank, attention share %, WCAG 2.1 contrast, legibility),
computes 0-100 Overall Attention Scores and rule-based Fix Lists, and exports 1-page PDFs + JSON.
"""

import json
import os
import sys
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.saliency import SaliencyEngine
from reports.generator import ReportGenerator


def _draw_case_1_coffee_packaging() -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Case 1: Artisanal Cold Brew Coffee Packaging (600x420, shelf distance 1.5m, width 14cm).
    - BEFORE: Cluttered background stripes, low-contrast beige price tag, small bottom-right logo.
    - AFTER: Clean dark matte background, Top-Left Golden Zone Logo, high-contrast gold Price & emerald CTA badge.
    """
    h, w = 420, 600
    aois = [
        {"name": "Marka Logosu (ROAST CO)", "aoi_type": "LOGO", "rect": (40, 35, 190, 65)},
        {"name": "Fiyat Etiketi (89 TL)", "aoi_type": "PRICE", "rect": (390, 40, 165, 65)},
        {"name": "Kampanya / CTA (%30 İNDİRİM)", "aoi_type": "CTA", "rect": (180, 315, 240, 68)},
    ]

    # BEFORE
    before = np.full((h, w, 3), (175, 185, 195), dtype=np.uint8)
    # High-clutter diagonal distraction lines & noisy badges across background
    for i in range(-h, w + h, 18):
        cv2.line(before, (i, 0), (i - h, h), (95, 110, 125), 3)
        cv2.line(before, (i, 0), (i + h, h), (130, 140, 155), 2)
    for y in range(70, 360, 75):
        for x in range(50, 550, 95):
            cv2.rectangle(before, (x, y), (x + 42, y + 32), (60, 190, 230), 2)

    # Low-contrast Logo box
    cv2.rectangle(before, (40, 35), (230, 100), (160, 170, 180), -1)
    cv2.putText(before, "ROAST CO", (52, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (130, 140, 150), 2)

    # Low-contrast Price box (beige on light gray -> ~1.6:1 contrast fail)
    cv2.rectangle(before, (390, 40), (555, 105), (185, 192, 198), -1)
    cv2.putText(before, "89 TL", (425, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (145, 152, 160), 2)

    # Low-contrast CTA box at bottom
    cv2.rectangle(before, (180, 315), (420, 383), (168, 176, 184), -1)
    cv2.putText(before, "SIMDI DENE", (225, 357), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (132, 140, 148), 2)

    # AFTER
    after = np.full((h, w, 3), (28, 32, 42), dtype=np.uint8)
    # Clean central product bottle silhouette
    cv2.rectangle(after, (235, 115), (365, 295), (45, 55, 72), -1)
    cv2.rectangle(after, (235, 115), (365, 295), (90, 110, 140), 2)
    cv2.putText(after, "COLD BREW", (245, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 240, 245), 2)

    # High-contrast Logo in Top-Left Golden Zone
    cv2.rectangle(after, (40, 35), (230, 100), (245, 248, 252), -1)
    cv2.putText(after, "ROAST CO", (50, 78), cv2.FONT_HERSHEY_DUPLEX, 0.9, (15, 20, 30), 2)

    # High-contrast Price badge (AAA contrast)
    cv2.rectangle(after, (390, 40), (555, 105), (20, 205, 255), -1)
    cv2.putText(after, "89 TL", (418, 84), cv2.FONT_HERSHEY_DUPLEX, 1.0, (12, 16, 24), 3)

    # High-contrast CTA button (AAA dark-slate text on vibrant amber/gold pop-out)
    cv2.rectangle(after, (180, 315), (420, 383), (20, 195, 255), -1)
    cv2.putText(after, "HEMEN AL", (212, 360), cv2.FONT_HERSHEY_DUPLEX, 0.95, (10, 14, 22), 2)

    return before, after, aois


def _draw_case_2_sneaker_ecommerce_banner() -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Case 2: E-Commerce Hero Sneaker Ad Creative (640x400, viewing distance 0.6m, width 22cm).
    - BEFORE: Busy grid of 12 competing badges, washed-out pastel CTA button (contrast ~1.8:1), tiny price.
    - AFTER: Minimal dark-slate hero layout, vibrant neon-orange CTA button (contrast >8:1), clear price & logo.
    """
    h, w = 400, 640
    aois = [
        {"name": "Marka Logosu (AEROX)", "aoi_type": "LOGO", "rect": (35, 30, 165, 58)},
        {"name": "Fiyat Etiketi (1.499 TL)", "aoi_type": "PRICE", "rect": (45, 195, 200, 62)},
        {"name": "Satın Al Butonu (SEPETE EKLE)", "aoi_type": "CTA", "rect": (45, 285, 235, 68)},
    ]

    # BEFORE
    before = np.full((h, w, 3), (215, 220, 225), dtype=np.uint8)
    for gx in range(280, 620, 55):
        for gy in range(25, 380, 50):
            cv2.circle(before, (gx, gy), 18, (80, 90, 220), 2)
            cv2.line(before, (gx - 15, gy - 15), (gx + 15, gy + 15), (40, 160, 60), 2)

    # Washed-out Logo
    cv2.rectangle(before, (35, 30), (200, 88), (205, 212, 218), -1)
    cv2.putText(before, "AEROX", (55, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (170, 178, 185), 2)

    # Low-contrast Price
    cv2.rectangle(before, (45, 195), (245, 257), (208, 214, 220), -1)
    cv2.putText(before, "1.499 TL", (65, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (165, 172, 180), 2)

    # Low-contrast CTA button (light cyan on light gray -> < 2.2:1)
    cv2.rectangle(before, (45, 285), (280, 353), (200, 212, 218), -1)
    cv2.putText(before, "SEPETE EKLE", (68, 326), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (158, 170, 178), 2)

    # AFTER
    after = np.full((h, w, 3), (18, 24, 38), dtype=np.uint8)
    # Hero sneaker showcase card on right
    cv2.circle(after, (465, 205), 115, (38, 52, 78), -1)
    cv2.rectangle(after, (365, 165), (565, 255), (240, 245, 250), -1)
    cv2.putText(after, "AEROX PRO", (392, 220), cv2.FONT_HERSHEY_DUPLEX, 0.85, (20, 28, 45), 2)

    # High-contrast Logo in Top-Left Golden Zone
    cv2.rectangle(after, (35, 30), (200, 88), (248, 250, 252), -1)
    cv2.putText(after, "AEROX", (52, 70), cv2.FONT_HERSHEY_DUPLEX, 0.95, (15, 23, 42), 2)

    # High-contrast Price Tag
    cv2.rectangle(after, (45, 195), (245, 257), (56, 189, 248), -1)
    cv2.putText(after, "1.499 TL", (60, 237), cv2.FONT_HERSHEY_DUPLEX, 0.95, (12, 18, 30), 2)

    # High-contrast Primary CTA Button
    cv2.rectangle(after, (45, 285), (280, 353), (16, 185, 129), -1)
    cv2.putText(after, "SEPETE EKLE", (58, 330), cv2.FONT_HERSHEY_DUPLEX, 0.85, (10, 15, 25), 2)

    return before, after, aois


def _draw_case_3_mobile_fintech_landing() -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Case 3: Mobile Fintech App Onboarding & Subscription Screen (480x640, distance 0.5m, width 12cm).
    - BEFORE: Overloaded promotional banners, low-contrast gray-on-gray CTA, buried price.
    - AFTER: Focused value proposition, high-contrast emerald CTA button, clear pricing tier.
    """
    h, w = 640, 480
    aois = [
        {"name": "Uygulama Logosu (PAYNOVA)", "aoi_type": "LOGO", "rect": (40, 35, 190, 58)},
        {"name": "Üyelik Fiyatı (49 TL/Ay)", "aoi_type": "PRICE", "rect": (90, 350, 300, 72)},
        {"name": "Ana CTA Butonu (ÜCRETSİZ BAŞLA)", "aoi_type": "CTA", "rect": (65, 490, 350, 78)},
    ]

    # BEFORE
    before = np.full((h, w, 3), (225, 228, 232), dtype=np.uint8)
    for y in range(105, 340, 42):
        cv2.rectangle(before, (25, y), (455, y + 32), (90, 100, 115), 2)
        cv2.line(before, (35, y + 16), (440, y + 16), (70, 140, 210), 2)

    # Weak Logo
    cv2.rectangle(before, (40, 35), (230, 93), (215, 220, 225), -1)
    cv2.putText(before, "PAYNOVA", (58, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (175, 182, 188), 2)

    # Weak Price
    cv2.rectangle(before, (90, 350), (390, 422), (216, 220, 226), -1)
    cv2.putText(before, "49 TL / AY", (150, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (172, 178, 184), 2)

    # Weak CTA
    cv2.rectangle(before, (65, 490), (415, 568), (210, 216, 222), -1)
    cv2.putText(before, "UCRETSIZ BASLA", (115, 538), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (168, 175, 182), 2)

    # AFTER
    after = np.full((h, w, 3), (15, 23, 42), dtype=np.uint8)
    # Clean central feature card
    cv2.rectangle(after, (45, 130), (435, 310), (30, 41, 59), -1)
    cv2.rectangle(after, (45, 130), (435, 310), (56, 189, 248), 2)
    cv2.putText(after, "SIFIR KOMISYON", (95, 228), cv2.FONT_HERSHEY_DUPLEX, 1.0, (248, 250, 252), 2)

    # Strong Logo
    cv2.rectangle(after, (40, 35), (230, 93), (248, 250, 252), -1)
    cv2.putText(after, "PAYNOVA", (52, 74), cv2.FONT_HERSHEY_DUPLEX, 0.9, (15, 23, 42), 2)

    # Strong Price Box
    cv2.rectangle(after, (90, 350), (390, 422), (245, 158, 11), -1)
    cv2.putText(after, "49 TL / AY", (135, 398), cv2.FONT_HERSHEY_DUPLEX, 1.0, (15, 23, 42), 2)

    # Strong CTA Button
    cv2.rectangle(after, (65, 490), (415, 568), (56, 189, 248), -1)
    cv2.putText(after, "UCRETSIZ BASLA", (92, 540), cv2.FONT_HERSHEY_DUPLEX, 0.95, (12, 18, 32), 2)

    return before, after, aois


def _draw_case_4_vitamin_supplement_box() -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Case 4: Pharmacy Vitamin C Supplement Box (520x440, pharmacy shelf distance 1.8m, width 12cm).
    - BEFORE: Busy scientific text texture across entire front face, low-contrast yellow-on-white dosage/price.
    - AFTER: High-hierarchy pharmacy packaging with bold brand header, clear price sticker, and high-contrast claim badge.
    """
    h, w = 440, 520
    aois = [
        {"name": "Marka Logosu (VITACARE)", "aoi_type": "LOGO", "rect": (40, 35, 200, 62)},
        {"name": "Fiyat Etiketi (195 TL)", "aoi_type": "PRICE", "rect": (320, 35, 160, 62)},
        {"name": "Ana Fayda / CTA (1000 MG GÜÇ)", "aoi_type": "CTA", "rect": (120, 320, 280, 72)},
    ]

    # BEFORE
    before = np.full((h, w, 3), (235, 240, 242), dtype=np.uint8)
    for y in range(110, 310, 16):
        cv2.line(before, (20, y), (500, y), (120, 130, 140), 2)
    for x in range(30, 490, 35):
        cv2.line(before, (x, 110), (x, 305), (140, 150, 160), 1)

    cv2.rectangle(before, (40, 35), (240, 97), (225, 232, 235), -1)
    cv2.putText(before, "VITACARE", (58, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (190, 198, 205), 2)

    cv2.rectangle(before, (320, 35), (480, 97), (222, 235, 240), -1)
    cv2.putText(before, "195 TL", (348, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (185, 200, 208), 2)

    cv2.rectangle(before, (120, 320), (400, 392), (220, 232, 238), -1)
    cv2.putText(before, "1000 MG GUC", (155, 366), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (182, 195, 202), 2)

    # AFTER
    after = np.full((h, w, 3), (20, 28, 44), dtype=np.uint8)
    cv2.circle(after, (260, 205), 72, (35, 120, 245), -1)
    cv2.putText(after, "VIT-C", (212, 216), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

    cv2.rectangle(after, (40, 35), (240, 97), (250, 252, 255), -1)
    cv2.putText(after, "VITACARE", (52, 76), cv2.FONT_HERSHEY_DUPLEX, 0.9, (15, 23, 42), 2)

    cv2.rectangle(after, (320, 35), (480, 97), (45, 225, 160), -1)
    cv2.putText(after, "195 TL", (342, 78), cv2.FONT_HERSHEY_DUPLEX, 0.95, (8, 12, 20), 2)

    cv2.rectangle(after, (120, 320), (400, 392), (20, 195, 250), -1)
    cv2.putText(after, "1000 MG GUC", (148, 368), cv2.FONT_HERSHEY_DUPLEX, 0.95, (12, 18, 30), 2)

    return before, after, aois


def evaluate_variant(
    bgr: np.ndarray,
    aois: List[Dict[str, Any]],
    distance_m: float,
    width_cm: float,
    out_prefix: str,
) -> Dict[str, Any]:
    proc = SaliencyEngine.simulate_shelf_distance(bgr, distance_m, width_cm) if distance_m > 0.5 else bgr
    sal, meta = SaliencyEngine.compute_saliency_map(proc)
    fixations = SaliencyEngine.compute_gaze_path(sal, 4)
    aoi_res = SaliencyEngine.calculate_aoi_metrics(
        sal,
        fixations,
        aois,
        bgr_image=proc,
        distance_meters=distance_m,
        package_width_cm=width_cm,
        lang="tr",
    )
    metrics = SaliencyEngine.calculate_metrics(
        proc,
        sal,
        fixations,
        meta,
        lang="tr",
        aoi_results=aoi_res,
        distance_meters=distance_m,
    )

    heat = SaliencyEngine.generate_heatmap_overlay(proc, sal)
    heat_gaze = SaliencyEngine.draw_gaze_path(heat, fixations)
    fog = SaliencyEngine.generate_fog_overlay(proc, sal)

    png_path = f"{out_prefix}_report.png"
    pdf_path = f"{out_prefix}_report.pdf"
    card = ReportGenerator.generate_single_report(proc, heat_gaze, fog, metrics, aoi_res)
    card.save(png_path, "PNG")
    ReportGenerator.save_pdf_report(card, pdf_path)

    return {
        "overall_attention_score": metrics["overall_attention_score"],
        "focus_score": metrics["focus_score"],
        "clutter_score": metrics["clutter_score"],
        "quadrant": metrics["quadrant"],
        "aoi_results": aoi_res,
        "fix_recommendations": metrics["fix_recommendations"],
        "report_png": os.path.basename(png_path),
        "report_pdf": os.path.basename(pdf_path),
    }


def run_all_case_studies(output_dir: str = "") -> Dict[str, Any]:
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "case_studies")
    os.makedirs(base_dir, exist_ok=True)
    artifacts_dir = output_dir or os.path.join(base_dir, "output")
    os.makedirs(artifacts_dir, exist_ok=True)

    specs = [
        (
            "case1_cold_brew_packaging",
            "1. Soğuk Kahve Ambalajı (Perakende Rafı - 1.5m)",
            1.5,
            14.0,
            _draw_case_1_coffee_packaging,
        ),
        (
            "case2_sneaker_ecommerce_ad",
            "2. E-Ticaret Spor Ayakkabı Bannerı (Masaüstü/Mobil - 0.6m)",
            0.6,
            22.0,
            _draw_case_2_sneaker_ecommerce_banner,
        ),
        (
            "case3_fintech_mobile_ui",
            "3. Mobil Fintech Abonelik Ekranı (Yakın Ekran - 0.5m)",
            0.5,
            12.0,
            _draw_case_3_mobile_fintech_landing,
        ),
        (
            "case4_vitamin_pharmacy_box",
            "4. Eczane Vitamin Kutusu (Uzak Raf - 1.8m)",
            1.8,
            12.0,
            _draw_case_4_vitamin_supplement_box,
        ),
    ]

    summary: Dict[str, Any] = {}
    for slug, title, dist_m, width_cm, builder_fn in specs:
        before_bgr, after_bgr, aois = builder_fn()
        before_prefix = os.path.join(artifacts_dir, f"{slug}_before")
        after_prefix = os.path.join(artifacts_dir, f"{slug}_after")

        res_before = evaluate_variant(before_bgr, aois, dist_m, width_cm, before_prefix)
        res_after = evaluate_variant(after_bgr, aois, dist_m, width_cm, after_prefix)

        ab = SaliencyEngine.compute_ab_comparison(before_bgr, after_bgr, lang="tr")
        ab_pdf = os.path.join(artifacts_dir, f"{slug}_ab_battle.pdf")
        ReportGenerator.generate_ab_pdf_report(before_bgr, after_bgr, ab, ab_pdf)

        summary[slug] = {
            "title": title,
            "distance_meters": dist_m,
            "package_width_cm": width_cm,
            "before": res_before,
            "after": res_after,
            "score_delta": res_after["overall_attention_score"] - res_before["overall_attention_score"],
            "clutter_delta": res_after["clutter_score"] - res_before["clutter_score"],
            "ab_winner": ab["overall_winner"],
            "ab_share_before": ab["variant_a"]["share"],
            "ab_share_after": ab["variant_b"]["share"],
            "ab_battle_pdf": os.path.basename(ab_pdf),
        }
        print(
            f"[OK] {title}: Önce Skor={res_before['overall_attention_score']}/100 -> "
            f"Sonra Skor={res_after['overall_attention_score']}/100 "
            f"(+{res_after['overall_attention_score'] - res_before['overall_attention_score']} puan)"
        )

    out_json = os.path.join(base_dir, "case_studies_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


if __name__ == "__main__":
    run_all_case_studies()
