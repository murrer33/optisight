"""
OptiSight CLI & Batch Processing Runner (Headless Agency / E-Commerce Workflow).
Operates with zero GUI / X11 dependencies.
"""

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional
import cv2

from core.metrics import SaliencyMetrics
from core.saliency import SaliencyEngine
from reports.generator import ReportGenerator

MAX_FILE_SIZE_BYTES = 80 * 1024 * 1024  # 80 MB safety cap


def run_batch_analysis(
    input_path: str,
    output_dir: str,
    distance_meters: float = 1.0,
    package_width_cm: float = 15.0,
    lang: str = "tr",
    engine_mode: str = "CLASSIC",
    reading_direction: str = "ltr",
    shelf_audit: bool = False,
) -> List[Dict[str, Any]]:
    """
    Processes a single image or an entire directory of images headlessly.
    Gracefully skips corrupted, unreadable, or oversized files with warnings.
    Generates 1920x1080 PNG executive report cards + 300 DPI PDF + summary.json + summary.csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    valid_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

    if os.path.isdir(input_path):
        files = [
            os.path.join(input_path, f)
            for f in sorted(os.listdir(input_path))
            if f.lower().endswith(valid_exts)
        ]
    elif os.path.isfile(input_path):
        files = [input_path]
    else:
        raise FileNotFoundError(f"Girdi yolu bulunamadi: {input_path}")

    summary_rows: List[Dict[str, Any]] = []
    for idx, fpath in enumerate(files, 1):
        fname = os.path.basename(fpath)
        try:
            fsize = os.path.getsize(fpath)
            if fsize == 0 or fsize > MAX_FILE_SIZE_BYTES:
                print(f"[{idx}/{len(files)}] [SKIP] {fname} -> Bos veya asiri buyuk dosya ({fsize} bytes)")
                continue

            bgr_raw = cv2.imread(fpath, cv2.IMREAD_COLOR)
            bgr = SaliencyEngine.validate_image(bgr_raw)
        except Exception as exc:
            print(f"[{idx}/{len(files)}] [SKIP] {fname} -> Bozuk veya okunamayan gorsel ({exc})")
            continue

        proc_bgr = (
            SaliencyEngine.simulate_shelf_distance(bgr, distance_meters, package_width_cm)
            if distance_meters > 0.5
            else bgr
        )

        sal_map, meta = SaliencyEngine.compute_saliency_map(
            proc_bgr, engine_mode=engine_mode, reading_direction=reading_direction
        )
        fixations = SaliencyEngine.compute_gaze_path(sal_map, 4, bgr_image=bgr)

        shelf_products: Optional[List[Dict[str, Any]]] = None
        if shelf_audit:
            planogram = SaliencyEngine.analyze_shelf_planogram(
                proc_bgr,
                saliency_map=sal_map,
                fixations=fixations,
                reading_direction=reading_direction,
                engine_mode=engine_mode,
                lang=lang,
            )
            shelf_products = planogram["products"]
            aoi_results = shelf_products
        else:
            detected_aois = SaliencyEngine.detect_text_and_cta_regions(
                proc_bgr,
                max_regions=5,
                distance_meters=distance_meters,
                package_width_cm=package_width_cm,
            )
            aoi_results = SaliencyEngine.calculate_aoi_metrics(
                sal_map,
                fixations,
                detected_aois,
                bgr_image=proc_bgr,
                distance_meters=distance_meters,
                package_width_cm=package_width_cm,
                lang=lang,
            )
        metrics = SaliencyEngine.calculate_metrics(
            proc_bgr,
            sal_map,
            fixations,
            meta,
            lang=lang,
            aoi_results=aoi_results,
            distance_meters=distance_meters,
            reading_direction=reading_direction,
            shelf_products=shelf_products,
        )

        heat = SaliencyEngine.generate_heatmap_overlay(proc_bgr, sal_map)
        heat_gaze = SaliencyEngine.draw_gaze_path(heat, fixations)
        fog = SaliencyEngine.generate_fog_overlay(proc_bgr, sal_map)

        report_img = ReportGenerator.generate_single_report(proc_bgr, heat_gaze, fog, metrics, aoi_results)
        stem = os.path.splitext(fname)[0]
        report_out_path = os.path.join(output_dir, f"{stem}_optisight_report.png")
        report_pdf_path = os.path.join(output_dir, f"{stem}_optisight_report.pdf")
        report_img.save(report_out_path, "PNG")
        ReportGenerator.save_pdf_report(report_img, report_pdf_path)

        wcag_ratios = [a["wcag_ratio"] for a in aoi_results if a.get("wcag_ratio") is not None]
        min_wcag = round(min(wcag_ratios), 2) if wcag_ratios else None
        low_contrast_count = sum(1 for a in aoi_results if a.get("wcag_grade") == "DÜŞÜK")

        row: Dict[str, Any] = {
            "file": fname,
            "report_png": os.path.basename(report_out_path),
            "report_pdf": os.path.basename(report_pdf_path),
            "engine_mode": metrics.get("engine_mode", engine_mode),
            "reading_direction": metrics.get("reading_direction", reading_direction),
            "overall_attention_score": metrics["overall_attention_score"],
            "focus_score": metrics["focus_score"],
            "clutter_score": metrics["clutter_score"],
            "primary_quadrant": metrics["quadrant_enum"].value,
            "primary_quadrant_label": metrics["quadrant"],
            "faces_detected": len(meta.get("faces", [])),
            "blind_spots_count": len(meta.get("blind_spots", [])),
            "auto_aoi_count": len(aoi_results),
            "shelf_product_count": len(shelf_products) if shelf_products is not None else 0,
            "min_wcag_contrast": min_wcag,
            "low_contrast_warnings": low_contrast_count,
            "distance_meters": distance_meters,
            "package_width_cm": package_width_cm,
            "insight": metrics["insight"],
            "fix_recommendations": " | ".join(metrics.get("fix_recommendations", [])),
        }
        summary_rows.append(row)
        print(
            f"[{idx}/{len(files)}] [OK] {fname} -> Genel Skor: {row['overall_attention_score']}/100 "
            f"| Odak: {row['focus_score']} | Karmasa: {row['clutter_score']} | AOI/Urun: {row['auto_aoi_count']}"
        )

    json_path = os.path.join(output_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(summary_rows, jf, ensure_ascii=False, indent=2)

    csv_path = os.path.join(output_dir, "summary.csv")
    if summary_rows:
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    return summary_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="optisight",
        description="OptiSight: AI Marketing, Packaging & Shelf Attention Analyzer (GUI & Headless CLI)",
    )
    parser.add_argument(
        "--batch",
        "-b",
        type=str,
        default=None,
        help="Toplu analiz icin gorsel klasoru veya tekil gorsel yolu (Headless CLI modu).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="optisight_reports",
        help="Toplu analiz raporlarinin (PNG, CSV, JSON) kaydedilecegi klasor.",
    )
    parser.add_argument(
        "--distance",
        "-d",
        type=float,
        default=0.5,
        help="Fiziksel bakis mesafesi (metre, orn: 0.5 yakin, 1.5 market koridoru).",
    )
    parser.add_argument(
        "--width-cm",
        "-w",
        type=float,
        default=15.0,
        help="Ambalajin veya ekran tasariminin gercek fiziksel genisligi (cm).",
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=["tr", "en"],
        default="tr",
        help="Rapor ve icgoru dili (tr / en).",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["classic", "onnx", "hybrid"],
        default="classic",
        help="Saliency motoru modu: classic (5-katman), onnx (DoG-ASPP ONNX), veya hybrid (%%65 klasik + %%35 ONNX).",
    )
    parser.add_argument(
        "--rtl",
        action="store_true",
        help="Sagdan-sola (Right-to-Left / RTL) okuma yonu ve altin bolge (Top-Right) onceligini etkinlestir.",
    )
    parser.add_argument(
        "--shelf-audit",
        action="store_true",
        help="Tam raf fotografinda urunleri (SKU) otomatik algila ve Urun Basina Dikkat Payi vs Raf Payi analizi yap.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Sentetik psikofiziksel validasyon benchmarkini (NSS, CC, SIM, AUC-Judd) calistir ve cik.",
    )
    parser.add_argument(
        "--eval-dataset",
        nargs=2,
        metavar=("IMAGES_DIR", "DENSITY_DIR"),
        help="MIT1003/CAT2000/SALICON/UEyes veri seti klasorlerinde NSS, CC, SIM, AUC-Judd olcumu yap.",
    )
    parser.add_argument(
        "--fixations-dir",
        type=str,
        default=None,
        help="Opsiyonel ikili sabitlenme noktalari (fixation map) klasoru.",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="--eval-dataset ile birlikte her katmani tek tek kapatarak Ablation calismasi yapar.",
    )
    parser.add_argument(
        "--optimize-weights",
        action="store_true",
        help="--eval-dataset uzerinde Grid Search ile katman agirliklarini optimize eder.",
    )
    return parser


def handle_cli_args(argv: Optional[List[str]] = None) -> bool:
    """
    Parses CLI arguments. If a headless CLI action (--benchmark, --eval-dataset, --batch)
    was requested, executes it and returns True. Returns False if GUI should be launched.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.benchmark:
        res = SaliencyMetrics.run_synthetic_psychophysical_benchmark(SaliencyEngine)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return True

    if args.eval_dataset:
        img_dir, den_dir = args.eval_dataset
        opt_w = None
        opt_info = None
        if args.optimize_weights:
            opt_info = SaliencyMetrics.grid_search_weights(
                SaliencyEngine, img_dir, den_dir, fixation_dir=args.fixations_dir
            )
            opt_w = opt_info.get("best_weights_tuple")
        res = SaliencyMetrics.evaluate_dataset_folder(
            SaliencyEngine,
            img_dir,
            den_dir,
            fixation_dir=args.fixations_dir,
            include_baselines=True,
            include_ablation=args.ablation,
            custom_weights=opt_w,
        )
        if opt_info is not None:
            res["grid_search"] = opt_info
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return True

    if args.batch:
        rows = run_batch_analysis(
            input_path=args.batch,
            output_dir=args.output,
            distance_meters=args.distance,
            package_width_cm=args.width_cm,
            lang=args.lang,
            engine_mode=args.engine.upper(),
            reading_direction="rtl" if args.rtl else "ltr",
            shelf_audit=bool(args.shelf_audit),
        )
        print(f"\nTamamlandi: {len(rows)} gorsel islendi -> '{os.path.abspath(args.output)}'")
        return True

    return False
