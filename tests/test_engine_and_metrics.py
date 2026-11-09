"""
Unit and regression test suite for OptiSight SaliencyEngine, SaliencyMetrics, CLI & Error Handling.
Run with:
    python -m unittest discover -s tests -v
or:
    pytest tests/
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.saliency import SaliencyEngine, Quadrant, _cached_center_bias
from core.metrics import SaliencyMetrics
from cli.runner import run_batch_analysis


class TestSaliencyEngine(unittest.TestCase):

    def test_weights_sum_to_one(self):
        total = (
            SaliencyEngine.WEIGHT_SPECTRAL
            + SaliencyEngine.WEIGHT_LAB
            + SaliencyEngine.WEIGHT_EDGE
            + SaliencyEngine.WEIGHT_BIAS
            + SaliencyEngine.WEIGHT_FACE
        )
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_saliency_map_shape_and_range(self):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.circle(img, (160, 120), 40, (0, 200, 255), -1)
        sal, meta = SaliencyEngine.compute_saliency_map(img)
        self.assertEqual(sal.shape, (240, 320))
        self.assertEqual(sal.dtype, np.float32)
        self.assertGreaterEqual(float(np.min(sal)), 0.0)
        self.assertLessEqual(float(np.max(sal)), 1.0)
        self.assertIn("faces", meta)
        self.assertIn("blind_spots", meta)

    def test_deterministic_regression_fixed_stimulus(self):
        """Deterministic regression test on a fixed synthetic packaging/UI stimulus."""
        rng = np.random.RandomState(42)
        img = rng.randint(20, 60, size=(240, 320, 3), dtype=np.uint8)
        cv2.rectangle(img, (180, 60), (290, 140), (10, 230, 250), -1)
        cv2.circle(img, (90, 170), 30, (240, 50, 200), -1)

        sal1, meta1 = SaliencyEngine.compute_saliency_map(img)
        sal2, meta2 = SaliencyEngine.compute_saliency_map(img)
        self.assertTrue(np.allclose(sal1, sal2, atol=1e-6))

        fix1 = SaliencyEngine.compute_gaze_path(sal1, num_fixations=4)
        fix2 = SaliencyEngine.compute_gaze_path(sal2, num_fixations=4)
        self.assertEqual(fix1, fix2)

        m1 = SaliencyEngine.calculate_metrics(img, sal1, fix1, meta1, lang="en")
        m2 = SaliencyEngine.calculate_metrics(img, sal2, fix2, meta2, lang="en")
        self.assertEqual(m1["focus_score"], m2["focus_score"])
        self.assertEqual(m1["clutter_score"], m2["clutter_score"])
        self.assertEqual(m1["quadrant"], m2["quadrant"])

    def test_itti_koch_and_spectral_baselines(self):
        img = np.full((200, 260, 3), 30, dtype=np.uint8)
        cv2.circle(img, (190, 80), 25, (20, 240, 255), -1)
        ik = SaliencyEngine.compute_itti_koch(img)
        sr = SaliencyEngine.compute_spectral_residual(img)
        self.assertEqual(ik.shape, (200, 260))
        self.assertEqual(sr.shape, (200, 260))
        self.assertAlmostEqual(float(np.max(ik)), 1.0, places=4)
        self.assertAlmostEqual(float(np.max(sr)), 1.0, places=4)

    def test_error_handling_corrupted_oversized_and_missing_ocr(self):
        # 1. Corrupted / None / tiny image
        with self.assertRaises(ValueError):
            SaliencyEngine.validate_image(None)
        with self.assertRaises(ValueError):
            SaliencyEngine.validate_image(np.zeros((4, 4, 3), dtype=np.uint8))

        # 2. Grayscale 2D and BGRA 4-channel auto-conversion
        gray = np.full((64, 64), 128, dtype=np.uint8)
        bgr_from_gray = SaliencyEngine.validate_image(gray)
        self.assertEqual(bgr_from_gray.shape, (64, 64, 3))

        bgra = np.full((64, 64, 4), 200, dtype=np.uint8)
        bgr_from_bgra = SaliencyEngine.validate_image(bgra)
        self.assertEqual(bgr_from_bgra.shape, (64, 64, 3))

        # 3. Oversized image (>2560px) auto-downscaling
        huge = np.zeros((3000, 4000, 3), dtype=np.uint8)
        capped = SaliencyEngine.validate_image(huge, max_dim=1280)
        self.assertLessEqual(max(capped.shape[:2]), 1280)

        # 4. Missing OCR library graceful MSER fallback
        orig_ocr = SaliencyEngine._ocr
        try:
            SaliencyEngine._ocr = False  # Simulate missing rapidocr_onnxruntime
            canvas = np.full((240, 400, 3), 245, dtype=np.uint8)
            cv2.rectangle(canvas, (80, 120), (320, 180), (20, 30, 220), -1)
            cv2.putText(canvas, "BUY NOW", (110, 162), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            regions = SaliencyEngine.detect_text_and_cta_regions(canvas, max_regions=3)
            self.assertIsInstance(regions, list)
        finally:
            SaliencyEngine._ocr = orig_ocr

    def test_popout_localization(self):
        img = np.full((300, 400, 3), 30, dtype=np.uint8)
        target_x, target_y = 280, 110
        cv2.circle(img, (target_x, target_y), 28, (20, 220, 255), -1)
        sal, _ = SaliencyEngine.compute_saliency_map(img)
        fixations = SaliencyEngine.compute_gaze_path(sal, num_fixations=4)
        self.assertGreaterEqual(len(fixations), 1)
        fx, fy = fixations[0]
        dist = np.hypot(fx - target_x, fy - target_y)
        self.assertLess(
            dist, 45.0, f"Primary fixation {(fx, fy)} too far from pop-out target {(target_x, target_y)}"
        )

    def test_aoi_metrics_arbitrary_fixations_no_index_error(self):
        sal = np.random.RandomState(0).rand(200, 300).astype(np.float32)
        fixations = SaliencyEngine.compute_gaze_path(sal, num_fixations=10)
        self.assertEqual(len(fixations), 10)
        aois = [
            {"name": "FullCanvas", "rect": (0, 0, 300, 200)},
            {"name": "EmptyCorner", "rect": (0, 0, 5, 5)},
        ]
        res = SaliencyEngine.calculate_aoi_metrics(sal, fixations, aois)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0]["fixation_rank"], 1)
        self.assertAlmostEqual(res[0]["share"], 100.0, places=0)

    def test_quadrant_enum_and_localization(self):
        self.assertEqual(SaliencyEngine.classify_quadrant(150, 150, 300, 300), Quadrant.CENTER)
        self.assertEqual(SaliencyEngine.classify_quadrant(20, 20, 300, 300), Quadrant.TOP_LEFT)
        self.assertEqual(SaliencyEngine.classify_quadrant(280, 20, 300, 300), Quadrant.TOP_RIGHT)
        self.assertEqual(SaliencyEngine.classify_quadrant(20, 280, 300, 300), Quadrant.BOTTOM_LEFT)
        self.assertEqual(SaliencyEngine.classify_quadrant(280, 280, 300, 300), Quadrant.BOTTOM_RIGHT)

        img = np.zeros((200, 200, 3), dtype=np.uint8)
        sal = np.zeros((200, 200), dtype=np.float32)
        sal[20, 20] = 1.0
        m_tr = SaliencyEngine.calculate_metrics(img, sal, [(20, 20)], lang="tr")
        m_en = SaliencyEngine.calculate_metrics(img, sal, [(20, 20)], lang="en")
        self.assertEqual(m_tr["quadrant_enum"], Quadrant.TOP_LEFT)
        self.assertIn("Altın Bölge", m_tr["quadrant"])
        self.assertIn("Golden Zone", m_en["quadrant"])
        self.assertIn("Golden Zone", m_en["insight"])

    def test_blind_spots_threshold(self):
        sal = np.zeros((200, 200), dtype=np.float32)
        sal[:100, :100] = 0.8
        spots = SaliencyEngine.detect_blind_spots(sal, threshold=0.15)
        zones = [b["quadrant"] for b in spots]
        self.assertNotIn(Quadrant.TOP_LEFT, zones)
        self.assertIn(Quadrant.BOTTOM_RIGHT, zones)

    def test_center_bias_lru_cache(self):
        _cached_center_bias.cache_clear()
        b1 = SaliencyEngine.compute_center_bias((180, 320))
        b2 = SaliencyEngine.compute_center_bias((180, 320))
        info = _cached_center_bias.cache_info()
        self.assertGreaterEqual(info.hits, 1)
        self.assertFalse(b1.flags.writeable)
        with self.assertRaises(ValueError):
            b2 *= 2.0

    def test_ab_comparison_size_invariance(self):
        img_a = np.full((200, 200, 3), 40, dtype=np.uint8)
        cv2.circle(img_a, (100, 100), 40, (20, 220, 255), -1)
        img_b = cv2.resize(img_a, (400, 400), interpolation=cv2.INTER_LINEAR)

        ab = SaliencyEngine.compute_ab_comparison(img_a, img_b)
        self.assertAlmostEqual(ab["variant_a"]["share"], 50.0, delta=4.5)
        self.assertAlmostEqual(ab["variant_b"]["share"], 50.0, delta=4.5)

    def test_wcag_only_on_text_cta_aois(self):
        canvas = np.full((240, 320, 3), 240, dtype=np.uint8)
        cv2.rectangle(canvas, (20, 20), (140, 80), (200, 210, 215), -1)
        sal = np.ones((240, 320), dtype=np.float32) * 0.5
        fix = [(60, 50)]
        aois = [
            {"name": "Ürün Fotoğrafı", "rect": (20, 20, 120, 60)},
            {"name": "Logo", "rect": (20, 20, 120, 60)},
            {"name": "Satın Al Butonu (CTA)", "rect": (20, 20, 120, 60)},
        ]
        res = SaliencyEngine.calculate_aoi_metrics(sal, fix, aois, bgr_image=canvas)
        self.assertIsNone(res[0]["wcag_ratio"])
        self.assertIsNone(res[1]["wcag_ratio"])
        self.assertIsNotNone(res[2]["wcag_ratio"])

    def test_physical_shelf_distance_and_color_blindness(self):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (150, 150), (40, 200, 80), -1)
        close_img = SaliencyEngine.simulate_shelf_distance(img, distance_meters=0.5, package_width_cm=15.0)
        far_img = SaliencyEngine.simulate_shelf_distance(img, distance_meters=2.5, package_width_cm=15.0)
        self.assertTrue(np.array_equal(img, close_img))
        self.assertFalse(np.array_equal(img, far_img))

        for mode in ("PROTAN", "DEUTAN", "TRITAN", "MONO", "NONE"):
            out = SaliencyEngine.simulate_color_blindness(img, mode)
            self.assertEqual(out.shape, img.shape)

    def test_wcag_contrast_calculation(self):
        bw = np.full((60, 120, 3), 255, dtype=np.uint8)
        bw[15:45, 20:100] = 0
        c_bw = SaliencyEngine.compute_wcag_contrast(bw)
        self.assertGreater(c_bw["ratio"], 15.0)
        self.assertEqual(c_bw["grade"], "AAA")
        self.assertTrue(c_bw["passed_aa"])

        low_c = np.full((60, 120, 3), 240, dtype=np.uint8)
        low_c[15:45, 20:100] = 200
        c_low = SaliencyEngine.compute_wcag_contrast(low_c)
        self.assertLess(c_low["ratio"], 3.0)
        self.assertEqual(c_low["grade"], "DÜŞÜK")
        self.assertFalse(c_low["passed_large"])

    def test_auto_detect_text_and_cta_regions(self):
        canvas = np.full((360, 640, 3), 245, dtype=np.uint8)
        cv2.rectangle(canvas, (180, 220), (460, 290), (20, 30, 220), -1)
        cv2.putText(canvas, "HEMEN AL", (215, 268), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(canvas, "SUPER INDIRIM %50", (120, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (15, 15, 15), 3)

        detected = SaliencyEngine.detect_text_and_cta_regions(
            canvas, max_regions=5, distance_meters=0.6, package_width_cm=20.0
        )
        self.assertGreaterEqual(len(detected), 1)
        first = detected[0]
        self.assertIn("wcag_ratio", first)
        self.assertIn("wcag_grade", first)
        self.assertIn("legible_at_distance", first)

        sal, _ = SaliencyEngine.compute_saliency_map(canvas)
        fix = SaliencyEngine.compute_gaze_path(sal, 4)
        aoi_res = SaliencyEngine.calculate_aoi_metrics(sal, fix, detected, bgr_image=canvas)
        self.assertEqual(len(aoi_res), len(detected))
        self.assertGreater(aoi_res[0]["wcag_ratio"], 1.0)


class TestSaliencyMetricsAndCLISmoke(unittest.TestCase):

    def test_identical_maps_perfect_scores(self):
        gt = SaliencyMetrics.fixation_points_to_density_map([(100, 80), (160, 120)], (200, 250))
        cc = SaliencyMetrics.compute_cc(gt, gt)
        sim = SaliencyMetrics.compute_sim(gt, gt)
        nss = SaliencyMetrics.compute_nss(gt, [(100, 80), (160, 120)])
        auc = SaliencyMetrics.compute_auc_judd(gt, [(100, 80), (160, 120)])

        self.assertAlmostEqual(cc, 1.0, places=4)
        self.assertAlmostEqual(sim, 1.0, places=4)
        self.assertGreater(nss, 2.0)
        self.assertGreater(auc, 0.95)

    def test_inverted_map_negative_correlation(self):
        gt = SaliencyMetrics.fixation_points_to_density_map([(100, 80)], (200, 250))
        inv = 1.0 - gt
        cc = SaliencyMetrics.compute_cc(inv, gt)
        nss = SaliencyMetrics.compute_nss(inv, [(100, 80)])
        auc = SaliencyMetrics.compute_auc_judd(inv, [(100, 80)])

        self.assertLess(cc, -0.9)
        self.assertLess(nss, -1.0)
        self.assertLess(auc, 0.1)

    def test_psychophysical_benchmark_outperforms_baseline(self):
        res = SaliencyMetrics.run_synthetic_psychophysical_benchmark(SaliencyEngine)
        opti = res["OptiSight_5Layer"]
        base = res["CenterBias_Baseline"]
        self.assertGreater(opti["NSS"], base["NSS"])
        self.assertGreater(opti["CC"], base["CC"])
        self.assertGreater(opti["AUC_Judd"], 0.85)

    def test_cli_batch_mode_handles_corrupted_file_gracefully(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            in_dir = os.path.join(tmpdir, "input")
            out_dir = os.path.join(tmpdir, "output")
            os.makedirs(in_dir, exist_ok=True)

            sample = np.full((300, 450, 3), 240, dtype=np.uint8)
            cv2.rectangle(sample, (100, 170), (350, 240), (20, 40, 220), -1)
            cv2.putText(sample, "SEPETE EKLE", (125, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            sample_path = os.path.join(in_dir, "test_ad.png")
            cv2.imwrite(sample_path, sample)

            # Write a corrupted image file alongside the valid one
            bad_path = os.path.join(in_dir, "corrupted_bad.png")
            with open(bad_path, "wb") as bf:
                bf.write(b"not_a_valid_png_header")

            rows = run_batch_analysis(
                input_path=in_dir,
                output_dir=out_dir,
                distance_meters=1.0,
                package_width_cm=15.0,
                lang="tr",
            )
            self.assertEqual(len(rows), 1)
            self.assertIn("overall_attention_score", rows[0])
            self.assertGreaterEqual(rows[0]["overall_attention_score"], 0)
            self.assertLessEqual(rows[0]["overall_attention_score"], 100)
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "summary.json")))
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "summary.csv")))
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "test_ad_optisight_report.png")))
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "test_ad_optisight_report.pdf")))

    def test_overall_score_fix_recommendations_and_aoi_roles(self):
        canvas = np.full((300, 420, 3), 230, dtype=np.uint8)
        # Low-contrast CTA button (gray on light gray)
        cv2.rectangle(canvas, (140, 210), (320, 265), (212, 216, 220), -1)
        cv2.putText(canvas, "SATIN AL", (165, 248), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 185, 190), 2)

        sal, meta = SaliencyEngine.compute_saliency_map(canvas)
        fix = SaliencyEngine.compute_gaze_path(sal, 4)
        aois = [
            {"name": "Marka Logosu", "aoi_type": "LOGO", "rect": (20, 20, 110, 50)},
            {"name": "Fiyat Etiketi", "aoi_type": "PRICE", "rect": (280, 25, 110, 50)},
            {"name": "Satın Al Butonu", "aoi_type": "CTA", "rect": (140, 210, 180, 55)},
        ]
        aoi_res = SaliencyEngine.calculate_aoi_metrics(sal, fix, aois, bgr_image=canvas)
        self.assertEqual(aoi_res[0]["aoi_type"], "LOGO")
        self.assertIsNone(aoi_res[0]["wcag_ratio"])
        self.assertEqual(aoi_res[1]["aoi_type"], "PRICE")
        self.assertIsNotNone(aoi_res[1]["wcag_ratio"])
        self.assertEqual(aoi_res[2]["aoi_type"], "CTA")
        self.assertLess(aoi_res[2]["wcag_ratio"], 3.0)

        metrics = SaliencyEngine.calculate_metrics(canvas, sal, fix, meta, aoi_results=aoi_res)
        self.assertIn("overall_attention_score", metrics)
        self.assertGreaterEqual(metrics["overall_attention_score"], 0)
        self.assertLessEqual(metrics["overall_attention_score"], 100)
        fixes = metrics["fix_recommendations"]
        self.assertGreaterEqual(len(fixes), 1)
        self.assertTrue(any("hedef" in f.lower() or "target" in f.lower() for f in fixes))

    def test_cli_smoke_benchmark_and_batch_subprocesses(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        main_py = os.path.join(repo_root, "main.py")

        # 1. Smoke test `--benchmark`
        proc_bench = subprocess.run(
            [sys.executable, main_py, "--benchmark"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(proc_bench.returncode, 0, msg=proc_bench.stderr)
        bench_data = json.loads(proc_bench.stdout)
        self.assertIn("OptiSight_5Layer", bench_data)
        self.assertIn("IttiKoch_Baseline", bench_data)

        # 2. Smoke test `--batch` via CLI subprocess
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "smoke.png")
            out_dir = os.path.join(tmpdir, "out")
            canvas = np.full((240, 320, 3), 230, dtype=np.uint8)
            cv2.circle(canvas, (160, 120), 40, (10, 180, 240), -1)
            cv2.imwrite(img_path, canvas)

            proc_batch = subprocess.run(
                [sys.executable, main_py, "--batch", img_path, "--output", out_dir, "--lang", "en"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(proc_batch.returncode, 0, msg=proc_batch.stderr)
            self.assertTrue(os.path.isfile(os.path.join(out_dir, "summary.json")))

    def test_onnx_and_hybrid_saliency_engines(self):
        img = np.full((300, 420, 3), 35, dtype=np.uint8)
        cv2.circle(img, (210, 150), 42, (20, 180, 250), -1)

        sal_classic, meta_c = SaliencyEngine.compute_saliency_map(img, engine_mode="CLASSIC")
        sal_onnx, meta_o = SaliencyEngine.compute_saliency_map(img, engine_mode="ONNX")
        sal_hybrid, meta_h = SaliencyEngine.compute_saliency_map(img, engine_mode="HYBRID")

        self.assertEqual(sal_classic.shape, (300, 420))
        self.assertEqual(sal_onnx.shape, (300, 420))
        self.assertEqual(sal_hybrid.shape, (300, 420))
        self.assertEqual(meta_c["engine_mode"], "CLASSIC")
        self.assertEqual(meta_o["engine_mode"], "ONNX")
        self.assertEqual(meta_h["engine_mode"], "HYBRID")
        self.assertGreater(
            float(np.mean(sal_hybrid[110:190, 170:250])), float(np.mean(sal_hybrid[0:60, 0:60]))
        )

    def test_rtl_reading_direction_shifts_golden_zone(self):
        bias_ltr = SaliencyEngine.compute_center_bias((200, 300), rtl=False)
        bias_rtl = SaliencyEngine.compute_center_bias((200, 300), rtl=True)
        # LTR center bias peaks left of center; RTL peaks right of center
        self.assertGreater(float(np.mean(bias_ltr[:, :150])), float(np.mean(bias_ltr[:, 150:])))
        self.assertGreater(float(np.mean(bias_rtl[:, 150:])), float(np.mean(bias_rtl[:, :150])))

        img = np.full((240, 360, 3), 30, dtype=np.uint8)
        cv2.rectangle(img, (230, 25), (335, 95), (30, 210, 255), -1)
        sal, meta = SaliencyEngine.compute_saliency_map(img, reading_direction="rtl")
        fix = SaliencyEngine.compute_gaze_path(sal, 4)
        m_rtl = SaliencyEngine.calculate_metrics(img, sal, fix, meta, reading_direction="rtl")
        self.assertEqual(m_rtl["reading_direction"], "rtl")
        self.assertIn("RTL", m_rtl["quadrant"])

    def test_shelf_product_detection_and_planogram_efficiency(self):
        shelf_img = np.full((420, 640, 3), 45, dtype=np.uint8)
        # Horizontal shelf rail at y=210
        cv2.rectangle(shelf_img, (0, 204), (640, 216), (200, 200, 205), -1)
        # Top shelf: 3 distinct product facings (one bright hero, two muted)
        cv2.rectangle(shelf_img, (40, 35), (190, 195), (60, 65, 70), -1)
        cv2.rectangle(shelf_img, (240, 35), (390, 195), (20, 195, 255), -1)
        cv2.putText(shelf_img, "HERO", (265, 120), cv2.FONT_HERSHEY_DUPLEX, 0.9, (15, 20, 25), 2)
        cv2.rectangle(shelf_img, (440, 35), (590, 195), (62, 66, 72), -1)
        # Bottom shelf: 2 product facings
        cv2.rectangle(shelf_img, (80, 235), (270, 395), (75, 80, 90), -1)
        cv2.rectangle(shelf_img, (350, 235), (540, 395), (180, 70, 40), -1)

        planogram = SaliencyEngine.analyze_shelf_planogram(shelf_img)
        self.assertGreaterEqual(planogram["product_count"], 3)
        products = planogram["products"]
        for p in products:
            self.assertIn("shelf_share", p)
            self.assertIn("attention_share", p)
            self.assertIn("efficiency_index", p)
        hero = planogram["hero_product"]
        self.assertIsNotNone(hero)
        self.assertGreater(hero["efficiency_index"], 1.0)

    def test_detector_precision_recall_and_statistical_summary(self):
        stats = SaliencyMetrics._summarize_metric_list([1.2, 1.6])
        self.assertAlmostEqual(stats["mean"], 1.4, places=3)
        self.assertIn("std", stats)
        self.assertIn("ci95", stats)

        iou = SaliencyMetrics.compute_box_iou((10, 10, 100, 100), (10, 10, 100, 100))
        self.assertAlmostEqual(iou, 1.0, places=4)

        det_bench = SaliencyMetrics.evaluate_shelf_and_text_detectors(SaliencyEngine)
        self.assertGreaterEqual(det_bench["shelf_planogram_detector"]["f1_score"], 0.70)
        self.assertGreaterEqual(det_bench["morphological_text_cta_detector"]["f1_score"], 0.85)


if __name__ == "__main__":
    unittest.main()
