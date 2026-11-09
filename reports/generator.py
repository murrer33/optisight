"""
OptiSight Executive Report Generator (1920x1080 Presentation Cards & 1-Page PDF Reports).
Pure Pillow + OpenCV + NumPy module (no GUI or X11 dependencies).
"""

import datetime
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class ReportGenerator:
    """Generates 1920x1080 single-region and A/B executive summary report cards and 1-page PDFs."""

    WIDTH = 1920
    HEIGHT = 1080

    C_BG = (15, 23, 42)  # #0F172A
    C_CARD = (30, 41, 59)  # #1E293B
    C_CARD_BORDER = (51, 65, 85)  # #334155
    C_CYAN = (56, 189, 248)  # #38BDF8
    C_ROSE = (244, 63, 94)  # #F43F5E
    C_EMERALD = (16, 185, 129)  # #10B981
    C_AMBER = (245, 158, 11)  # #F59E0B
    C_WHITE = (248, 250, 252)  # #F8FAFC
    C_MUTED = (148, 163, 184)  # #94A3B8

    @classmethod
    def _get_font(cls, size: int, bold: bool = False) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
        font_names = [
            "segoeui.ttf" if not bold else "segoeuib.ttf",
            "arial.ttf" if not bold else "arialbd.ttf",
            "calibri.ttf" if not bold else "calibrib.ttf",
            "DejaVuSans.ttf" if not bold else "DejaVuSans-Bold.ttf",
        ]
        for name in font_names:
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                continue
        return ImageFont.load_default()

    @classmethod
    def save_pdf_report(
        cls,
        report_img: Image.Image,
        pdf_path: str,
        title: str = "OptiSight Executive Attention & Legibility Audit",
        author: str = "OptiSight v2.2",
    ) -> str:
        """
        Exports a single-page executive report `Image.Image` to a high-resolution (300 DPI)
        1-page PDF document without external PDF library dependencies.
        """
        rgb_img = report_img.convert("RGB")
        rgb_img.save(
            pdf_path,
            "PDF",
            resolution=300.0,
            title=title,
            author=author,
        )
        return pdf_path

    @classmethod
    def generate_single_pdf_report(
        cls,
        orig_bgr: np.ndarray,
        heatmap_gaze_bgr: np.ndarray,
        fog_bgr: np.ndarray,
        metrics: dict,
        pdf_path: str,
        aoi_results: Optional[List[dict]] = None,
    ) -> str:
        """Generates a 1-page single-region executive audit report and saves it directly as a PDF."""
        card = cls.generate_single_report(orig_bgr, heatmap_gaze_bgr, fog_bgr, metrics, aoi_results)
        return cls.save_pdf_report(card, pdf_path)

    @classmethod
    def generate_ab_pdf_report(
        cls,
        var_a_orig: np.ndarray,
        var_b_orig: np.ndarray,
        ab_data: dict,
        pdf_path: str,
    ) -> str:
        """Generates a 1-page A/B Battle executive audit report and saves it directly as a PDF."""
        card = cls.generate_ab_report(var_a_orig, var_b_orig, ab_data)
        return cls.save_pdf_report(card, pdf_path, title="OptiSight A/B Visual Attention Battle Report")

    @classmethod
    def generate_single_report(
        cls,
        orig_bgr: np.ndarray,
        heatmap_gaze_bgr: np.ndarray,
        fog_bgr: np.ndarray,
        metrics: dict,
        aoi_results: Optional[List[dict]] = None,
    ) -> Image.Image:
        card = Image.new("RGB", (cls.WIDTH, cls.HEIGHT), cls.C_BG)
        draw = ImageDraw.Draw(card)

        cls._draw_header(
            draw,
            title="OPTISIGHT // TAHMİNİ GÖZ TAKİBİ VE DİKKAT DENETİM RAPORU",
            mode_tag="TEKLİ BÖLGE DİKKAT DENETİMİ",
        )

        # Top-right Overall Attention Score Badge (0-100)
        overall_score = int(metrics.get("overall_attention_score", metrics.get("focus_score", 0)))
        badge_col = (
            cls.C_EMERALD if overall_score >= 70 else (cls.C_AMBER if overall_score >= 50 else cls.C_ROSE)
        )
        draw.rounded_rectangle(
            [(1450, 40), (1870, 96)], radius=12, fill=cls.C_CARD, outline=badge_col, width=2
        )
        f_score_hdr = cls._get_font(18, bold=True)
        draw.text(
            (1475, 55),
            f"GENEL DİKKAT SKORU: {overall_score} / 100",
            fill=badge_col,
            font=f_score_hdr,
        )

        col_w = 570
        col_h = 580
        y_top = 125
        gap = 25
        x_start = 50

        cls._draw_image_card(
            card,
            draw,
            orig_bgr,
            "1. ORİJİNAL TASARIM / RAF BAĞLAMI",
            x_start,
            y_top,
            col_w,
            col_h,
            cls.C_MUTED,
        )

        cls._draw_image_card(
            card,
            draw,
            heatmap_gaze_bgr,
            "2. TERMAL ISI HARİTASI VE BAKIŞ YOLU (1->4)",
            x_start + col_w + gap,
            y_top,
            col_w,
            col_h,
            cls.C_CYAN,
        )

        cls._draw_image_card(
            card,
            draw,
            fog_bgr,
            "3. ODAK SPOTU (AYRIŞTIRILMIŞ DİKKAT MASKESİ)",
            x_start + (col_w + gap) * 2,
            y_top,
            col_w,
            col_h,
            cls.C_CYAN,
        )

        dash_y = 725
        dash_w = cls.WIDTH - 100
        dash_h = 315

        draw.rounded_rectangle(
            [(50, dash_y), (50 + dash_w, dash_y + dash_h)],
            radius=16,
            fill=cls.C_CARD,
            outline=cls.C_CARD_BORDER,
            width=2,
        )

        f_sub = cls._get_font(13, bold=True)
        draw.text(
            (80, dash_y + 16),
            "TAHMİNİ METRİKLER, HEDEF AOI DENETİMİ VE KURAL TABANLI 'DÜZELT' LİSTESİ",
            fill=cls.C_CYAN,
            font=f_sub,
        )

        # 4 Compact Metric Tiles on Left Half (80..960)
        cls._draw_metric_tile(
            draw,
            80,
            dash_y + 46,
            205,
            105,
            f"{overall_score} / 100",
            "GENEL DİKKAT SKORU",
            "Odak, netlik ve hiyerarşi skoru",
            badge_col,
        )

        cls._draw_metric_tile(
            draw,
            300,
            dash_y + 46,
            205,
            105,
            f"{metrics.get('focus_score', 0)} / 100",
            "ODAK YOĞUNLUĞU",
            "Birincil odak toplanma oranı",
            cls.C_CYAN,
        )

        clutter = metrics.get("clutter_score", 0)
        c_color = cls.C_ROSE if clutter > 50 else cls.C_EMERALD
        cls._draw_metric_tile(
            draw,
            520,
            dash_y + 46,
            205,
            105,
            f"{clutter} / 100",
            "KARMAŞA ENDEKSİ",
            "Kenar ve yönelim entropisi",
            c_color,
        )

        quad = (
            metrics.get("quadrant", "Merkez Odak").replace(" (Altın Bölge)", "").replace(" (Golden Zone)", "")
        )
        cls._draw_metric_tile(
            draw,
            740,
            dash_y + 46,
            220,
            105,
            quad,
            "İLK BAKIŞ BÖLGESİ",
            "Gözün ilk temas noktası (#1)",
            cls.C_WHITE,
        )

        # Right Half Top: AOI Audit Table (Logo, Price, CTA)
        right_card_x = 980
        right_card_w = dash_w - 950
        right_card_h = 105

        draw.rounded_rectangle(
            [(right_card_x, dash_y + 46), (right_card_x + right_card_w, dash_y + 46 + right_card_h)],
            radius=10,
            fill=cls.C_BG,
            outline=cls.C_CARD_BORDER,
            width=1,
        )

        f_rtitle = cls._get_font(11, bold=True)
        f_rtext = cls._get_font(10, bold=False)

        if aoi_results and len(aoi_results) > 0:
            is_shelf = any(a.get("aoi_type") == "PRODUCT" for a in aoi_results)
            hdr_title = (
                "RAF PLANOGRAM DENETİMİ (ÜRÜN BAŞINA DİKKAT PAYI vs. RAF PAYI VE VERİMLİLİK)"
                if is_shelf
                else "HEDEF AOI DENETİMİ (LOGO / FİYAT / CTA — SABİTLENME SIRASI VE DİKKAT PAYI)"
            )
            draw.text(
                (right_card_x + 16, dash_y + 56),
                hdr_title,
                fill=cls.C_CYAN,
                font=f_rtitle,
            )
            y_aoi = dash_y + 78
            for a in aoi_results[:3]:
                if a.get("aoi_type") == "PRODUCT" and "shelf_share" in a:
                    eff = float(a.get("efficiency_index", 1.0))
                    txt = (
                        f"• {a['name']}: %{a['attention_share']} Dikkat vs %{a['shelf_share']} Raf Payı"
                        f"  |  Verim: {eff}x  |  {a['ttff']}"
                    )
                    row_color = cls.C_ROSE if eff < 0.85 else (cls.C_EMERALD if eff >= 1.15 else cls.C_WHITE)
                else:
                    ratio_val = a.get("wcag_ratio")
                    role_tag = f"[{a.get('aoi_type', 'AOI')}] " if a.get("aoi_type") else ""
                    wcag_str = f"  |  WCAG {ratio_val}:1 ({a['wcag_grade']})" if ratio_val is not None else ""
                    txt = f"• {role_tag}{a['name']}: %{a['share']} Dikkat Payı  |  {a['ttff']}{wcag_str}"
                    row_color = cls.C_ROSE if (ratio_val is not None and ratio_val < 4.5) else cls.C_WHITE
                draw.text((right_card_x + 16, y_aoi), txt[:108], fill=row_color, font=f_rtext)
                y_aoi += 21
        else:
            blind_spots = metrics.get("blind_spots", [])
            draw.text(
                (right_card_x + 16, dash_y + 56),
                "GÖRSEL KÖR NOKTALAR VE BANNER KÖRLÜĞÜ DENETİMİ",
                fill=cls.C_AMBER,
                font=f_rtitle,
            )
            if blind_spots:
                b = blind_spots[0]
                draw.text(
                    (right_card_x + 16, dash_y + 80),
                    f"⚠️ Kritik Kör Nokta: {b['zone']} (<%{b['energy']} Dikkat)",
                    fill=cls.C_ROSE,
                    font=f_rtext,
                )
                draw.text(
                    (right_card_x + 16, dash_y + 102),
                    "Tavsiye: Kritik 'Satın Al/CTA' veya Fiyat öğesini bu ölü bölgeden uzaklaştırın.",
                    fill=cls.C_MUTED,
                    font=f_rtext,
                )
            else:
                draw.text(
                    (right_card_x + 16, dash_y + 80),
                    "✓ Dengeli Görsel Enerji: Ciddi bir kör nokta tespit edilmedi.",
                    fill=cls.C_EMERALD,
                    font=f_rtext,
                )
                draw.text(
                    (right_card_x + 16, dash_y + 102),
                    "Tasarım kadranlar arasında akıcı bir bilişsel göz hareketi sağlıyor.",
                    fill=cls.C_MUTED,
                    font=f_rtext,
                )

        # Bottom Section (dash_y + 162 .. dash_y + 298): Rule-Based "DÜZELT" (Fix) Action List + Insight
        fix_box_y = dash_y + 162
        fix_box_h = 136
        draw.rounded_rectangle(
            [(80, fix_box_y), (50 + dash_w - 30, fix_box_y + fix_box_h)],
            radius=10,
            fill=cls.C_BG,
            outline=cls.C_CARD_BORDER,
            width=1,
        )

        f_fix_hdr = cls._get_font(12, bold=True)
        f_fix_item = cls._get_font(12, bold=False)
        draw.text(
            (100, fix_box_y + 12),
            "🛠️ KURAL TABANLI 'DÜZELT' EYLEM LİSTESİ (AKSİYON ÖNERİLERİ):",
            fill=cls.C_AMBER,
            font=f_fix_hdr,
        )

        fix_list = metrics.get("fix_recommendations", [])
        if not fix_list:
            fix_list = [f"💡 {metrics.get('insight', '')}"]

        y_fix = fix_box_y + 38
        for item in fix_list[:3]:
            item_col = cls.C_EMERALD if item.startswith("✓") else cls.C_WHITE
            draw.text((105, y_fix), f"▸ {item[:135]}", fill=item_col, font=f_fix_item)
            y_fix += 24

        # Subtle insight footer line inside the box
        f_ins_small = cls._get_font(11, bold=False)
        draw.text(
            (105, fix_box_y + 110),
            f"💡 Özet İçgörü: {metrics.get('insight', '')[:140]}",
            fill=cls.C_MUTED,
            font=f_ins_small,
        )

        return card

    @classmethod
    def generate_ab_report(
        cls,
        var_a_orig: np.ndarray,
        var_b_orig: np.ndarray,
        ab_data: dict,
    ) -> Image.Image:
        card = Image.new("RGB", (cls.WIDTH, cls.HEIGHT), cls.C_BG)
        draw = ImageDraw.Draw(card)

        winner = ab_data["overall_winner"]
        var_a = ab_data["variant_a"]
        var_b = ab_data["variant_b"]

        cls._draw_header(
            draw,
            title="OPTISIGHT // A/B GÖRSEL REKABET DENETİM RAPORU",
            mode_tag="BAŞA BAŞ RAF MÜCADELESİ (A/B BATTLE)",
        )

        w_text = f"🏆 KAZANAN: {winner.upper()}"
        f_win = cls._get_font(18, bold=True)
        draw.rounded_rectangle(
            [(1470, 40), (1870, 92)],
            radius=12,
            fill=cls.C_CYAN if winner in ("A Varyantı", "Variant A") else cls.C_ROSE,
        )
        draw.text((1500, 52), w_text, fill=cls.C_BG, font=f_win)

        col_w = 880
        col_h = 580
        y_top = 130
        x_a = 50
        x_b = 990

        m_a = var_a["metrics"]
        m_b = var_b["metrics"]
        score_a = m_a.get("overall_attention_score", m_a["focus_score"])
        score_b = m_b.get("overall_attention_score", m_b["focus_score"])

        cls._draw_image_card(
            card,
            draw,
            var_a["heatmap_gaze"],
            f"A VARYANTI (SKOR: {score_a}/100) — %{var_a['share']} DİKKAT PAYI",
            x_a,
            y_top,
            col_w,
            col_h,
            cls.C_CYAN,
        )

        cls._draw_image_card(
            card,
            draw,
            var_b["heatmap_gaze"],
            f"B VARYANTI (SKOR: {score_b}/100) — %{var_b['share']} DİKKAT PAYI",
            x_b,
            y_top,
            col_w,
            col_h,
            cls.C_ROSE,
        )

        dash_y = 740
        dash_w = cls.WIDTH - 100
        dash_h = 295

        draw.rounded_rectangle(
            [(50, dash_y), (50 + dash_w, dash_y + dash_h)],
            radius=16,
            fill=cls.C_CARD,
            outline=cls.C_CARD_BORDER,
            width=2,
        )

        f_bar_title = cls._get_font(14, bold=True)
        draw.text((80, dash_y + 18), "DİKKAT PAYI DAĞILIMI (%)", fill=cls.C_WHITE, font=f_bar_title)

        draw.text((80, dash_y + 44), f"A Varyantı: %{var_a['share']}", fill=cls.C_CYAN, font=f_bar_title)
        draw.text(
            (50 + dash_w - 230, dash_y + 44),
            f"B Varyantı: %{var_b['share']}",
            fill=cls.C_ROSE,
            font=f_bar_title,
        )

        bar_x = 80
        bar_y = dash_y + 72
        bar_w = dash_w - 60
        bar_h = 20

        draw.rounded_rectangle([(bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h)], radius=10, fill=cls.C_ROSE)
        a_chunk_w = int(round(bar_w * (var_a["share"] / 100.0)))
        if a_chunk_w > 0:
            draw.rounded_rectangle(
                [(bar_x, bar_y), (bar_x + a_chunk_w, bar_y + bar_h)], radius=10, fill=cls.C_CYAN
            )

        sub_y = dash_y + 115

        cls._draw_metric_tile(
            draw,
            80,
            sub_y,
            400,
            95,
            f"A: {score_a}/100  vs  B: {score_b}/100",
            "GENEL DİKKAT SKORU (0-100)",
            f"Genel dikkat skorunda {'A Varyantı' if score_a >= score_b else 'B Varyantı'} önde",
            cls.C_CYAN if score_a >= score_b else cls.C_ROSE,
        )

        cls._draw_metric_tile(
            draw,
            520,
            sub_y,
            400,
            95,
            f"A: {m_a['clutter_score']}  vs  B: {m_b['clutter_score']}",
            "GÖRSEL KARMAŞA ENDEKSİ",
            "Düşük karmaşa tüketicinin daha hızlı algılamasını sağlar",
            cls.C_EMERALD if m_a["clutter_score"] < m_b["clutter_score"] else cls.C_ROSE,
        )

        first_win = ab_data.get("first_fixation_winner", "A Varyantı")
        cls._draw_metric_tile(
            draw,
            960,
            sub_y,
            420,
            95,
            f"{first_win} #1. Sabitlenmeyi Kazandı",
            "İLK BAKIŞ YAKALAMA GALİBİ",
            "Birincil sabitlenme noktasında en yüksek odak enerjisini yakaladı",
            cls.C_WHITE,
        )

        rec_y = dash_y + 225
        draw.rounded_rectangle(
            [(80, rec_y), (50 + dash_w - 30, rec_y + 55)],
            radius=10,
            fill=cls.C_BG,
            outline=cls.C_CARD_BORDER,
            width=1,
        )
        f_rec = cls._get_font(14, bold=False)
        margin_val = ab_data.get("margin", 0)
        if winner in ("Berabere", "Tie"):
            rec_text = (
                f"💡 YÖNETİCİ ÖZETİ: Tasarımlar başa baş performans gösterdi "
                f"(Dikkat Payı: %{var_a['share']} vs %{var_b['share']}, Genel Skor: {score_a} vs {score_b})."
            )
        else:
            rec_text = (
                f"💡 YÖNETİCİ ÖZETİ: {winner} genel görsel rekabette öne çıktı "
                f"(Bileşik Üstünlük: +%{margin_val}, Dikkat Payı: %{var_a['share']} vs %{var_b['share']}, "
                f"Genel Skor: {score_a} vs {score_b}). Birincil varyant olarak önerilir."
            )
        draw.text((105, rec_y + 18), rec_text, fill=cls.C_WHITE, font=f_rec)

        return card

    @classmethod
    def _draw_header(cls, draw: ImageDraw.ImageDraw, title: str, mode_tag: str) -> None:
        f_title = cls._get_font(26, bold=True)
        f_tag = cls._get_font(12, bold=True)
        f_time = cls._get_font(12, bold=False)

        draw.text((50, 40), title, fill=cls.C_WHITE, font=f_title)

        # Dynamic mode_tag badge width calculation to prevent text clipping/overlap
        bbox = draw.textbbox((0, 0), mode_tag, font=f_tag)
        tag_w = bbox[2] - bbox[0]
        badge_w = max(180, tag_w + 28)
        draw.rounded_rectangle(
            [(50, 85), (50 + badge_w, 114)], radius=6, fill=cls.C_CARD, outline=cls.C_CYAN, width=1
        )
        draw.text((64, 91), mode_tag, fill=cls.C_CYAN, font=f_tag)

        now_str = datetime.datetime.now().strftime("%Y-%m-%d  •  %H:%M:%S")
        time_x = 50 + badge_w + 20
        draw.text((time_x, 91), f"ZAMAN DAMGASI: {now_str}", fill=cls.C_MUTED, font=f_time)

    @classmethod
    def _draw_image_card(
        cls,
        base_img: Image.Image,
        draw: ImageDraw.ImageDraw,
        bgr_array: np.ndarray,
        caption: str,
        x: int,
        y: int,
        w: int,
        h: int,
        border_color: Tuple[int, int, int],
    ) -> None:
        draw.rounded_rectangle(
            [(x, y), (x + w, y + h)],
            radius=14,
            fill=cls.C_CARD,
            outline=cls.C_CARD_BORDER,
            width=2,
        )

        f_cap = cls._get_font(12, bold=True)
        draw.text((x + 18, y + 16), caption, fill=border_color, font=f_cap)

        inner_x = x + 16
        inner_y = y + 48
        inner_w = w - 32
        inner_h = h - 64

        rgb = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)
        pil_sub = Image.fromarray(rgb)

        sub_w, sub_h = pil_sub.size
        scale = min(inner_w / sub_w, inner_h / sub_h)
        new_w = max(1, int(round(sub_w * scale)))
        new_h = max(1, int(round(sub_h * scale)))
        pil_resized = pil_sub.resize((new_w, new_h), Image.Resampling.LANCZOS)

        paste_x = inner_x + (inner_w - new_w) // 2
        paste_y = inner_y + (inner_h - new_h) // 2
        base_img.paste(pil_resized, (paste_x, paste_y))

    @classmethod
    def _draw_metric_tile(
        cls,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        w: int,
        h: int,
        val_str: str,
        label_str: str,
        desc_str: str,
        val_color: Tuple[int, int, int],
    ) -> None:
        draw.rounded_rectangle(
            [(x, y), (x + w, y + h)],
            radius=10,
            fill=cls.C_BG,
            outline=cls.C_CARD_BORDER,
            width=1,
        )

        f_val = cls._get_font(20, bold=True)
        f_lbl = cls._get_font(11, bold=True)
        f_desc = cls._get_font(10, bold=False)

        draw.text((x + 14, y + 12), val_str, fill=val_color, font=f_val)
        draw.text((x + 14, y + 44), label_str, fill=cls.C_MUTED, font=f_lbl)
        draw.text((x + 14, y + 64), desc_str, fill=(100, 116, 139), font=f_desc)
