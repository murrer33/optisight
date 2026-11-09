"""
OptiSight GUI Module (ui/app.py)
Contains PyQt6 overlays, floating HUD, A/B Battle Modal, system tray controller, and global hotkeys.
"""

import sys
import os
from typing import Optional

import numpy as np
import cv2
from PIL import Image

try:
    from pynput import keyboard
except Exception:
    keyboard = None  # Headless Linux / Wayland fallback

from PyQt6.QtCore import Qt, QPoint, QRect, QObject, pyqtSignal, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QPixmap, QImage, QIcon, QAction
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QProgressBar,
    QMenu,
    QSystemTrayIcon,
    QFileDialog,
    QSlider,
    QInputDialog,
)

# ---------------------------------------------------------------------------
# GLOBAL CONSTANTS & STYLING
# ---------------------------------------------------------------------------
COLOR_BG_DARK = "#0F172A"  # Deep slate 900
COLOR_CARD_DARK = "#1E293B"  # Slate 800
COLOR_CARD_HOVER = "#334155"  # Slate 700
COLOR_CYAN = "#38BDF8"  # Tailwind Sky 400
COLOR_ROSE = "#F43F5E"  # Tailwind Rose 500
COLOR_EMERALD = "#10B981"  # Tailwind Emerald 500
COLOR_AMBER = "#F59E0B"  # Tailwind Amber 500
COLOR_TEXT_PRIMARY = "#F8FAFC"  # Slate 50
COLOR_TEXT_MUTED = "#94A3B8"  # Slate 400
COLOR_BORDER = "#334155"

HUD_STYLESHEET = f"""
QWidget {{
    background-color: {COLOR_BG_DARK};
    color: {COLOR_TEXT_PRIMARY};
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
}}
QFrame#cardFrame {{
    background-color: {COLOR_CARD_DARK};
    border: 1px solid {COLOR_BORDER};
    border-radius: 12px;
}}
QLabel#titleLabel {{
    font-size: 13px;
    font-weight: 800;
    color: {COLOR_CYAN};
    letter-spacing: 1px;
}}
QLabel#metricValue {{
    font-size: 24px;
    font-weight: 800;
    color: {COLOR_TEXT_PRIMARY};
}}
QLabel#metricLabel {{
    font-size: 11px;
    font-weight: 600;
    color: {COLOR_TEXT_MUTED};
    text-transform: uppercase;
}}
QPushButton.modeBtn {{
    background-color: {COLOR_CARD_DARK};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    padding: 7px 12px;
    font-size: 11px;
    font-weight: 700;
    color: {COLOR_TEXT_PRIMARY};
}}
QPushButton.modeBtn:hover {{
    background-color: {COLOR_CARD_HOVER};
    border-color: {COLOR_CYAN};
}}
QPushButton.modeBtn:checked {{
    background-color: {COLOR_CYAN};
    color: #0F172A;
    border-color: {COLOR_CYAN};
}}
QPushButton.actionBtn {{
    background-color: {COLOR_CARD_DARK};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 11px;
    font-weight: 700;
    color: {COLOR_TEXT_PRIMARY};
}}
QPushButton.actionBtn:hover {{
    background-color: {COLOR_CARD_HOVER};
    border-color: {COLOR_CYAN};
    color: {COLOR_CYAN};
}}
QPushButton.specialBtn {{
    background-color: #0369A1;
    border: 1px solid {COLOR_CYAN};
    border-radius: 8px;
    padding: 7px 12px;
    font-size: 11px;
    font-weight: 800;
    color: #FFFFFF;
}}
QPushButton.specialBtn:hover {{
    background-color: {COLOR_CYAN};
    color: #0F172A;
}}
QPushButton.closeBtn {{
    background-color: transparent;
    border: none;
    color: {COLOR_TEXT_MUTED};
    font-size: 14px;
    font-weight: 800;
}}
QPushButton.closeBtn:hover {{
    color: {COLOR_ROSE};
}}
QSlider::groove:horizontal {{
    height: 6px;
    background: {COLOR_CARD_HOVER};
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background: {COLOR_CYAN};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {COLOR_TEXT_PRIMARY};
    border: 2px solid {COLOR_CYAN};
    width: 14px;
    margin-top: -4px;
    margin-bottom: -4px;
    border-radius: 7px;
}}
"""


def cv2_to_qpixmap(bgr_array: np.ndarray) -> QPixmap:
    if bgr_array is None or bgr_array.size == 0:
        return QPixmap()
    h, w, ch = bgr_array.shape
    bytes_per_line = ch * w
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB))
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def pil_to_qimage(pil_img: Image.Image) -> QImage:
    rgba = pil_img.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimg = QImage(data, rgba.size[0], rgba.size[1], QImage.Format.Format_RGBA8888)
    return qimg.copy()


from core.saliency import SaliencyEngine
from reports.generator import ReportGenerator


# ---------------------------------------------------------------------------
# 3. SCREEN CAPTURE & SNIPPING OVERLAY
# ---------------------------------------------------------------------------
class SnippingOverlay(QWidget):
    single_snip_completed = pyqtSignal(np.ndarray, QRect)
    ab_snip_completed = pyqtSignal(np.ndarray, np.ndarray, QRect, QRect)
    cancelled = pyqtSignal()

    MODE_SINGLE = "SINGLE"
    MODE_AB = "AB"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode = self.MODE_SINGLE
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
        )
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self.bg_pixmap = None
        self.bgr_full = None
        self.scale_x = 1.0
        self.scale_y = 1.0

        self.is_mouse_down = False
        self.start_pos = QPoint()
        self.current_pos = QPoint()
        self.mouse_cursor_pos = QPoint()

        self.ab_step = 1
        self.rect_a = None
        self.rect_b = None
        self.bgr_a = None
        self.bgr_b = None

    def start_capture(self, mode=MODE_SINGLE):
        self.mode = mode
        self.ab_step = 1
        self.rect_a = None
        self.rect_b = None
        self.bgr_a = None
        self.bgr_b = None
        self.is_mouse_down = False

        primary = QApplication.primaryScreen()
        if not primary:
            return

        self.bg_pixmap = primary.grabWindow(0)
        qimg = self.bg_pixmap.toImage().convertToFormat(QImage.Format.Format_BGR888)
        ptr = qimg.bits()
        ptr.setsize(qimg.sizeInBytes())
        self.bgr_full = np.array(ptr, dtype=np.uint8).reshape((qimg.height(), qimg.width(), 3)).copy()

        geo = primary.geometry()
        self.setGeometry(geo)

        self.scale_x = self.bgr_full.shape[1] / float(max(1, self.width()))
        self.scale_y = self.bgr_full.shape[0] / float(max(1, self.height()))

        self.show()
        self.activateWindow()
        self.raise_()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            self.cancelled.emit()
        else:
            super().keyPressEvent(event)

    def mouseMoveEvent(self, event):
        self.mouse_cursor_pos = event.pos()
        if self.is_mouse_down:
            self.current_pos = event.pos()
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_mouse_down = True
            self.start_pos = event.pos()
            self.current_pos = event.pos()
            self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            self.close()
            self.cancelled.emit()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.is_mouse_down:
            self.is_mouse_down = False
            self.current_pos = event.pos()
            selection = QRect(self.start_pos, self.current_pos).normalized()

            if selection.width() < 25 or selection.height() < 25:
                self.update()
                return

            cropped_bgr = self._extract_bgr(selection)

            if self.mode == self.MODE_SINGLE:
                self.close()
                self.single_snip_completed.emit(cropped_bgr, selection)

            elif self.mode == self.MODE_AB:
                if self.ab_step == 1:
                    self.rect_a = selection
                    self.bgr_a = cropped_bgr
                    self.ab_step = 2
                    self.update()
                elif self.ab_step == 2:
                    self.rect_b = selection
                    self.bgr_b = cropped_bgr
                    self.close()
                    self.ab_snip_completed.emit(self.bgr_a, self.bgr_b, self.rect_a, self.rect_b)

    def _extract_bgr(self, rect: QRect) -> np.ndarray:
        px_x = int(round(rect.x() * self.scale_x))
        px_y = int(round(rect.y() * self.scale_y))
        px_w = int(round(rect.width() * self.scale_x))
        px_h = int(round(rect.height() * self.scale_y))

        img_h, img_w = self.bgr_full.shape[:2]
        x1 = max(0, min(img_w - 1, px_x))
        y1 = max(0, min(img_h - 1, px_y))
        x2 = max(x1 + 1, min(img_w, px_x + px_w))
        y2 = max(y1 + 1, min(img_h, px_y + px_h))

        return self.bgr_full[y1:y2, x1:x2].copy()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.bg_pixmap is not None:
            painter.drawPixmap(self.rect(), self.bg_pixmap)

        painter.fillRect(self.rect(), QColor(15, 23, 42, 150))

        current_rect = QRect()
        if self.is_mouse_down:
            current_rect = QRect(self.start_pos, self.current_pos).normalized()

        if self.rect_a is not None and self.bg_pixmap is not None:
            painter.drawPixmap(self.rect_a, self.bg_pixmap, self.rect_a)
            self._draw_box(
                painter,
                self.rect_a,
                label="A Varyantı (Referans)",
                color=QColor(56, 189, 248),
                is_active=False,
            )

        if self.is_mouse_down and current_rect.width() > 3 and current_rect.height() > 3:
            if self.bg_pixmap is not None:
                painter.drawPixmap(current_rect, self.bg_pixmap, current_rect)
            label = (
                "B Varyantı (Meydan Okuyan)"
                if (self.mode == self.MODE_AB and self.ab_step == 2)
                else "Odak Alanı"
            )
            color = (
                QColor(244, 63, 94)
                if (self.mode == self.MODE_AB and self.ab_step == 2)
                else QColor(56, 189, 248)
            )
            self._draw_box(painter, current_rect, label=label, color=color, is_active=True)

        self._draw_header_banner(painter)
        self._draw_crosshairs(painter, self.mouse_cursor_pos)

    def _draw_box(self, painter: QPainter, rect: QRect, label: str, color: QColor, is_active: bool):
        fill_color = QColor(color.red(), color.green(), color.blue(), 20 if not is_active else 35)
        painter.fillRect(rect, fill_color)

        pen = QPen(color, 2 if not is_active else 2.5, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)

        accent_len = min(16, min(rect.width(), rect.height()) // 4)
        thick_pen = QPen(color, 4)
        painter.setPen(thick_pen)
        painter.drawLine(rect.left(), rect.top(), rect.left() + accent_len, rect.top())
        painter.drawLine(rect.left(), rect.top(), rect.left(), rect.top() + accent_len)
        painter.drawLine(rect.right(), rect.top(), rect.right() - accent_len, rect.top())
        painter.drawLine(rect.right(), rect.top(), rect.right(), rect.top() + accent_len)
        painter.drawLine(rect.left(), rect.bottom(), rect.left() + accent_len, rect.bottom())
        painter.drawLine(rect.left(), rect.bottom(), rect.left(), rect.bottom() - accent_len)
        painter.drawLine(rect.right(), rect.bottom(), rect.right() - accent_len, rect.bottom())
        painter.drawLine(rect.right(), rect.bottom(), rect.right(), rect.bottom() - accent_len)

        badge_text = f"{label}  [{rect.width()} x {rect.height()}]"
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(font)
        fm = painter.fontMetrics()
        text_w = fm.horizontalAdvance(badge_text) + 16
        text_h = fm.height() + 8

        badge_x = rect.left()
        badge_y = rect.top() - text_h - 4
        if badge_y < 50:
            badge_y = rect.top() + 6

        badge_rect = QRect(badge_x, badge_y, text_w, text_h)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(15, 23, 42, 225))
        painter.drawRoundedRect(badge_rect, 4, 4)

        painter.setPen(QPen(color, 1))
        painter.drawRoundedRect(badge_rect, 4, 4)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_text)

    def _draw_header_banner(self, painter: QPainter):
        if self.mode == self.MODE_SINGLE:
            title = "OPTISIGHT // TEKLİ DİKKAT VE ODAK TARAMASI"
            subtitle = "Hedef marketing/raf alanını seçmek için sürükleyin  •  [Esc] İptal"
            accent_color = QColor(56, 189, 248)
        else:
            if self.ab_step == 1:
                title = "A/B TESTİ // ADIM 1/2: A VARYANTINI SEÇİN (REFERANS)"
                subtitle = "A Ürününü / Tasarımını çerçevelemek için sürükleyin  •  [Esc] İptal"
                accent_color = QColor(56, 189, 248)
            else:
                title = "A/B TESTİ // ADIM 2/2: B VARYANTINI SEÇİN (MEYDAN OKUYAN)"
                subtitle = "B Ürününü / Tasarımını çerçevelemek için sürükleyin  •  [Esc] İptal"
                accent_color = QColor(244, 63, 94)

        banner_w = 560
        banner_h = 60
        banner_x = (self.width() - banner_w) // 2
        banner_y = 28
        banner_rect = QRect(banner_x, banner_y, banner_w, banner_h)

        painter.setPen(QPen(accent_color, 1.5))
        painter.setBrush(QColor(15, 23, 42, 230))
        painter.drawRoundedRect(banner_rect, 10, 10)

        title_font = QFont("Segoe UI", 11, QFont.Weight.ExtraBold)
        painter.setFont(title_font)
        painter.setPen(accent_color)
        painter.drawText(QRect(banner_x, banner_y + 8, banner_w, 24), Qt.AlignmentFlag.AlignCenter, title)

        sub_font = QFont("Segoe UI", 9, QFont.Weight.Medium)
        painter.setFont(sub_font)
        painter.setPen(QColor(203, 213, 225))
        painter.drawText(QRect(banner_x, banner_y + 32, banner_w, 20), Qt.AlignmentFlag.AlignCenter, subtitle)

    def _draw_crosshairs(self, painter: QPainter, pos: QPoint):
        crosshair_pen = QPen(QColor(56, 189, 248, 90), 1, Qt.PenStyle.DashLine)
        painter.setPen(crosshair_pen)
        painter.drawLine(0, pos.y(), self.width(), pos.y())
        painter.drawLine(pos.x(), 0, pos.x(), self.height())


# ---------------------------------------------------------------------------
# 4. LIVE OVERLAYS & ADVANCED HUD DIALOGS
# ---------------------------------------------------------------------------
class LiveRegionOverlay(QWidget):
    aoi_updated = pyqtSignal(list)

    def __init__(
        self,
        rect: QRect,
        orig_bgr: np.ndarray,
        saliency_map: np.ndarray,
        fixations: list,
        meta: dict,
        raw_bgr: Optional[np.ndarray] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
        )
        self.setGeometry(rect)
        self.setMouseTracking(True)

        self.orig_bgr = orig_bgr
        self.raw_bgr = raw_bgr if raw_bgr is not None else orig_bgr
        self.saliency_map = saliency_map
        self.fixations = fixations
        self.meta = meta
        self.current_display_bgr = orig_bgr

        # Advanced Interactive States
        self.aois = []
        self.aoi_results = []
        self.is_drawing_aoi = False
        self.pending_aoi_role = "CUSTOM"
        self.aoi_start = QPoint()
        self.aoi_curr = QPoint()

        self.show_blind_spots = False
        self.shelf_distance = 0.5  # meters
        self.package_width_cm = 15.0  # physical package width in cm
        self.color_blind_mode = "NONE"

        # Cached Pixmaps for Instant, High-Performance Zero-Flicker Repaints
        self._cached_base_pixmap: Optional[QPixmap] = None
        self._cached_raw_pixmap: Optional[QPixmap] = None
        self._recompute_cached_pixmaps()

        # Scanpath Replay Animation (3.0 seconds, 100 frames)
        self.is_replaying = False
        self.replay_step = 0
        self.replay_timer = QTimer(self)
        self.replay_timer.setInterval(30)
        self.replay_timer.timeout.connect(self._on_replay_tick)

    def _recompute_cached_pixmaps(self):
        base = self.current_display_bgr
        if base is not None:
            if self.shelf_distance > 0.5:
                base = SaliencyEngine.simulate_shelf_distance(
                    base, self.shelf_distance, self.package_width_cm
                )
            if self.color_blind_mode and self.color_blind_mode != "NONE":
                base = SaliencyEngine.simulate_color_blindness(base, self.color_blind_mode)
            self._cached_base_pixmap = cv2_to_qpixmap(base)
        else:
            self._cached_base_pixmap = QPixmap()

        if self.raw_bgr is not None:
            self._cached_raw_pixmap = cv2_to_qpixmap(self.raw_bgr)
        else:
            self._cached_raw_pixmap = self._cached_base_pixmap

    def update_image(self, bgr_array: np.ndarray):
        self.current_display_bgr = bgr_array
        self._recompute_cached_pixmaps()
        self.update()

    def set_shelf_distance(self, distance: float):
        self.shelf_distance = distance
        self._recompute_cached_pixmaps()
        self.update()

    def set_package_width_cm(self, width_cm: float):
        self.package_width_cm = max(2.0, float(width_cm))
        self._recompute_cached_pixmaps()
        self.update()

    def set_color_blind_mode(self, mode: str):
        self.color_blind_mode = mode
        self._recompute_cached_pixmaps()
        self.update()

    def set_show_blind_spots(self, enabled: bool):
        self.show_blind_spots = enabled
        self.update()

    def start_aoi_draw_mode(self, role: str = "CUSTOM"):
        self.is_drawing_aoi = True
        self.pending_aoi_role = role or "CUSTOM"
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.raise_()
        self.activateWindow()
        self.update()

    def clear_aois(self):
        self.aois = []
        self.aoi_results = []
        self.aoi_updated.emit([])
        self.update()

    def run_auto_text_cta_scan(self):
        detected = SaliencyEngine.detect_text_and_cta_regions(
            self.raw_bgr,
            max_regions=5,
            distance_meters=self.shelf_distance,
            package_width_cm=self.package_width_cm,
        )
        for d in detected:
            bx, by, bw, bh = d["rect"]
            self.aois.append(
                {
                    "name": d["name"],
                    "rect": (bx, by, bw, bh),
                    "qrect": QRect(bx, by, bw, bh),
                    "is_text_or_cta": True,
                    "num_lines": d.get("num_lines", 1),
                    "wcag_ratio": d["wcag_ratio"],
                    "wcag_grade": d["wcag_grade"],
                }
            )
        self.aoi_results = SaliencyEngine.calculate_aoi_metrics(
            self.saliency_map,
            self.fixations,
            self.aois,
            bgr_image=self.raw_bgr,
            distance_meters=self.shelf_distance,
            package_width_cm=self.package_width_cm,
        )
        self.aoi_updated.emit(self.aoi_results)
        self.update()

    def run_auto_shelf_product_scan(self, reading_direction: str = "ltr"):
        planogram = SaliencyEngine.analyze_shelf_planogram(
            self.raw_bgr,
            saliency_map=self.saliency_map,
            fixations=self.fixations,
            reading_direction=reading_direction,
        )
        products = planogram.get("products", [])
        self.aois = []
        for p in products:
            bx, by, bw, bh = p["rect"]
            self.aois.append(
                {
                    "name": p["name"],
                    "aoi_type": "PRODUCT",
                    "rect": (bx, by, bw, bh),
                    "qrect": QRect(bx, by, bw, bh),
                    "is_text_or_cta": False,
                    "shelf_share": p["shelf_share"],
                    "attention_share": p["attention_share"],
                    "efficiency_index": p["efficiency_index"],
                    "status": p["status"],
                }
            )
        self.aoi_results = products
        self.aoi_updated.emit(self.aoi_results)
        self.update()

    def start_scanpath_replay(self):
        if not self.fixations:
            return
        self.is_replaying = True
        self.replay_step = 0
        self.replay_timer.start()
        self.update()

    def _on_replay_tick(self):
        self.replay_step += 1
        if self.replay_step >= 100:
            self.replay_timer.stop()
            self.is_replaying = False
        self.update()

    def mousePressEvent(self, event):
        if self.is_drawing_aoi:
            if event.button() == Qt.MouseButton.LeftButton:
                self.aoi_start = event.pos()
                self.aoi_curr = event.pos()
                self.update()
                return
            elif event.button() == Qt.MouseButton.RightButton:
                self.is_drawing_aoi = False
                self.setCursor(Qt.CursorShape.ArrowCursor)
                self.update()
                return
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        if self.is_drawing_aoi and event.key() == Qt.Key.Key_Escape:
            self.is_drawing_aoi = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()
        else:
            super().keyPressEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_drawing_aoi and (event.buttons() & Qt.MouseButton.LeftButton):
            self.aoi_curr = event.pos()
            self.update()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.is_drawing_aoi and event.button() == Qt.MouseButton.LeftButton:
            self.is_drawing_aoi = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            r = QRect(self.aoi_start, self.aoi_curr).normalized()
            if r.width() > 15 and r.height() > 15:
                role = getattr(self, "pending_aoi_role", "CUSTOM")
                role_defaults = {
                    "LOGO": "Marka Logosu",
                    "PRICE": "Fiyat Etiketi",
                    "CTA": "Satın Al / CTA Butonu",
                }
                if role in role_defaults:
                    count_same = sum(1 for a in self.aois if a.get("aoi_type") == role)
                    suffix = f" #{count_same + 1}" if count_same > 0 else ""
                    name = f"{role_defaults[role]}{suffix}"
                    aoi_type = role
                    is_text_flag = role in ("PRICE", "CTA")
                else:
                    name, ok = QInputDialog.getText(
                        self, "Hedef AOI Belirle", "İlgi Alanı Adı (örn: Logo, Satın Al Butonu, Fiyat):"
                    )
                    if not ok or not name.strip():
                        name = f"AOI {len(self.aois) + 1}"
                    name = name.strip()
                    aoi_type = SaliencyEngine._infer_aoi_type({"name": name})
                    is_text_flag = aoi_type in ("PRICE", "CTA", "TEXT")

                self.aois.append(
                    {
                        "name": name,
                        "aoi_type": aoi_type,
                        "is_text_or_cta": is_text_flag,
                        "rect": (r.x(), r.y(), r.width(), r.height()),
                        "qrect": r,
                    }
                )
                # Recompute AOI metrics with WCAG contrast
                self.aoi_results = SaliencyEngine.calculate_aoi_metrics(
                    self.saliency_map,
                    self.fixations,
                    self.aois,
                    bgr_image=self.raw_bgr,
                    distance_meters=self.shelf_distance,
                    package_width_cm=self.package_width_cm,
                )
                self.aoi_updated.emit(self.aoi_results)
            self.update()
        else:
            super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 1. Base Image: maintains whatever mode is active (Heatmap, etc.) seamlessly
        if self._cached_base_pixmap and not self._cached_base_pixmap.isNull():
            painter.drawPixmap(self.rect(), self._cached_base_pixmap)

        # 2. Blind Spots Visual Alert
        if self.show_blind_spots:
            blind_spots = self.meta.get("blind_spots", [])
            for b in blind_spots:
                bx, by, bw, bh = b["rect"]
                brect = QRect(bx, by, bw, bh)
                painter.fillRect(brect, QColor(244, 63, 94, 60))
                pen = QPen(QColor(244, 63, 94), 2, Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.drawRect(brect)
                # Warning badge
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(15, 23, 42, 200))
                badge = QRect(bx + 8, by + 8, 180, 22)
                painter.drawRoundedRect(badge, 4, 4)
                painter.setPen(QColor(244, 63, 94))
                font = QFont("Segoe UI", 8, QFont.Weight.Bold)
                painter.setFont(font)
                painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, f"⚠️ KÖR NOKTA (<%{b['energy']})")

        # 3. User & Auto-Detected AOI Boxes (with Non-Obstructive Translucent Badges)
        for idx, aoi in enumerate(self.aois):
            r = aoi["qrect"]
            res = self.aoi_results[idx] if idx < len(self.aoi_results) else None
            share_val = res.get("share") if res else None
            wcag_ratio = res.get("wcag_ratio") if res else None
            wcag_grade = res.get("wcag_grade") if res else None

            is_product = bool(res and res.get("aoi_type") == "PRODUCT" and "shelf_share" in res)
            is_low_contrast = wcag_grade == "DÜŞÜK"

            if is_product:
                eff_v = float(res.get("efficiency_index", 1.0))
                box_color = (
                    QColor(16, 185, 129)
                    if eff_v >= 1.15
                    else (QColor(244, 63, 94) if eff_v < 0.85 else QColor(56, 189, 248))
                )
                att_s = res.get("attention_share", 0)
                badge_text = f"📦 {aoi['name']} • %{att_s:.0f} • {eff_v:.1f}x"
            else:
                box_color = QColor(244, 63, 94) if is_low_contrast else QColor(56, 189, 248)
                aoi_role = aoi.get("aoi_type", "AOI")
                icon = (
                    "🏷️"
                    if aoi_role == "LOGO"
                    else ("💰" if aoi_role == "PRICE" else ("🛒" if aoi_role == "CTA" else "🎯"))
                )
                badge_text = f"{icon} {aoi['name']}"
                if share_val is not None:
                    badge_text += f" • %{share_val:.0f}"
                if wcag_ratio is not None:
                    badge_text += f" • {wcag_ratio}:1"

            # Draw box frame ONLY (No fill brush so product/image underneath NEVER darkens!)
            pen = QPen(box_color, 2, Qt.PenStyle.DashDotLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(r)

            # Badge measurement
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            fm = painter.fontMetrics()
            tw = min(fm.horizontalAdvance(badge_text) + 14, max(40, self.width() - 8))
            th = fm.height() + 6

            # Smart placement: above box if space permits, else neatly inside top of box
            if r.top() >= th + 4:
                badge_y = r.top() - th - 2
            else:
                badge_y = min(self.height() - th - 4, r.top() + 4)

            badge_x = max(4, min(self.width() - tw - 4, r.left()))
            badge_r = QRect(badge_x, badge_y, tw, th)

            # Translucent dark pill badge with accent border: NEVER SOLID BLACK
            painter.setPen(QPen(box_color, 1))
            painter.setBrush(QColor(15, 23, 42, 180))
            painter.drawRoundedRect(badge_r, 4, 4)
            painter.setPen(box_color)
            painter.drawText(badge_r, Qt.AlignmentFlag.AlignCenter, badge_text)

        # 4. Drawing Mode Visual Banner & Dragging Box
        if self.is_drawing_aoi:
            banner_h = 28
            banner_r = QRect(0, 0, self.width(), banner_h)
            painter.fillRect(banner_r, QColor(15, 23, 42, 225))
            painter.setPen(QPen(QColor(56, 189, 248), 1))
            painter.drawLine(0, banner_h - 1, self.width(), banner_h - 1)

            role_titles = {
                "LOGO": "🏷️ LOGO SEÇİMİ: Fareyle logonun etrafını kutu içine alın (İptal: ESC)",
                "PRICE": "💰 FİYAT SEÇİMİ: Fareyle fiyat etiketini kutu içine alın (İptal: ESC)",
                "CTA": "🛒 CTA SEÇİMİ: Fareyle butonun etrafını kutu içine alın (İptal: ESC)",
                "CUSTOM": "🎯 ÖZEL AOI: Fareyle alanı kutu içine alın (İptal: ESC)",
            }
            hint = role_titles.get(self.pending_aoi_role, "🎯 Alanı seçmek için fareyle sürükleyin")
            painter.setPen(QColor(241, 245, 249))
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.drawText(banner_r, Qt.AlignmentFlag.AlignCenter, hint)

            if self.aoi_curr != self.aoi_start:
                r_temp = QRect(self.aoi_start, self.aoi_curr).normalized()
                painter.setPen(QPen(QColor(245, 158, 11), 2, Qt.PenStyle.DashLine))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(r_temp)

        # 5. Scanpath Replay Live Animation
        if self.is_replaying and self.fixations:
            self._draw_replay_animation(painter)

        # Active boundary frame
        painter.setPen(QColor(56, 189, 248))
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)

    def _draw_replay_animation(self, painter: QPainter):
        # 100 ticks = 3.0s
        t = self.replay_step / 100.0
        n_fix = len(self.fixations)
        if n_fix < 2:
            return

        # Segments: 0->1, 1->2, 2->3
        # Timeline allocation
        cur_x, cur_y = self.fixations[0]
        active_idx = 0

        if t < 0.22:
            cur_x, cur_y = self.fixations[0]
            active_idx = 0
        elif t < 0.32:
            sub_t = (t - 0.22) / 0.10
            cur_x = int(round(self.fixations[0][0] + (self.fixations[1][0] - self.fixations[0][0]) * sub_t))
            cur_y = int(round(self.fixations[0][1] + (self.fixations[1][1] - self.fixations[0][1]) * sub_t))
            active_idx = 1
        elif t < 0.54:
            cur_x, cur_y = self.fixations[1]
            active_idx = 1
        elif t < 0.64:
            sub_t = (t - 0.54) / 0.10
            p1 = self.fixations[1]
            p2 = self.fixations[2] if n_fix > 2 else self.fixations[1]
            cur_x = int(round(p1[0] + (p2[0] - p1[0]) * sub_t))
            cur_y = int(round(p1[1] + (p2[1] - p1[1]) * sub_t))
            active_idx = 2
        elif t < 0.84:
            cur_x, cur_y = self.fixations[2] if n_fix > 2 else self.fixations[1]
            active_idx = 2
        elif t < 0.92:
            sub_t = (t - 0.84) / 0.08
            p2 = self.fixations[2] if n_fix > 2 else self.fixations[1]
            p3 = self.fixations[3] if n_fix > 3 else p2
            cur_x = int(round(p2[0] + (p3[0] - p2[0]) * sub_t))
            cur_y = int(round(p2[1] + (p3[1] - p2[1]) * sub_t))
            active_idx = 3
        else:
            cur_x, cur_y = self.fixations[3] if n_fix > 3 else self.fixations[-1]
            active_idx = 3

        # Draw pulsing foveal spotlight
        pulse_r = 32 + int(np.sin(self.replay_step * 0.35) * 8)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(56, 189, 248, 80))
        painter.drawEllipse(QPoint(cur_x, cur_y), pulse_r + 14, pulse_r + 14)

        painter.setBrush(QColor(56, 189, 248, 160))
        painter.drawEllipse(QPoint(cur_x, cur_y), pulse_r, pulse_r)

        painter.setBrush(QColor(255, 255, 255, 220))
        painter.drawEllipse(QPoint(cur_x, cur_y), 10, 10)

        # Dynamic floating badge
        elapsed_sec = t * 3.0
        badge_text = f"{active_idx + 1}. Sabitlenme • {elapsed_sec:.1f} sn Göz Odaklanması"
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        fm = painter.fontMetrics()
        bw = fm.horizontalAdvance(badge_text) + 20
        bh = fm.height() + 8
        badge_rect = QRect(cur_x - bw // 2, max(10, cur_y - pulse_r - bh - 6), bw, bh)

        painter.setBrush(QColor(15, 23, 42, 240))
        painter.setPen(QPen(QColor(56, 189, 248), 1.5))
        painter.drawRoundedRect(badge_rect, 6, 6)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_text)


class FloatingControlHUD(QWidget):
    view_changed = pyqtSignal(str, bool)
    shelf_distance_changed = pyqtSignal(float)
    package_width_changed = pyqtSignal(float)
    color_blind_changed = pyqtSignal(str)
    blind_spots_toggled = pyqtSignal(bool)
    add_aoi_requested = pyqtSignal(str)
    clear_aoi_requested = pyqtSignal()
    auto_cta_requested = pyqtSignal()
    shelf_scan_requested = pyqtSignal()
    engine_or_rtl_changed = pyqtSignal(str, str)
    replay_requested = pyqtSignal()
    copy_requested = pyqtSignal()
    save_requested = pyqtSignal()
    pdf_requested = pyqtSignal()
    closed = pyqtSignal()

    def __init__(self, metrics: dict, parent=None):
        super().__init__(parent)
        self.metrics = metrics
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
        )
        self.setStyleSheet(HUD_STYLESHEET)
        self.current_view_mode = "HEATMAP"
        self.show_gaze = True
        self.package_width_cm = 15.0
        self.engine_mode = "CLASSIC"
        self.reading_direction = "ltr"
        self._drag_pos = QPoint()
        self.aoi_results = []
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        card = QFrame()
        card.setObjectName("cardFrame")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 14, 16, 14)
        card_layout.setSpacing(12)

        # 1. Header Row
        header_row = QHBoxLayout()
        header_row.setSpacing(8)

        title = QLabel("⚡ OPTISIGHT // DİKKAT VE BAKIŞ PANELİ")
        title.setObjectName("titleLabel")
        header_row.addWidget(title)
        header_row.addStretch()

        close_btn = QPushButton("✕")
        close_btn.setObjectName("closeBtn")
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self._on_close)
        header_row.addWidget(close_btn)
        card_layout.addLayout(header_row)

        # 2. View Mode Toggles
        toggle_row = QHBoxLayout()
        toggle_row.setSpacing(6)

        self.btn_heatmap = QPushButton("🔥 Isı Haritası")
        self.btn_heatmap.setObjectName("btnHeatmap")
        self.btn_heatmap.setProperty("class", "modeBtn")
        self.btn_heatmap.setCheckable(True)
        self.btn_heatmap.setChecked(True)
        self.btn_heatmap.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_heatmap.clicked.connect(lambda: self._set_view_mode("HEATMAP"))

        self.btn_fog = QPushButton("🌫 Odak Spotu")
        self.btn_fog.setProperty("class", "modeBtn")
        self.btn_fog.setCheckable(True)
        self.btn_fog.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_fog.clicked.connect(lambda: self._set_view_mode("FOG"))

        self.btn_orig = QPushButton("📷 Orijinal")
        self.btn_orig.setProperty("class", "modeBtn")
        self.btn_orig.setCheckable(True)
        self.btn_orig.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_orig.clicked.connect(lambda: self._set_view_mode("ORIGINAL"))

        self.btn_gaze = QPushButton("🎯 Bakış [1-4]")
        self.btn_gaze.setProperty("class", "modeBtn")
        self.btn_gaze.setCheckable(True)
        self.btn_gaze.setChecked(True)
        self.btn_gaze.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_gaze.clicked.connect(self._toggle_gaze)

        toggle_row.addWidget(self.btn_heatmap)
        toggle_row.addWidget(self.btn_fog)
        toggle_row.addWidget(self.btn_orig)
        toggle_row.addWidget(self.btn_gaze)
        card_layout.addLayout(toggle_row)

        # 3. Interactive Tools Row (Scanpath Replay, Blind Spots, AOI)
        tools_row = QHBoxLayout()
        tools_row.setSpacing(6)

        btn_replay = QPushButton("▶ Oynat (3 sn)")
        btn_replay.setProperty("class", "specialBtn")
        btn_replay.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_replay.clicked.connect(self.replay_requested.emit)

        self.btn_blind = QPushButton("⚠️ Kör Noktalar")
        self.btn_blind.setProperty("class", "modeBtn")
        self.btn_blind.setCheckable(True)
        self.btn_blind.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_blind.clicked.connect(lambda: self.blind_spots_toggled.emit(self.btn_blind.isChecked()))

        btn_auto_cta = QPushButton("🔍 Oto Metin/CTA")
        btn_auto_cta.setProperty("class", "actionBtn")
        btn_auto_cta.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_auto_cta.clicked.connect(self.auto_cta_requested.emit)

        btn_shelf_scan = QPushButton("📦 Raf Ürünleri")
        btn_shelf_scan.setProperty("class", "actionBtn")
        btn_shelf_scan.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_shelf_scan.clicked.connect(self.shelf_scan_requested.emit)

        tools_row.addWidget(btn_replay)
        tools_row.addWidget(self.btn_blind)
        tools_row.addWidget(btn_auto_cta)
        tools_row.addWidget(btn_shelf_scan)
        card_layout.addLayout(tools_row)

        # 3a. Engine Mode (Klasik / ONNX / Hibrit) & Reading Direction (LTR / RTL) Row
        eng_row = QHBoxLayout()
        eng_row.setSpacing(5)
        eng_lbl = QLabel("🧠 Motor / Yön:")
        eng_lbl.setStyleSheet(f"font-size: 11px; color: {COLOR_TEXT_MUTED}; font-weight: 700;")
        eng_row.addWidget(eng_lbl)

        self.engine_buttons = {}
        for eng_code, eng_txt in [
            ("CLASSIC", "Klasik 5-K"),
            ("ONNX", "ONNX MSI"),
            ("HYBRID", "Hibrit"),
        ]:
            b_eng = QPushButton(eng_txt)
            b_eng.setProperty("class", "modeBtn")
            b_eng.setCheckable(True)
            b_eng.setChecked(eng_code == "CLASSIC")
            b_eng.setCursor(Qt.CursorShape.PointingHandCursor)
            b_eng.clicked.connect(lambda checked=False, ec=eng_code: self._set_engine_mode(ec))
            self.engine_buttons[eng_code] = b_eng
            eng_row.addWidget(b_eng)

        self.btn_rtl = QPushButton("⬅ RTL Okuma")
        self.btn_rtl.setProperty("class", "modeBtn")
        self.btn_rtl.setCheckable(True)
        self.btn_rtl.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_rtl.clicked.connect(self._toggle_rtl_direction)
        eng_row.addWidget(self.btn_rtl)
        card_layout.addLayout(eng_row)

        # 3b. Dedicated AOI Tagging Row (Logo, Fiyat, CTA, Özel)
        aoi_btn_row = QHBoxLayout()
        aoi_btn_row.setSpacing(5)
        for role_code, btn_txt in [
            ("LOGO", "🏷️ +Logo Seç"),
            ("PRICE", "💰 +Fiyat Seç"),
            ("CTA", "🛒 +CTA Seç"),
            ("CUSTOM", "🎯 +Özel AOI"),
        ]:
            b_role = QPushButton(btn_txt)
            b_role.setProperty("class", "actionBtn")
            b_role.setCursor(Qt.CursorShape.PointingHandCursor)
            b_role.clicked.connect(lambda checked=False, rc=role_code: self.add_aoi_requested.emit(rc))
            aoi_btn_row.addWidget(b_role)

        btn_clear_aoi = QPushButton("🗑️ Temizle")
        btn_clear_aoi.setProperty("class", "modeBtn")
        btn_clear_aoi.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_clear_aoi.setToolTip("Eklenen tüm AOI ve raf ürün kutularını sıfırla")
        btn_clear_aoi.clicked.connect(self.clear_aoi_requested.emit)
        aoi_btn_row.addWidget(btn_clear_aoi)
        card_layout.addLayout(aoi_btn_row)

        # 4. Physical Shelf Distance Slider (0.5m to 3.0m + Package Width in cm)
        dist_row = QVBoxLayout()
        dist_row.setSpacing(4)
        dist_label_row = QHBoxLayout()
        dist_title = QLabel("📏 Açısal Raf Mesafesi Simülasyonu:")
        dist_title.setStyleSheet(f"font-size: 11px; color: {COLOR_TEXT_MUTED}; font-weight: 700;")
        self.dist_val_lbl = QLabel("0.5m (Yakın • 15 cm)")
        self.dist_val_lbl.setStyleSheet(f"font-size: 11px; color: {COLOR_CYAN}; font-weight: 800;")

        btn_pkg_w = QPushButton("📐 Genişlik (cm)")
        btn_pkg_w.setProperty("class", "modeBtn")
        btn_pkg_w.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_pkg_w.clicked.connect(self._prompt_package_width)

        dist_label_row.addWidget(dist_title)
        dist_label_row.addStretch()
        dist_label_row.addWidget(self.dist_val_lbl)
        dist_label_row.addWidget(btn_pkg_w)
        dist_row.addLayout(dist_label_row)

        self.dist_slider = QSlider(Qt.Orientation.Horizontal)
        self.dist_slider.setRange(5, 30)  # 0.5m to 3.0m in 0.1m steps
        self.dist_slider.setValue(5)
        self.dist_slider.valueChanged.connect(self._on_distance_changed)
        dist_row.addWidget(self.dist_slider)
        card_layout.addLayout(dist_row)

        # 4b. Color Blindness / Accessibility Filter Row
        cvd_row = QHBoxLayout()
        cvd_row.setSpacing(4)
        cvd_lbl = QLabel("👁️ Erişilebilirlik:")
        cvd_lbl.setStyleSheet(f"font-size: 11px; color: {COLOR_TEXT_MUTED}; font-weight: 700;")
        cvd_row.addWidget(cvd_lbl)

        self.cvd_buttons = {}
        for mode_code, mode_text in [
            ("NONE", "Normal"),
            ("PROTAN", "Protanopi"),
            ("DEUTAN", "Deuteranopi"),
            ("TRITAN", "Tritanopi"),
            ("MONO", "Monokrom"),
        ]:
            b_cvd = QPushButton(mode_text)
            b_cvd.setProperty("class", "modeBtn")
            b_cvd.setCheckable(True)
            b_cvd.setChecked(mode_code == "NONE")
            b_cvd.setCursor(Qt.CursorShape.PointingHandCursor)
            b_cvd.clicked.connect(lambda checked, m=mode_code: self._set_color_blind_mode(m))
            self.cvd_buttons[mode_code] = b_cvd
            cvd_row.addWidget(b_cvd)
        card_layout.addLayout(cvd_row)

        # 5. Live Metrics Row (Overall 0-100 Attention Score + Focus + Clutter + Quadrant)
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(8)

        overall_val = self.metrics.get("overall_attention_score", self.metrics.get("focus_score", 0))
        overall_col = (
            COLOR_EMERALD if overall_val >= 70 else (COLOR_AMBER if overall_val >= 50 else COLOR_ROSE)
        )
        self.overall_box = self._create_metric_box(f"{overall_val}", "Genel Skor", overall_col)
        focus_box = self._create_metric_box(f"{self.metrics.get('focus_score', 0)}", "Odak", COLOR_CYAN)
        clutter_box = self._create_metric_box(
            f"{self.metrics.get('clutter_score', 0)}",
            "Karmaşa",
            COLOR_ROSE if self.metrics.get("clutter_score", 0) > 50 else COLOR_EMERALD,
        )
        quad_box = self._create_metric_box(
            self.metrics.get("quadrant", "Merkez Odak")
            .replace(" (Altın Bölge)", "")
            .replace(" (Golden Zone)", ""),
            "İlk Bakış",
            COLOR_TEXT_PRIMARY,
            is_text=True,
        )

        metrics_row.addWidget(self.overall_box)
        metrics_row.addWidget(focus_box)
        metrics_row.addWidget(clutter_box)
        metrics_row.addWidget(quad_box)
        card_layout.addLayout(metrics_row)

        # 6. AOI Table (Expands dynamically when AOIs added)
        self.aoi_frame = QFrame()
        self.aoi_frame.setStyleSheet(f"background-color: {COLOR_BG_DARK}; border-radius: 8px; padding: 4px;")
        self.aoi_layout = QVBoxLayout(self.aoi_frame)
        self.aoi_layout.setContentsMargins(8, 8, 8, 8)
        self.aoi_layout.setSpacing(4)

        aoi_hdr = QLabel("🎯 HEDEF ODAK ALANLARI (AOI):")
        aoi_hdr.setStyleSheet(f"color: {COLOR_CYAN}; font-size: 10px; font-weight: 800;")
        self.aoi_layout.addWidget(aoi_hdr)

        self.aoi_list_lbl = QLabel(
            "Henüz AOI alanı çizilmedi. Logo, CTA veya Fiyatı ölçmek için '+ AOI Alanı Ekle' butonuna tıklayın."
        )
        self.aoi_list_lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")
        self.aoi_list_lbl.setWordWrap(True)
        self.aoi_layout.addWidget(self.aoi_list_lbl)
        card_layout.addWidget(self.aoi_frame)

        # 7. Rule-Based 'Düzelt' (Fix) Recommendations Box
        fix_frame = QFrame()
        fix_frame.setStyleSheet(f"background-color: {COLOR_BG_DARK}; border-radius: 8px; padding: 6px;")
        fix_layout = QVBoxLayout(fix_frame)
        fix_layout.setContentsMargins(10, 8, 10, 8)
        fix_layout.setSpacing(4)

        fix_hdr = QLabel("🛠️ KURAL TABANLI 'DÜZELT' LİSTESİ:")
        fix_hdr.setStyleSheet(f"color: {COLOR_AMBER}; font-size: 10px; font-weight: 800;")
        fix_layout.addWidget(fix_hdr)

        fixes = self.metrics.get("fix_recommendations", [self.metrics.get("insight", "")])
        self.fix_list_lbl = QLabel("<br>".join(f"▸ {fx}" for fx in fixes[:3]))
        self.fix_list_lbl.setWordWrap(True)
        self.fix_list_lbl.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY}; font-size: 11px;")
        fix_layout.addWidget(self.fix_list_lbl)
        card_layout.addWidget(fix_frame)

        # 8. Export Action Buttons (Copy, PNG, 1-Page PDF)
        actions_row = QHBoxLayout()
        actions_row.setSpacing(6)

        btn_copy = QPushButton("📋 Kopyala")
        btn_copy.setProperty("class", "actionBtn")
        btn_copy.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_copy.clicked.connect(self.copy_requested.emit)

        btn_save = QPushButton("💾 PNG Kaydet")
        btn_save.setProperty("class", "actionBtn")
        btn_save.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_save.clicked.connect(self.save_requested.emit)

        btn_pdf = QPushButton("📄 Tek Sayfa PDF")
        btn_pdf.setProperty("class", "specialBtn")
        btn_pdf.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_pdf.clicked.connect(self.pdf_requested.emit)

        actions_row.addWidget(btn_copy)
        actions_row.addWidget(btn_save)
        actions_row.addWidget(btn_pdf)
        card_layout.addLayout(actions_row)

        main_layout.addWidget(card)
        self.setFixedWidth(490)

    def _on_distance_changed(self, val: int):
        dist_m = val / 10.0
        zone_txt = "Yakın" if dist_m <= 0.8 else ("Koridor Yürüyüşü" if dist_m <= 1.8 else "Uzak Raf")
        self.dist_val_lbl.setText(f"{dist_m:.1f}m ({zone_txt} • {self.package_width_cm:.0f} cm)")
        self.shelf_distance_changed.emit(dist_m)

    def _prompt_package_width(self):
        val, ok = QInputDialog.getDouble(
            self,
            "Fiziksel Ambalaj/Ekran Genişliği",
            "Seçilen alanın gerçek fiziksel genişliğini (cm) girin:",
            self.package_width_cm,
            2.0,
            300.0,
            1,
        )
        if ok:
            self.package_width_cm = float(val)
            self._on_distance_changed(self.dist_slider.value())
            self.package_width_changed.emit(self.package_width_cm)

    def _set_color_blind_mode(self, mode: str):
        for k, btn in self.cvd_buttons.items():
            btn.setChecked(k == mode)
        self.color_blind_changed.emit(mode)

    def update_aoi_table(self, aoi_results: list, updated_metrics: Optional[dict] = None):
        self.aoi_results = aoi_results
        if updated_metrics is not None:
            self.metrics = updated_metrics
            fixes = self.metrics.get("fix_recommendations", [])
            if fixes:
                self.fix_list_lbl.setText("<br>".join(f"▸ {fx}" for fx in fixes[:4]))

        if not aoi_results:
            self.aoi_list_lbl.setText(
                "Henüz AOI alanı çizilmedi. '+Logo', '+Fiyat', '+CTA' veya '🔍 Oto Metin/CTA' butonuna tıklayın."
            )
            return

        lines = []
        for a in aoi_results:
            if a.get("aoi_type") == "PRODUCT" and "shelf_share" in a:
                eff_v = float(a.get("efficiency_index", 1.0))
                eff_col = COLOR_EMERALD if eff_v >= 1.15 else (COLOR_ROSE if eff_v < 0.85 else COLOR_CYAN)
                lines.append(
                    f"• <b>📦 {a['name']}</b>: %{a['attention_share']} Dikkat vs %{a['shelf_share']} Raf"
                    f"  |  <span style='color:{eff_col}; font-weight:700;'>Verim: {eff_v}x</span>  |  {a['ttff']}"
                )
            else:
                wcag_txt = ""
                ratio_val = a.get("wcag_ratio")
                role_tag = f"[{a.get('aoi_type', 'AOI')}] " if a.get("aoi_type") else ""
                if ratio_val is not None:
                    grade_col = COLOR_ROSE if ratio_val < 4.5 else COLOR_EMERALD
                    wcag_txt = (
                        f"  |  <span style='color:{grade_col};'>WCAG {ratio_val}:1 ({a['wcag_grade']})</span>"
                    )
                lines.append(
                    f"• <b>{role_tag}{a['name']}</b>: %{a['share']} Dikkat  |  {a['ttff']}{wcag_txt}"
                )
        self.aoi_list_lbl.setText("<br>".join(lines))

    def _set_engine_mode(self, mode_code: str):
        self.engine_mode = mode_code
        for k, btn in self.engine_buttons.items():
            btn.setChecked(k == mode_code)
        self.engine_or_rtl_changed.emit(self.engine_mode, self.reading_direction)

    def _toggle_rtl_direction(self):
        self.reading_direction = "rtl" if self.btn_rtl.isChecked() else "ltr"
        self.engine_or_rtl_changed.emit(self.engine_mode, self.reading_direction)

    def _create_metric_box(self, value: str, label: str, color_hex: str, is_text: bool = False) -> QFrame:
        box = QFrame()
        box.setStyleSheet(f"background-color: {COLOR_BG_DARK}; border-radius: 8px; padding: 6px;")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(2)

        val_lbl = QLabel(value)
        val_lbl.setObjectName("metricValue")
        val_lbl.setStyleSheet(
            f"color: {color_hex}; font-size: {'14px' if is_text else '22px'}; font-weight: 800;"
        )
        val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        sub_lbl = QLabel(label)
        sub_lbl.setObjectName("metricLabel")
        sub_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(val_lbl)
        layout.addWidget(sub_lbl)
        return box

    def _set_view_mode(self, mode: str):
        self.current_view_mode = mode
        self.btn_heatmap.setChecked(mode == "HEATMAP")
        self.btn_fog.setChecked(mode == "FOG")
        self.btn_orig.setChecked(mode == "ORIGINAL")
        self.view_changed.emit(self.current_view_mode, self.show_gaze)

    def _toggle_gaze(self):
        self.show_gaze = self.btn_gaze.isChecked()
        self.view_changed.emit(self.current_view_mode, self.show_gaze)

    def _on_close(self):
        self.close()
        self.closed.emit()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._on_close()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()


class ABBattleModal(QDialog):
    copy_requested = pyqtSignal()
    save_requested = pyqtSignal()

    def __init__(self, ab_data: dict, parent=None):
        super().__init__(parent)
        self.ab_data = ab_data
        self.setWindowTitle("OptiSight // A/B Görsel Rekabet Denetimi")
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet(HUD_STYLESHEET)
        self.resize(1100, 720)
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        var_a = self.ab_data["variant_a"]
        var_b = self.ab_data["variant_b"]
        winner = self.ab_data["overall_winner"]
        winner_tr = "A Varyantı" if "A" in winner else ("B Varyantı" if "B" in winner else "Berabere")

        header = QFrame()
        header.setObjectName("cardFrame")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(18, 12, 18, 12)

        title_lbl = QLabel("⚔️ A/B GÖRSEL REKABET MÜCADELESİ")
        title_lbl.setObjectName("titleLabel")
        h_layout.addWidget(title_lbl)
        h_layout.addStretch()

        winner_pill = QLabel(f"🏆 KAZANAN: {winner_tr.upper()}")
        winner_pill.setStyleSheet(
            f"background-color: {COLOR_CYAN if 'A' in winner else COLOR_ROSE}; "
            f"color: #0F172A; font-weight: 800; font-size: 13px; padding: 6px 16px; border-radius: 8px;"
        )
        h_layout.addWidget(winner_pill)
        main_layout.addWidget(header)

        images_row = QHBoxLayout()
        images_row.setSpacing(16)

        col_a = self._create_variant_column("A Varyantı (Referans)", var_a, COLOR_CYAN)
        col_b = self._create_variant_column("B Varyantı (Meydan Okuyan)", var_b, COLOR_ROSE)

        images_row.addWidget(col_a)
        images_row.addWidget(col_b)
        main_layout.addLayout(images_row, stretch=1)

        comp_card = QFrame()
        comp_card.setObjectName("cardFrame")
        comp_layout = QVBoxLayout(comp_card)
        comp_layout.setContentsMargins(18, 14, 18, 14)
        comp_layout.setSpacing(10)

        bar_lbl_row = QHBoxLayout()
        lbl_a = QLabel(f"A Varyantı: %{var_a['share']} Dikkat")
        lbl_a.setStyleSheet(f"color: {COLOR_CYAN}; font-weight: 800; font-size: 13px;")
        lbl_b = QLabel(f"B Varyantı: %{var_b['share']} Dikkat")
        lbl_b.setStyleSheet(f"color: {COLOR_ROSE}; font-weight: 800; font-size: 13px;")
        bar_lbl_row.addWidget(lbl_a)
        bar_lbl_row.addStretch()
        bar_lbl_row.addWidget(lbl_b)
        comp_layout.addLayout(bar_lbl_row)

        pbar = QProgressBar()
        pbar.setRange(0, 100)
        pbar.setValue(int(round(var_a["share"])))
        pbar.setTextVisible(False)
        pbar.setFixedHeight(16)
        pbar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {COLOR_ROSE};
                border: none;
                border-radius: 8px;
            }}
            QProgressBar::chunk {{
                background-color: {COLOR_CYAN};
                border-radius: 8px;
            }}
        """)
        comp_layout.addWidget(pbar)

        first_win = self.ab_data.get("first_fixation_winner", "A Varyantı")
        first_win_tr = "A Varyantı" if "A" in first_win else "B Varyantı"
        takeaway = QLabel(
            f"📌 <b>İlk Göz Teması:</b> {first_win_tr} ilk bakışı yakaladı (1. Sabitlenme). "
            f"<b>Genel Raf Hakimiyeti:</b> {winner_tr} %{self.ab_data.get('margin', 0)} farkla daha yüksek görsel ağırlık elde etti."
        )
        takeaway.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 12px;")
        comp_layout.addWidget(takeaway)
        main_layout.addWidget(comp_card)

        actions = QHBoxLayout()
        actions.setSpacing(10)

        btn_copy = QPushButton("📋 A/B Raporunu Kopyala")
        btn_copy.setProperty("class", "actionBtn")
        btn_copy.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_copy.clicked.connect(self.copy_requested.emit)

        btn_save = QPushButton("💾 Yönetici A/B PNG Kaydet")
        btn_save.setProperty("class", "actionBtn")
        btn_save.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_save.clicked.connect(self.save_requested.emit)

        btn_close = QPushButton("Kapat")
        btn_close.setProperty("class", "actionBtn")
        btn_close.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_close.clicked.connect(self.close)

        actions.addWidget(btn_copy)
        actions.addWidget(btn_save)
        actions.addStretch()
        actions.addWidget(btn_close)
        main_layout.addLayout(actions)

    def _create_variant_column(self, title: str, var_info: dict, accent_hex: str) -> QFrame:
        card = QFrame()
        card.setObjectName("cardFrame")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        header = QLabel(title)
        header.setStyleSheet(f"color: {accent_hex}; font-size: 14px; font-weight: 800;")
        layout.addWidget(header)

        img_lbl = QLabel()
        img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pix = cv2_to_qpixmap(var_info["heatmap_gaze"])
        img_lbl.setPixmap(
            pix.scaled(
                480, 320, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        )
        layout.addWidget(img_lbl, stretch=1)

        m = var_info["metrics"]
        chips_row = QHBoxLayout()
        chips_row.setSpacing(8)

        focus_lbl = QLabel(f"Odak: <b>{m['focus_score']}/100</b>")
        focus_lbl.setStyleSheet(
            f"background-color: {COLOR_BG_DARK}; padding: 6px 10px; border-radius: 6px; font-size: 11px;"
        )

        clutter_lbl = QLabel(f"Karmaşa: <b>{m['clutter_score']}/100</b>")
        clutter_lbl.setStyleSheet(
            f"background-color: {COLOR_BG_DARK}; padding: 6px 10px; border-radius: 6px; font-size: 11px;"
        )

        quad_lbl = QLabel(
            f"Sıcak Bölge: <b>{m['quadrant'].replace(' (Altın Bölge)', '').replace(' (Golden Zone)', '')}</b>"
        )
        quad_lbl.setStyleSheet(
            f"background-color: {COLOR_BG_DARK}; padding: 6px 10px; border-radius: 6px; font-size: 11px;"
        )

        chips_row.addWidget(focus_lbl)
        chips_row.addWidget(clutter_lbl)
        chips_row.addWidget(quad_lbl)
        layout.addLayout(chips_row)

        return card


class MiniLauncherPill(QWidget):
    single_scan_clicked = pyqtSignal()
    ab_test_clicked = pyqtSignal()
    quit_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._drag_pos = QPoint()
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        container = QFrame()
        container.setStyleSheet(f"""
            QFrame {{
                background-color: {COLOR_BG_DARK};
                border: 1.5px solid {COLOR_CYAN};
                border-radius: 20px;
            }}
            QPushButton {{
                background: transparent;
                border: none;
                color: {COLOR_TEXT_PRIMARY};
                font-family: 'Segoe UI';
                font-size: 11px;
                font-weight: 700;
                padding: 6px 12px;
                border-radius: 14px;
            }}
            QPushButton:hover {{
                background-color: {COLOR_CARD_DARK};
                color: {COLOR_CYAN};
            }}
        """)
        c_layout = QHBoxLayout(container)
        c_layout.setContentsMargins(8, 4, 8, 4)
        c_layout.setSpacing(4)

        logo = QLabel("👁 OPTISIGHT")
        logo.setStyleSheet(f"color: {COLOR_CYAN}; font-weight: 900; font-size: 11px; padding-right: 4px;")
        c_layout.addWidget(logo)

        btn_single = QPushButton("Tekli [Alt+H]")
        btn_single.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_single.clicked.connect(self.single_scan_clicked.emit)
        c_layout.addWidget(btn_single)

        btn_ab = QPushButton("A/B Mücadelesi [Alt+B]")
        btn_ab.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_ab.clicked.connect(self.ab_test_clicked.emit)
        c_layout.addWidget(btn_ab)

        btn_quit = QPushButton("✕")
        btn_quit.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_quit.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-weight: 800;")
        btn_quit.clicked.connect(self.quit_clicked.emit)
        c_layout.addWidget(btn_quit)

        layout.addWidget(container)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()


# ---------------------------------------------------------------------------
# 5. HOTKEY MANAGER & THREAD BRIDGE
# ---------------------------------------------------------------------------
class HotkeyBridge(QObject):
    trigger_single = pyqtSignal()
    trigger_ab = pyqtSignal()
    trigger_quit = pyqtSignal()


# ---------------------------------------------------------------------------
# 6. MAIN APPLICATION CONTROLLER
# ---------------------------------------------------------------------------
class OptiSightApp(QObject):
    def __init__(self, app: QApplication):
        super().__init__()
        self.app = app

        self.single_data = {}
        self.ab_data = {}

        self.snipper = SnippingOverlay()
        self.snipper.single_snip_completed.connect(self.on_single_snip_completed)
        self.snipper.ab_snip_completed.connect(self.on_ab_snip_completed)

        self.live_overlay = None
        self.hud = None
        self.ab_modal = None

        self.mini_pill = MiniLauncherPill()
        self.mini_pill.single_scan_clicked.connect(self.trigger_single_scan)
        self.mini_pill.ab_test_clicked.connect(self.trigger_ab_test)
        self.mini_pill.quit_clicked.connect(self.quit_application)
        self._position_mini_pill()
        self.mini_pill.show()

        self._setup_tray_icon()

        self.hotkey_bridge = HotkeyBridge()
        self.hotkey_bridge.trigger_single.connect(self.trigger_single_scan)
        self.hotkey_bridge.trigger_ab.connect(self.trigger_ab_test)
        self.hotkey_bridge.trigger_quit.connect(self.quit_application)
        self._start_global_hotkeys()

    def _position_mini_pill(self):
        screen = self.app.primaryScreen().geometry()
        pill_w = 340
        pill_h = 48
        self.mini_pill.setGeometry(
            screen.width() - pill_w - 40, screen.height() - pill_h - 60, pill_w, pill_h
        )

    def _setup_tray_icon(self):
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.setBrush(QColor(15, 23, 42))
        painter.setPen(QPen(QColor(56, 189, 248), 3))
        painter.drawEllipse(4, 4, 56, 56)

        painter.setBrush(QColor(56, 189, 248))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(18, 18, 28, 28)

        painter.setBrush(QColor(15, 23, 42))
        painter.drawEllipse(26, 26, 12, 12)
        painter.end()

        tray_icon = QIcon(pixmap)
        self.tray = QSystemTrayIcon(tray_icon, self.app)
        self.tray.setToolTip("OptiSight: Yapay Zeka Marketing & Raf Dikkat Analizörü")

        menu = QMenu()
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {COLOR_BG_DARK};
                color: {COLOR_TEXT_PRIMARY};
                border: 1px solid {COLOR_BORDER};
                padding: 6px;
            }}
            QMenu::item:selected {{
                background-color: {COLOR_CARD_DARK};
                color: {COLOR_CYAN};
            }}
        """)

        act_single = QAction("⚡ Tekli Dikkat Taraması (Alt+H)", menu)
        act_single.triggered.connect(self.trigger_single_scan)
        menu.addAction(act_single)

        act_ab = QAction("⚔️ A/B Rekabet Mücadelesi (Alt+B)", menu)
        act_ab.triggered.connect(self.trigger_ab_test)
        menu.addAction(act_ab)

        menu.addSeparator()

        act_pill = QAction("Mini Başlatıcı Hapı Göster/Gizle", menu)
        act_pill.triggered.connect(self._toggle_mini_pill)
        menu.addAction(act_pill)

        menu.addSeparator()

        act_quit = QAction("OptiSight'tan Çıkış (Alt+Q)", menu)
        act_quit.triggered.connect(self.quit_application)
        menu.addAction(act_quit)

        self.tray.setContextMenu(menu)
        self.tray.show()

    def _toggle_mini_pill(self):
        if self.mini_pill.isVisible():
            self.mini_pill.hide()
        else:
            self.mini_pill.show()

    def _start_global_hotkeys(self):
        def on_alt_h():
            self.hotkey_bridge.trigger_single.emit()

        def on_alt_b():
            self.hotkey_bridge.trigger_ab.emit()

        def on_alt_q():
            self.hotkey_bridge.trigger_quit.emit()

        hotkeys = {"<alt>+h": on_alt_h, "<alt>+b": on_alt_b, "<alt>+q": on_alt_q}
        if keyboard is not None:
            self.listener = keyboard.GlobalHotKeys(hotkeys)
            self.listener.daemon = True
            self.listener.start()
        else:
            self.listener = None

    def trigger_single_scan(self):
        self._close_active_overlays()
        self.snipper.start_capture(SnippingOverlay.MODE_SINGLE)

    def trigger_ab_test(self):
        self._close_active_overlays()
        self.snipper.start_capture(SnippingOverlay.MODE_AB)

    def _close_active_overlays(self):
        if self.live_overlay:
            self.live_overlay.close()
            self.live_overlay = None
        if self.hud:
            self.hud.close()
            self.hud = None
        if self.ab_modal:
            self.ab_modal.close()
            self.ab_modal = None

    def on_single_snip_completed(self, bgr_cropped: np.ndarray, rect: QRect):
        sal_map, meta = SaliencyEngine.compute_saliency_map(bgr_cropped)
        fixations = SaliencyEngine.compute_gaze_path(sal_map, 4, bgr_image=bgr_cropped)
        metrics = SaliencyEngine.calculate_metrics(bgr_cropped, sal_map, fixations, meta)

        heat = SaliencyEngine.generate_heatmap_overlay(bgr_cropped, sal_map)
        heat_gaze = SaliencyEngine.draw_gaze_path(heat, fixations)
        fog = SaliencyEngine.generate_fog_overlay(bgr_cropped, sal_map)
        fog_gaze = SaliencyEngine.draw_gaze_path(fog, fixations)
        orig_gaze = SaliencyEngine.draw_gaze_path(bgr_cropped, fixations)

        self.single_data = {
            "orig": bgr_cropped,
            "orig_gaze": orig_gaze,
            "saliency": sal_map,
            "heatmap": heat,
            "heatmap_gaze": heat_gaze,
            "fog": fog,
            "fog_gaze": fog_gaze,
            "fixations": fixations,
            "metrics": metrics,
            "meta": meta,
            "rect": rect,
            "aoi_results": [],
        }

        # Live Region Overlay
        self.live_overlay = LiveRegionOverlay(rect, heat_gaze, sal_map, fixations, meta, raw_bgr=bgr_cropped)
        self.live_overlay.aoi_updated.connect(self._on_aoi_updated)
        self.live_overlay.show()

        # Floating Control HUD
        self.hud = FloatingControlHUD(metrics)
        self.hud.view_changed.connect(self._on_hud_view_changed)
        self.hud.shelf_distance_changed.connect(self.live_overlay.set_shelf_distance)
        self.hud.package_width_changed.connect(self.live_overlay.set_package_width_cm)
        self.hud.color_blind_changed.connect(self.live_overlay.set_color_blind_mode)
        self.hud.blind_spots_toggled.connect(self.live_overlay.set_show_blind_spots)
        self.hud.add_aoi_requested.connect(self.live_overlay.start_aoi_draw_mode)
        self.hud.clear_aoi_requested.connect(self.live_overlay.clear_aois)
        self.hud.auto_cta_requested.connect(self.live_overlay.run_auto_text_cta_scan)
        self.hud.shelf_scan_requested.connect(
            lambda: self.live_overlay.run_auto_shelf_product_scan(
                self.hud.reading_direction if self.hud else "ltr"
            )
        )
        self.hud.engine_or_rtl_changed.connect(self._on_engine_or_rtl_changed)
        self.hud.replay_requested.connect(self.live_overlay.start_scanpath_replay)
        self.hud.copy_requested.connect(self._copy_single_report)
        self.hud.save_requested.connect(self._save_single_report)
        self.hud.pdf_requested.connect(self._save_single_pdf_report)
        self.hud.closed.connect(self._close_active_overlays)

        hud_x = rect.right() + 15
        hud_y = rect.top()
        screen_geo = self.app.primaryScreen().geometry()
        if hud_x + 510 > screen_geo.width():
            hud_x = max(10, rect.left() - 510)
        if hud_y + 440 > screen_geo.height():
            hud_y = max(10, screen_geo.height() - 460)

        self.hud.move(hud_x, hud_y)
        self.hud.show()

    def _on_engine_or_rtl_changed(self, engine_mode: str, reading_direction: str):
        if not self.single_data or not self.live_overlay:
            return
        bgr_cropped = self.single_data["orig"]
        sal_map, meta = SaliencyEngine.compute_saliency_map(
            bgr_cropped, engine_mode=engine_mode, reading_direction=reading_direction
        )
        fixations = SaliencyEngine.compute_gaze_path(sal_map, 4, bgr_image=bgr_cropped)
        heat = SaliencyEngine.generate_heatmap_overlay(bgr_cropped, sal_map)
        heat_gaze = SaliencyEngine.draw_gaze_path(heat, fixations)
        fog = SaliencyEngine.generate_fog_overlay(bgr_cropped, sal_map)
        fog_gaze = SaliencyEngine.draw_gaze_path(fog, fixations)
        orig_gaze = SaliencyEngine.draw_gaze_path(bgr_cropped, fixations)

        self.single_data["saliency"] = sal_map
        self.single_data["meta"] = meta
        self.single_data["fixations"] = fixations
        self.single_data["heatmap"] = heat
        self.single_data["heatmap_gaze"] = heat_gaze
        self.single_data["fog"] = fog
        self.single_data["fog_gaze"] = fog_gaze
        self.single_data["orig_gaze"] = orig_gaze

        self.live_overlay.saliency_map = sal_map
        self.live_overlay.fixations = fixations
        self.live_overlay.meta = meta
        if self.hud:
            self._on_hud_view_changed(self.hud.current_view_mode, self.hud.show_gaze)
        self._on_aoi_updated(self.single_data.get("aoi_results", []))

    def _on_aoi_updated(self, aoi_results: list):
        self.single_data["aoi_results"] = aoi_results
        dist_m = self.live_overlay.shelf_distance if self.live_overlay else 1.0
        r_dir = self.hud.reading_direction if self.hud else "ltr"
        shelf_prods = [a for a in aoi_results if a.get("aoi_type") == "PRODUCT"]
        updated_metrics = SaliencyEngine.calculate_metrics(
            self.single_data["orig"],
            self.single_data["saliency"],
            self.single_data["fixations"],
            self.single_data["meta"],
            aoi_results=aoi_results,
            distance_meters=dist_m,
            reading_direction=r_dir,
            shelf_products=shelf_prods if shelf_prods else None,
        )
        self.single_data["metrics"] = updated_metrics
        if self.hud:
            self.hud.update_aoi_table(aoi_results, updated_metrics)

    def _on_hud_view_changed(self, mode: str, show_gaze: bool):
        if not self.live_overlay or not self.single_data:
            return

        if mode == "HEATMAP":
            img = self.single_data["heatmap_gaze"] if show_gaze else self.single_data["heatmap"]
        elif mode == "FOG":
            img = self.single_data["fog_gaze"] if show_gaze else self.single_data["fog"]
        else:
            img = self.single_data["orig_gaze"] if show_gaze else self.single_data["orig"]

        self.live_overlay.update_image(img)

    def _copy_single_report(self):
        if not self.single_data:
            return
        report_img = ReportGenerator.generate_single_report(
            self.single_data["orig"],
            self.single_data["heatmap_gaze"],
            self.single_data["fog"],
            self.single_data["metrics"],
            self.single_data.get("aoi_results"),
        )
        qimg = pil_to_qimage(report_img)
        QApplication.clipboard().setImage(qimg)
        if self.tray:
            self.tray.showMessage(
                "OptiSight",
                "Yönetici Rapor Kartı panoya kopyalandı!",
                QSystemTrayIcon.MessageIcon.Information,
                2500,
            )

    def _save_single_report(self):
        if not self.single_data:
            return
        file_path, _ = QFileDialog.getSaveFileName(
            None, "Yönetici Rapor Kartını Kaydet", "OptiSight_Dikkat_Raporu.png", "PNG Dosyaları (*.png)"
        )
        if file_path:
            report_img = ReportGenerator.generate_single_report(
                self.single_data["orig"],
                self.single_data["heatmap_gaze"],
                self.single_data["fog"],
                self.single_data["metrics"],
                self.single_data.get("aoi_results"),
            )
            report_img.save(file_path, "PNG")
            if self.tray:
                self.tray.showMessage(
                    "OptiSight",
                    f"Rapor başarıyla kaydedildi: {os.path.basename(file_path)}",
                    QSystemTrayIcon.MessageIcon.Information,
                    2500,
                )

    def _save_single_pdf_report(self):
        if not self.single_data:
            return
        file_path, _ = QFileDialog.getSaveFileName(
            None,
            "Tek Sayfalık PDF Raporu Kaydet",
            "OptiSight_Dikkat_Raporu.pdf",
            "PDF Dosyaları (*.pdf)",
        )
        if file_path:
            ReportGenerator.generate_single_pdf_report(
                self.single_data["orig"],
                self.single_data["heatmap_gaze"],
                self.single_data["fog"],
                self.single_data["metrics"],
                file_path,
                self.single_data.get("aoi_results"),
            )
            if self.tray:
                self.tray.showMessage(
                    "OptiSight",
                    f"Tek sayfalık PDF rapor kaydedildi: {os.path.basename(file_path)}",
                    QSystemTrayIcon.MessageIcon.Information,
                    2500,
                )

    def on_ab_snip_completed(self, bgr_a: np.ndarray, bgr_b: np.ndarray, rect_a: QRect, rect_b: QRect):
        self.ab_data = SaliencyEngine.compute_ab_comparison(bgr_a, bgr_b)
        self.ab_data["orig_a"] = bgr_a
        self.ab_data["orig_b"] = bgr_b

        self.ab_modal = ABBattleModal(self.ab_data)
        self.ab_modal.copy_requested.connect(self._copy_ab_report)
        self.ab_modal.save_requested.connect(self._save_ab_report)
        self.ab_modal.show()

    def _copy_ab_report(self):
        if not self.ab_data:
            return
        report_img = ReportGenerator.generate_ab_report(
            self.ab_data["orig_a"], self.ab_data["orig_b"], self.ab_data
        )
        qimg = pil_to_qimage(report_img)
        QApplication.clipboard().setImage(qimg)
        if self.tray:
            self.tray.showMessage(
                "OptiSight",
                "A/B Rekabet Raporu panoya kopyalandı!",
                QSystemTrayIcon.MessageIcon.Information,
                2500,
            )

    def _save_ab_report(self):
        if not self.ab_data:
            return
        file_path, _ = QFileDialog.getSaveFileName(
            None, "A/B Rekabet Raporunu Kaydet", "OptiSight_AB_Rekabet_Raporu.png", "PNG Dosyaları (*.png)"
        )
        if file_path:
            report_img = ReportGenerator.generate_ab_report(
                self.ab_data["orig_a"], self.ab_data["orig_b"], self.ab_data
            )
            report_img.save(file_path, "PNG")
            if self.tray:
                self.tray.showMessage(
                    "OptiSight",
                    f"A/B Raporu başarıyla kaydedildi: {os.path.basename(file_path)}",
                    QSystemTrayIcon.MessageIcon.Information,
                    2500,
                )

    def quit_application(self):
        self._close_active_overlays()
        if self.mini_pill:
            self.mini_pill.close()
        if self.tray:
            self.tray.hide()
        self.app.quit()


def launch_gui() -> None:
    """Initializes and executes the PyQt6 desktop overlay application."""
    if hasattr(Qt.ApplicationAttribute, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    _controller = OptiSightApp(app)
    sys.exit(app.exec())
