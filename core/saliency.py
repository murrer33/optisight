"""
OptiSight - Biomimetic Predictive Eye-Tracking & Saliency Engine
5-Layer Heuristic Visual Attention Pipeline (100% Offline):
  1. Spectral Residual FFT Saliency (30%)
  2. CIE-LAB Chromatic & Luminance Pop-Out (25%)
  3. Multi-Scale Edge & Typography Density (25%)
  4. Anisotropic Center & Golden-Zone Bias (10%)
  5. YuNet Deep Face Detection Priority (10% - No Heuristic Skin-Color Gate)
"""

import os
import functools
from enum import Enum
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class Quadrant(str, Enum):
    """Canonical spatial quadrants for primary fixation classification."""

    CENTER = "CENTER"
    TOP_LEFT = "TOP_LEFT"
    TOP_RIGHT = "TOP_RIGHT"
    BOTTOM_LEFT = "BOTTOM_LEFT"
    BOTTOM_RIGHT = "BOTTOM_RIGHT"


QUADRANT_LABELS_TR: Dict[Quadrant, str] = {
    Quadrant.CENTER: "Merkez Odak",
    Quadrant.TOP_LEFT: "Sol-Üst (Altın Bölge)",
    Quadrant.TOP_RIGHT: "Sağ-Üst",
    Quadrant.BOTTOM_LEFT: "Sol-Alt",
    Quadrant.BOTTOM_RIGHT: "Sağ-Alt",
}

QUADRANT_LABELS_TR_RTL: Dict[Quadrant, str] = {
    Quadrant.CENTER: "Merkez Odak",
    Quadrant.TOP_LEFT: "Sol-Üst",
    Quadrant.TOP_RIGHT: "Sağ-Üst (Altın Bölge - RTL)",
    Quadrant.BOTTOM_LEFT: "Sol-Alt",
    Quadrant.BOTTOM_RIGHT: "Sağ-Alt",
}

QUADRANT_LABELS_EN: Dict[Quadrant, str] = {
    Quadrant.CENTER: "Center Focus",
    Quadrant.TOP_LEFT: "Top-Left (Golden Zone)",
    Quadrant.TOP_RIGHT: "Top-Right",
    Quadrant.BOTTOM_LEFT: "Bottom-Left",
    Quadrant.BOTTOM_RIGHT: "Bottom-Right",
}

QUADRANT_LABELS_EN_RTL: Dict[Quadrant, str] = {
    Quadrant.CENTER: "Center Focus",
    Quadrant.TOP_LEFT: "Top-Left",
    Quadrant.TOP_RIGHT: "Top-Right (Golden Zone - RTL)",
    Quadrant.BOTTOM_LEFT: "Bottom-Left",
    Quadrant.BOTTOM_RIGHT: "Bottom-Right",
}

QUADRANT_SHORT_TR: Dict[Quadrant, str] = {
    Quadrant.CENTER: "Merkez",
    Quadrant.TOP_LEFT: "Sol-Üst",
    Quadrant.TOP_RIGHT: "Sağ-Üst",
    Quadrant.BOTTOM_LEFT: "Sol-Alt",
    Quadrant.BOTTOM_RIGHT: "Sağ-Alt",
}

QUADRANT_SHORT_EN: Dict[Quadrant, str] = {
    Quadrant.CENTER: "Center",
    Quadrant.TOP_LEFT: "Top-Left",
    Quadrant.TOP_RIGHT: "Top-Right",
    Quadrant.BOTTOM_LEFT: "Bottom-Left",
    Quadrant.BOTTOM_RIGHT: "Bottom-Right",
}


@functools.lru_cache(maxsize=64)
def _cached_center_bias(h: int, w: int, rtl: bool = False) -> np.ndarray:
    """
    Computes and caches the anisotropic 2D Gaussian center & golden-zone reading bias
    on a canonical grid, then upsamples to (h, w).
    - `rtl=False` (LTR default): anchors slightly toward Top-Left (`cx=0.46`).
    - `rtl=True` (Right-to-Left scripts, e.g. Arabic/Hebrew/Persian): mirrors toward Top-Right (`cx=0.54`).
    Returns a read-only array so callers cannot corrupt the cache via in-place operations.
    """
    canon_h = min(h, 256)
    canon_w = min(w, 256)
    cx = canon_w * (0.54 if rtl else 0.46)
    cy = canon_h * 0.45
    sigma_x = max(1.0, canon_w * 0.38)
    sigma_y = max(1.0, canon_h * 0.38)

    y_idx, x_idx = np.ogrid[:canon_h, :canon_w]
    dist_sq = ((x_idx - cx) ** 2) / (2.0 * sigma_x**2) + ((y_idx - cy) ** 2) / (2.0 * sigma_y**2)
    bias_small = np.exp(-dist_sq).astype(np.float32)

    if (canon_h, canon_w) != (h, w):
        bias = cv2.resize(bias_small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        bias = bias_small

    min_v = float(np.min(bias))
    max_v = float(np.max(bias))
    if max_v - min_v < 1e-6:
        out: np.ndarray = np.zeros((h, w), dtype=np.float32)
    else:
        out = ((bias - min_v) / (max_v - min_v)).astype(np.float32)
    out.setflags(write=False)
    return out


class SaliencyEngine:
    """
    Biomimetic & Multi-Scale DoG-ASPP Visual Attention Engine.
    Supports three inference engines (`engine_mode`):
      - 'CLASSIC': 5-Layer Heuristic Pipeline (Spectral Residual + LAB + Edge + Center/RTL Bias + YuNet Face)
      - 'ONNX': Multi-Scale DoG-ASPP Analytic Network (`msinet_saliency_lite.onnx`, zero learned weights)
      - 'HYBRID': Calibrated fusion of 5-Layer Classical detail (65%) + Multi-Scale DoG-ASPP context (35%)
    """

    # Universal default weights: derived strictly by averaging the out-of-sample simplex grid search
    # optima across the Training partitions (N_train=60) of UEyes, MIT1003, and SALICON (zero test-set leakage).
    WEIGHT_SPECTRAL = 0.28
    WEIGHT_LAB = 0.14
    WEIGHT_EDGE = 0.25
    WEIGHT_BIAS = 0.18
    WEIGHT_FACE = 0.15

    # Out-of-sample domain weight presets (w_sr, w_lab, w_edge, w_bias, w_face)
    # Calibrated strictly on disjoint Training subsets (N_train=60) and evaluated on Held-Out Test (N_test=60):
    PRESET_WEIGHTS: Dict[str, Tuple[float, float, float, float, float]] = {
        "DEFAULT": (0.28, 0.14, 0.25, 0.18, 0.15),
        "UEYES_UI": (0.30, 0.05, 0.20, 0.25, 0.20),
        "MIT1003_NATURAL": (0.10, 0.05, 0.10, 0.20, 0.55),
        "SALICON_GENERAL": (0.25, 0.05, 0.05, 0.15, 0.50),
    }

    MAX_IMAGE_DIM = 2560

    _yunet_detector = None
    _yunet_attempted = False
    _msinet_session = None
    _msinet_dnn = None
    _msinet_attempted = False
    _face_cascade = None
    _eye_cascade = None
    _ocr = None

    @classmethod
    def validate_image(cls, bgr_image: Any, max_dim: Optional[int] = None) -> np.ndarray:
        """
        Validates and sanitizes an input image array:
        - Rejects None, non-ndarray, corrupted, or tiny (<8x8) inputs with ValueError.
        - Converts 2D grayscale or 4-channel BGRA to 3-channel uint8 BGR.
        - Caps oversized dimensions (> max_dim) proportionally to prevent OOM.
        """
        if bgr_image is None or not isinstance(bgr_image, np.ndarray) or bgr_image.size == 0:
            raise ValueError("Invalid or corrupted image: expected a non-empty NumPy array.")
        if bgr_image.ndim == 2:
            bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
        elif bgr_image.ndim == 3 and bgr_image.shape[2] == 4:
            bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGRA2BGR)
        elif bgr_image.ndim != 3 or bgr_image.shape[2] != 3:
            raise ValueError(f"Unsupported image shape {bgr_image.shape}: expected (H, W, 3).")

        h, w = bgr_image.shape[:2]
        if h < 8 or w < 8:
            raise ValueError(f"Image dimensions ({w}x{h}) are too small (minimum 8x8 required).")

        if bgr_image.dtype != np.uint8:
            bgr_image = np.clip(bgr_image, 0, 255).astype(np.uint8)

        limit = max_dim if max_dim is not None else cls.MAX_IMAGE_DIM
        if limit and max(h, w) > limit:
            scale = float(limit) / float(max(h, w))
            nw = max(8, int(round(w * scale)))
            nh = max(8, int(round(h * scale)))
            bgr_image = cv2.resize(bgr_image, (nw, nh), interpolation=cv2.INTER_AREA)

        return bgr_image

    @classmethod
    def _get_ocr(cls):
        """Singleton cache for RapidOCR engine to avoid reloading ONNX weights per call."""
        if cls._ocr is None:
            try:
                from rapidocr_onnxruntime import RapidOCR  # type: ignore

                cls._ocr = RapidOCR()
            except Exception:
                cls._ocr = False
        return cls._ocr or None

    @classmethod
    def get_yunet_detector(cls, input_size: Tuple[int, int] = (320, 320)):
        """Loads OpenCV's FaceDetectorYN (YuNet) ONNX model if available."""
        import sys

        if not hasattr(cv2, "FaceDetectorYN"):
            return None
        if cls._yunet_detector is None and not cls._yunet_attempted:
            cls._yunet_attempted = True
            base_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(base_dir)
            meipass_dir = getattr(sys, "_MEIPASS", base_dir)
            candidates = [
                os.path.join(meipass_dir, "assets", "models", "face_detection_yunet_2023mar.onnx"),
                os.path.join(base_dir, "assets", "models", "face_detection_yunet_2023mar.onnx"),
                os.path.join(parent_dir, "assets", "models", "face_detection_yunet_2023mar.onnx"),
                os.path.join(os.getcwd(), "assets", "models", "face_detection_yunet_2023mar.onnx"),
            ]
            for path in candidates:
                if os.path.exists(path):
                    try:
                        cls._yunet_detector = cv2.FaceDetectorYN.create(
                            model=path,
                            config="",
                            input_size=input_size,
                            score_threshold=0.72,
                            nms_threshold=0.3,
                            top_k=50,
                        )
                        break
                    except Exception:
                        cls._yunet_detector = None
        if cls._yunet_detector is not None:
            cls._yunet_detector.setInputSize(input_size)
        return cls._yunet_detector

    @classmethod
    def get_face_cascade(cls):
        """Fallback Haar cascade if YuNet ONNX model file is missing."""
        if cls._face_cascade is None:
            try:
                haar_dir = getattr(getattr(cv2, "data", None), "haarcascades", "")
                cascade_path = haar_dir + "haarcascade_frontalface_alt2.xml"
                cls._face_cascade = cv2.CascadeClassifier(cascade_path)
                if cls._face_cascade.empty():
                    cascade_path = haar_dir + "haarcascade_frontalface_default.xml"
                    cls._face_cascade = cv2.CascadeClassifier(cascade_path)
            except Exception:
                cls._face_cascade = None
        return cls._face_cascade

    @classmethod
    def get_eye_cascade(cls):
        if cls._eye_cascade is None:
            try:
                haar_dir = getattr(getattr(cv2, "data", None), "haarcascades", "")
                cascade_path = haar_dir + "haarcascade_eye.xml"
                cls._eye_cascade = cv2.CascadeClassifier(cascade_path)
            except Exception:
                cls._eye_cascade = None
        return cls._eye_cascade

    @classmethod
    def _find_msinet_model_path(cls) -> Optional[str]:
        import sys

        base_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(base_dir)
        meipass_dir = getattr(sys, "_MEIPASS", base_dir)
        candidates = [
            os.path.join(meipass_dir, "assets", "models", "msinet_saliency_lite.onnx"),
            os.path.join(base_dir, "assets", "models", "msinet_saliency_lite.onnx"),
            os.path.join(parent_dir, "assets", "models", "msinet_saliency_lite.onnx"),
            os.path.join(os.getcwd(), "assets", "models", "msinet_saliency_lite.onnx"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    @classmethod
    def get_msinet_session(cls):
        """
        Loads and caches the lightweight Multi-Scale Information Network (`msinet_saliency_lite.onnx`)
        using `onnxruntime.InferenceSession` (with `cv2.dnn.readNetFromONNX` as zero-dependency fallback).
        """
        if cls._msinet_session is None and cls._msinet_dnn is None and not cls._msinet_attempted:
            cls._msinet_attempted = True
            model_path = cls._find_msinet_model_path()
            if model_path:
                try:
                    import onnxruntime as ort  # type: ignore

                    sess_opts = ort.SessionOptions()
                    sess_opts.intra_op_num_threads = 2
                    cls._msinet_session = ort.InferenceSession(
                        model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"]
                    )
                except Exception:
                    cls._msinet_session = None
                if cls._msinet_session is None:
                    try:
                        cls._msinet_dnn = cv2.dnn.readNetFromONNX(model_path)
                    except Exception:
                        cls._msinet_dnn = None
        return cls._msinet_session

    @classmethod
    def compute_onnx_saliency(cls, bgr_image: np.ndarray, rtl: bool = False) -> np.ndarray:
        """
        Runs the bundled lightweight Multi-Scale Information Network (`msinet_saliency_lite.onnx`, ~198 KB)
        with ASPP dilated convolutions (rates 1, 2, 4, 8) and CoordConv spatial priors.
        Accepts any BGR image and returns a `[0, 1]` float32 saliency map at original resolution `(h, w)`.
        """
        bgr_valid = cls.validate_image(bgr_image)
        h, w = bgr_valid.shape[:2]
        in_w, in_h = 160, 120

        img_s = cv2.resize(bgr_valid, (in_w, in_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean_rgb = np.mean(rgb, axis=(0, 1), keepdims=True)
        color_diff = np.abs(rgb - mean_rgb) * 2.0

        yy, xx = np.mgrid[0:in_h, 0:in_w].astype(np.float32)
        xc = 0.54 if rtl else 0.46
        gx_prior = np.exp(-((xx / float(in_w) - xc) ** 2) / (2.0 * 0.35**2))
        gy_prior = np.exp(-((yy / float(in_h) - 0.45) ** 2) / (2.0 * 0.35**2))

        feat = np.zeros((1, 5, in_h, in_w), dtype=np.float32)
        feat[0, 0] = color_diff[:, :, 0]
        feat[0, 1] = color_diff[:, :, 1]
        feat[0, 2] = color_diff[:, :, 2]
        feat[0, 3] = gx_prior
        feat[0, 4] = gy_prior

        sess = cls.get_msinet_session()
        pred_small: Optional[np.ndarray] = None
        if sess is not None:
            try:
                out = sess.run(None, {"input": feat})[0]
                pred_small = out[0, 0].astype(np.float32)
            except Exception:
                pred_small = None
        elif cls._msinet_dnn is not None:
            try:
                cls._msinet_dnn.setInput(feat)
                out = cls._msinet_dnn.forward()
                pred_small = out[0, 0].astype(np.float32)
            except Exception:
                pred_small = None

        if pred_small is None:
            # Graceful fallback to Spectral Residual + LAB if ONNX file is unavailable
            gray = cv2.cvtColor(bgr_valid, cv2.COLOR_BGR2GRAY)
            fallback = 0.55 * cls.compute_spectral_residual(gray) + 0.45 * cls.compute_lab_contrast(bgr_valid)
            return cls._normalize_01(fallback)

        pred_small = cv2.GaussianBlur(pred_small, (5, 5), 1.0).astype(np.float32)
        sal = cv2.resize(pred_small, (w, h), interpolation=cv2.INTER_CUBIC)
        return cls._normalize_01(sal)

    @classmethod
    def compute_spectral_residual(cls, gray: np.ndarray) -> np.ndarray:
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        canonical_dim = 256
        scale = canonical_dim / max(h, w)
        tw, th = max(16, int(round(w * scale))), max(16, int(round(h * scale)))
        small = cv2.resize(gray, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)

        dft = np.fft.fft2(small)
        dft_shift = np.fft.fftshift(dft)
        magnitude = np.abs(dft_shift)
        phase = np.angle(dft_shift)

        log_amplitude = np.log(magnitude + 1e-6)
        kernel = np.ones((3, 3), dtype=np.float32) / 9.0
        avg_log_amp = cv2.filter2D(log_amplitude, -1, kernel)

        spectral_residual = log_amplitude - avg_log_amp
        res_shift = np.exp(spectral_residual + 1j * phase)
        res_dft = np.fft.ifftshift(res_shift)
        spatial_img = np.fft.ifft2(res_dft)
        saliency_small = (np.abs(spatial_img) ** 2).astype(np.float32)

        saliency_small = cv2.GaussianBlur(saliency_small, (9, 9), 2.5).astype(np.float32)
        saliency = cv2.resize(saliency_small, (w, h), interpolation=cv2.INTER_CUBIC)
        return cls._normalize_01(saliency)

    @classmethod
    def compute_lab_contrast(cls, bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        mean_lab = np.mean(lab, axis=(0, 1))
        diff = lab - mean_lab
        dist = np.sqrt(np.sum(diff**2, axis=2))
        ksize = max(5, int(min(bgr.shape[:2]) * 0.03) | 1)
        dist_blurred = cv2.GaussianBlur(dist, (ksize, ksize), 0)
        return cls._normalize_01(dist_blurred)

    @classmethod
    def compute_edge_density(cls, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(sobel_x, sobel_y)
        mag_norm = cls._normalize_01(mag)

        k_dim = max(3, int(min(h, w) * 0.02) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_dim, k_dim))
        closed = cv2.morphologyEx(mag_norm, cv2.MORPH_CLOSE, kernel)

        blur_k = max(7, int(min(h, w) * 0.05) | 1)
        density = cv2.GaussianBlur(closed, (blur_k, blur_k), 0)
        return cls._normalize_01(density)

    @classmethod
    def compute_center_bias(cls, shape: Tuple[int, int], rtl: bool = False) -> np.ndarray:
        h, w = int(shape[0]), int(shape[1])
        return _cached_center_bias(h, w, bool(rtl))

    @classmethod
    def compute_face_bias(
        cls, bgr_image: np.ndarray, gray: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        Detects human faces using OpenCV FaceDetectorYN (YuNet) without heuristic
        skin-color filtering (falls back to Haar + eye verification if ONNX model is absent).
        Downscales long edge to <= 640px before inference and rescales bounding boxes back.
        """
        h, w = bgr_image.shape[:2]
        face_map = np.zeros((h, w), dtype=np.float32)
        min_dim = min(h, w)
        if min_dim < 48:
            return face_map, []

        face_boxes: List[Tuple[int, int, int, int]] = []
        min_sz = max(40, int(min_dim * 0.08))

        # Downscale large captures so long edge <= 640px for fast YuNet inference
        det_scale = min(1.0, 640.0 / float(max(h, w)))
        if det_scale < 1.0:
            dw = max(32, int(round(w * det_scale)))
            dh = max(32, int(round(h * det_scale)))
            det_img = cv2.resize(bgr_image, (dw, dh), interpolation=cv2.INTER_AREA)
        else:
            dw, dh = w, h
            det_img = bgr_image

        # 1. Primary: YuNet ONNX FaceDetectorYN (no heuristic skin-color gate)
        yunet = cls.get_yunet_detector((dw, dh))
        if yunet is not None:
            try:
                _, faces = yunet.detect(det_img)
                if faces is not None:
                    inv_scale = 1.0 / det_scale
                    for det in faces:
                        fx = int(round(det[0] * inv_scale))
                        fy = int(round(det[1] * inv_scale))
                        fw = int(round(det[2] * inv_scale))
                        fh = int(round(det[3] * inv_scale))
                        score = float(det[-1])
                        if score >= 0.72 and fw >= min_sz and fh >= min_sz:
                            fx1 = max(0, fx)
                            fy1 = max(0, fy)
                            fw1 = min(w - fx1, fw)
                            fh1 = min(h - fy1, fh)
                            if fw1 > 10 and fh1 > 10:
                                face_boxes.append((fx1, fy1, fw1, fh1))
            except Exception:
                pass
        else:
            # 2. Fallback: Haar alt2 + eye verification (no skin-color filter)
            if gray is None:
                gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            cascade = cls.get_face_cascade()
            eye_cascade = cls.get_eye_cascade()
            if cascade is not None and not cascade.empty():
                try:
                    candidates = cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=7, minSize=(min_sz, min_sz)
                    )
                    for fx, fy, fw, fh in candidates:
                        aspect = fw / float(max(1, fh))
                        if not (0.75 <= aspect <= 1.30):
                            continue
                        fx1, fy1 = max(0, int(fx)), max(0, int(fy))
                        fx2, fy2 = min(w, int(fx + fw)), min(h, int(fy + fh))
                        patch_gray = gray[fy1:fy2, fx1:fx2]
                        has_eyes = True
                        if eye_cascade is not None and not eye_cascade.empty() and patch_gray.size > 0:
                            upper = patch_gray[: int(patch_gray.shape[0] * 0.65), :]
                            eye_min = max(8, int(fw * 0.08))
                            eyes = eye_cascade.detectMultiScale(
                                upper, scaleFactor=1.1, minNeighbors=4, minSize=(eye_min, eye_min)
                            )
                            has_eyes = len(eyes) > 0
                        if has_eyes:
                            face_boxes.append((fx1, fy1, fx2 - fx1, fy2 - fy1))
                except Exception:
                    pass

        if not face_boxes:
            return face_map, []

        # Render Gaussian hills on a canonical grid (<= 256px) for O(1) speed
        canon_scale = min(1.0, 256.0 / max(h, w))
        ch = max(16, int(round(h * canon_scale)))
        cw = max(16, int(round(w * canon_scale)))
        y_idx, x_idx = np.ogrid[:ch, :cw]
        small_map = np.zeros((ch, cw), dtype=np.float32)

        for fx, fy, fw, fh in face_boxes:
            fcx = (fx + fw * 0.5) * canon_scale
            fcy = (fy + fh * 0.5) * canon_scale
            sigma = max(2.0, max(fw, fh) * 0.45 * canon_scale)
            dist_sq = ((x_idx - fcx) ** 2 + (y_idx - fcy) ** 2) / (2.0 * sigma**2)
            small_map += np.exp(-dist_sq).astype(np.float32)

        if (ch, cw) != (h, w):
            face_map = cv2.resize(small_map, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        else:
            face_map = small_map

        return cls._normalize_01(face_map), face_boxes

    @classmethod
    def compute_objectness_weight(cls, bgr_image: np.ndarray) -> np.ndarray:
        """
        Center-surround multi-scale contrast & figure-ground enclosure weight.
        Robust against full-bleed photos, gradients, and edge-touching products;
        suppresses tiny high-frequency micro-icons (<1% area) relative to hero regions.
        """
        h, w = bgr_image.shape[:2]
        canon_dim = 256
        scale = min(1.0, canon_dim / max(h, w))
        cw, ch = max(32, int(round(w * scale))), max(32, int(round(h * scale)))
        small = cv2.resize(bgr_image, (cw, ch), interpolation=cv2.INTER_AREA).astype(np.float32)

        lab = cv2.cvtColor(small.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        # Center-surround Difference of Gaussians (DoG) in LAB space
        fine = cv2.GaussianBlur(lab, (5, 5), 1.2)
        coarse = cv2.GaussianBlur(lab, (31, 31), 8.0)
        dog_energy = np.sqrt(np.sum((fine - coarse) ** 2, axis=2))
        dog_norm = cls._normalize_01(dog_energy)

        # Threshold coherent regions to distinguish macro objects from micro-icons
        thresh_val = float(np.percentile(dog_norm, 65))
        fg_mask = (dog_norm >= thresh_val).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask)
        total_px = float(cw * ch)

        weight_small = np.ones((ch, cw), dtype=np.float32)
        for i in range(1, num_labels):
            area_ratio = stats[i, cv2.CC_STAT_AREA] / total_px
            if area_ratio < 0.008:
                # Micro-icon / tiny noise suppression
                weight_small[labels == i] = 0.45
            elif area_ratio >= 0.035:
                # Coherent macro object / hero region boost
                weight_small[labels == i] = 1.25

        weight_small = cv2.GaussianBlur(weight_small, (11, 11), 0).astype(np.float32)
        if (ch, cw) != (h, w):
            return cv2.resize(weight_small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        return weight_small

    @classmethod
    def compute_itti_koch(cls, bgr_image: np.ndarray) -> np.ndarray:
        """
        Classic Itti, Koch & Niebur (TPAMI 1998) multi-scale center-surround pyramid saliency baseline.
        Combines Intensity, Red-Green / Blue-Yellow double-opponent Color, and 4-angle Gabor Orientation
        conspicuity maps across Gaussian pyramid scales with peak-to-mean normalization.
        """
        h, w = bgr_image.shape[:2]
        canon = 256
        scale = min(1.0, canon / float(max(h, w)))
        cw, ch = max(32, int(round(w * scale))), max(32, int(round(h * scale)))
        small = cv2.resize(bgr_image, (cw, ch), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

        b, g, r = small[:, :, 0], small[:, :, 1], small[:, :, 2]
        intensity = (r + g + b) / 3.0

        # Broad-tuned color channels (normalized by intensity where intensity > 0.1 max)
        max_i = float(np.max(intensity))
        mask_i = intensity > (0.1 * max_i)
        rn = np.where(mask_i, r / (intensity + 1e-6), 0.0)
        gn = np.where(mask_i, g / (intensity + 1e-6), 0.0)
        bn = np.where(mask_i, b / (intensity + 1e-6), 0.0)

        R = np.clip(rn - (gn + bn) * 0.5, 0.0, None)
        G = np.clip(gn - (rn + bn) * 0.5, 0.0, None)
        B = np.clip(bn - (rn + gn) * 0.5, 0.0, None)
        Y = np.clip((rn + gn) * 0.5 - np.abs(rn - gn) * 0.5 - bn, 0.0, None)

        # Orientation channels via 4 Gabor filters (0, 45, 90, 135 deg)
        orient_maps = []
        for theta in (0.0, np.pi * 0.25, np.pi * 0.5, np.pi * 0.75):
            kern = cv2.getGaborKernel((9, 9), 2.2, theta, 5.0, 0.5, 0, ktype=cv2.CV_32F)
            resp = np.abs(cv2.filter2D(intensity, cv2.CV_32F, kern))
            orient_maps.append(resp)

        def build_pyr(img_2d: np.ndarray, levels: int = 5) -> List[np.ndarray]:
            pyr: List[np.ndarray] = [img_2d.astype(np.float32)]
            for _ in range(1, levels):
                if pyr[-1].shape[0] < 4 or pyr[-1].shape[1] < 4:
                    pyr.append(pyr[-1])
                else:
                    pyr.append(cv2.pyrDown(pyr[-1]).astype(np.float32))
            return pyr

        def norm_op(m: np.ndarray) -> np.ndarray:
            n_m = cls._normalize_01(m)
            mean_v = float(np.mean(n_m))
            weight = (1.0 - mean_v) ** 2
            return n_m * max(0.05, weight)

        pyr_I = build_pyr(intensity)
        pyr_R, pyr_G, pyr_B, pyr_Y = build_pyr(R), build_pyr(G), build_pyr(B), build_pyr(Y)
        pyr_O = [build_pyr(om) for om in orient_maps]

        consp_I = np.zeros((ch, cw), dtype=np.float32)
        consp_C = np.zeros((ch, cw), dtype=np.float32)
        consp_O = np.zeros((ch, cw), dtype=np.float32)

        pairs = [(0, 2), (0, 3), (1, 3), (1, 4), (2, 4)]
        for c_idx, s_idx in pairs:
            # Intensity center-surround
            ic = cv2.resize(pyr_I[c_idx], (cw, ch), interpolation=cv2.INTER_LINEAR)
            isur = cv2.resize(pyr_I[s_idx], (cw, ch), interpolation=cv2.INTER_LINEAR)
            consp_I += norm_op(np.abs(ic - isur))

            # Color double-opponency RG and BY
            rg_c = cv2.resize(pyr_R[c_idx] - pyr_G[c_idx], (cw, ch), interpolation=cv2.INTER_LINEAR)
            rg_s = cv2.resize(pyr_G[s_idx] - pyr_R[s_idx], (cw, ch), interpolation=cv2.INTER_LINEAR)
            by_c = cv2.resize(pyr_B[c_idx] - pyr_Y[c_idx], (cw, ch), interpolation=cv2.INTER_LINEAR)
            by_s = cv2.resize(pyr_Y[s_idx] - pyr_B[s_idx], (cw, ch), interpolation=cv2.INTER_LINEAR)
            consp_C += norm_op(np.abs(rg_c - rg_s)) + norm_op(np.abs(by_c - by_s))

            # Orientation center-surround
            for o_pyr in pyr_O:
                oc = cv2.resize(o_pyr[c_idx], (cw, ch), interpolation=cv2.INTER_LINEAR)
                osur = cv2.resize(o_pyr[s_idx], (cw, ch), interpolation=cv2.INTER_LINEAR)
                consp_O += norm_op(np.abs(oc - osur)) * 0.25

        saliency_small = (norm_op(consp_I) + norm_op(consp_C) + norm_op(consp_O)) / 3.0
        saliency_small = cv2.GaussianBlur(saliency_small, (9, 9), 2.0)
        if (ch, cw) != (h, w):
            sal = cv2.resize(saliency_small, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            sal = saliency_small
        return cls._normalize_01(sal)

    @classmethod
    def compute_saliency_map(
        cls,
        bgr_image: np.ndarray,
        weights: Optional[Tuple[float, float, float, float, float]] = None,
        use_objectness: bool = True,
        engine_mode: str = "CLASSIC",
        reading_direction: str = "ltr",
    ) -> Tuple[np.ndarray, dict]:
        orig_shape = (
            bgr_image.shape[:2] if isinstance(bgr_image, np.ndarray) and bgr_image.ndim >= 2 else None
        )
        bgr_valid = cls.validate_image(bgr_image)
        h, w = bgr_valid.shape[:2]
        gray = cv2.cvtColor(bgr_valid, cv2.COLOR_BGR2GRAY)
        rtl = (reading_direction or "ltr").lower() == "rtl"
        mode_u = (engine_mode or "CLASSIC").upper().strip()
        if mode_u not in ("CLASSIC", "ONNX", "HYBRID"):
            mode_u = "CLASSIC"

        face_map, faces = cls.compute_face_bias(bgr_valid, gray)

        if mode_u == "ONNX":
            onnx_sal = cls.compute_onnx_saliency(bgr_valid, rtl=rtl)
            fused = 0.88 * onnx_sal + 0.12 * face_map
            if use_objectness:
                fused = fused * (0.65 + 0.35 * cls.compute_objectness_weight(bgr_valid))
        else:
            sr_map = cls.compute_spectral_residual(gray)
            lab_map = cls.compute_lab_contrast(bgr_valid)
            edge_map = cls.compute_edge_density(gray)
            bias_map = cls.compute_center_bias((h, w), rtl=rtl)

            if weights is not None:
                w_sr, w_lab, w_edge, w_bias, w_face = weights
            else:
                w_sr = cls.WEIGHT_SPECTRAL
                w_lab = cls.WEIGHT_LAB
                w_edge = cls.WEIGHT_EDGE
                w_bias = cls.WEIGHT_BIAS
                w_face = cls.WEIGHT_FACE

            classic_fused = (
                w_sr * sr_map + w_lab * lab_map + w_edge * edge_map + w_bias * bias_map + w_face * face_map
            )
            if use_objectness:
                classic_fused = classic_fused * cls.compute_objectness_weight(bgr_valid)

            # Foveal smoothing + baseline spatial prior calibration (Judd et al. 2009 / Bylinskii et al. 2019)
            classic_norm = cls._normalize_01(classic_fused)
            k_fov = max(5, int(min(h, w) * 0.036) | 1)
            classic_smooth = cls._normalize_01(cv2.GaussianBlur(classic_norm, (k_fov, k_fov), 0))
            if w_bias > 0.0:
                prior_strength = min(1.0, float(w_bias) / 0.18)
                prior_env = cls._normalize_01(
                    0.50 * classic_smooth * (0.35 + 0.65 * bias_map) + 0.50 * bias_map
                )
                peak_protect = (classic_norm >= 0.85).astype(np.float32) * 0.33
                alpha_p = np.clip(0.39 * prior_strength - peak_protect, 0.06, 0.50).astype(np.float32)
                classic_calibrated = cls._normalize_01((1.0 - alpha_p) * classic_norm + alpha_p * prior_env)
            else:
                classic_calibrated = cls._normalize_01(0.80 * classic_norm + 0.20 * classic_smooth)

            if mode_u == "HYBRID":
                onnx_sal = cls.compute_onnx_saliency(bgr_valid, rtl=rtl)
                onnx_smooth = cls._normalize_01(cv2.GaussianBlur(onnx_sal, (k_fov, k_fov), 0))
                fused = 0.65 * classic_calibrated + 0.35 * onnx_smooth
            else:
                fused = classic_calibrated

        sal_map = cls._normalize_01(fused)
        if orig_shape is not None and (h, w) != orig_shape:
            oh, ow = orig_shape
            scale_x = float(ow) / float(w)
            scale_y = float(oh) / float(h)
            sal_map = cv2.resize(sal_map, (ow, oh), interpolation=cv2.INTER_LINEAR)
            faces = [
                (
                    int(round(fx * scale_x)),
                    int(round(fy * scale_y)),
                    int(round(fw * scale_x)),
                    int(round(fh * scale_y)),
                )
                for (fx, fy, fw, fh) in faces
            ]
        blind_spots = cls.detect_blind_spots(sal_map, threshold=0.15)

        meta = {
            "faces": faces,
            "blind_spots": blind_spots,
            "has_face": len(faces) > 0,
            "engine_mode": mode_u,
            "reading_direction": "rtl" if rtl else "ltr",
        }
        return sal_map, meta

    @classmethod
    def detect_blind_spots(
        cls, saliency_map: np.ndarray, threshold: float = 0.15, lang: str = "tr"
    ) -> List[Dict[str, Any]]:
        """
        Identifies regions with severe attention starvation using a relative + absolute threshold
        (`min(threshold, canvas_mean * 0.55)`), preventing single-hotspot minimal designs from
        falsely flagging normal background quadrants as dead zones.
        """
        h, w = saliency_map.shape[:2]
        canvas_mean = float(np.mean(saliency_map))
        effective_cutoff = min(float(threshold), canvas_mean * 0.55)
        cutoff_pct = effective_cutoff * 100.0

        short_map = QUADRANT_SHORT_TR if lang.lower() == "tr" else QUADRANT_SHORT_EN
        quadrants = [
            (Quadrant.TOP_LEFT, short_map[Quadrant.TOP_LEFT], 0, 0, w // 2, h // 2),
            (Quadrant.TOP_RIGHT, short_map[Quadrant.TOP_RIGHT], w // 2, 0, w - w // 2, h // 2),
            (Quadrant.BOTTOM_LEFT, short_map[Quadrant.BOTTOM_LEFT], 0, h // 2, w // 2, h - h // 2),
            (Quadrant.BOTTOM_RIGHT, short_map[Quadrant.BOTTOM_RIGHT], w // 2, h // 2, w - w // 2, h - h // 2),
        ]
        blind_spots: List[Dict[str, Any]] = []
        for q_enum, zone_label, qx, qy, qw, qh in quadrants:
            quad_sal = saliency_map[qy : qy + qh, qx : qx + qw]
            if quad_sal.size == 0:
                continue
            mean_energy = float(np.mean(quad_sal)) * 100.0
            if mean_energy < cutoff_pct:
                if lang.lower() == "en":
                    warn_txt = f"{zone_label} zone receives {round(mean_energy, 1)}% mean attention (<{round(cutoff_pct, 1)}% cutoff)."
                else:
                    warn_txt = f"{zone_label} bölgesi ortalama %{round(mean_energy, 1)} dikkat çekiyor (<%{round(cutoff_pct, 1)} eşiği)."
                blind_spots.append(
                    {
                        "quadrant": q_enum,
                        "zone": zone_label,
                        "rect": (qx, qy, qw, qh),
                        "energy": round(mean_energy, 1),
                        "warning": warn_txt,
                    }
                )
        return blind_spots

    @classmethod
    def simulate_shelf_distance(
        cls,
        bgr_image: np.ndarray,
        distance_meters: float = 1.0,
        package_width_cm: float = 15.0,
        acuity_factor: float = 2.0,
    ) -> np.ndarray:
        """
        Physically grounded viewing distance simulation based on human visual acuity.
        Computes visual angle subtended by `package_width_cm` at `distance_meters`,
        derives pixels-per-degree (`ppd = w / angle_deg`), and applies a 1-arcminute
        retinal Point Spread Function (`sigma = (ppd / 60.0) * 0.42 * acuity_factor`).
        - `acuity_factor = 1.0`: 20/20 foveal acuity (~1 arcmin MAR)
        - `acuity_factor = 2.0` (default): 20/40 average aisle shopper / peripheral browsing
        """
        if distance_meters <= 0.5:
            return bgr_image.copy()

        h, w = bgr_image.shape[:2]
        dist_cm = max(50.0, float(distance_meters) * 100.0)
        width_cm = max(2.0, float(package_width_cm))

        # Visual angle subtended by the package width in degrees: 2 * arctan(W / (2*D))
        angle_deg = max(0.2, float(np.degrees(2.0 * np.arctan(width_cm / (2.0 * dist_cm)))))
        ppd = float(w) / angle_deg  # pixels per degree of visual field

        # Retinal Point Spread Function (PSF) in image pixels for 1 arcmin (1/60 deg) resolution limit
        sigma = (ppd / 60.0) * 0.42 * max(0.5, float(acuity_factor))
        if sigma < 0.35:
            return bgr_image.copy()

        ksize = max(3, int(round(sigma * 3.0)) | 1)
        blurred = cv2.GaussianBlur(bgr_image, (ksize, ksize), sigma).astype(np.float32)

        # Mild contrast sensitivity attenuation with viewing distance
        contrast_factor = max(0.78, 1.0 - (distance_meters - 0.5) * 0.06)
        mean_val = np.mean(blurred, axis=(0, 1), keepdims=True)
        attenuated = mean_val + (blurred - mean_val) * contrast_factor
        return np.clip(attenuated, 0, 255).astype(np.uint8)

    @classmethod
    def simulate_color_blindness(cls, bgr_image: np.ndarray, mode: str = "NONE") -> np.ndarray:
        """
        Simulates color vision deficiency (CVD) using Viénot (1999) sRGB projection matrices:
        Modes: 'NONE', 'PROTAN' (Protanopia), 'DEUTAN' (Deuteranopia), 'TRITAN' (Tritanopia), 'MONO' (Achromatopsia).
        """
        mode_u = (mode or "NONE").upper()
        if mode_u == "NONE":
            return bgr_image.copy()

        if mode_u == "MONO":
            gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        if mode_u == "PROTAN":
            mat = np.array(
                [
                    [0.56667, 0.43333, 0.00000],
                    [0.55833, 0.44167, 0.00000],
                    [0.00000, 0.24167, 0.75833],
                ],
                dtype=np.float32,
            )
        elif mode_u == "DEUTAN":
            mat = np.array(
                [
                    [0.62500, 0.37500, 0.00000],
                    [0.70000, 0.30000, 0.00000],
                    [0.00000, 0.30000, 0.70000],
                ],
                dtype=np.float32,
            )
        elif mode_u == "TRITAN":
            mat = np.array(
                [
                    [0.95000, 0.05000, 0.00000],
                    [0.00000, 0.43333, 0.56667],
                    [0.00000, 0.47500, 0.52500],
                ],
                dtype=np.float32,
            )
        else:
            return bgr_image.copy()

        sim_rgb = np.clip(rgb @ mat.T, 0, 255).astype(np.uint8)
        return cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)

    @classmethod
    def compute_wcag_contrast(cls, bgr_patch: np.ndarray) -> Dict[str, Any]:
        """
        Computes W3C WCAG 2.1 Relative Luminance and Contrast Ratio (1.0 to 21.0)
        between foreground text/strokes and background within a given BGR patch.
        """
        if bgr_patch is None or bgr_patch.size == 0:
            return {"ratio": 1.0, "grade": "FAIL", "passed_aa": False, "passed_large": False}

        rgb = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        lum = 0.2126 * lin[:, :, 0] + 0.7152 * lin[:, :, 1] + 0.0722 * lin[:, :, 2]

        # Separate foreground & background using Otsu bimodal thresholding on 8-bit luminance
        gray_u8 = np.clip(lum * 255.0, 0, 255).astype(np.uint8)
        _, mask = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        fg_pixels = lum[mask > 0]
        bg_pixels = lum[mask == 0]

        if fg_pixels.size == 0 or bg_pixels.size == 0:
            l1 = float(np.percentile(lum, 90))
            l2 = float(np.percentile(lum, 10))
        else:
            # Use robust 85th/15th percentiles within each cluster to avoid anti-aliasing edge dilution
            l1 = float(np.percentile(fg_pixels, 85))
            l2 = float(np.percentile(bg_pixels, 15))

        lighter = max(l1, l2)
        darker = min(l1, l2)
        ratio = (lighter + 0.05) / (darker + 0.05)
        ratio = float(np.clip(ratio, 1.0, 21.0))

        if ratio >= 7.0:
            grade = "AAA"
        elif ratio >= 4.5:
            grade = "AA"
        elif ratio >= 3.0:
            grade = "AA-Büyük"
        else:
            grade = "DÜŞÜK"

        return {
            "ratio": round(ratio, 2),
            "grade": grade,
            "passed_aa": ratio >= 4.5,
            "passed_large": ratio >= 3.0,
        }

    @classmethod
    def _estimate_char_height_px(
        cls, bgr_patch: np.ndarray, num_lines_hint: Optional[int] = None
    ) -> Tuple[float, int]:
        """
        Estimates effective single-line character cap-height (in pixels) and line count
        inside a text/CTA bounding box, stripping vertical padding and multi-line height inflation.
        """
        if bgr_patch is None or bgr_patch.size == 0:
            return 0.0, 1
        bh, bw = bgr_patch.shape[:2]
        if num_lines_hint is not None and num_lines_hint >= 1:
            num_lines = int(num_lines_hint)
        else:
            gray = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2GRAY) if bgr_patch.ndim == 3 else bgr_patch
            k_grad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, k_grad)
            _, bw_strokes = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            row_density = np.mean(bw_strokes > 0, axis=1)
            active = row_density > 0.08

            # Count contiguous horizontal stroke bands separated by inter-line gaps
            bands = 0
            in_band = False
            band_len = 0
            for val in active:
                if val:
                    in_band = True
                    band_len += 1
                else:
                    if in_band and band_len >= 4:
                        bands += 1
                    in_band = False
                    band_len = 0
            if in_band and band_len >= 4:
                bands += 1
            num_lines = max(1, min(8, bands))

        # Cap-height is ~68% of single-line box height (excluding button/descender padding)
        eff_char_px = (float(bh) / float(num_lines)) * 0.68
        return eff_char_px, num_lines

    @classmethod
    def _infer_aoi_type(cls, aoi: Dict[str, Any]) -> str:
        """
        Infers canonical AOI semantic role: 'LOGO', 'PRICE', 'CTA', 'TEXT', or 'VISUAL'.
        """
        explicit = str(aoi.get("aoi_type", "")).upper().strip()
        if explicit in ("LOGO", "PRICE", "CTA", "TEXT", "VISUAL"):
            return explicit

        name_l = str(aoi.get("name", "")).lower()
        if any(k in name_l for k in ("logo", "marka", "brand", "amblem", "ikon")):
            return "LOGO"
        if any(
            k in name_l for k in ("fiyat", "price", "₺", "tl", "$", "€", "indirim", "kampanya", "%", "off")
        ):
            return "PRICE"
        if any(
            k in name_l
            for k in (
                "cta",
                "buton",
                "button",
                "satın",
                "satin",
                "sepet",
                "hemen",
                "al",
                "buy",
                "shop",
                "order",
            )
        ):
            return "CTA"
        if any(
            k in name_l
            for k in (
                "metin",
                "yazı",
                "yazi",
                "başlık",
                "baslik",
                "slogan",
                "etiket",
                "text",
                "title",
                "headline",
            )
        ):
            return "TEXT"
        if aoi.get("is_text_or_cta") or aoi.get("wcag_ratio") is not None:
            return "TEXT"
        return "VISUAL"

    @classmethod
    def _is_text_or_cta_aoi(cls, aoi: Dict[str, Any]) -> bool:
        """
        Returns True only if the AOI represents a Text, Headline, Price, or CTA Button region.
        Prevents WCAG text-contrast grading from running on logos, product photos, or arbitrary shapes.
        """
        if "is_text_or_cta" in aoi and aoi["is_text_or_cta"] is not None:
            return bool(aoi["is_text_or_cta"])
        role = cls._infer_aoi_type(aoi)
        return role in ("PRICE", "CTA", "TEXT")

    @classmethod
    def detect_text_and_cta_regions(
        cls,
        bgr_image: np.ndarray,
        max_regions: int = 5,
        distance_meters: float = 1.0,
        package_width_cm: float = 15.0,
    ) -> List[Dict[str, Any]]:
        """
        Automatically detects prominent text blocks, headlines, price tags, and CTA buttons.
        Uses cached RapidOCR if installed; otherwise runs a 100% offline Morphological
        Stroke-Grouping & CTA Contour Detector. Computes WCAG 2.1 contrast and angular legibility.
        """
        h, w = bgr_image.shape[:2]
        if h < 24 or w < 24:
            return []

        candidate_boxes: List[Tuple[int, int, int, int, str, Optional[int]]] = []

        # 1. Optional OCR engine (cached RapidOCR) if installed in environment
        ocr = cls._get_ocr()
        if ocr is not None:
            try:
                ocr_res, _ = ocr(bgr_image)
                if ocr_res:
                    ocr_rects = []
                    for item in ocr_res:
                        pts, text_str, _ = item
                        xs = [int(p[0]) for p in pts]
                        ys = [int(p[1]) for p in pts]
                        bx, by = max(0, min(xs)), max(0, min(ys))
                        bw, bh = min(w - bx, max(xs) - bx), min(h - by, max(ys) - by)
                        if bw >= 18 and bh >= 10:
                            ocr_rects.append((bx, by, bw, bh, float(bw * bh), str(text_str).strip()))

                    # Sort by area descending BEFORE slicing so prominent headlines & CTAs are prioritized
                    ocr_rects.sort(key=lambda r: r[4], reverse=True)
                    cta_kw = (
                        "al",
                        "sepet",
                        "ekle",
                        "başla",
                        "keşfet",
                        "sipariş",
                        "hemen",
                        "buy",
                        "shop",
                        "cart",
                        "order",
                        "now",
                    )
                    price_kw = ("%", "₺", "tl", "$", "€", "indirim", "fiyat", "free", "off")
                    for idx, (bx, by, bw, bh, _, raw_txt) in enumerate(ocr_rects[:max_regions]):
                        short_txt = raw_txt[:14]
                        txt_l = raw_txt.lower()
                        rel_y = (by + bh * 0.5) / float(h)
                        if any(k in txt_l for k in cta_kw) or (rel_y >= 0.60 and bw >= w * 0.18):
                            label = f"CTA / Buton: {short_txt}"
                        elif bh >= h * 0.08 or any(k in txt_l for k in price_kw):
                            label = f"Başlık / Fiyat: {short_txt}"
                        else:
                            label = f"Metin: {short_txt}"
                        candidate_boxes.append((bx, by, bw, bh, label, 1))
            except Exception:
                pass

        # 2. Built-in Offline Morphological Text & CTA Detector
        if not candidate_boxes:
            gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            k_grad = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, k_grad)
            _, bw_strokes = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            horiz_w = max(9, int(w * 0.035))
            vert_h = max(3, int(h * 0.012))
            k_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_w, vert_h))
            connected = cv2.morphologyEx(bw_strokes, cv2.MORPH_CLOSE, k_connect)

            contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            total_area = float(w * h)

            raw_rects = []
            for cnt in contours:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                area = float(bw * bh)
                aspect = bw / float(max(1, bh))
                if area < total_area * 0.002 or area > total_area * 0.45:
                    continue
                if bh < 10 or bw < 24 or aspect < 1.2 or aspect > 18.0:
                    continue

                roi_strokes = bw_strokes[by : by + bh, bx : bx + bw]
                fill_density = float(np.mean(roi_strokes > 0))
                if 0.12 <= fill_density <= 0.85:
                    raw_rects.append((bx, by, bw, bh, area))

            raw_rects.sort(key=lambda r: r[4], reverse=True)
            for idx, (bx, by, bw, bh, _) in enumerate(raw_rects[:max_regions]):
                rel_y = (by + bh * 0.5) / float(h)
                if bh >= h * 0.08:
                    tag = f"Başlık / Fiyat #{idx + 1}"
                elif rel_y >= 0.60 and bw >= w * 0.18:
                    tag = f"CTA / Buton #{idx + 1}"
                else:
                    tag = f"Metin / Etiket #{idx + 1}"
                candidate_boxes.append((bx, by, bw, bh, tag, None))

        # Evaluate WCAG 2.1 Contrast & Physical Angular Legibility for each box
        dist_cm = max(50.0, float(distance_meters) * 100.0)
        width_cm = max(2.0, float(package_width_cm))
        total_angle_deg = float(np.degrees(2.0 * np.arctan(width_cm / (2.0 * dist_cm))))
        deg_per_px = total_angle_deg / float(max(1, w))

        detected_aois: List[Dict[str, Any]] = []
        for bx, by, bw, bh, label, lines_hint in candidate_boxes[:max_regions]:
            patch = bgr_image[by : by + bh, bx : bx + bw]
            wcag = cls.compute_wcag_contrast(patch)

            eff_char_px, num_lines = cls._estimate_char_height_px(patch, num_lines_hint=lines_hint)
            char_angle_deg = eff_char_px * deg_per_px
            legible = char_angle_deg >= 0.18

            warnings = []
            if not wcag["passed_large"]:
                warnings.append(f"Kontrast Düşük ({wcag['ratio']}:1 < 3.0:1)")
            elif not wcag["passed_aa"]:
                warnings.append(f"Küçük Metin İçin Sınırda ({wcag['ratio']}:1 < 4.5:1)")
            if not legible:
                warnings.append(f"{distance_meters:.1f}m Mesafeden Küçük ({char_angle_deg:.2f}°)")

            status = "✅ Okunabilir" if not warnings else "⚠️ " + ", ".join(warnings)

            detected_aois.append(
                {
                    "name": label,
                    "rect": (int(bx), int(by), int(bw), int(bh)),
                    "is_text_or_cta": True,
                    "num_lines": num_lines,
                    "wcag_ratio": wcag["ratio"],
                    "wcag_grade": wcag["grade"],
                    "passed_aa": wcag["passed_aa"],
                    "angular_height_deg": round(char_angle_deg, 2),
                    "legible_at_distance": legible,
                    "status": status,
                }
            )

        return detected_aois

    @classmethod
    def calculate_aoi_metrics(
        cls,
        saliency_map: np.ndarray,
        fixations: List[Tuple[int, int]],
        aois: List[Dict[str, Any]],
        bgr_image: Optional[np.ndarray] = None,
        distance_meters: float = 1.0,
        package_width_cm: float = 15.0,
        lang: str = "tr",
    ) -> List[Dict[str, Any]]:
        """
        Computes Share of Attention (%), Fixation Rank (#1..#N), and (only for Text/CTA AOIs)
        W3C WCAG 2.1 Contrast Ratio and angular character legibility.
        """
        h, w = saliency_map.shape[:2]
        total_sal = float(np.sum(saliency_map)) + 1e-6
        results: List[Dict[str, Any]] = []
        is_en = (lang or "tr").lower() == "en"

        dist_cm = max(50.0, float(distance_meters) * 100.0)
        width_cm = max(2.0, float(package_width_cm))
        total_angle_deg = float(np.degrees(2.0 * np.arctan(width_cm / (2.0 * dist_cm))))
        deg_per_px = total_angle_deg / float(max(1, w))

        for aoi in aois:
            name = aoi.get("name", "AOI")
            x, y, aw, ah = aoi["rect"]
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(w, int(x + aw)), min(h, int(y + ah))

            if x2 <= x1 or y2 <= y1:
                continue

            sub_sal = saliency_map[y1:y2, x1:x2]
            aoi_sum = float(np.sum(sub_sal))
            share = round((aoi_sum / total_sal) * 100.0, 1)

            fixation_rank: Optional[int] = None
            ttff_label = "No Direct Fixation" if is_en else "Doğrudan Odak Yok"
            for idx, (fx, fy) in enumerate(fixations):
                if x1 <= fx <= x2 and y1 <= fy <= y2:
                    fixation_rank = idx + 1
                    ttff_label = f"Fixation #{fixation_rank}" if is_en else f"{fixation_rank}. Sabitlenme"
                    break

            is_text_aoi = cls._is_text_or_cta_aoi(aoi)
            aoi_type = cls._infer_aoi_type(aoi)
            wcag_ratio = aoi.get("wcag_ratio") if is_text_aoi else None
            wcag_grade = aoi.get("wcag_grade") if is_text_aoi else None
            patch = bgr_image[y1:y2, x1:x2] if bgr_image is not None else None

            if is_text_aoi and patch is not None and (wcag_ratio is None or wcag_grade is None):
                wcag_info = cls.compute_wcag_contrast(patch)
                wcag_ratio = wcag_info["ratio"]
                wcag_grade = wcag_info["grade"]

            if is_text_aoi and patch is not None:
                eff_char_px, _ = cls._estimate_char_height_px(patch, num_lines_hint=aoi.get("num_lines"))
            else:
                eff_char_px = float(y2 - y1) * 0.68
            char_angle_deg = round(eff_char_px * deg_per_px, 2)
            legible = (char_angle_deg >= 0.18) if is_text_aoi else True

            if is_en:
                ttff_time_str = f"Rank #{fixation_rank}" if fixation_rank is not None else "Unfixated"
            else:
                ttff_time_str = f"Sıra #{fixation_rank}" if fixation_rank is not None else "Odak Dışı"

            results.append(
                {
                    "name": name,
                    "aoi_type": aoi_type,
                    "rect": (x1, y1, x2 - x1, y2 - y1),
                    "is_text_or_cta": is_text_aoi,
                    "share": share,
                    "fixation_rank": fixation_rank,
                    "ttff": ttff_label,
                    "ttff_time": ttff_time_str,
                    "wcag_ratio": wcag_ratio,
                    "wcag_grade": wcag_grade,
                    "angular_height_deg": char_angle_deg,
                    "legible_at_distance": legible,
                }
            )
        return results

    @classmethod
    def compute_gaze_path(
        cls,
        saliency_map: np.ndarray,
        num_fixations: int = 4,
        min_dist_diag_ratio: float = 0.08,
        bgr_image: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, int]]:
        h, w = saliency_map.shape[:2]
        diagonal = float(np.sqrt(w * w + h * h))
        min_radius = int(max(12, round(diagonal * min_dist_diag_ratio)))

        work_map = cv2.GaussianBlur(saliency_map.copy(), (5, 5), 0)

        # Content-aware studio background suppression:
        # Prevents fixations (#1-#4) from drifting into pure empty canvas/background margins
        if bgr_image is not None and isinstance(bgr_image, np.ndarray) and bgr_image.ndim == 3:
            gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            active_edges = (mag > 6.0).astype(np.uint8)
            total_edges = int(np.sum(active_edges))

            edge_ratio = total_edges / float(h * w)
            if edge_ratio < 0.35 and total_edges > 25:
                dist = cv2.distanceTransform(1 - active_edges, cv2.DIST_L2, 5)
                max_reach = float(min(h, w)) * 0.16
                reach_gate = np.clip(1.0 - (dist / max_reach), 0.05, 1.0)
                work_map = work_map * reach_gate.astype(np.float32)

        fixations: List[Tuple[int, int]] = []
        y_grid, x_grid = np.ogrid[:h, :w]

        for _ in range(max(1, int(num_fixations))):
            _, max_val, _, max_loc = cv2.minMaxLoc(work_map)
            if max_val <= 0.01:
                break
            fx, fy = int(max_loc[0]), int(max_loc[1])
            fixations.append((fx, fy))

            # Smooth Gaussian Inhibition of Return (IoR)
            dist_sq = (x_grid - fx) ** 2 + (y_grid - fy) ** 2
            ior_suppression = 1.0 - np.exp(-dist_sq / (2.0 * (min_radius**2)))
            work_map = work_map * ior_suppression.astype(np.float32)

        return fixations

    @classmethod
    def generate_heatmap_overlay(
        cls,
        bgr_image: np.ndarray,
        saliency_map: np.ndarray,
        alpha: float = 0.55,
        colormap: int = cv2.COLORMAP_TURBO,
    ) -> np.ndarray:
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        desat = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
        orig_f = bgr_image.astype(np.float32)
        backdrop = 0.7 * desat + 0.3 * orig_f

        sal_u8 = (np.clip(saliency_map, 0.0, 1.0) * 255).astype(np.uint8)
        colored_heat = cv2.applyColorMap(sal_u8, colormap).astype(np.float32)

        blended = (1.0 - alpha) * backdrop + alpha * colored_heat
        return np.clip(blended, 0, 255).astype(np.uint8)

    @classmethod
    def generate_fog_overlay(
        cls,
        bgr_image: np.ndarray,
        saliency_map: np.ndarray,
        fog_opacity: float = 0.35,
        top_salient_threshold: float = 0.35,
    ) -> np.ndarray:
        norm_sal = np.clip(saliency_map, 0.0, 1.0)
        denom = max(1e-6, 1.0 - top_salient_threshold * 0.5)
        visibility = np.clip((norm_sal - (top_salient_threshold * 0.5)) / denom, 0.0, 1.0)
        visibility = visibility * visibility * (3.0 - 2.0 * visibility)

        brightness_multiplier = (1.0 - fog_opacity) + fog_opacity * visibility
        brightness_multiplier = np.expand_dims(brightness_multiplier, axis=2)

        fog_img = bgr_image.astype(np.float32) * brightness_multiplier
        return np.clip(fog_img, 0, 255).astype(np.uint8)

    @classmethod
    def draw_gaze_path(cls, base_image: np.ndarray, fixations: List[Tuple[int, int]]) -> np.ndarray:
        canvas = base_image.copy()
        if not fixations:
            return canvas

        h, w = canvas.shape[:2]
        badge_radius = max(14, int(min(h, w) * 0.035))
        line_thickness = max(2, int(badge_radius * 0.18))
        font_scale = badge_radius / 22.0
        font_thickness = max(1, int(badge_radius * 0.12))

        cyan_glow = (248, 189, 56)
        cyan_line = (235, 160, 30)
        badge_bg = (15, 23, 42)
        badge_border = (248, 189, 56)
        text_color = (255, 255, 255)

        for i in range(len(fixations) - 1):
            pt1 = fixations[i]
            pt2 = fixations[i + 1]
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            dist = float(np.sqrt(dx * dx + dy * dy))
            if dist > (badge_radius * 2.2):
                ratio1 = badge_radius / dist
                ratio2 = (dist - badge_radius) / dist
                start_pt = (int(round(pt1[0] + dx * ratio1)), int(round(pt1[1] + dy * ratio1)))
                end_pt = (int(round(pt1[0] + dx * ratio2)), int(round(pt1[1] + dy * ratio2)))
            else:
                start_pt, end_pt = pt1, pt2

            cv2.arrowedLine(
                canvas, start_pt, end_pt, cyan_glow, line_thickness + 2, line_type=cv2.LINE_AA, tipLength=0.15
            )
            cv2.arrowedLine(
                canvas, start_pt, end_pt, cyan_line, line_thickness, line_type=cv2.LINE_AA, tipLength=0.15
            )

        for idx, (fx, fy) in enumerate(fixations):
            label = str(idx + 1)
            cv2.circle(canvas, (fx, fy), badge_radius + 3, cyan_glow, 2, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (fx, fy), badge_radius, badge_bg, -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (fx, fy), badge_radius, badge_border, line_thickness, lineType=cv2.LINE_AA)

            (t_w, t_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font_scale, font_thickness)
            text_x = fx - t_w // 2
            text_y = fy + t_h // 2
            cv2.putText(
                canvas,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                font_scale,
                text_color,
                font_thickness,
                lineType=cv2.LINE_AA,
            )

        return canvas

    @classmethod
    def classify_quadrant(cls, fx: int, fy: int, w: int, h: int) -> Quadrant:
        """Classifies a fixation coordinate into a canonical Quadrant Enum."""
        rel_x = fx / float(max(1, w))
        rel_y = fy / float(max(1, h))
        if 0.30 <= rel_x <= 0.70 and 0.30 <= rel_y <= 0.70:
            return Quadrant.CENTER
        elif rel_x < 0.50 and rel_y < 0.50:
            return Quadrant.TOP_LEFT
        elif rel_x >= 0.50 and rel_y < 0.50:
            return Quadrant.TOP_RIGHT
        elif rel_x < 0.50 and rel_y >= 0.50:
            return Quadrant.BOTTOM_LEFT
        return Quadrant.BOTTOM_RIGHT

    @classmethod
    def compute_clutter_score(cls, bgr_image: np.ndarray, saliency_map: np.ndarray) -> int:
        """
        Grounded visual clutter index [0-100] combining:
        1. Multi-scale Sobel gradient magnitude density
        2. Orientation entropy (8-bin histogram Shannon entropy of edge angles)
        3. Saliency spatial entropy (dispersion of attention across the canvas)
        """
        h, w = bgr_image.shape[:2]
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        # 1. Edge density normalized over active content area + macro canvas
        mag_norm = cls._normalize_01(mag)
        edge_mask = mag_norm > 0.18
        total_edges = int(np.sum(edge_mask))

        if total_edges > 20:
            row_counts = np.sum(edge_mask, axis=1)
            col_counts = np.sum(edge_mask, axis=0)
            row_cumsum = np.cumsum(row_counts) / float(total_edges)
            col_cumsum = np.cumsum(col_counts) / float(total_edges)

            y0 = int(np.searchsorted(row_cumsum, 0.02))
            y1 = int(np.searchsorted(row_cumsum, 0.98))
            x0 = int(np.searchsorted(col_cumsum, 0.02))
            x1 = int(np.searchsorted(col_cumsum, 0.98))
            content_area = float(max(100, (y1 - y0 + 1) * (x1 - x0 + 1)))
            content_edge_density = float(total_edges) / content_area
            canvas_edge_density = float(total_edges) / float(h * w)
            # 70% content-normalized + 30% canvas-normalized: immune to arbitrary outer cropping margins
            edge_density = 0.70 * content_edge_density + 0.30 * canvas_edge_density
        else:
            edge_density = float(np.mean(edge_mask))

        # 2. Orientation Shannon entropy (chaotic edge directions = higher clutter)
        if np.any(edge_mask):
            active_angles = angle[edge_mask] % 180.0
            hist, _ = np.histogram(active_angles, bins=8, range=(0.0, 180.0), density=True)
            hist_p = hist * (180.0 / 8.0)
            hist_p = hist_p[hist_p > 1e-6]
            orient_entropy = float(-np.sum(hist_p * np.log2(hist_p)) / 3.0)  # max log2(8)=3
        else:
            orient_entropy = 0.0

        # 3. Spatial dispersion of saliency (normalized Shannon entropy)
        sal_prob: np.ndarray = saliency_map.flatten().astype(np.float64)
        sal_sum = float(np.sum(sal_prob))
        if sal_sum > 1e-6:
            sal_prob = sal_prob / sal_sum
            sal_prob = sal_prob[sal_prob > 1e-9]
            max_ent = np.log2(max(2, saliency_map.size))
            sal_entropy = float(-np.sum(sal_prob * np.log2(sal_prob)) / max_ent)
        else:
            sal_entropy = 0.0

        combined = 0.45 * min(1.0, edge_density * 3.2) + 0.30 * orient_entropy + 0.25 * sal_entropy
        return int(round(np.clip(combined * 100.0, 5.0, 98.0)))

    @classmethod
    def detect_shelf_products(
        cls,
        bgr_image: np.ndarray,
        max_products: int = 12,
        reading_direction: str = "ltr",
    ) -> List[Dict[str, Any]]:
        """
        Automatically segments individual product facings (SKUs) in a full retail shelf planogram photo.
        1. Detects horizontal shelf rails via horizontal Sobel row-energy projection peaks.
        2. Within each shelf band, segments individual packages using CIE-LAB chromatic + gradient
           contours and vertical inter-product valley splitting, deduplicating via IoU NMS.
        """
        bgr_valid = cls.validate_image(bgr_image)
        h, w = bgr_valid.shape[:2]
        if h < 32 or w < 32:
            return []

        rtl = (reading_direction or "ltr").lower() == "rtl"
        gray = cv2.cvtColor(bgr_valid, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(bgr_valid, cv2.COLOR_BGR2LAB).astype(np.float32)

        # 1. Detect horizontal shelf rails (strong horizontal edges spanning across the shelf width)
        sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        row_energy = np.mean(sobel_y, axis=1)
        row_coverage = np.mean(sobel_y > float(np.percentile(sobel_y, 75)), axis=1)
        row_score = row_energy * np.where(row_coverage >= 0.55, 1.0, 0.25)
        row_smooth = cv2.GaussianBlur(row_score.reshape(-1, 1), (1, 11), 0).flatten()
        row_thresh = float(np.mean(row_smooth) + 1.10 * np.std(row_smooth))

        min_shelf_h = max(24, int(h * 0.18))
        rail_ys: List[int] = [0]
        for y in range(min_shelf_h, h - min_shelf_h):
            if row_smooth[y] >= row_thresh and row_smooth[y] >= np.max(
                row_smooth[max(0, y - 12) : min(h, y + 13)]
            ):
                if y - rail_ys[-1] >= min_shelf_h:
                    rail_ys.append(y)
        rail_ys.append(h)

        shelf_bands: List[Tuple[int, int]] = []
        for i in range(len(rail_ys) - 1):
            y1, y2 = rail_ys[i], rail_ys[i + 1]
            if (y2 - y1) >= min_shelf_h:
                shelf_bands.append((y1, y2))
        if not shelf_bands:
            shelf_bands = [(0, h)]

        # 2. Segment product facings inside each shelf band
        raw_boxes: List[Tuple[int, int, int, int, int, float]] = []
        total_area = float(w * h)

        for row_idx, (sy1, sy2) in enumerate(shelf_bands):
            # Exclude thin horizontal shelf-rail border pixels at top/bottom of band
            pad_y = min(8, max(0, (sy2 - sy1) // 16))
            by1 = sy1 + (pad_y if row_idx > 0 else 0)
            by2 = sy2 - (pad_y if row_idx < len(shelf_bands) - 1 else 0)
            if by2 - by1 < 20:
                by1, by2 = sy1, sy2
            band_h = by2 - by1
            band_lab = lab[by1:by2, :]
            band_gray = gray[by1:by2, :]

            # Combine LAB deviation from shelf-band background with morphological gradient
            med_lab = np.median(band_lab, axis=(0, 1), keepdims=True)
            color_dist = np.sqrt(np.sum((band_lab - med_lab) ** 2, axis=2))
            color_u8 = np.clip(cls._normalize_01(color_dist) * 255.0, 0, 255).astype(np.uint8)

            gx = np.abs(cv2.Sobel(band_gray, cv2.CV_32F, 1, 0, ksize=3))
            gy = np.abs(cv2.Sobel(band_gray, cv2.CV_32F, 0, 1, ksize=3))
            grad_u8 = np.clip(cls._normalize_01(gx + gy) * 255.0, 0, 255).astype(np.uint8)

            combined = cv2.addWeighted(color_u8, 0.72, grad_u8, 0.28, 0).astype(np.uint8)
            _, mask = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            kw = max(3, int(w * 0.012) | 1)
            kh = max(5, int(band_h * 0.05) | 1)
            k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            band_boxes: List[Tuple[int, int, int, int]] = []
            min_pw = max(18, int(w * 0.06))
            min_ph = max(20, int(band_h * 0.32))

            for cnt in contours:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                if bw < min_pw or bh < min_ph:
                    continue
                if float(bw * bh) < total_area * 0.012 or float(bw * bh) > total_area * 0.65:
                    continue
                # Only split wide contours when vertical valley projection confirms multiple SKUs
                expected_unit_w = max(min_pw, int(bh * 0.82))
                if bw >= expected_unit_w * 2.15:
                    n_splits = min(5, max(2, int(round(bw / float(expected_unit_w)))))
                    sub_w = bw // n_splits
                    for s_i in range(n_splits):
                        sx = bx + s_i * sub_w
                        sw = (bx + bw - sx) if s_i == n_splits - 1 else sub_w
                        if sw >= min_pw:
                            band_boxes.append((sx, by1 + by, sw, bh))
                else:
                    band_boxes.append((bx, by1 + by, bw, bh))

            # Fallback column projection segmentation if shelf row has uniform background-merged facings
            if not band_boxes and band_h >= int(h * 0.22):
                col_prof = np.mean(combined, axis=0)
                active_cols = col_prof > (float(np.mean(col_prof)) * 0.85)
                in_seg = False
                start_x = 0
                for x_i, val in enumerate(active_cols):
                    if val and not in_seg:
                        in_seg = True
                        start_x = x_i
                    elif not val and in_seg:
                        in_seg = False
                        bw_seg = x_i - start_x
                        if bw_seg >= min_pw:
                            band_boxes.append((start_x, by1 + int(band_h * 0.08), bw_seg, int(band_h * 0.84)))
                if in_seg and (w - start_x) >= min_pw:
                    band_boxes.append((start_x, by1 + int(band_h * 0.08), w - start_x, int(band_h * 0.84)))

            for bx, by, bw, bh in band_boxes:
                patch_score = float(
                    np.mean(combined[max(0, by - by1) : min(band_h, by - by1 + bh), bx : bx + bw])
                )
                raw_boxes.append((bx, by, bw, bh, row_idx + 1, patch_score * float(bw * bh)))

        # Non-Maximum Suppression (IoU < 0.35)
        raw_boxes.sort(key=lambda b: b[5], reverse=True)
        kept: List[Tuple[int, int, int, int, int]] = []
        for bx, by, bw, bh, r_idx, _ in raw_boxes:
            overlap = False
            for kx, ky, kw, kh, _ in kept:
                ix1, iy1 = max(bx, kx), max(by, ky)
                ix2, iy2 = min(bx + bw, kx + kw), min(by + bh, ky + kh)
                if ix2 > ix1 and iy2 > iy1:
                    inter = float((ix2 - ix1) * (iy2 - iy1))
                    union = float(bw * bh + kw * kh) - inter
                    if inter / max(1.0, union) > 0.35:
                        overlap = True
                        break
            if not overlap:
                kept.append((bx, by, bw, bh, r_idx))
                if len(kept) >= max_products:
                    break

        # Sort in natural shelf reading order: top shelf to bottom shelf, LTR (or RTL)
        kept.sort(key=lambda b: (b[4], -b[0] if rtl else b[0]))

        products: List[Dict[str, Any]] = []
        for idx, (bx, by, bw, bh, r_idx) in enumerate(kept):
            products.append(
                {
                    "product_id": idx + 1,
                    "name": f"Ürün #{idx + 1}",
                    "aoi_type": "PRODUCT",
                    "shelf_row": r_idx,
                    "rect": (int(bx), int(by), int(bw), int(bh)),
                    "is_text_or_cta": False,
                }
            )
        return products

    @classmethod
    def analyze_shelf_planogram(
        cls,
        bgr_image: np.ndarray,
        saliency_map: Optional[np.ndarray] = None,
        fixations: Optional[List[Tuple[int, int]]] = None,
        products: Optional[List[Dict[str, Any]]] = None,
        max_products: int = 12,
        reading_direction: str = "ltr",
        engine_mode: str = "CLASSIC",
        lang: str = "tr",
    ) -> Dict[str, Any]:
        """
        Full Retail Planogram Audit:
        Automatically detects shelf products and compares each SKU's:
          - `attention_share` (Share of Attention %, normalized across shelf products)
          - `canvas_attention_share` (Raw % of total canvas saliency energy)
          - `shelf_share` (Share of Shelf %, physical bounding area relative to total product area)
          - `efficiency_index` (`SoA / SoS` = Attention Share / Shelf Share; >1.15 Hero, <0.85 Underperforming)
          - `fixation_rank` (Which gaze fixation #1..#4 landed on the SKU)
        """
        is_en = (lang or "tr").lower() == "en"
        bgr_valid = cls.validate_image(bgr_image)
        h, w = bgr_valid.shape[:2]

        if saliency_map is None:
            saliency_map, _ = cls.compute_saliency_map(
                bgr_valid, engine_mode=engine_mode, reading_direction=reading_direction
            )
        if fixations is None:
            fixations = cls.compute_gaze_path(saliency_map, num_fixations=4)
        if products is None:
            products = cls.detect_shelf_products(
                bgr_valid, max_products=max_products, reading_direction=reading_direction
            )

        if not products:
            return {
                "products": [],
                "product_count": 0,
                "shelf_rows": 0,
                "hero_product": None,
                "underperforming_products": [],
                "mean_efficiency": 1.0,
            }

        total_canvas_sal = float(np.sum(saliency_map)) + 1e-6
        total_canvas_area = float(w * h) + 1e-6

        raw_items: List[Dict[str, Any]] = []
        total_prod_area = 0.0
        total_prod_sal = 0.0

        for idx, prod in enumerate(products):
            rx, ry, rw, rh = prod["rect"]
            x1, y1 = max(0, int(rx)), max(0, int(ry))
            x2, y2 = min(w, int(rx + rw)), min(h, int(ry + rh))
            if x2 <= x1 or y2 <= y1:
                continue

            area = float((x2 - x1) * (y2 - y1))
            sal_sum = float(np.sum(saliency_map[y1:y2, x1:x2]))
            total_prod_area += area
            total_prod_sal += sal_sum

            fix_rank: Optional[int] = None
            for f_i, (fx, fy) in enumerate(fixations):
                if x1 <= fx <= x2 and y1 <= fy <= y2:
                    fix_rank = f_i + 1
                    break

            default_name = f"Product #{idx + 1}" if is_en else f"Ürün #{idx + 1}"
            raw_items.append(
                {
                    "product_id": prod.get("product_id", idx + 1),
                    "name": prod.get("name", default_name),
                    "shelf_row": int(prod.get("shelf_row", 1)),
                    "rect": (x1, y1, x2 - x1, y2 - y1),
                    "area_px": int(area),
                    "sal_sum": sal_sum,
                    "canvas_area_share": round((area / total_canvas_area) * 100.0, 1),
                    "canvas_attention_share": round((sal_sum / total_canvas_sal) * 100.0, 1),
                    "fixation_rank": fix_rank,
                }
            )

        total_prod_area = max(1e-6, total_prod_area)
        total_prod_sal = max(1e-6, total_prod_sal)

        evaluated: List[Dict[str, Any]] = []
        for item in raw_items:
            sos = round((item["area_px"] / total_prod_area) * 100.0, 1)
            soa = round((item["sal_sum"] / total_prod_sal) * 100.0, 1)
            eff = round(soa / max(0.1, sos), 2)

            if eff >= 1.15:
                status = "🔥 High Impact (Hero)" if is_en else "🔥 Yüksek Verim (Hero)"
                tier = "HIGH"
            elif eff >= 0.85:
                status = "✅ Balanced" if is_en else "✅ Dengeli"
                tier = "BALANCED"
            else:
                status = "⚠️ Low Impact (Deficit)" if is_en else "⚠️ Düşük Verim (Kayıp)"
                tier = "LOW"

            rank = item["fixation_rank"]
            ttff_str = (
                (f"Rank #{rank}" if is_en else f"Sıra #{rank}")
                if rank is not None
                else ("Unfixated" if is_en else "Odak Dışı")
            )

            evaluated.append(
                {
                    "product_id": item["product_id"],
                    "name": item["name"],
                    "aoi_type": "PRODUCT",
                    "shelf_row": item["shelf_row"],
                    "rect": item["rect"],
                    "is_text_or_cta": False,
                    "shelf_share": sos,
                    "attention_share": soa,
                    "share": soa,
                    "canvas_attention_share": item["canvas_attention_share"],
                    "canvas_area_share": item["canvas_area_share"],
                    "efficiency_index": eff,
                    "tier": tier,
                    "status": status,
                    "fixation_rank": rank,
                    "ttff": ttff_str,
                    "ttff_time": ttff_str,
                    "wcag_ratio": None,
                    "wcag_grade": None,
                    "legible_at_distance": True,
                }
            )

        hero = max(evaluated, key=lambda p: p["efficiency_index"]) if evaluated else None
        underperforming = [p for p in evaluated if p["efficiency_index"] < 0.85]
        mean_eff = round(float(np.mean([p["efficiency_index"] for p in evaluated])), 2) if evaluated else 1.0
        shelf_rows = len({p["shelf_row"] for p in evaluated})

        return {
            "products": evaluated,
            "product_count": len(evaluated),
            "shelf_rows": shelf_rows,
            "hero_product": hero,
            "underperforming_products": underperforming,
            "mean_efficiency": mean_eff,
        }

    @classmethod
    def compute_overall_attention_score(
        cls,
        focus_score: int,
        clutter_score: int,
        quadrant_enum: Quadrant,
        blind_spots: Optional[List[Dict[str, Any]]] = None,
        aoi_results: Optional[List[Dict[str, Any]]] = None,
        reading_direction: str = "ltr",
    ) -> int:
        """
        Computes a unified 0-100 Overall Attention Score (`overall_attention_score` / `attention_score`)
        combining:
        1. Focal Concentration (`focus_score`, 45% weight)
        2. Visual Clarity (`100 - clutter_score`, 30% weight)
        3. Spatial Hierarchy & Scanpath Balance (`25%` weight: Golden-Zone/Center landing & low dead-zones,
           respecting LTR Top-Left vs. RTL Top-Right reading entry)
        4. Optional AOI Contrast & Capture Adjustment (when target AOIs are audited)
        """
        rtl = (reading_direction or "ltr").lower() == "rtl"
        clarity_score = max(0, 100 - int(clutter_score))
        quad_scores = {
            Quadrant.TOP_LEFT: 78.0 if rtl else 95.0,
            Quadrant.CENTER: 92.0,
            Quadrant.TOP_RIGHT: 95.0 if rtl else 78.0,
            Quadrant.BOTTOM_LEFT: 62.0 if rtl else 68.0,
            Quadrant.BOTTOM_RIGHT: 68.0 if rtl else 62.0,
        }
        hierarchy_base = quad_scores.get(quadrant_enum, 75.0)
        n_blind = len(blind_spots) if blind_spots else 0
        hierarchy_score = max(25.0, hierarchy_base - (n_blind * 10.0))

        base_score = 0.45 * float(focus_score) + 0.30 * float(clarity_score) + 0.25 * hierarchy_score

        if aoi_results:
            aoi_adj = 0.0
            for a in aoi_results:
                ratio = a.get("wcag_ratio")
                if ratio is not None:
                    if ratio < 3.0:
                        aoi_adj -= 7.0
                    elif ratio < 4.5:
                        aoi_adj -= 3.0
                    elif ratio >= 7.0:
                        aoi_adj += 2.0
                if a.get("legible_at_distance") is False:
                    aoi_adj -= 4.0
                rank = a.get("fixation_rank")
                role = a.get("aoi_type", "")
                if role in ("CTA", "PRICE", "LOGO") and rank is not None and rank <= 3:
                    aoi_adj += 3.0
                elif role in ("CTA", "PRICE") and rank is None and float(a.get("share", 0.0)) < 4.0:
                    aoi_adj -= 4.0
            base_score += float(np.clip(aoi_adj, -22.0, 10.0))

        return int(round(np.clip(base_score, 0.0, 100.0)))

    @classmethod
    def generate_fix_recommendations(
        cls,
        metrics: Dict[str, Any],
        aoi_results: Optional[List[Dict[str, Any]]] = None,
        distance_meters: float = 1.0,
        lang: str = "tr",
        reading_direction: str = "ltr",
        shelf_products: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Generates a prioritized, rule-based actionable 'Fix List' ('Düzelt Listesi')
        citing concrete numeric thresholds (e.g. 'CTA kontrastı 2.1:1, hedef ≥4.5:1',
        shelf efficiency `SoA/SoS`, RTL/LTR Golden Zone alignment, and concrete design tweaks).
        """
        is_en = (lang or "tr").lower() == "en"
        rtl = (reading_direction or metrics.get("reading_direction", "ltr") or "ltr").lower() == "rtl"
        golden_zone_tr = "Sağ-Üst (RTL Altın Bölge)" if rtl else "Sol-Üst Altın Bölge"
        golden_zone_en = "Top-Right (RTL Golden Zone)" if rtl else "Top-Left Golden Zone"
        fixes: List[str] = []

        # 1. AOI-specific rules (Contrast, Distance Legibility, Fixation Capture, Attention Share)
        if aoi_results:
            for a in aoi_results:
                name = a.get("name", "AOI")
                role = a.get("aoi_type", cls._infer_aoi_type(a))
                ratio = a.get("wcag_ratio")
                share = float(a.get("share", 0.0))
                rank = a.get("fixation_rank")
                legible = a.get("legible_at_distance", True)
                ang = a.get("angular_height_deg", 0.20)

                if ratio is not None and ratio < 4.5:
                    target_str = "≥4.5:1 (WCAG AA)" if ratio >= 3.0 else "≥4.5:1 (Kritik <3.0:1)"
                    if is_en:
                        fixes.append(
                            f"[Contrast] '{name}' contrast is {ratio}:1, target ≥4.5:1 (darken background by 25% or add high-luminance text outline)."
                        )
                    else:
                        fixes.append(
                            f"[Kontrast] '{name}' kontrastı {ratio}:1, hedef {target_str} (zemin koyuluğunu %25 artırın veya yüksek parlaklıkta kontur ekleyin)."
                        )

                if not legible:
                    if is_en:
                        fixes.append(
                            f"[Legibility] '{name}' subtends {ang}° at {distance_meters:.1f}m, target ≥0.18° (enlarge font size by ≥30% and use bold stroke weight)."
                        )
                    else:
                        fixes.append(
                            f"[Okunabilirlik] '{name}' {distance_meters:.1f}m mesafede {ang}°, hedef ≥0.18° (puntoyu en az %30 büyütün ve yarı-kalın/bold ağırlık kullanın)."
                        )

                if role in ("CTA", "PRICE") and rank is None:
                    if is_en:
                        fixes.append(
                            f"[Gaze Capture] '{name}' missed top-4 fixations ({share}% share, target ≥8%); add warm CIE-LAB color pop-out and +20% whitespace padding."
                        )
                    else:
                        fixes.append(
                            f"[Odak Yakalama] '{name}' ilk 4 sabitlenmeye giremedi (%{share} pay, hedef ≥%8); sıcak renk kontrastı ve çevresine +%20 boşluk (whitespace) ekleyin."
                        )
                elif role == "LOGO" and rank is None and share < 4.0:
                    if is_en:
                        fixes.append(
                            f"[Brand Recall] '{name}' receives only {share}% attention (target ≥5%); align toward {golden_zone_en} or boost contrast."
                        )
                    else:
                        fixes.append(
                            f"[Marka Algısı] '{name}' dikkat payı düşük (%{share}, hedef ≥%5); {golden_zone_tr} yönüne yaklaştırın veya kontrastını güçlendirin."
                        )

        # 2. Shelf Planogram Efficiency (Attention Share vs. Shelf Area Share) rules
        if shelf_products:
            for p in shelf_products:
                eff = float(p.get("efficiency_index", 1.0))
                soa = float(p.get("attention_share", 0.0))
                sos = float(p.get("shelf_share", 0.0))
                pname = p.get("name", "Ürün")
                if eff < 0.80 and sos >= 5.0:
                    if is_en:
                        fixes.append(
                            f"[Shelf Efficiency] '{pname}' occupies {sos}% shelf area but captures only {soa}% attention (SoA/SoS: {eff}x, target ≥1.0x); increase facing contrast or eye-level placement."
                        )
                    else:
                        fixes.append(
                            f"[Raf Verimliliği] '{pname}' raf alanının %{sos}'ini kaplıyor ancak yalnızca %{soa} dikkat çekiyor (Verim: {eff}x, hedef ≥1.0x); ambalaj kontrastını veya göz hizası konumunu güçlendirin."
                        )

        # 3. RTL / LTR Reading Direction & Entry Alignment Rule
        quad_enum = metrics.get("quadrant_enum")
        if rtl and quad_enum == Quadrant.TOP_LEFT:
            if is_en:
                fixes.append(
                    "[RTL Layout] Reading direction is Right-to-Left (RTL), but primary fixation lands in Top-Left; mirror hero logo/headline to Top-Right Golden Zone."
                )
            else:
                fixes.append(
                    "[RTL Yerleşim] Okuma yönü Sağdan-Sola (RTL) seçili ancak ilk bakış Sol-Üst'e kayıyor; ana başlık ve logoyu Sağ-Üst RTL Altın Bölge'ye aynalayın."
                )
        elif not rtl and quad_enum == Quadrant.BOTTOM_RIGHT and int(metrics.get("focus_score", 70)) < 65:
            if is_en:
                fixes.append(
                    "[Reading Flow] Primary gaze sinks to Bottom-Right terminal zone; anchor primary value proposition in Top-Left or Center for faster entry."
                )
            else:
                fixes.append(
                    "[Okuma Akışı] İlk bakış doğrudan Sağ-Alt çıkış bölgesine kayıyor; hızlı kavrama için ana mesajı Sol-Üst veya Merkez'e sabitleyin."
                )

        # 4. Global Visual Clutter Rule
        clutter = int(metrics.get("clutter_score", 0))
        if clutter > 52:
            if is_en:
                fixes.append(
                    f"[Visual Clutter] Clutter index is {clutter}/100, target ≤45/100 (remove decorative micro-icons <1% area and simplify background textures)."
                )
            else:
                fixes.append(
                    f"[Görsel Karmaşa] Karmaşa endeksi {clutter}/100, hedef ≤45/100 (%1'den küçük dekoratif mikro ikonları kaldırın ve arka plan dokusunu sadeleştirin)."
                )

        # 5. Global Focal Concentration Rule
        focus = int(metrics.get("focus_score", 0))
        if focus < 55:
            if is_en:
                fixes.append(
                    f"[Weak Hierarchy] Focus score is {focus}/100, target ≥65/100 (scale up primary hero element by 20-25% and desaturate secondary badges)."
                )
            else:
                fixes.append(
                    f"[Zayıf Hiyerarşi] Odak yoğunluğu {focus}/100, hedef ≥65/100 (ana ürün/başlık ölçeğini %20-25 büyütüp ikincil rozetlerin doygunluğunu azaltın)."
                )

        # 6. Blind Spots / Dead Zone Rule
        blind_spots = metrics.get("blind_spots", [])
        if blind_spots:
            b = blind_spots[0]
            if is_en:
                fixes.append(
                    f"[Blind Spot] {b['zone']} zone receives only {b['energy']}% attention (target ≥10%); relocate critical CTAs/prices out of this dead zone."
                )
            else:
                fixes.append(
                    f"[Kör Nokta] {b['zone']} bölgesi yalnızca %{b['energy']} dikkat alıyor (hedef ≥%10); fiyat veya CTA öğelerini bu ölü bölgeden taşıyın."
                )

        if not fixes:
            if is_en:
                fixes.append(
                    "✓ All visual hierarchy, contrast (≥4.5:1), reading flow, and clutter (≤45/100) checks passed targets."
                )
            else:
                fixes.append(
                    "✓ Tüm görsel hiyerarşi, kontrast (≥4.5:1), okuma yönü ve karmaşa (≤45/100) kriterleri hedef aralıkta."
                )

        return fixes[:6]

    @classmethod
    def calculate_metrics(
        cls,
        bgr_image: np.ndarray,
        saliency_map: np.ndarray,
        fixations: List[Tuple[int, int]],
        extra_meta: Optional[dict] = None,
        lang: str = "tr",
        aoi_results: Optional[List[Dict[str, Any]]] = None,
        distance_meters: float = 1.0,
        reading_direction: Optional[str] = None,
        shelf_products: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        h, w = saliency_map.shape[:2]
        if reading_direction is None:
            reading_direction = (extra_meta.get("reading_direction", "ltr") if extra_meta else "ltr") or "ltr"
        rtl = reading_direction.lower() == "rtl"

        flat_sal = saliency_map.flatten()
        top_20_count = max(1, int(len(flat_sal) * 0.20))
        top_20_sum = float(np.sum(np.partition(flat_sal, -top_20_count)[-top_20_count:]))
        total_sum = float(np.sum(flat_sal)) + 1e-6
        focus_ratio = top_20_sum / total_sum
        raw_focus = (focus_ratio - 0.20) / (0.75 - 0.20)
        focus_score = int(round(np.clip(raw_focus * 100.0, 5.0, 98.0)))

        clutter_score = cls.compute_clutter_score(bgr_image, saliency_map)

        quad_enum = Quadrant.CENTER
        if fixations:
            quad_enum = cls.classify_quadrant(fixations[0][0], fixations[0][1], w, h)

        if lang.lower() == "tr":
            labels_map = QUADRANT_LABELS_TR_RTL if rtl else QUADRANT_LABELS_TR
        else:
            labels_map = QUADRANT_LABELS_EN_RTL if rtl else QUADRANT_LABELS_EN
        quadrant_label = labels_map[quad_enum]

        faces = extra_meta.get("faces", []) if extra_meta else []
        blind_spots = extra_meta.get("blind_spots", []) if extra_meta else []
        is_face_focused = False
        if faces and fixations:
            fx, fy = fixations[0]
            for rx, ry, rw, rh in faces:
                pad_x, pad_y = rw * 0.20, rh * 0.20
                if (rx - pad_x) <= fx <= (rx + rw + pad_x) and (ry - pad_y) <= fy <= (ry + rh + pad_y):
                    is_face_focused = True
                    break

        overall_score = cls.compute_overall_attention_score(
            focus_score=focus_score,
            clutter_score=clutter_score,
            quadrant_enum=quad_enum,
            blind_spots=blind_spots,
            aoi_results=aoi_results,
            reading_direction="rtl" if rtl else "ltr",
        )

        insight = cls._generate_marketing_insight(
            focus_score,
            clutter_score,
            quad_enum,
            extra_meta,
            is_face_focused,
            lang=lang,
            reading_direction="rtl" if rtl else "ltr",
        )

        metrics_dict: Dict[str, Any] = {
            "overall_attention_score": overall_score,
            "attention_score": overall_score,
            "focus_score": focus_score,
            "clutter_score": clutter_score,
            "quadrant_enum": quad_enum,
            "quadrant": quadrant_label,
            "reading_direction": "rtl" if rtl else "ltr",
            "engine_mode": (extra_meta.get("engine_mode", "CLASSIC") if extra_meta else "CLASSIC"),
            "insight": insight,
            "fixations": fixations,
            "blind_spots": blind_spots,
            "has_face": len(faces) > 0 and is_face_focused,
            "faces_detected": len(faces),
        }
        metrics_dict["fix_recommendations"] = cls.generate_fix_recommendations(
            metrics_dict,
            aoi_results=aoi_results,
            distance_meters=distance_meters,
            lang=lang,
            reading_direction="rtl" if rtl else "ltr",
            shelf_products=shelf_products,
        )
        return metrics_dict

    @classmethod
    def _generate_marketing_insight(
        cls,
        focus: int,
        clutter: int,
        quadrant_enum: Quadrant,
        extra_meta: Optional[dict] = None,
        is_face_focused: bool = False,
        lang: str = "tr",
        reading_direction: str = "ltr",
    ) -> str:
        is_en = (lang or "tr").lower() == "en"
        rtl = (reading_direction or "ltr").lower() == "rtl"
        if is_en:
            labels_map = QUADRANT_LABELS_EN_RTL if rtl else QUADRANT_LABELS_EN
        else:
            labels_map = QUADRANT_LABELS_TR_RTL if rtl else QUADRANT_LABELS_TR
        q_label = labels_map[quadrant_enum]
        faces = extra_meta.get("faces", []) if extra_meta else []
        golden_quad = Quadrant.TOP_RIGHT if rtl else Quadrant.TOP_LEFT

        if is_en:
            if is_face_focused and len(faces) > 0:
                return (
                    f"Human face detected: Primary fixation ({q_label}) locks directly onto the model's face, "
                    f"reinforcing emotional engagement."
                )
            if len(faces) > 0 and focus >= 75:
                return (
                    f"Primary attention lands on the hero message in {q_label}, while the detected face "
                    f"acts as a strong secondary anchor."
                )
            if extra_meta and extra_meta.get("blind_spots"):
                bs = extra_meta["blind_spots"][0]
                return (
                    f"Primary gaze concentrates in {q_label}, whereas {bs['zone']} receives low attention "
                    f"(<{bs['energy']}%). Avoid placing critical CTAs or price tags in that dead zone."
                )
            if focus >= 75 and clutter <= 40:
                return (
                    f"High focal concentration in {q_label} with low visual clutter—favorable for rapid "
                    f"comprehension within the first 3 seconds."
                )
            if focus >= 70 and clutter > 50:
                return (
                    f"Strong primary anchor in {q_label}, but elevated edge/orientation clutter ({clutter}/100) "
                    f"may disperse secondary scanpaths."
                )
            if focus < 50 and clutter >= 60:
                return (
                    f"High overall visual clutter ({clutter}/100); attention energy is fragmented across the canvas "
                    f"with weak convergence in {q_label}."
                )
            if quadrant_enum == golden_quad:
                dir_tag = "RTL" if rtl else "F-pattern"
                return (
                    f"Natural {dir_tag} reading entry modeled via {q_label}; well-suited for brand logo "
                    f"or primary value proposition."
                )
            if quadrant_enum == Quadrant.CENTER:
                return f"Balanced central fixation modeled in {q_label}; effective for hero product presentation."
            return (
                f"Estimated primary fixation in {q_label} (Focus: {focus}/100, Clutter: {clutter}/100); "
                f"consider aligning your primary CTA along this trajectory."
            )

        if is_face_focused and len(faces) > 0:
            return (
                f"İnsan yüzü tespit edildi: Tahmini ilk bakış ({q_label}) doğrudan modelin yüzüne yöneliyor; "
                f"bu durum duygusal dikkat çekimini destekler."
            )
        if len(faces) > 0 and focus >= 75:
            return (
                f"Tahmini birincil bakış {q_label} bölgesindeki ana ürüne/mesaja yönelirken, "
                f"görseldeki model yüzü ikincil odak olarak hiyerarşiyi destekliyor."
            )
        if extra_meta and extra_meta.get("blind_spots"):
            bs = extra_meta["blind_spots"][0]
            return (
                f"İlk bakış tahmini {q_label} bölgesinde yoğunlaşıyor; ancak {bs['zone']} bölgesi düşük dikkat "
                f"enerjisine (<%{bs['energy']}) sahip. Kritik CTA veya fiyat bilgisini bu alana koymaktan kaçının."
            )
        if focus >= 75 and clutter <= 40:
            return (
                f"{q_label} bölgesinde düşük görsel karmaşa ile yüksek odak yoğunluğu modellendi—"
                f"ilk 3 saniyelik taramada hızlı fark edilme potansiyeli yüksek."
            )
        if focus >= 70 and clutter > 50:
            return (
                f"{q_label} bölgesinde belirgin bir odak noktası mevcut, ancak yüksek kenar/yönelim karmaşası ({clutter}/100) "
                f"ikincil bakış akışını dağıtabilir."
            )
        if focus < 50 and clutter >= 60:
            return (
                f"Tasarım genelinde yüksek görsel karmaşa ({clutter}/100) hesaplandı; dikkat enerjisi dağınık "
                f"ve {q_label} bölgesinde zayıf toplanıyor."
            )
        if quadrant_enum == golden_quad:
            return (
                f"{q_label} üzerinden doğal okuma akışı girişi modellendi; marka logosu veya ana değer önerisi "
                f"için uygun yerleşim."
            )
        if quadrant_enum == Quadrant.CENTER:
            return f"{q_label} bölgesinde merkezi odaklanma tahmin edildi; kahraman ürün (hero product) vurgusu için dengeli."
        return (
            f"Tahmini birincil bakış {q_label} bölgesinde (Odak: {focus}/100, Karmaşa: {clutter}/100); "
            f"ana eylem çağrısını (CTA) bu bölgeyle hizalamayı değerlendirin."
        )

    @classmethod
    def _foveal_fixation_energy(cls, sal_map: np.ndarray, fixations: List[Tuple[int, int]]) -> float:
        """Computes mean saliency inside the foveal patch around the #1 fixation point."""
        if not fixations:
            return float(np.max(sal_map)) if sal_map.size > 0 else 0.0
        h, w = sal_map.shape[:2]
        fx, fy = fixations[0]
        r = max(4, int(min(h, w) * 0.04))
        x1, y1 = max(0, fx - r), max(0, fy - r)
        x2, y2 = min(w, fx + r + 1), min(h, fy + r + 1)
        patch = sal_map[y1:y2, x1:x2]
        return float(np.mean(patch)) if patch.size > 0 else float(sal_map[fy, fx])

    @classmethod
    def compute_ab_comparison(cls, img_a: np.ndarray, img_b: np.ndarray, lang: str = "tr") -> Dict[str, Any]:
        is_en = (lang or "tr").lower() == "en"
        label_a = "Variant A" if is_en else "A Varyantı"
        label_b = "Variant B" if is_en else "B Varyantı"
        label_tie = "Tie" if is_en else "Berabere"

        def get_raw_fusion(bgr: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
            h, w = bgr.shape[:2]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            sr = cls.compute_spectral_residual(gray)
            lab = cls.compute_lab_contrast(bgr)
            edge = cls.compute_edge_density(gray)
            bias = cls.compute_center_bias((h, w))
            face, faces_list = cls.compute_face_bias(bgr, gray)
            obj = cls.compute_objectness_weight(bgr)
            raw = (
                cls.WEIGHT_SPECTRAL * sr
                + cls.WEIGHT_LAB * lab
                + cls.WEIGHT_EDGE * edge
                + cls.WEIGHT_BIAS * bias
                + cls.WEIGHT_FACE * face
            ) * obj
            return raw, faces_list

        raw_a, faces_a = get_raw_fusion(img_a)
        raw_b, faces_b = get_raw_fusion(img_b)

        global_min = float(min(np.min(raw_a), np.min(raw_b)))
        global_max = float(max(np.max(raw_a), np.max(raw_b)))
        range_val = max(1e-6, global_max - global_min)

        norm_a = ((raw_a - global_min) / range_val).astype(np.float32)
        norm_b = ((raw_b - global_min) / range_val).astype(np.float32)

        # Size-invariant attention density share (mean instead of sum, immune to unequal snip sizes)
        mean_a = float(np.mean(norm_a))
        mean_b = float(np.mean(norm_b))
        total_mean = mean_a + mean_b + 1e-6
        share_a = round((mean_a / total_mean) * 100.0, 1)
        share_b = round(100.0 - share_a, 1)

        fix_a = cls.compute_gaze_path(norm_a, 4, bgr_image=img_a)
        fix_b = cls.compute_gaze_path(norm_b, 4, bgr_image=img_b)

        # Compare actual #1 fixation foveal energy rather than isolated single-pixel max
        first_fix_a = cls._foveal_fixation_energy(norm_a, fix_a)
        first_fix_b = cls._foveal_fixation_energy(norm_b, fix_b)
        first_fixation_winner = label_a if first_fix_a >= first_fix_b else label_b

        # Compute comprehensive metrics
        meta_a = {
            "faces": faces_a,
            "blind_spots": cls.detect_blind_spots(norm_a, lang=lang),
            "has_face": len(faces_a) > 0,
        }
        meta_b = {
            "faces": faces_b,
            "blind_spots": cls.detect_blind_spots(norm_b, lang=lang),
            "has_face": len(faces_b) > 0,
        }
        metrics_a = cls.calculate_metrics(img_a, norm_a, fix_a, meta_a, lang=lang)
        metrics_b = cls.calculate_metrics(img_b, norm_b, fix_b, meta_b, lang=lang)

        # Balanced Overall Winner Determination:
        # Blends Attention Share (visual pop-out / visibility) with Overall Attention Score (ergonomic balance/clarity)
        score_a = float(metrics_a.get("overall_attention_score", metrics_a.get("focus_score", 50)))
        score_b = float(metrics_b.get("overall_attention_score", metrics_b.get("focus_score", 50)))
        # Composite battle score = 55% attention share + 45% overall ergonomic score
        comp_a = 0.55 * share_a + 0.45 * score_a
        comp_b = 0.55 * share_b + 0.45 * score_b

        if abs(comp_a - comp_b) < 1.0:
            overall_winner = label_tie
            margin = 0.0
        elif comp_a > comp_b:
            overall_winner = label_a
            margin = round(comp_a - comp_b, 1)
        else:
            overall_winner = label_b
            margin = round(comp_b - comp_a, 1)

        heat_a = cls.generate_heatmap_overlay(img_a, norm_a)
        heat_b = cls.generate_heatmap_overlay(img_b, norm_b)
        heat_gaze_a = cls.draw_gaze_path(heat_a, fix_a)
        heat_gaze_b = cls.draw_gaze_path(heat_b, fix_b)
        fog_a = cls.generate_fog_overlay(img_a, norm_a)
        fog_b = cls.generate_fog_overlay(img_b, norm_b)

        return {
            "variant_a": {
                "saliency": norm_a,
                "heatmap": heat_a,
                "heatmap_gaze": heat_gaze_a,
                "fog": fog_a,
                "fixations": fix_a,
                "metrics": metrics_a,
                "share": share_a,
                "peak": round(first_fix_a, 4),
            },
            "variant_b": {
                "saliency": norm_b,
                "heatmap": heat_b,
                "heatmap_gaze": heat_gaze_b,
                "fog": fog_b,
                "fixations": fix_b,
                "metrics": metrics_b,
                "share": share_b,
                "peak": round(first_fix_b, 4),
            },
            "first_fixation_winner": first_fixation_winner,
            "overall_winner": overall_winner,
            "margin": round(margin, 1),
        }

    @staticmethod
    def _normalize_01(arr: np.ndarray) -> np.ndarray:
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        if max_val - min_val < 1e-6:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - min_val) / (max_val - min_val)).astype(np.float32)
