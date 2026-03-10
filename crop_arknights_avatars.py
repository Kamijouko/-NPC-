#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import Request, urlopen


try:
    import cv2
    import numpy as np
except Exception as exc:  # noqa: BLE001
    raise RuntimeError(
        "Missing dependencies. Create conda env from "
        "'environment_arknights_avatar.yml' first."
    ) from exc


ANIME_CASCADE_URL = (
    "https://raw.githubusercontent.com/nagadomi/"
    "lbpcascade_animeface/master/lbpcascade_animeface.xml"
)
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
USER_AGENT = "ArkAvatars-Cropper/2.0"
VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def list_images(root: Path) -> List[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS
    )


def read_image(path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None, None
    decoded = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        return None, None

    # Preserve alpha mask for better fallback on transparent portraits.
    if decoded.ndim == 2:
        bgr = cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR)
        return bgr, estimate_foreground_mask(bgr)
    if decoded.shape[2] == 4:
        bgr = decoded[:, :, :3].astype(np.float32)
        alpha = decoded[:, :, 3].astype(np.float32) / 255.0
        bg = np.full_like(bgr, 255.0)
        merged = bgr * alpha[..., None] + bg * (1.0 - alpha[..., None])
        mask = (decoded[:, :, 3] > 8).astype(np.uint8) * 255
        return merged.astype(np.uint8), mask
    return decoded, estimate_foreground_mask(decoded)


def estimate_foreground_mask(image: np.ndarray) -> Optional[np.ndarray]:
    h, w = image.shape[:2]
    if h < 16 or w < 16:
        return None

    m = max(8, int(round(min(h, w) * 0.04)))
    corners = np.concatenate(
        [
            image[:m, :m].reshape(-1, 3),
            image[:m, -m:].reshape(-1, 3),
            image[-m:, :m].reshape(-1, 3),
            image[-m:, -m:].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(corners, axis=0).astype(np.int16)
    diff = np.abs(image.astype(np.int16) - bg[None, None, :])
    dist = np.max(diff, axis=2).astype(np.uint8)

    # Most exported portraits have near-uniform background.
    fg = (dist > 18).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if num_labels <= 1:
        return None

    min_area = int(h * w * 0.03)
    best_label = -1
    best_area = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > best_area and area >= min_area:
            best_area = area
            best_label = label
    if best_label < 0:
        return None

    mask = np.zeros_like(fg)
    mask[labels == best_label] = 255
    return mask


def write_jpg(path: Path, image: np.ndarray, quality: int) -> None:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Failed to encode JPEG: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=45) as resp:
        data = resp.read()
    path.write_bytes(data)


def find_haar_cascade() -> Optional[Path]:
    filename = "haarcascade_frontalface_default.xml"
    candidates: List[Path] = []

    cv2_data = getattr(cv2, "data", None)
    if cv2_data is not None and hasattr(cv2_data, "haarcascades"):
        candidates.append(Path(cv2_data.haarcascades) / filename)

    cv2_pkg = Path(cv2.__file__).resolve().parent
    candidates.extend(
        [
            cv2_pkg / "data" / filename,
            cv2_pkg / "haarcascades" / filename,
            Path(sys.prefix) / "Library" / "etc" / "haarcascades" / filename,
            Path(sys.prefix) / "share" / "opencv4" / "haarcascades" / filename,
            Path(sys.prefix) / "Lib" / "site-packages" / "cv2" / "data" / filename,
        ]
    )

    seen = set()
    for p in candidates:
        key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)
        if p.exists():
            return p
    return None


def parse_detector_list(text: str) -> List[str]:
    allowed = {"anime", "yunet", "haar"}
    items = [x.strip().lower() for x in text.split(",") if x.strip()]
    items = [x for x in items if x in allowed]
    if not items:
        raise ValueError("No valid detectors in --detectors (use anime,yunet,haar)")
    return items


def load_detectors(
    detector_order: List[str],
    anime_cascade_path: Path,
    yunet_model_path: Path,
) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for name in detector_order:
        if name == "anime":
            if not anime_cascade_path.exists():
                download_file(ANIME_CASCADE_URL, anime_cascade_path)
            anime = cv2.CascadeClassifier(str(anime_cascade_path))
            if anime.empty():
                raise RuntimeError("Failed to load anime cascade.")
            out["anime"] = anime
        elif name == "haar":
            haar_path = find_haar_cascade()
            if haar_path is None:
                print("warn: haarcascade_frontalface_default.xml not found, skipping haar.")
                continue
            haar = cv2.CascadeClassifier(str(haar_path))
            if haar.empty():
                print("warn: failed to load haar cascade, skipping haar.")
                continue
            out["haar"] = haar
        elif name == "yunet":
            if not hasattr(cv2, "FaceDetectorYN_create"):
                print("warn: OpenCV build has no FaceDetectorYN, skipping yunet.")
                continue
            if not yunet_model_path.exists():
                download_file(YUNET_URL, yunet_model_path)
            try:
                yunet = cv2.FaceDetectorYN_create(
                    str(yunet_model_path),
                    "",
                    (320, 320),
                    0.7,
                    0.3,
                    5000,
                )
                out["yunet"] = yunet
            except Exception as exc:  # noqa: BLE001
                print(f"warn: failed to init yunet: {exc}")
    return out


def _detect_cascade(
    cascade,
    gray: np.ndarray,
    scale_factor: float,
    min_neighbors: int,
    min_size: int,
) -> List[Tuple[int, int, int, int]]:
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size),
    )
    return [tuple(map(int, face)) for face in faces]


def detect_candidates(detectors: Dict[str, object], image: np.ndarray, relaxed: bool = False):
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    candidates = []

    cascade_scale = 1.08 if relaxed else 1.1
    anime_neighbors = 3 if relaxed else 4
    anime_min_size = 18 if relaxed else 24
    haar_neighbors = 4 if relaxed else 5
    haar_min_size = 20 if relaxed else 24
    yunet_min_score = 0.42 if relaxed else 0.55

    def run_yunet(img: np.ndarray, min_score: float, scale_back: float = 1.0):
        local = []
        yunet_local = detectors.get("yunet")
        if yunet_local is None:
            return local
        ih, iw = img.shape[:2]
        try:
            yunet_local.setInputSize((iw, ih))
            _, out = yunet_local.detect(img)
            if out is not None:
                for row in out:
                    score = float(row[-1])
                    if score < min_score:
                        continue
                    x, y, bw, bh = row[:4]
                    if scale_back != 1.0:
                        x /= scale_back
                        y /= scale_back
                        bw /= scale_back
                        bh /= scale_back
                    box = (
                        int(round(x)),
                        int(round(y)),
                        int(round(bw)),
                        int(round(bh)),
                    )
                    local.append(("yunet", score, box))
        except Exception as exc:  # noqa: BLE001
            print(f"warn: yunet detection failed on one image: {exc}")
        return local

    anime = detectors.get("anime")
    if anime is not None:
        for box in _detect_cascade(anime, gray_eq, cascade_scale, anime_neighbors, anime_min_size):
            candidates.append(("anime", 0.72, box))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        for box in _detect_cascade(anime, clahe, cascade_scale, anime_neighbors, anime_min_size):
            candidates.append(("anime", 0.68, box))
        # Upscaled pass improves tiny-face recall on full portraits.
        up_limit = 2200 if relaxed else 1600
        if max(h, w) <= up_limit:
            scale = 1.8 if relaxed else 1.5
            up = cv2.resize(gray_eq, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            up_neighbors = 3 if relaxed else 4
            up_min_size = 24 if relaxed else 30
            for x, y, bw, bh in _detect_cascade(anime, up, cascade_scale, up_neighbors, up_min_size):
                box = (
                    int(round(x / scale)),
                    int(round(y / scale)),
                    int(round(bw / scale)),
                    int(round(bh / scale)),
                )
                candidates.append(("anime", 0.66, box))

    haar = detectors.get("haar")
    if haar is not None:
        for box in _detect_cascade(haar, gray_eq, cascade_scale, haar_neighbors, haar_min_size):
            candidates.append(("haar", 0.58, box))

    candidates.extend(run_yunet(image, yunet_min_score))

    # Extra recall pass: in relaxed mode, try upscaled image for tiny faces.
    if relaxed:
        scale = 1.5
        up = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        candidates.extend(run_yunet(up, max(0.35, yunet_min_score - 0.05), scale_back=scale))

    # Extra recall pass: detect on horizontal flip and map boxes back.
    flipped = cv2.flip(image, 1)
    fgray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
    fgray_eq = cv2.equalizeHist(fgray)

    if anime is not None:
        for fx, fy, fbw, fbh in _detect_cascade(
            anime, fgray_eq, cascade_scale, anime_neighbors, anime_min_size
        ):
            box = (w - (fx + fbw), fy, fbw, fbh)
            candidates.append(("anime", 0.64 if relaxed else 0.6, box))
    if haar is not None:
        for fx, fy, fbw, fbh in _detect_cascade(
            haar, fgray_eq, cascade_scale, haar_neighbors, haar_min_size
        ):
            box = (w - (fx + fbw), fy, fbw, fbh)
            candidates.append(("haar", 0.5 if relaxed else 0.46, box))
    for _, score, (fx, fy, fbw, fbh) in run_yunet(
        flipped, max(0.35, yunet_min_score - 0.05)
    ):
        box = (w - (fx + fbw), fy, fbw, fbh)
        candidates.append(("yunet", max(0.35, score - 0.04), box))

    valid = []
    for det_name, conf, (x, y, bw, bh) in candidates:
        if bw <= 0 or bh <= 0:
            continue
        if x >= w or y >= h:
            continue
        valid.append((det_name, conf, (x, y, bw, bh)))
    return valid


def box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter = float((inter_x2 - inter_x1) * (inter_y2 - inter_y1))
    area_a = float(max(1, aw * ah))
    area_b = float(max(1, bw * bh))
    return inter / (area_a + area_b - inter)


def dedupe_candidates_nms(
    candidates: List[Tuple[str, float, Tuple[int, int, int, int]]],
    iou_thresh: float = 0.42,
) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
    if not candidates:
        return []

    det_weight = {"yunet": 1.0, "anime": 0.9, "haar": 0.75}
    # Keep strongest candidate among heavily overlapping boxes.
    ordered = sorted(
        candidates,
        key=lambda x: (x[2][2] * x[2][3]) * (det_weight.get(x[0], 0.7) * x[1]),
        reverse=True,
    )
    kept: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
    for cand in ordered:
        _, _, box = cand
        if all(box_iou(box, kb) < iou_thresh for _, _, kb in kept):
            kept.append(cand)
    return kept


def score_candidate(
    det_name: str,
    conf: float,
    box: Tuple[int, int, int, int],
    width: int,
    height: int,
) -> float:
    # Prefer larger/centered faces and more reliable detectors.
    det_weight = {"yunet": 1.0, "anime": 0.9, "haar": 0.75}
    x, y, bw, bh = box
    cx_ref = width / 2.0
    cy_ref = height * 0.38
    diag = (width ** 2 + height ** 2) ** 0.5

    area = bw * bh
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    dist_norm = (((cx - cx_ref) ** 2 + (cy - cy_ref) ** 2) ** 0.5) / max(1.0, diag)
    return area + (det_weight.get(det_name, 0.7) * conf * 120000.0) - (dist_norm * 65000.0)


def rank_candidates(
    candidates: List[Tuple[str, float, Tuple[int, int, int, int]]],
    width: int,
    height: int,
) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
    return sorted(
        candidates,
        key=lambda c: score_candidate(c[0], c[1], c[2], width, height),
        reverse=True,
    )


def is_plausible_face(
    box: Tuple[int, int, int, int],
    width: int,
    height: int,
    mask: Optional[np.ndarray],
) -> bool:
    x, y, bw, bh = box
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    area_ratio = (bw * bh) / float(max(1, width * height))
    cx_ratio = cx / float(max(1, width))
    cy_ratio = cy / float(max(1, height))

    # Reject obvious false positives (e.g. boots/props near lower body).
    if cy_ratio > 0.62:
        return False
    if not (0.08 <= cx_ratio <= 0.92):
        return False
    if not (0.002 <= area_ratio <= 0.22):
        return False

    if mask is not None and np.count_nonzero(mask) > 64:
        mh, mw = mask.shape[:2]
        px = int(min(max(0, round(cx)), mw - 1))
        py = int(min(max(0, round(cy)), mh - 1))
        if mask[py, px] == 0:
            # Center not even on subject silhouette, very likely false.
            return False

    return True


def crop_square(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    side: int,
    border_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    h, w = image.shape[:2]
    half = side / 2.0
    x1 = int(round(center_x - half))
    y1 = int(round(center_y - half))
    x2 = x1 + side
    y2 = y1 + side

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=border_color,
        )
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top

    return image[y1:y2, x1:x2]


def smart_fallback(alpha_mask: Optional[np.ndarray], image: np.ndarray) -> Tuple[float, float, int]:
    h, w = image.shape[:2]
    if alpha_mask is not None and np.count_nonzero(alpha_mask) > 64:
        x, y, bw, bh = cv2.boundingRect(alpha_mask)
        side = int(round(max(bw * 0.6, bh * 0.45)))
        side = max(64, min(side, int(min(w, h) * 0.95)))
        cx = x + bw * 0.5
        cy = y + bh * 0.2
        return cx, cy, side

    side = int(round(min(w, h) * 0.9))
    cx = w / 2.0
    cy = h * 0.38
    return cx, cy, side


def heuristic_head_box(mask: Optional[np.ndarray], width: int, height: int) -> Optional[Tuple[float, float, int]]:
    if mask is None or np.count_nonzero(mask) < 64:
        return None
    x, y, bw, bh = cv2.boundingRect(mask)
    if bw <= 0 or bh <= 0:
        return None

    # Upper-body portrait prior: head is near upper center of foreground bbox.
    side = int(round(max(bw * 0.42, bh * 0.28)))
    side = max(64, min(side, int(min(width, height) * 0.85)))
    cx = x + bw * 0.5
    cy = y + bh * 0.16
    return cx, cy, side


def process_one(
    detectors: Dict[str, object],
    src: Path,
    dst: Path,
    size: int,
    padding: float,
    y_offset_ratio: float,
    jpeg_quality: int,
    use_heuristic: bool,
) -> Tuple[str, int]:
    image, alpha_mask = read_image(src)
    if image is None:
        return "error_read", 0
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return "error_read", 0

    candidates = dedupe_candidates_nms(
        detect_candidates(detectors, image, relaxed=False),
        iou_thresh=0.42,
    )
    relaxed_used = False
    if not candidates:
        candidates = dedupe_candidates_nms(
            detect_candidates(detectors, image, relaxed=True),
            iou_thresh=0.38,
        )
        relaxed_used = len(candidates) > 0

    if candidates:
        ranked = rank_candidates(candidates, w, h)
        cand_count = len(ranked)
        selected = None
        for det_name, _conf, (x, y, fw, fh) in ranked:
            if is_plausible_face((x, y, fw, fh), w, h, alpha_mask):
                selected = (det_name, x, y, fw, fh)
                break
        if selected is not None:
            det_name, x, y, fw, fh = selected
            side = max(16, int(round(max(fw, fh) * padding)))
            cx = x + fw / 2.0
            cy = y + fh / 2.0 + side * y_offset_ratio
            crop = crop_square(image, cx, cy, side)
            status = f"ok_{det_name}"
            if cand_count > 1:
                status += "_multi"
            if relaxed_used:
                status += "_relaxed"
            face_count = cand_count
        else:
            # Detections exist but all look implausible: keep for manual review.
            cx, cy, side = smart_fallback(alpha_mask, image)
            crop = crop_square(image, cx, cy, side)
            status = "suspicious_auto"
            face_count = cand_count
    else:
        heur = heuristic_head_box(alpha_mask, w, h) if use_heuristic else None
        if use_heuristic and heur is not None:
            cx, cy, side = heur
            crop = crop_square(image, cx, cy, side)
            status = "ok_heuristic"
            face_count = 0
        else:
            cx, cy, side = smart_fallback(alpha_mask, image)
            crop = crop_square(image, cx, cy, side)
            status = "fallback_no_face"
            face_count = 0

    interp = cv2.INTER_AREA if crop.shape[0] >= size else cv2.INTER_CUBIC
    avatar = cv2.resize(crop, (size, size), interpolation=interp)
    write_jpg(dst, avatar, jpeg_quality)
    return status, face_count


def manual_review_fallbacks(
    fallback_items: List[Tuple[Path, Path]],
    size: int,
    jpeg_quality: int,
) -> Tuple[Dict[str, str], bool]:
    if not fallback_items:
        return {}, False

    window_name = "Avatar Manual Review"
    zoom_trackbar = "ViewZoom(%)"
    updated_status: Dict[str, str] = {}
    aborted = False

    def wheel_delta(flags: int) -> int:
        # Prefer OpenCV's official decoder if available.
        if hasattr(cv2, "getMouseWheelDelta"):
            try:
                d = int(cv2.getMouseWheelDelta(flags))
                if d != 0:
                    return d
            except Exception:
                pass
        raw = (int(flags) >> 16) & 0xFFFF
        if raw >= 0x8000:
            raw -= 0x10000
        return int(raw)

    viewport_w = 1200
    viewport_h = 1000

    try:
        # GUI_NORMAL removes Qt expanded tool controls that may hijack wheel events.
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(window_name, viewport_w, viewport_h)
        cv2.createTrackbar(zoom_trackbar, window_name, 100, 300, lambda _x: None)
    except Exception as exc:  # noqa: BLE001
        print(f"warn: cannot open GUI window, skip manual review: {exc}")
        return {}, False

    try:
        for idx, (src, dst) in enumerate(fallback_items, start=1):
            image, alpha_mask = read_image(src)
            if image is None:
                updated_status[str(src)] = "error_read"
                continue
            h, w = image.shape[:2]
            if h <= 0 or w <= 0:
                updated_status[str(src)] = "error_read"
                continue

            init_cx, init_cy, init_side = smart_fallback(alpha_mask, image)
            min_dim = max(1, min(w, h))
            state = {
                "cx": float(init_cx),
                "cy": float(init_cy),
                "side": int(max(32, init_side)),
                "view_scale": 1.0,
                "map_scale": 1.0,
                "map_crop_x": 0,
                "map_crop_y": 0,
                "map_pad_x": 0,
                "map_pad_y": 0,
                "map_crop_w": 0,
                "map_crop_h": 0,
            }

            base_scale = min(float(viewport_w) / float(w), float(viewport_h) / float(h))
            cv2.setTrackbarPos(zoom_trackbar, window_name, 100)

            def on_mouse(event, x, y, flags, _param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    scale = max(1e-6, float(state["map_scale"]))
                    dx = int(x) + int(state["map_crop_x"]) - int(state["map_pad_x"])
                    dy = int(y) + int(state["map_crop_y"]) - int(state["map_pad_y"])
                    if (
                        dx < 0
                        or dy < 0
                        or dx >= int(state["map_crop_w"])
                        or dy >= int(state["map_crop_h"])
                    ):
                        return
                    state["cx"] = float(dx / scale)
                    state["cy"] = float(dy / scale)
                elif event == cv2.EVENT_MOUSEWHEEL:
                    delta = wheel_delta(flags)
                    if delta > 0:
                        # Wheel up: enlarge box
                        state["side"] = int(round(state["side"] * 1.08))
                    elif delta < 0:
                        # Wheel down: shrink box
                        state["side"] = int(round(state["side"] * 0.92))

            cv2.setMouseCallback(window_name, on_mouse)

            while True:
                zoom_pct = cv2.getTrackbarPos(zoom_trackbar, window_name)
                zoom_pct = min(300, max(25, int(zoom_pct)))
                state["view_scale"] = base_scale * (zoom_pct / 100.0)
                view = image.copy()
                side = int(max(32, min(max(w, h) * 2, state["side"])))
                cx = float(state["cx"])
                cy = float(state["cy"])
                half = side / 2.0
                x1 = int(round(cx - half))
                y1 = int(round(cy - half))
                x2 = x1 + side
                y2 = y1 + side

                cv2.rectangle(view, (x1, y1), (x2, y2), (20, 210, 255), 2)
                cv2.drawMarker(
                    view,
                    (int(round(cx)), int(round(cy))),
                    (255, 120, 20),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=20,
                    thickness=2,
                )

                scale = float(state["view_scale"])
                disp_w = max(1, int(round(w * scale)))
                disp_h = max(1, int(round(h * scale)))
                interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
                disp = cv2.resize(view, (disp_w, disp_h), interpolation=interp)

                # Keep current face box around viewport center when zoomed in.
                focus_x = int(round(cx * scale))
                focus_y = int(round(cy * scale))
                crop_x = 0
                crop_y = 0
                if disp_w > viewport_w:
                    crop_x = max(0, min(disp_w - viewport_w, focus_x - viewport_w // 2))
                if disp_h > viewport_h:
                    crop_y = max(0, min(disp_h - viewport_h, focus_y - viewport_h // 2))

                crop_w = min(viewport_w, disp_w - crop_x)
                crop_h = min(viewport_h, disp_h - crop_y)
                disp_crop = disp[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
                canvas = np.full((viewport_h, viewport_w, 3), 235, dtype=np.uint8)
                pad_x = (viewport_w - crop_w) // 2
                pad_y = (viewport_h - crop_h) // 2
                canvas[pad_y : pad_y + crop_h, pad_x : pad_x + crop_w] = disp_crop

                state["map_scale"] = scale
                state["map_crop_x"] = crop_x
                state["map_crop_y"] = crop_y
                state["map_pad_x"] = pad_x
                state["map_pad_y"] = pad_y
                state["map_crop_w"] = disp_w
                state["map_crop_h"] = disp_h

                tip1 = (
                    f"{idx}/{len(fallback_items)} {src.name} | "
                    "Click center | wheel/[ ] resize box | slider zoom view | WASD move | Enter save | Q quit"
                )
                tip2 = f"box={side}px  zoom={zoom_pct}%"
                cv2.putText(
                    canvas,
                    tip1,
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (20, 20, 20),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    tip1,
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    tip2,
                    (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (20, 20, 20),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    tip2,
                    (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                cv2.imshow(window_name, canvas)
                key_ex = cv2.waitKeyEx(30)
                key = key_ex & 0xFF
                if key in (13, 10, 32):  # Enter / Space
                    crop = crop_square(image, state["cx"], state["cy"], state["side"])
                    interp = cv2.INTER_AREA if crop.shape[0] >= size else cv2.INTER_CUBIC
                    avatar = cv2.resize(crop, (size, size), interpolation=interp)
                    write_jpg(dst, avatar, jpeg_quality)
                    updated_status[str(src)] = "ok_manual_click"
                    break
                if key in (ord("q"), 27):  # q / Esc
                    aborted = True
                    break
                if key in (ord("["), ord("-"), ord("_")):
                    state["side"] = int(round(state["side"] * 0.93))
                if key in (ord("]"), ord("="), ord("+")):
                    state["side"] = int(round(state["side"] * 1.07))
                move_step = max(2, int(round(state["side"] * 0.03)))
                # Arrow keys (waitKeyEx on Windows): left/right/up/down
                if key_ex in (2424832,):
                    state["cx"] -= move_step
                if key_ex in (2555904,):
                    state["cx"] += move_step
                if key_ex in (2490368,):
                    state["cy"] -= move_step
                if key_ex in (2621440,):
                    state["cy"] += move_step
                if key in (ord("a"), ord("A")):
                    state["cx"] -= move_step
                if key in (ord("d"), ord("D")):
                    state["cx"] += move_step
                if key in (ord("w"), ord("W")):
                    state["cy"] -= move_step
                if key in (ord("s"), ord("S")):
                    state["cy"] += move_step
                if key in (ord("r"), ord("R")):
                    state["cx"], state["cy"], state["side"] = float(init_cx), float(init_cy), int(init_side)
                    cv2.setTrackbarPos(zoom_trackbar, window_name, 100)

                state["cx"] = float(min(max(0, state["cx"]), w - 1))
                state["cy"] = float(min(max(0, state["cy"]), h - 1))
                state["side"] = int(min(max(32, state["side"]), max(w, h) * 2))

            if aborted:
                break
    finally:
        cv2.destroyWindow(window_name)

    return updated_status, aborted


def summarize_status_counts(report_rows: List[Tuple[str, str, str, int]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for _src, _dst, status, _face_count in report_rows:
        out[status] = out.get(status, 0) + 1
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch crop Arknights portraits into fixed-size avatars."
    )
    parser.add_argument("--input-dir", default="Subjects", help="Source root directory.")
    parser.add_argument("--output-dir", default="Subjects-avatar", help="Output root directory.")
    parser.add_argument("--size", type=int, default=512, help="Avatar width/height.")
    parser.add_argument("--padding", type=float, default=2.25, help="Face box expansion ratio.")
    parser.add_argument(
        "--y-offset-ratio",
        type=float,
        default=0.10,
        help="Vertical center offset, as a ratio of crop side. Positive places face higher in frame.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality 1-100.")
    parser.add_argument(
        "--detectors",
        default="anime,yunet,haar",
        help="Comma-separated detector order. Options: anime,yunet,haar",
    )
    parser.add_argument(
        "--anime-cascade-path",
        default=".models/lbpcascade_animeface.xml",
        help="Path to anime cascade XML.",
    )
    parser.add_argument(
        "--yunet-model-path",
        default=".models/face_detection_yunet_2023mar.onnx",
        help="Path to YuNet ONNX model.",
    )
    parser.add_argument("--max", type=int, default=0, help="Limit number of images.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JPG output.")
    parser.add_argument("--dry-run", action="store_true", help="Only scan and print summary.")
    parser.add_argument(
        "--report-csv",
        default="avatar_crop_report.csv",
        help="CSV report path for statuses.",
    )
    parser.add_argument(
        "--use-heuristic",
        action="store_true",
        help="Enable foreground-based head heuristic when no face is detected.",
    )
    parser.add_argument(
        "--manual-fallback-review",
        action="store_true",
        help="After auto pass, open clickable UI to manually fix fallback_no_face images.",
    )
    args = parser.parse_args()

    if args.size <= 0:
        raise ValueError("--size must be > 0")
    if not (1 <= args.jpeg_quality <= 100):
        raise ValueError("--jpeg-quality must be between 1 and 100")
    if args.padding <= 0:
        raise ValueError("--padding must be > 0")

    detector_order = parse_detector_list(args.detectors)
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    report_csv = Path(args.report_csv).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    images = list_images(input_dir)
    if args.max > 0:
        images = images[: args.max]

    print(f"Found images: {len(images)}")
    if not images:
        return 0

    if args.dry_run:
        for p in images[:20]:
            print(p)
        if len(images) > 20:
            print(f"... ({len(images) - 20} more)")
        return 0

    detectors = load_detectors(
        detector_order=detector_order,
        anime_cascade_path=Path(args.anime_cascade_path),
        yunet_model_path=Path(args.yunet_model_path),
    )
    if not detectors:
        raise RuntimeError("No detector loaded. Check model download/network settings.")

    report_rows = []
    total = len(images)
    for i, src in enumerate(images, start=1):
        rel = src.relative_to(input_dir)
        dst = (output_dir / rel).with_suffix(".jpg")

        if dst.exists() and not args.overwrite:
            print(f"[{i}/{total}] skip: {dst}")
            report_rows.append((str(src), str(dst), "skip_exists", 0))
            continue

        try:
            status, face_count = process_one(
                detectors=detectors,
                src=src,
                dst=dst,
                size=args.size,
                padding=args.padding,
                y_offset_ratio=args.y_offset_ratio,
                jpeg_quality=args.jpeg_quality,
                use_heuristic=args.use_heuristic,
            )
            report_rows.append((str(src), str(dst), status, face_count))
            print(f"[{i}/{total}] {status}: {dst}")
        except Exception as exc:  # noqa: BLE001
            report_rows.append((str(src), str(dst), "error_write", 0))
            print(f"[{i}/{total}] error_write: {src} ({exc})")

    if args.manual_fallback_review:
        fallback_items: List[Tuple[Path, Path]] = []
        for src_s, dst_s, status, _fc in report_rows:
            if status in {"fallback_no_face", "suspicious_auto"}:
                fallback_items.append((Path(src_s), Path(dst_s)))
        if fallback_items:
            print(f"manual review: {len(fallback_items)} images (fallback + suspicious)")
            update_map, aborted = manual_review_fallbacks(
                fallback_items=fallback_items,
                size=args.size,
                jpeg_quality=args.jpeg_quality,
            )
            if aborted:
                print("manual review aborted by user (remaining items stay unchanged).")
            if update_map:
                new_rows = []
                for src_s, dst_s, status, fc in report_rows:
                    if status in {"fallback_no_face", "suspicious_auto"} and src_s in update_map:
                        new_rows.append((src_s, dst_s, update_map[src_s], fc))
                    else:
                        new_rows.append((src_s, dst_s, status, fc))
                report_rows = new_rows
        else:
            print("manual review: no fallback/suspicious images to process")

    counts = summarize_status_counts(report_rows)

    report_csv.parent.mkdir(parents=True, exist_ok=True)
    with report_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "output", "status", "face_count"])
        writer.writerows(report_rows)

    needs_review = report_csv.with_name(report_csv.stem + "_needs_review.txt")
    with needs_review.open("w", encoding="utf-8") as f:
        for src, dst, status, face_count in report_rows:
            if status in {
                "fallback_no_face",
                "suspicious_auto",
                "error_read",
                "error_write",
                "ok_heuristic",
            } or "_multi" in status:
                f.write(f"{status}\t{face_count}\t{src}\t{dst}\n")

    print("Done.")
    print("Summary:")
    for k in sorted(counts.keys()):
        print(f"  {k}: {counts[k]}")
    print(f"Report CSV: {report_csv}")
    print(f"Needs review: {needs_review}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
