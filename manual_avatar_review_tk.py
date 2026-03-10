#!/usr/bin/env python3
import argparse
import csv
import tkinter as tk
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


REVIEW_DEFAULT = {"fallback_no_face", "suspicious_auto"}
REVIEW_EXTRA = {"error_read", "error_write", "ok_heuristic"}


def norm_path(p: str) -> str:
    return str(Path(p)).replace("\\", "/").lower().strip()


def normalize_key(text: str) -> str:
    s = str(text).lower()
    return "".join(ch for ch in s if ch.isalnum())


def row_matches_status(
    status: str,
    review_statuses: Optional[set[str]],
    include_multi: bool,
    include_extra: bool,
    force_review_image: bool,
    review_image_raw: str,
) -> bool:
    if force_review_image and review_image_raw:
        return True
    if review_statuses is not None:
        if status in review_statuses:
            return True
        if "_multi" in review_statuses and "_multi" in status:
            return True
        return False
    if status in REVIEW_DEFAULT:
        return True
    if include_extra and status in REVIEW_EXTRA:
        return True
    return include_multi and "_multi" in status


def row_matches_image(row: dict, review_image_raw: str) -> bool:
    if not review_image_raw:
        return True
    token_norm = norm_path(review_image_raw)
    token_name = Path(review_image_raw).name.lower()
    token_stem = Path(review_image_raw).stem.lower()
    token_key = normalize_key(token_stem or token_name or token_norm)
    src = row.get("source", "")
    out = row.get("output", "")
    src_norm = norm_path(src)
    out_norm = norm_path(out)
    src_name = Path(src).name.lower()
    out_name = Path(out).name.lower()
    src_stem = Path(src).stem.lower()
    out_stem = Path(out).stem.lower()
    src_key = normalize_key(src_stem or src_name or src_norm)
    out_key = normalize_key(out_stem or out_name or out_norm)

    if src_norm == token_norm or out_norm == token_norm:
        return True
    if src_name == token_name or out_name == token_name:
        return True
    if token_stem and (src_stem == token_stem or out_stem == token_stem):
        return True
    if token_norm and (token_norm in src_norm or token_norm in out_norm):
        return True
    if token_name and (token_name in src_name or token_name in out_name):
        return True
    if token_stem and (token_stem in src_stem or token_stem in out_stem):
        return True
    if token_key and (
        token_key in src_key
        or token_key in out_key
        or src_key in token_key
        or out_key in token_key
    ):
        return True
    return False


def row_matches_category(row: dict, review_category_raw: str) -> bool:
    if not review_category_raw:
        return True
    token_norm = norm_path(review_category_raw).strip("/")
    token_name = Path(review_category_raw).name.lower().strip()
    token_key = normalize_key(token_name or token_norm)
    src_norm = norm_path(row.get("source", ""))
    out_norm = norm_path(row.get("output", ""))
    if token_norm and (token_norm in src_norm or token_norm in out_norm):
        return True
    if token_name:
        if f"/{token_name}/" in src_norm or f"/{token_name}/" in out_norm:
            return True
        if src_norm.endswith(f"/{token_name}") or out_norm.endswith(f"/{token_name}"):
            return True
    if token_key:
        src_key = normalize_key(src_norm)
        out_key = normalize_key(out_norm)
        if token_key in src_key or token_key in out_key:
            return True
    return False


def read_image(path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None, None
    decoded = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        return None, None
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
    fg = (dist > 18).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return None
    min_area = int(h * w * 0.03)
    best, best_area = -1, 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > best_area and area >= min_area:
            best, best_area = i, area
    if best < 0:
        return None
    out = np.zeros_like(fg)
    out[labels == best] = 255
    return out


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
    return w / 2.0, h * 0.38, side


def extract_foreground_square(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if h < 8 or w < 8:
        return image
    m = max(4, int(round(min(h, w) * 0.08)))
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
    diff = np.max(np.abs(image.astype(np.int16) - bg[None, None, :]), axis=2)
    fg = (diff > 14).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    ys, xs = np.where(fg > 0)
    if ys.size < max(64, int(h * w * 0.02)):
        return image
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    bw = x2 - x1
    bh = y2 - y1
    side = int(round(max(bw, bh) * 1.12))
    side = max(32, min(side, max(h, w)))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    ix1 = int(round(cx - side / 2.0))
    iy1 = int(round(cy - side / 2.0))
    ix1 = max(0, min(ix1, w - side))
    iy1 = max(0, min(iy1, h - side))
    return image[iy1 : iy1 + side, ix1 : ix1 + side]


def estimate_box_from_existing_avatar(
    source_img: np.ndarray,
    output_path: Path,
) -> Optional[Tuple[float, float, int, float]]:
    if not output_path.exists():
        return None
    avatar_img, _ = read_image(output_path)
    if avatar_img is None:
        return None

    src_h, src_w = source_img.shape[:2]
    src_min = min(src_h, src_w)
    if src_min < 96:
        return None

    template = extract_foreground_square(avatar_img)
    if min(template.shape[:2]) < 24:
        return None

    src_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    max_dim = max(src_h, src_w)
    src_scale = 1.0
    if max_dim > 1500:
        src_scale = 1500.0 / float(max_dim)
        new_w = max(32, int(round(src_w * src_scale)))
        new_h = max(32, int(round(src_h * src_scale)))
        src_gray = cv2.resize(src_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    edge_src = cv2.Canny(src_gray, 60, 180)
    min_small = float(min(src_gray.shape[:2]))
    coarse_fracs = [
        0.18, 0.22, 0.26, 0.30, 0.35, 0.40, 0.46, 0.53, 0.61, 0.70, 0.80, 0.90
    ]

    def evaluate(side_small: int) -> Optional[Tuple[float, Tuple[int, int], int]]:
        if side_small < 40:
            return None
        if side_small >= src_gray.shape[0] or side_small >= src_gray.shape[1]:
            return None
        interp = cv2.INTER_AREA if tpl_gray.shape[0] > side_small else cv2.INTER_CUBIC
        tpl = cv2.resize(tpl_gray, (side_small, side_small), interpolation=interp)
        if tpl.shape[0] >= src_gray.shape[0] or tpl.shape[1] >= src_gray.shape[1]:
            return None
        res1 = cv2.matchTemplate(src_gray, tpl, cv2.TM_CCOEFF_NORMED)
        _, max1, _, loc1 = cv2.minMaxLoc(res1)

        edge_tpl = cv2.Canny(tpl, 60, 180)
        res2 = cv2.matchTemplate(edge_src, edge_tpl, cv2.TM_CCOEFF_NORMED)
        _, max2, _, _ = cv2.minMaxLoc(res2)

        score = float(0.72 * max1 + 0.28 * max2)
        return score, loc1, side_small

    best: Optional[Tuple[float, Tuple[int, int], int]] = None
    for frac in coarse_fracs:
        candidate = evaluate(int(round(min_small * frac)))
        if candidate is None:
            continue
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return None

    best_side = best[2]
    fine_start = max(40, int(round(best_side * 0.84)))
    fine_end = int(round(best_side * 1.18))
    for side_small in range(fine_start, fine_end + 1, max(6, int(round(best_side * 0.03)))):
        candidate = evaluate(side_small)
        if candidate is None:
            continue
        if candidate[0] > best[0]:
            best = candidate

    if best[0] < 0.20:
        return None

    loc_x, loc_y = best[1]
    side_small = best[2]
    cx_small = float(loc_x + side_small / 2.0)
    cy_small = float(loc_y + side_small / 2.0)
    cx = cx_small / src_scale
    cy = cy_small / src_scale
    side = int(round(side_small / src_scale))
    side = max(32, min(side, int(round(src_min * 1.15))))
    return cx, cy, side, float(best[0])


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
    pl = max(0, -x1)
    pt = max(0, -y1)
    pr = max(0, x2 - w)
    pb = max(0, y2 - h)
    if pl or pt or pr or pb:
        image = cv2.copyMakeBorder(
            image, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=border_color
        )
        x1 += pl
        y1 += pt
        x2 += pl
        y2 += pt
    return image[y1:y2, x1:x2]


def write_jpg(path: Path, image: np.ndarray, quality: int) -> None:
    ok, enc = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError(f"JPEG encode failed: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    enc.tofile(str(path))


def render_preview(
    image: np.ndarray,
    cx: float,
    cy: float,
    side: int,
    viewport_w: int,
    viewport_h: int,
    zoom_pct: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    h, w = image.shape[:2]
    base = min(float(viewport_w) / float(w), float(viewport_h) / float(h))
    scale = base * (float(zoom_pct) / 100.0)
    scale = max(0.05, scale)

    view = image.copy()
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

    disp_w = max(1, int(round(w * scale)))
    disp_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    disp = cv2.resize(view, (disp_w, disp_h), interpolation=interp)

    focus_x = int(round(cx * scale))
    focus_y = int(round(cy * scale))
    crop_x = max(0, min(max(0, disp_w - viewport_w), focus_x - viewport_w // 2))
    crop_y = max(0, min(max(0, disp_h - viewport_h), focus_y - viewport_h // 2))
    crop_w = min(viewport_w, disp_w - crop_x)
    crop_h = min(viewport_h, disp_h - crop_y)

    disp_crop = disp[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
    canvas = np.full((viewport_h, viewport_w, 3), 235, dtype=np.uint8)
    pad_x = (viewport_w - crop_w) // 2
    pad_y = (viewport_h - crop_h) // 2
    canvas[pad_y : pad_y + crop_h, pad_x : pad_x + crop_w] = disp_crop

    mapping = {
        "scale": scale,
        "crop_x": crop_x,
        "crop_y": crop_y,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "disp_w": disp_w,
        "disp_h": disp_h,
    }
    return canvas, mapping


def np_to_photoimage_bgr(img_bgr: np.ndarray) -> tk.PhotoImage:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    header = f"P6 {w} {h} 255\n".encode("ascii")
    data = header + rgb.tobytes()
    return tk.PhotoImage(data=data, format="PPM")


class ManualReviewer:
    def __init__(
        self,
        rows: List[dict],
        report_path: Path,
        viewport_w: int,
        viewport_h: int,
        size: int,
        jpeg_quality: int,
        limit: int,
        include_multi: bool,
        include_extra: bool,
        review_statuses: Optional[set[str]],
        review_image: str,
        force_review_image: bool,
        review_category: str,
        preselected_targets: Optional[List[int]] = None,
    ) -> None:
        self.rows = rows
        self.report_path = report_path
        self.viewport_w = viewport_w
        self.viewport_h = viewport_h
        self.size = size
        self.jpeg_quality = jpeg_quality
        self.include_multi = include_multi
        self.include_extra = include_extra
        self.review_statuses = review_statuses
        self.force_review_image = force_review_image
        self.review_image_raw = review_image.strip()
        self.review_category_raw = review_category.strip()
        self.targets = preselected_targets if preselected_targets is not None else [
            i
            for i, r in enumerate(rows)
            if self.should_review(r)
            and (self.match_review_image(r) or self.match_review_image_output(r))
            and row_matches_category(r, self.review_category_raw)
        ]
        if limit > 0:
            self.targets = self.targets[:limit]
        self.target_pos = 0
        self.cur_img = None
        self.cur_mask = None
        self.cur_h = 0
        self.cur_w = 0
        self.cx = 0.0
        self.cy = 0.0
        self.side = 100
        self.mapping = {}
        self.photo = None
        self.init_note = "init=fallback"
        self.done = False

        self.root = tk.Tk()
        self.root.title("Avatar Manual Review (Tk)")
        self.root.geometry(f"{viewport_w}x{viewport_h + 120}")

        top = tk.Frame(self.root)
        top.pack(fill=tk.X, padx=8, pady=6)
        self.info_var = tk.StringVar(value="")
        tk.Label(top, textvariable=self.info_var, anchor="w").pack(fill=tk.X)

        ctl = tk.Frame(self.root)
        ctl.pack(fill=tk.X, padx=8)
        tk.Label(ctl, text="ViewZoom").pack(side=tk.LEFT)
        self.zoom_scale = tk.Scale(
            ctl,
            from_=25,
            to=300,
            orient=tk.HORIZONTAL,
            length=260,
            command=lambda _v: self.redraw(),
        )
        self.zoom_scale.set(100)
        self.zoom_scale.pack(side=tk.LEFT, padx=8)
        tk.Label(
            ctl,
            text="Left click center | Wheel/[ ] box size | WASD/Arrows move | Enter save-next | N skip | Q quit",
            anchor="w",
        ).pack(side=tk.LEFT, padx=10)

        self.canvas = tk.Canvas(
            self.root, width=viewport_w, height=viewport_h, bg="#ebebeb", highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self.on_wheel_linux(+120))
        self.canvas.bind("<Button-5>", lambda e: self.on_wheel_linux(-120))
        self.root.bind("<Key>", self.on_key)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        if not self.targets:
            self.info_var.set("No matched rows to review (check --review-statuses/--review-image).")
            self.done = True
            self.redraw_blank()
        else:
            print(f"manual-review targets: {len(self.targets)}")
            for i in self.targets[:10]:
                print("  -", self.rows[i].get("source", ""))
            self.load_current()

    def should_review(self, row: dict) -> bool:
        return row_matches_status(
            status=row.get("status", ""),
            review_statuses=self.review_statuses,
            include_multi=self.include_multi,
            include_extra=self.include_extra,
            force_review_image=self.force_review_image,
            review_image_raw=self.review_image_raw,
        )

    def match_review_image(self, row: dict) -> bool:
        return row_matches_image(row, self.review_image_raw)

    def match_review_image_output(self, row: dict) -> bool:
        return row_matches_image(row, self.review_image_raw)

    def load_current(self) -> None:
        if self.target_pos >= len(self.targets):
            self.done = True
            self.info_var.set("Review complete. You can close this window.")
            self.redraw_blank()
            self.write_reports()
            return
        idx = self.targets[self.target_pos]
        row = self.rows[idx]
        src = Path(row["source"])
        img, mask = read_image(src)
        if img is None:
            row["status"] = "error_read"
            self.target_pos += 1
            self.load_current()
            return
        self.cur_img = img
        self.cur_mask = mask
        self.cur_h, self.cur_w = img.shape[:2]
        self.cx, self.cy, self.side, self.init_note = self.init_box_for_row(row, img, mask)
        self.zoom_scale.set(100)
        self.redraw()

    def init_box_for_row(
        self,
        row: dict,
        img: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Tuple[float, float, int, str]:
        fallback_cx, fallback_cy, fallback_side = smart_fallback(mask, img)
        fallback_side = int(max(32, fallback_side))
        output_path = Path(row.get("output", ""))
        matched = estimate_box_from_existing_avatar(img, output_path)
        if matched is None:
            return fallback_cx, fallback_cy, fallback_side, "init=fallback"
        cx, cy, side, score = matched
        if not (0 <= cx < img.shape[1] and 0 <= cy < img.shape[0]):
            return fallback_cx, fallback_cy, fallback_side, "init=fallback"
        return cx, cy, int(max(32, side)), f"init=from_avatar({score:.2f})"

    def redraw_blank(self) -> None:
        blank = np.full((self.viewport_h, self.viewport_w, 3), 235, dtype=np.uint8)
        self.photo = np_to_photoimage_bgr(blank)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

    def redraw(self) -> None:
        if self.done or self.cur_img is None:
            return
        canvas_img, self.mapping = render_preview(
            image=self.cur_img,
            cx=self.cx,
            cy=self.cy,
            side=int(self.side),
            viewport_w=self.viewport_w,
            viewport_h=self.viewport_h,
            zoom_pct=int(self.zoom_scale.get()),
        )
        self.photo = np_to_photoimage_bgr(canvas_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

        idx = self.targets[self.target_pos]
        row = self.rows[idx]
        self.info_var.set(
            f"{self.target_pos + 1}/{len(self.targets)} | {Path(row['source']).name} | "
            f"status={row.get('status')} | box={int(self.side)}px | "
            f"zoom={int(self.zoom_scale.get())}% | {self.init_note}"
        )

    def move(self, dx: float, dy: float) -> None:
        self.cx = float(min(max(0.0, self.cx + dx), self.cur_w - 1))
        self.cy = float(min(max(0.0, self.cy + dy), self.cur_h - 1))
        self.redraw()

    def resize_box(self, factor: float) -> None:
        self.side = int(min(max(32, round(self.side * factor)), max(self.cur_w, self.cur_h) * 2))
        self.redraw()

    def map_canvas_to_image(self, x: int, y: int) -> Optional[Tuple[float, float]]:
        m = self.mapping
        if not m:
            return None
        dx = int(x) + int(m["crop_x"]) - int(m["pad_x"])
        dy = int(y) + int(m["crop_y"]) - int(m["pad_y"])
        if dx < 0 or dy < 0 or dx >= int(m["disp_w"]) or dy >= int(m["disp_h"]):
            return None
        scale = max(1e-6, float(m["scale"]))
        ix = dx / scale
        iy = dy / scale
        if ix < 0 or iy < 0 or ix >= self.cur_w or iy >= self.cur_h:
            return None
        return float(ix), float(iy)

    def on_click(self, event) -> None:
        if self.done or self.cur_img is None:
            return
        pt = self.map_canvas_to_image(event.x, event.y)
        if pt is None:
            return
        self.cx, self.cy = pt
        self.redraw()

    def on_wheel_linux(self, delta: int) -> None:
        class E:
            pass
        e = E()
        e.delta = delta
        self.on_wheel(e)

    def on_wheel(self, event) -> None:
        if self.done or self.cur_img is None:
            return
        delta = int(getattr(event, "delta", 0))
        if delta > 0:
            self.resize_box(1.08)
        elif delta < 0:
            self.resize_box(0.92)

    def save_current_and_next(self) -> None:
        if self.done or self.cur_img is None:
            return
        idx = self.targets[self.target_pos]
        row = self.rows[idx]
        crop = crop_square(self.cur_img, self.cx, self.cy, int(self.side))
        interp = cv2.INTER_AREA if crop.shape[0] >= self.size else cv2.INTER_CUBIC
        avatar = cv2.resize(crop, (self.size, self.size), interpolation=interp)
        dst = Path(row["output"])
        write_jpg(dst, avatar, self.jpeg_quality)
        row["status"] = "ok_manual_click"
        self.target_pos += 1
        self.load_current()

    def skip_current(self) -> None:
        if self.done:
            return
        self.target_pos += 1
        self.load_current()

    def on_key(self, event) -> None:
        if self.done:
            if event.keysym in {"q", "Q", "Escape", "Return"}:
                self.on_close()
            return
        key = event.keysym
        step = max(2, int(round(self.side * 0.03)))
        if key in {"Return", "KP_Enter", "space"}:
            self.save_current_and_next()
        elif key in {"q", "Q", "Escape"}:
            self.on_close()
        elif key in {"n", "N"}:
            self.skip_current()
        elif key in {"a", "A", "Left"}:
            self.move(-step, 0)
        elif key in {"d", "D", "Right"}:
            self.move(step, 0)
        elif key in {"w", "W", "Up"}:
            self.move(0, -step)
        elif key in {"s", "S", "Down"}:
            self.move(0, step)
        elif key in {"bracketleft", "minus", "underscore"}:
            self.resize_box(0.93)
        elif key in {"bracketright", "equal", "plus"}:
            self.resize_box(1.07)
        elif key in {"r", "R"}:
            idx = self.targets[self.target_pos]
            row = self.rows[idx]
            self.cx, self.cy, self.side, self.init_note = self.init_box_for_row(
                row, self.cur_img, self.cur_mask
            )
            self.zoom_scale.set(100)
            self.redraw()

    def write_reports(self) -> None:
        with self.report_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["source", "output", "status", "face_count"])
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)

        needs_review = self.report_path.with_name(self.report_path.stem + "_needs_review.txt")
        with needs_review.open("w", encoding="utf-8") as f:
            for row in self.rows:
                status = row.get("status", "")
                if status in REVIEW_DEFAULT:
                    f.write(
                        f"{status}\t{row.get('face_count', 0)}\t{row.get('source','')}\t"
                        f"{row.get('output','')}\n"
                    )
                elif self.include_extra and status in REVIEW_EXTRA:
                    f.write(
                        f"{status}\t{row.get('face_count', 0)}\t{row.get('source','')}\t"
                        f"{row.get('output','')}\n"
                    )
                elif self.include_multi and "_multi" in status:
                    f.write(
                        f"{status}\t{row.get('face_count', 0)}\t{row.get('source','')}\t"
                        f"{row.get('output','')}\n"
                    )

    def on_close(self) -> None:
        self.write_reports()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def load_report_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        out.append(
            {
                "source": row.get("source", ""),
                "output": row.get("output", ""),
                "status": row.get("status", ""),
                "face_count": row.get("face_count", "0"),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manual avatar review UI (Tkinter) for fallback/suspicious rows."
    )
    parser.add_argument("--report-csv", default="avatar_crop_report.csv")
    parser.add_argument("--viewport-width", type=int, default=1200)
    parser.add_argument("--viewport-height", type=int, default=1000)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--include-multi",
        action="store_true",
        help="Also review statuses ending with _multi.",
    )
    parser.add_argument(
        "--include-extra",
        action="store_true",
        help="Also review error_read/error_write/ok_heuristic.",
    )
    parser.add_argument(
        "--review-statuses",
        default="",
        help=(
            "Comma-separated statuses to review. "
            "If set, overrides --include-multi/--include-extra defaults. "
            "Example: fallback_no_face,suspicious_auto or fallback_no_face,_multi"
        ),
    )
    parser.add_argument(
        "--review-image",
        default="",
        help="Only review one image (source path or filename).",
    )
    parser.add_argument(
        "--review-category",
        default="",
        help=(
            "Only review one category directory (match source/output path token). "
            "When set, status filters are ignored unless --review-index is used."
        ),
    )
    parser.add_argument(
        "--force-review-image",
        action="store_true",
        help="When used with --review-image, ignore status filters and review matched image(s) directly.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print matched review targets, do not open UI.",
    )
    parser.add_argument(
        "--review-index",
        default="",
        help=(
            "Review specific CSV row index(es), 1-based. "
            "Examples: 12 or 12,27,35. Overrides status/image filters."
        ),
    )
    args = parser.parse_args()

    report_path = Path(args.report_csv).resolve()
    if not report_path.exists():
        raise FileNotFoundError(f"Report CSV not found: {report_path}")

    rows = load_report_rows(report_path)
    review_statuses: Optional[set[str]] = None
    if args.review_statuses.strip():
        review_statuses = {
            x.strip()
            for x in args.review_statuses.split(",")
            if x.strip()
        }

    preselected_targets: List[int]
    if args.review_index.strip():
        wanted: List[int] = []
        for part in args.review_index.split(","):
            s = part.strip()
            if not s:
                continue
            try:
                idx = int(s)
            except ValueError:
                continue
            if 1 <= idx <= len(rows):
                wanted.append(idx - 1)
        preselected_targets = sorted(set(wanted))
    else:
        if args.review_category.strip():
            preselected_targets = [
                i
                for i, r in enumerate(rows)
                if row_matches_category(r, args.review_category.strip())
                and row_matches_image(r, args.review_image.strip())
            ]
        else:
            preselected_targets = [
                i
                for i, r in enumerate(rows)
                if row_matches_status(
                    status=r.get("status", ""),
                    review_statuses=review_statuses,
                    include_multi=args.include_multi,
                    include_extra=args.include_extra,
                    force_review_image=args.force_review_image,
                    review_image_raw=args.review_image.strip(),
                )
                and row_matches_image(r, args.review_image.strip())
            ]
    if args.limit > 0:
        preselected_targets = preselected_targets[: args.limit]

    print(f"matched targets: {len(preselected_targets)}")
    for i in preselected_targets[:20]:
        print(
            "  -",
            rows[i].get("status", ""),
            rows[i].get("source", ""),
            "=>",
            rows[i].get("output", ""),
        )

    if args.dry_run:
        return 0

    app = ManualReviewer(
        rows=rows,
        report_path=report_path,
        viewport_w=args.viewport_width,
        viewport_h=args.viewport_height,
        size=args.size,
        jpeg_quality=args.jpeg_quality,
        limit=args.limit,
        include_multi=args.include_multi,
        include_extra=args.include_extra,
        review_statuses=review_statuses,
        review_image=args.review_image,
        force_review_image=args.force_review_image,
        review_category=args.review_category,
        preselected_targets=preselected_targets,
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
