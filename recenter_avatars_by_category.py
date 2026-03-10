#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import cv2
    import numpy as np
except Exception as exc:  # noqa: BLE001
    raise RuntimeError(
        "Missing dependencies. Use your conda env with opencv-python and numpy."
    ) from exc


VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def norm_path(p: str) -> str:
    return str(Path(p)).replace("\\", "/").lower().strip()


def normalize_key(text: str) -> str:
    s = str(text).lower()
    return "".join(ch for ch in s if ch.isalnum())


def row_matches_category(row: dict, category_raw: str) -> bool:
    if not category_raw:
        return True
    token_norm = norm_path(category_raw).strip("/")
    token_name = Path(category_raw).name.lower().strip()
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


def write_jpg(path: Path, image: np.ndarray, quality: int) -> None:
    ok, enc = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError(f"JPEG encode failed: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    enc.tofile(str(path))


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
        src_gray = cv2.resize(
            src_gray,
            (max(32, int(round(src_w * src_scale))), max(32, int(round(src_h * src_scale)))),
            interpolation=cv2.INTER_AREA,
        )

    edge_src = cv2.Canny(src_gray, 60, 180)
    min_small = float(min(src_gray.shape[:2]))
    coarse_fracs = [0.18, 0.22, 0.26, 0.30, 0.35, 0.40, 0.46, 0.53, 0.61, 0.70, 0.80, 0.90]

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
        return float(0.72 * max1 + 0.28 * max2), loc1, side_small

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
    step = max(6, int(round(best_side * 0.03)))
    for side_small in range(max(40, int(round(best_side * 0.84))), int(round(best_side * 1.18)) + 1, step):
        candidate = evaluate(side_small)
        if candidate is None:
            continue
        if candidate[0] > best[0]:
            best = candidate

    if best[0] < 0.20:
        return None
    loc_x, loc_y = best[1]
    side_small = best[2]
    cx = float(loc_x + side_small / 2.0) / src_scale
    cy = float(loc_y + side_small / 2.0) / src_scale
    side = int(round(side_small / src_scale))
    side = max(32, min(side, int(round(src_min * 1.15))))
    return cx, cy, side, float(best[0])


def parse_center_offset(text: str) -> Tuple[float, float]:
    s = text.strip()
    if not s:
        return 0.0, 0.0
    if "," in s:
        parts = [x.strip() for x in s.split(",")]
        if len(parts) != 2:
            raise ValueError("--center-offset must be one float or dx,dy")
        return float(parts[0]), float(parts[1])
    return 0.0, float(s)


def build_name_index(root: Optional[Path]) -> Dict[str, List[Path]]:
    if root is None:
        return {}
    index: Dict[str, List[Path]] = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in VALID_EXTS:
            continue
        index.setdefault(p.name.lower(), []).append(p.resolve())
    return index


def resolve_existing_path(
    raw_path: str,
    root: Optional[Path],
    name_index: Dict[str, List[Path]],
    category_raw: str,
) -> Optional[Path]:
    p = Path(raw_path)
    if p.exists():
        return p.resolve()
    if root is None:
        return None
    if not p.is_absolute():
        q = (root / p)
        if q.exists():
            return q.resolve()
    q = root / p.name
    if q.exists():
        return q.resolve()
    cands = list(name_index.get(p.name.lower(), []))
    if not cands:
        return None
    token_norm = norm_path(category_raw).strip("/")
    if token_norm:
        filtered = [c for c in cands if token_norm in norm_path(str(c))]
        if filtered:
            cands = filtered
    token_key = normalize_key(Path(category_raw).name or category_raw)
    if token_key and len(cands) > 1:
        filtered = [c for c in cands if token_key in normalize_key(str(c))]
        if filtered:
            cands = filtered
    cands = sorted(cands, key=lambda x: len(str(x)))
    return cands[0]


def infer_output_size(path: Path, default_size: int) -> int:
    img, _ = read_image(path)
    if img is None:
        return default_size
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return default_size
    return int(max(32, min(h, w)))


def load_rows(path: Path) -> Tuple[List[dict], List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    for need in ["source", "output", "status", "face_count"]:
        if need not in fieldnames:
            fieldnames.append(need)
    return rows, fieldnames


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto re-center processed avatars by category and regenerate from source image."
    )
    parser.add_argument("--report-csv", default="avatar_crop_report.csv")
    parser.add_argument("--report-out", default="")
    parser.add_argument("--category", required=True, help="Category token/path to match.")
    parser.add_argument(
        "--center-offset",
        default="0",
        help=(
            "Center offset ratio relative to crop side. "
            "Use one value for dy, or dx,dy. Positive y moves crop center down."
        ),
    )
    parser.add_argument("--size", type=int, default=0, help="Output avatar size. 0 = infer from old avatar.")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--source-root", default="", help="Optional source root remap for moved files.")
    parser.add_argument("--output-root", default="", help="Optional output root remap for moved files.")
    parser.add_argument("--status-label", default="ok_recenter_auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--keep-original-report-paths",
        action="store_true",
        help="Do not rewrite source/output paths in report to resolved local paths.",
    )
    args = parser.parse_args()

    if not (1 <= args.jpeg_quality <= 100):
        raise ValueError("--jpeg-quality must be between 1 and 100")
    if args.size < 0:
        raise ValueError("--size must be >= 0")

    offset_x_ratio, offset_y_ratio = parse_center_offset(args.center_offset)
    report_path = Path(args.report_csv).resolve()
    if not report_path.exists():
        raise FileNotFoundError(f"Report CSV not found: {report_path}")
    report_out = Path(args.report_out).resolve() if args.report_out.strip() else report_path

    rows, fieldnames = load_rows(report_path)
    targets = [i for i, r in enumerate(rows) if row_matches_category(r, args.category)]
    if args.limit > 0:
        targets = targets[: args.limit]

    print(f"matched targets: {len(targets)}")
    for i in targets[:20]:
        row = rows[i]
        print("  -", row.get("source", ""), "=>", row.get("output", ""))
    if args.dry_run:
        return 0

    src_root = Path(args.source_root).resolve() if args.source_root.strip() else None
    out_root = Path(args.output_root).resolve() if args.output_root.strip() else None
    src_index = build_name_index(src_root)
    out_index = build_name_index(out_root)

    counts: Dict[str, int] = {}

    for n, idx in enumerate(targets, start=1):
        row = rows[idx]
        src = resolve_existing_path(row.get("source", ""), src_root, src_index, args.category)
        out = resolve_existing_path(row.get("output", ""), out_root, out_index, args.category)

        if src is None:
            row["status"] = "error_source_missing"
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            print(f"[{n}/{len(targets)}] source missing: {row.get('source', '')}")
            continue
        if out is None:
            row["status"] = "error_output_missing"
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            print(f"[{n}/{len(targets)}] output missing: {row.get('output', '')}")
            continue

        if not args.keep_original_report_paths:
            row["source"] = str(src)
            row["output"] = str(out)

        src_img, _ = read_image(src)
        if src_img is None:
            row["status"] = "error_read_source"
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            print(f"[{n}/{len(targets)}] read source fail: {src}")
            continue

        matched = estimate_box_from_existing_avatar(src_img, out)
        if matched is None:
            row["status"] = "recenter_match_fail"
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            print(f"[{n}/{len(targets)}] match fail: {src.name}")
            continue

        cx, cy, side, score = matched
        cx += side * offset_x_ratio
        cy += side * offset_y_ratio
        crop = crop_square(src_img, cx, cy, int(side))
        out_size = args.size if args.size > 0 else infer_output_size(out, default_size=512)
        interp = cv2.INTER_AREA if crop.shape[0] >= out_size else cv2.INTER_CUBIC
        avatar = cv2.resize(crop, (out_size, out_size), interpolation=interp)
        write_jpg(out, avatar, args.jpeg_quality)

        row["status"] = args.status_label
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        print(
            f"[{n}/{len(targets)}] ok: {src.name} score={score:.2f} side={int(side)} "
            f"offset=({offset_x_ratio:.3f},{offset_y_ratio:.3f})"
        )

    with report_out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("Summary:")
    for k in sorted(counts.keys()):
        print(f"  {k}: {counts[k]}")
    print(f"Report written: {report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
