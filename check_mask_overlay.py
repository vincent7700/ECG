#!/usr/bin/env python3
"""Visualize mask / image alignment — especially for P2 augmented samples.

Creates one figure per sample showing:
  - Original image
  - Image with ALL masks overlaid (color-coded per mask type)
  - Image + grid_combined only (blue)
  - Image + all_signals only (red)
  - Image + label_centers only (yellow)
  - Individual per-lead signal masks combined (rainbow)

Usage:
    # Single P2 sample
    python stage/check_mask_overlay.py \
        --image data/output_augmentation/images/ECG_031_09_p0_aug.webp \
        --masks data/output_augmentation/masks/ECG_031_09_p0_aug \
        --output stage/overlays/ECG_031_09_p0_aug.png

    # Batch: all (or N random) augmented samples
    python stage/check_mask_overlay.py \
        --dataset data/output_augmentation \
        --output stage/overlays \
        --count 20

    # P1 base samples
    python stage/check_mask_overlay.py \
        --dataset data/output_impression \
        --output stage/overlays_p1 \
        --count 10

    # Side-by-side P1 vs P2 comparison
    python stage/check_mask_overlay.py \
        --compare \
        --p1 data/output_impression \
        --p2 data/output_augmentation \
        --output stage/overlays_compare \
        --count 10
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MASK_SUFFIX_AUG = "_aug"

COLOR_GRID_MAJOR = (30, 144, 255)       # dodger blue
COLOR_GRID_MINOR = (135, 206, 250)      # light sky blue
COLOR_GRID_INTERSEC = (0, 255, 255)     # cyan
COLOR_SIGNAL = (220, 30, 30)            # red
COLOR_LABEL = (255, 215, 0)             # gold
COLOR_HANDWRITING = (255, 140, 0)       # orange
COLOR_PAPER_BOUNDARY = (0, 200, 0)      # green
COLOR_BLACK_SQUARE = (128, 0, 128)      # purple
COLOR_REFERENCE_PULSE = (255, 20, 147)  # pink
COLOR_MEDICAL_TEXT = (255, 105, 180)    # hot pink

IMAGE_EXTS = (".webp", ".png", ".jpg", ".jpeg")


def load_image_rgb(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(mask_dir: Path, name: str) -> np.ndarray | None:
    p = mask_dir / name
    if not p.exists():
        return None
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return m


def fit_mask_to(img_shape: tuple[int, int], mask: np.ndarray) -> np.ndarray:
    h, w = img_shape
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def blend_mask(img: np.ndarray, mask: np.ndarray, color: tuple[int, int, int],
               alpha: float = 0.55, threshold: int = 30) -> np.ndarray:
    """Alpha-blend a binary/grayscale mask onto img in place-safe fashion."""
    if mask is None:
        return img
    mask = fit_mask_to(img.shape[:2], mask)
    # Use mask intensity as a weight, scaled by alpha
    w = (mask.astype(np.float32) / 255.0) * alpha
    w = np.where(mask > threshold, w, 0.0)[:, :, None]
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out = img.astype(np.float32) * (1 - w) + color_arr * w
    return np.clip(out, 0, 255).astype(np.uint8)


def collect_signal_masks(mask_dir: Path) -> list[tuple[str, np.ndarray]]:
    out = []
    for p in sorted(mask_dir.glob("mask_signal_*.png")):
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            out.append((p.stem.replace("mask_signal_", ""), m))
    return out


def per_lead_rainbow(img: np.ndarray, signal_masks: list[tuple[str, np.ndarray]],
                     alpha: float = 0.7) -> np.ndarray:
    """Draw each lead signal in a distinct rainbow color."""
    if not signal_masks:
        return img.copy()
    cmap = plt.get_cmap("hsv")
    n = len(signal_masks)
    out = img.copy()
    for i, (name, m) in enumerate(signal_masks):
        rgba = cmap(i / max(n, 1))
        color = (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
        out = blend_mask(out, m, color, alpha=alpha)
    return out


def pct_nonzero(mask: np.ndarray | None) -> float:
    if mask is None:
        return 0.0
    return 100.0 * np.count_nonzero(mask) / mask.size


def find_mask_bbox(mask: np.ndarray | None) -> tuple[int, int, int, int] | None:
    """Return (x0, y0, x1, y1) bounding box of nonzero pixels, or None."""
    if mask is None:
        return None
    ys, xs = np.where(mask > 30)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def make_zoom_figure(image_path: Path, mask_dir: Path, out_path: Path,
                     n_crops: int = 4, crop_size: int = 600) -> bool:
    """Zoom into random windows of the image+signal overlay to inspect pixel-level alignment."""
    img = load_image_rgb(image_path)
    if img is None:
        return False
    if not mask_dir.is_dir():
        return False

    m_all_sig = load_mask(mask_dir, "mask_all_signals.png")
    m_grid = load_mask(mask_dir, "mask_grid_combined.png")
    if m_all_sig is None:
        print(f"  SKIP zoom (no all_signals mask): {image_path.name}")
        return False
    m_all_sig = fit_mask_to(img.shape[:2], m_all_sig)

    bbox = find_mask_bbox(m_all_sig)
    if bbox is None:
        print(f"  SKIP zoom (empty signal mask): {image_path.name}")
        return False
    x0, y0, x1, y1 = bbox

    overlay = img.copy()
    overlay = blend_mask(overlay, m_grid, COLOR_GRID_MAJOR, alpha=0.35)
    overlay = blend_mask(overlay, m_all_sig, COLOR_SIGNAL, alpha=0.75)

    # Pick crops: evenly along x within the signal bbox, vertically centered
    h, w = img.shape[:2]
    cs = min(crop_size, y1 - y0 + 50, x1 - x0 + 50)
    cs = max(cs, 300)
    xs = np.linspace(x0 + cs // 2, x1 - cs // 2, n_crops).astype(int)
    ys_mid = (y0 + y1) // 2

    fig, axes = plt.subplots(2, n_crops, figsize=(5 * n_crops, 10))
    if n_crops == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    fig.suptitle(f"Zoom {image_path.stem}  —  {w}x{h}  —  sig={pct_nonzero(m_all_sig):.2f}%",
                 fontsize=12, fontweight="bold")

    for i, cx in enumerate(xs):
        cx0 = max(0, min(w - cs, cx - cs // 2))
        cy0 = max(0, min(h - cs, ys_mid - cs // 2))
        raw_crop = img[cy0:cy0 + cs, cx0:cx0 + cs]
        over_crop = overlay[cy0:cy0 + cs, cx0:cx0 + cs]
        axes[0, i].imshow(raw_crop); axes[0, i].set_title(f"crop {i+1} — image")
        axes[1, i].imshow(over_crop); axes[1, i].set_title(f"crop {i+1} — +grille +signaux")

    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return True


def make_figure(image_path: Path, mask_dir: Path, out_path: Path,
                title_prefix: str = "") -> bool:
    img = load_image_rgb(image_path)
    if img is None:
        print(f"  SKIP (cannot read image): {image_path.name}")
        return False
    if not mask_dir.is_dir():
        print(f"  SKIP (mask dir missing): {mask_dir}")
        return False

    # Load all common masks
    m_grid_major = load_mask(mask_dir, "mask_grid_major.png")
    m_grid_minor = load_mask(mask_dir, "mask_grid_minor.png")
    m_grid_combined = load_mask(mask_dir, "mask_grid_combined.png")
    m_grid_inter = load_mask(mask_dir, "mask_grid_intersections.png")
    m_all_signals = load_mask(mask_dir, "mask_all_signals.png")
    m_labels = load_mask(mask_dir, "mask_label_centers.png")
    m_lead_labels = load_mask(mask_dir, "mask_lead_labels.png")
    m_handwriting = load_mask(mask_dir, "mask_handwriting.png")
    m_paper = load_mask(mask_dir, "mask_paper_boundary.png")
    m_black_sq = load_mask(mask_dir, "mask_black_square.png")
    m_ref_pulse = load_mask(mask_dir, "mask_reference_pulse.png")
    m_medical = load_mask(mask_dir, "mask_medical_text.png")

    signal_masks = collect_signal_masks(mask_dir)

    # Composite with ALL masks
    composite = img.copy()
    composite = blend_mask(composite, m_grid_minor, COLOR_GRID_MINOR, alpha=0.25)
    composite = blend_mask(composite, m_grid_major, COLOR_GRID_MAJOR, alpha=0.45)
    composite = blend_mask(composite, m_paper, COLOR_PAPER_BOUNDARY, alpha=0.35)
    composite = blend_mask(composite, m_black_sq, COLOR_BLACK_SQUARE, alpha=0.5)
    composite = blend_mask(composite, m_medical, COLOR_MEDICAL_TEXT, alpha=0.55)
    composite = blend_mask(composite, m_handwriting, COLOR_HANDWRITING, alpha=0.55)
    composite = blend_mask(composite, m_ref_pulse, COLOR_REFERENCE_PULSE, alpha=0.6)
    composite = blend_mask(composite, m_all_signals, COLOR_SIGNAL, alpha=0.6)
    composite = blend_mask(composite, m_lead_labels, COLOR_LABEL, alpha=0.6)
    composite = blend_mask(composite, m_labels, COLOR_LABEL, alpha=0.8)

    grid_only = blend_mask(img.copy(), m_grid_combined, COLOR_GRID_MAJOR, alpha=0.55)
    sig_only = blend_mask(img.copy(), m_all_signals, COLOR_SIGNAL, alpha=0.7)
    lbl_only = blend_mask(img.copy(), m_labels, COLOR_LABEL, alpha=0.9)
    lbl_only = blend_mask(lbl_only, m_lead_labels, COLOR_LABEL, alpha=0.6)
    rainbow = per_lead_rainbow(img, signal_masks, alpha=0.75)

    h, w = img.shape[:2]
    sig_pct = pct_nonzero(m_all_signals)
    grid_pct = pct_nonzero(m_grid_major)
    lbl_pct = pct_nonzero(m_labels)

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    title = f"{title_prefix}{image_path.stem}  —  {w}x{h}  —  signals={sig_pct:.2f}%  grid={grid_pct:.2f}%  labels={lbl_pct:.2f}%"
    fig.suptitle(title, fontsize=12, fontweight="bold")

    axes[0, 0].imshow(img); axes[0, 0].set_title("Image seule")
    axes[0, 1].imshow(composite); axes[0, 1].set_title("Tous masques superposés")
    axes[0, 2].imshow(rainbow); axes[0, 2].set_title(f"Signaux par dérivation ({len(signal_masks)} leads)")
    axes[1, 0].imshow(grid_only); axes[1, 0].set_title("Grille (bleu)")
    axes[1, 1].imshow(sig_only); axes[1, 1].set_title("all_signals (rouge)")
    axes[1, 2].imshow(lbl_only); axes[1, 2].set_title("Labels (or)")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return True


def make_compare_figure(p1_img_path: Path, p1_mask_dir: Path,
                        p2_img_path: Path, p2_mask_dir: Path,
                        out_path: Path) -> bool:
    p1_img = load_image_rgb(p1_img_path)
    p2_img = load_image_rgb(p2_img_path)
    if p1_img is None or p2_img is None:
        print(f"  SKIP compare (missing image): {p1_img_path.name} / {p2_img_path.name}")
        return False

    def build(img, mask_dir):
        if not mask_dir.is_dir():
            return img.copy(), img.copy(), img.copy()
        m_grid = load_mask(mask_dir, "mask_grid_combined.png")
        m_sig = load_mask(mask_dir, "mask_all_signals.png")
        m_lbl = load_mask(mask_dir, "mask_label_centers.png")
        m_ll = load_mask(mask_dir, "mask_lead_labels.png")
        composite = img.copy()
        composite = blend_mask(composite, m_grid, COLOR_GRID_MAJOR, alpha=0.45)
        composite = blend_mask(composite, m_sig, COLOR_SIGNAL, alpha=0.6)
        composite = blend_mask(composite, m_ll, COLOR_LABEL, alpha=0.5)
        composite = blend_mask(composite, m_lbl, COLOR_LABEL, alpha=0.85)
        return img, composite, (m_grid, m_sig, m_lbl)

    p1_raw, p1_over, p1_stats = build(p1_img, p1_mask_dir)
    p2_raw, p2_over, p2_stats = build(p2_img, p2_mask_dir)

    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    fig.suptitle(f"P1 vs P2 — {p2_img_path.stem}", fontsize=13, fontweight="bold")

    p1_sig_pct = pct_nonzero(p1_stats[1]) if isinstance(p1_stats, tuple) else 0
    p2_sig_pct = pct_nonzero(p2_stats[1]) if isinstance(p2_stats, tuple) else 0

    axes[0, 0].imshow(p1_raw); axes[0, 0].set_title(f"P1 image ({p1_raw.shape[1]}x{p1_raw.shape[0]})")
    axes[0, 1].imshow(p2_raw); axes[0, 1].set_title(f"P2 image ({p2_raw.shape[1]}x{p2_raw.shape[0]})")
    axes[1, 0].imshow(p1_over); axes[1, 0].set_title(f"P1 + masks  (sig={p1_sig_pct:.2f}%)")
    color_p2 = "red" if p2_sig_pct < 0.05 else "black"
    axes[1, 1].imshow(p2_over); axes[1, 1].set_title(f"P2 + masks  (sig={p2_sig_pct:.2f}%)", color=color_p2)

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return True


def find_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def run_single(image_path: Path, mask_dir: Path, output: Path) -> None:
    if output.is_dir() or not output.suffix:
        output = output / f"{image_path.stem}_overlay.png"
    ok = make_figure(image_path, mask_dir, output)
    if ok:
        print(f"OK: {output}")


def run_dataset(dataset: Path, output: Path, count: int | None, shuffle: bool) -> None:
    images_dir = dataset / "images"
    masks_dir = dataset / "masks"
    if not images_dir.is_dir():
        print(f"ERR: no images dir at {images_dir}")
        sys.exit(1)

    images = find_images(images_dir)
    if not images:
        print(f"ERR: no images in {images_dir}")
        sys.exit(1)

    if shuffle:
        random.shuffle(images)
    if count:
        images = images[:count]

    output.mkdir(parents=True, exist_ok=True)
    ok = skip = 0
    for i, img_path in enumerate(images, 1):
        mdir = masks_dir / img_path.stem
        out = output / f"{img_path.stem}_overlay.png"
        if make_figure(img_path, mdir, out):
            ok += 1
        else:
            skip += 1
        if i % 5 == 0 or i == len(images):
            print(f"  [{i}/{len(images)}]  ok={ok}  skip={skip}")
    print(f"Done: {ok} overlays in {output}  (skipped={skip})")


def strip_aug(stem: str) -> str:
    return stem[:-len(MASK_SUFFIX_AUG)] if stem.endswith(MASK_SUFFIX_AUG) else stem


def run_compare(p1_dir: Path, p2_dir: Path, output: Path,
                count: int | None, shuffle: bool) -> None:
    p2_images = find_images(p2_dir / "images")
    if not p2_images:
        print(f"ERR: no images in {p2_dir / 'images'}")
        sys.exit(1)

    if shuffle:
        random.shuffle(p2_images)
    if count:
        p2_images = p2_images[:count]

    output.mkdir(parents=True, exist_ok=True)
    ok = skip = 0
    for i, p2_img in enumerate(p2_images, 1):
        base = strip_aug(p2_img.stem)
        p1_img = None
        for ext in IMAGE_EXTS:
            cand = p1_dir / "images" / f"{base}{ext}"
            if cand.exists():
                p1_img = cand
                break
        if p1_img is None:
            skip += 1
            continue
        p1_mask = p1_dir / "masks" / base
        p2_mask = p2_dir / "masks" / p2_img.stem
        out = output / f"{p2_img.stem}_compare.png"
        if make_compare_figure(p1_img, p1_mask, p2_img, p2_mask, out):
            ok += 1
        else:
            skip += 1
        if i % 5 == 0 or i == len(p2_images):
            print(f"  [{i}/{len(p2_images)}]  ok={ok}  skip={skip}")
    print(f"Done: {ok} comparisons in {output}  (skipped={skip})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, help="Single image to process")
    parser.add_argument("--masks", type=Path, help="Mask directory for single image")
    parser.add_argument("--dataset", type=Path, help="Dataset dir with images/ and masks/ subdirs")
    parser.add_argument("--compare", action="store_true", help="Side-by-side P1 vs P2 overlays")
    parser.add_argument("--p1", type=Path, help="P1 dataset dir (compare mode)")
    parser.add_argument("--p2", type=Path, help="P2 dataset dir (compare mode)")
    parser.add_argument("--output", type=Path, required=True, help="Output file or directory")
    parser.add_argument("--count", type=int, default=None, help="Max number of samples (batch modes)")
    parser.add_argument("--shuffle", action="store_true", help="Random sample order")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffling")
    parser.add_argument("--zoom", action="store_true", help="Produce zoomed crops instead of full overlay")
    parser.add_argument("--n-crops", type=int, default=4, help="Number of zoom crops (--zoom mode)")
    parser.add_argument("--crop-size", type=int, default=700, help="Zoom crop size in px (--zoom mode)")
    args = parser.parse_args()

    random.seed(args.seed)

    if args.compare:
        if not args.p1 or not args.p2:
            parser.error("--compare requires --p1 and --p2")
        run_compare(args.p1, args.p2, args.output, args.count, args.shuffle)
        return

    if args.image and args.masks:
        if args.zoom:
            out = args.output if args.output.suffix else args.output / f"{args.image.stem}_zoom.png"
            make_zoom_figure(args.image, args.masks, out, args.n_crops, args.crop_size)
        else:
            run_single(args.image, args.masks, args.output)
        return

    if args.dataset:
        if args.zoom:
            images_dir = args.dataset / "images"
            masks_dir = args.dataset / "masks"
            imgs = find_images(images_dir)
            if args.shuffle:
                random.shuffle(imgs)
            if args.count:
                imgs = imgs[:args.count]
            args.output.mkdir(parents=True, exist_ok=True)
            ok = skip = 0
            for i, p in enumerate(imgs, 1):
                out = args.output / f"{p.stem}_zoom.png"
                if make_zoom_figure(p, masks_dir / p.stem, out, args.n_crops, args.crop_size):
                    ok += 1
                else:
                    skip += 1
                if i % 5 == 0 or i == len(imgs):
                    print(f"  [{i}/{len(imgs)}]  ok={ok}  skip={skip}")
            print(f"Done zoom: {ok} in {args.output} (skipped={skip})")
        else:
            run_dataset(args.dataset, args.output, args.count, args.shuffle)
        return

    parser.error("Provide either (--image + --masks), --dataset, or --compare with --p1/--p2")


if __name__ == "__main__":
    main()
