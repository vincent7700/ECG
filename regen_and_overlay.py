#!/usr/bin/env python3
"""Regenerate P2 masks with the fix applied, then produce overlays.

Equivalent to re-running the fixed pipeline — uses the existing NPZs
(which already contain the correct post-AR-crop inverse map) and calls
generate_p2_masks_remap to produce masks at the right size. Then applies
orientation rotation (as the fixed workflow_engine step 4 does) and
renders overlays so alignment can be judged visually.

Saves regenerated masks to stage/fixed_masks/<name>/ and overlays to
stage/fixed_overlays/.

Usage:
    python stage/regen_and_overlay.py --count 10 --shuffle --seed 42
    python stage/regen_and_overlay.py --names ECG_032_875_p0_aug ECG_032_666_p0_aug
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from shared.mask_generator import generate_p2_masks_remap


COLOR_GRID = (30, 144, 255)
COLOR_SIGNAL = (220, 30, 30)
COLOR_LABEL = (255, 215, 0)


def load_rgb(path: Path):
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def blend(img, mask, color, alpha=0.6):
    if mask is None:
        return img
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    w = ((mask > 30).astype(np.float32) * alpha)[:, :, None]
    c = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(img.astype(np.float32) * (1 - w) + c * w, 0, 255).astype(np.uint8)


def apply_portrait_rot(mask, portrait_rot):
    if portrait_rot == "cw":
        return cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    if portrait_rot == "ccw":
        return cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if portrait_rot == "180":
        return cv2.rotate(mask, cv2.ROTATE_180)
    return mask


def regen_one(name: str, out_mask_dir: Path) -> dict:
    base = name[:-4] if name.endswith("_aug") else name
    p2_npz = ROOT / "data/output_augmentation/npz" / f"{name}.npz"
    p1_mask_dir = ROOT / "data/output_impression/masks" / base
    meta_path = ROOT / "data/output_augmentation/json" / f"{name}_metadata.json"

    if not p2_npz.exists() or not p1_mask_dir.is_dir():
        return {}

    out_mask_dir.mkdir(parents=True, exist_ok=True)

    # Clean prior regen output for this sample
    for p in out_mask_dir.iterdir():
        if p.suffix == ".png":
            p.unlink()

    # Step A: remap (produces masks at NPZ post-crop, pre-rotation dims)
    results = generate_p2_masks_remap(str(p2_npz), str(p1_mask_dir), str(out_mask_dir))

    # Step B: apply orientation rotation (matches fixed workflow_engine step 4)
    portrait_rot = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        portrait_rot = meta.get("portrait_rotation")

    if portrait_rot:
        for p in out_mask_dir.glob("*.png"):
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            m2 = apply_portrait_rot(m, portrait_rot)
            cv2.imwrite(str(p), m2, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    return results


def make_overlay(img_path: Path, mask_dir: Path, out_path: Path, title: str):
    img = load_rgb(img_path)
    if img is None:
        return False

    def load(name):
        p = mask_dir / name
        return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.exists() else None

    m_grid = load("mask_grid_combined.png")
    m_sig = load("mask_all_signals.png")
    m_lbl = load("mask_label_centers.png")
    m_ll = load("mask_lead_labels.png")

    composite = img.copy()
    composite = blend(composite, m_grid, COLOR_GRID, alpha=0.4)
    composite = blend(composite, m_sig, COLOR_SIGNAL, alpha=0.65)
    composite = blend(composite, m_ll, COLOR_LABEL, alpha=0.55)
    composite = blend(composite, m_lbl, COLOR_LABEL, alpha=0.85)

    grid_only = blend(img.copy(), m_grid, COLOR_GRID, alpha=0.5)
    sig_only = blend(img.copy(), m_sig, COLOR_SIGNAL, alpha=0.7)

    h, w = img.shape[:2]
    pct_sig = 100 * np.count_nonzero(m_sig) / m_sig.size if m_sig is not None else 0
    pct_grid = 100 * np.count_nonzero(m_grid) / m_grid.size if m_grid is not None else 0

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle(f"{title}   {w}x{h}   signals={pct_sig:.2f}%  grid={pct_grid:.2f}%",
                 fontsize=13, fontweight="bold")
    axes[0, 0].imshow(img); axes[0, 0].set_title("Image P2")
    axes[0, 1].imshow(composite); axes[0, 1].set_title("grille (bleu) + signaux (rouge) + labels (or)")
    axes[1, 0].imshow(grid_only); axes[1, 0].set_title("Grille seule")
    axes[1, 1].imshow(sig_only); axes[1, 1].set_title("Signaux seuls")
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--names", nargs="*", default=None, help="Explicit sample names")
    parser.add_argument("--mask-out", type=Path, default=ROOT / "stage/fixed_masks")
    parser.add_argument("--overlay-out", type=Path, default=ROOT / "stage/fixed_overlays")
    parser.add_argument("--only-ar-crop", action="store_true",
                        help="Only process samples whose metadata has ar_crop set (the ones affected by the bug)")
    args = parser.parse_args()

    random.seed(args.seed)

    images_dir = ROOT / "data/output_augmentation/images"
    json_dir = ROOT / "data/output_augmentation/json"

    if args.names:
        names = list(args.names)
    else:
        all_names = sorted(p.stem for p in images_dir.glob("*.webp"))
        if args.only_ar_crop:
            filtered = []
            for n in all_names:
                meta_p = json_dir / f"{n}_metadata.json"
                if meta_p.exists():
                    try:
                        meta = json.loads(meta_p.read_text(encoding="utf-8"))
                        if meta.get("ar_crop"):
                            filtered.append(n)
                    except Exception:
                        pass
            all_names = filtered
        if args.shuffle:
            random.shuffle(all_names)
        names = all_names[: args.count]

    print(f"Processing {len(names)} samples")
    args.mask_out.mkdir(parents=True, exist_ok=True)
    args.overlay_out.mkdir(parents=True, exist_ok=True)

    ok = skip = 0
    for i, name in enumerate(names, 1):
        img_path = images_dir / f"{name}.webp"
        if not img_path.exists():
            skip += 1
            continue
        mask_dir = args.mask_out / name
        results = regen_one(name, mask_dir)
        if not results:
            skip += 1
            continue
        out_path = args.overlay_out / f"{name}_fixed_overlay.png"
        if make_overlay(img_path, mask_dir, out_path, f"FIXED — {name}"):
            ok += 1
        else:
            skip += 1
        if i % 5 == 0 or i == len(names):
            print(f"  [{i}/{len(names)}]  ok={ok}  skip={skip}")

    print(f"Done: {ok} overlays in {args.overlay_out}  (skipped={skip})")


if __name__ == "__main__":
    main()
