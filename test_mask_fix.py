#!/usr/bin/env python3
"""Empirical test: regenerate P2 masks WITHOUT the double ar_crop bug.

Bypasses workflow_engine.py:3437-3475 and calls generate_p2_masks_remap
directly. The remap output should already be at post-AR-crop dims (since
the NPZ stores post-crop page_width/height and the inverse map is computed
at those dims by save_pipeline2_npz).

If alignment is perfect with the regenerated mask, the double-crop diagnosis
is confirmed and the fix is to skip the ar_crop block in workflow_engine.

Usage:
    python stage/test_mask_fix.py                      # Test 875 by default
    python stage/test_mask_fix.py --name ECG_032_666_p0_aug
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.mask_generator import generate_p2_masks_remap


def overlay(img_rgb: np.ndarray, mask: np.ndarray, color, alpha=0.7) -> np.ndarray:
    out = img_rgb.astype(np.float32)
    mask = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]),
                      interpolation=cv2.INTER_NEAREST) if mask.shape[:2] != img_rgb.shape[:2] else mask
    w = (mask > 30).astype(np.float32) * alpha
    w = w[:, :, None]
    c = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(out * (1 - w) + c * w, 0, 255).astype(np.uint8)


def row_heatmap(mask: np.ndarray) -> np.ndarray:
    """Per-row mask pixel count (for diff profile)."""
    return (mask > 30).sum(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="ECG_032_875_p0_aug")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.root
    name = args.name
    base = name[:-4] if name.endswith("_aug") else name  # strip _aug for P1

    p2_npz = root / "data/output_augmentation/npz" / f"{name}.npz"
    p1_mask_dir = root / "data/output_impression/masks" / base
    p2_img = root / "data/output_augmentation/images" / f"{name}.webp"
    orig_mask_dir = root / "data/output_augmentation/masks" / name

    assert p2_npz.exists(), f"NPZ missing: {p2_npz}"
    assert p1_mask_dir.is_dir(), f"P1 mask dir missing: {p1_mask_dir}"
    assert p2_img.exists(), f"P2 image missing: {p2_img}"

    # Load image
    img_bgr = cv2.imread(str(p2_img), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    print(f"image: {w}x{h}")

    # Load "buggy" mask (current pipeline output)
    buggy = cv2.imread(str(orig_mask_dir / "mask_all_signals.png"), cv2.IMREAD_GRAYSCALE)
    print(f"buggy mask (current pipeline): {buggy.shape[1]}x{buggy.shape[0]}")

    # Regenerate using remap only — no double crop
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        results = generate_p2_masks_remap(str(p2_npz), str(p1_mask_dir), str(tmp_path))
        fixed_path = tmp_path / "mask_all_signals.png"
        assert fixed_path.exists(), f"Regen failed: {list(tmp_path.iterdir())}"
        fixed = cv2.imread(str(fixed_path), cv2.IMREAD_GRAYSCALE)

    print(f"fixed mask (remap only): {fixed.shape[1]}x{fixed.shape[0]}")

    # Per-row heatmap diagnosis
    buggy_rows = row_heatmap(buggy)
    fixed_rows = row_heatmap(fixed) if fixed.shape[0] == buggy.shape[0] else None

    # Where does the actual ECG content live in the image?
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    dark = (gray[50:-50, 50:-50] < 130).sum(axis=1)
    dark_top = np.argsort(-dark)[:20]
    print(f"\nImage heavy rows (top 20 dark): {sorted(dark_top)[:10]}")
    print(f"Buggy mask heavy rows       : {sorted(np.argsort(-buggy_rows)[:20])[:10]}")
    if fixed_rows is not None:
        print(f"Fixed mask heavy rows       : {sorted(np.argsort(-fixed_rows)[:20])[:10]}")

    # Visual comparison
    ov_buggy = overlay(img_rgb, buggy, (220, 30, 30), alpha=0.7)
    ov_fixed = overlay(img_rgb, fixed, (30, 180, 30), alpha=0.7) if fixed.shape[:2] == (h, w) else None

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle(f"Fix test — {name}   (img {w}x{h}, buggy {buggy.shape[1]}x{buggy.shape[0]}, fixed {fixed.shape[1]}x{fixed.shape[0]})",
                 fontsize=13, fontweight="bold")
    axes[0, 0].imshow(img_rgb); axes[0, 0].set_title("Image P2")
    axes[0, 1].imshow(ov_buggy); axes[0, 1].set_title("Image + masque BUGGY (rouge)", color="red")
    axes[1, 0].imshow(ov_fixed if ov_fixed is not None else img_rgb)
    axes[1, 0].set_title("Image + masque FIXED (vert)", color="green")

    # Overlay both to show the diff directly
    combo = overlay(img_rgb, buggy, (220, 30, 30), alpha=0.5)
    if fixed.shape[:2] == (h, w):
        combo = overlay(combo, fixed, (30, 180, 30), alpha=0.5)
    axes[1, 1].imshow(combo); axes[1, 1].set_title("Buggy (rouge) + Fixed (vert) superposés")

    for ax in axes.flat:
        ax.axis("off")

    out_path = args.output or (root / "stage" / "fix_test" / f"{name}_fix_test.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved: {out_path}")

    # Also save a zoom on a specific region for pixel-level inspection
    # Find a row where both masks have content, crop 500x500 around it
    if fixed.shape == buggy.shape:
        y_peak = int(np.argmax(buggy_rows))
        x_peak = int(np.argmax((buggy > 30).sum(axis=0)))
        cs = 500
        y0 = max(0, min(h - cs, y_peak - cs // 2))
        x0 = max(0, min(w - cs, x_peak - cs // 2))
        raw_crop = img_rgb[y0:y0 + cs, x0:x0 + cs]
        bug_crop = ov_buggy[y0:y0 + cs, x0:x0 + cs]
        fix_crop = ov_fixed[y0:y0 + cs, x0:x0 + cs]
        fig2, ax2 = plt.subplots(1, 3, figsize=(18, 6))
        fig2.suptitle(f"{name} — zoom pixel-level  crop @ ({x0},{y0})", fontsize=12)
        ax2[0].imshow(raw_crop); ax2[0].set_title("Image")
        ax2[1].imshow(bug_crop); ax2[1].set_title("Masque BUGGY (rouge)")
        ax2[2].imshow(fix_crop); ax2[2].set_title("Masque FIXED (vert)")
        for a in ax2:
            a.axis("off")
        zoom_out = out_path.parent / f"{name}_fix_zoom.png"
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        fig2.savefig(zoom_out, dpi=120, bbox_inches="tight")
        plt.close(fig2)
        print(f"saved: {zoom_out}")


if __name__ == "__main__":
    main()
