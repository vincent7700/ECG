# Mesure quantitative du decalage GT vs centre visuel des intersections.
# Pour chaque point GT du NPZ, trouve le centre reel de l'intersection rouge dans une petite ROI,
# et calcule l'offset.
#
# Ne modifie aucun fichier. Sortie : statistiques + histogramme dans stage/_diagnose_out/

import os, sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = r"C:\Users\v\Desktop\ECGPerturb-main"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from shared.npz_schema import load_unified_npz

OUT_DIR = os.path.join(PROJECT_ROOT, "stage", "_diagnose_out")
os.makedirs(OUT_DIR, exist_ok=True)

NPZ_KEY = "grid_major_5mm"
ROI_RADIUS = 8   # rayon en pixels autour du GT pour chercher le vrai centre


def detect_intersection_center_in_roi(img_gray, gx_float, gy_float, roi_radius):
    # Trouve le centre de l'intersection rouge dans une petite ROI autour de (gx, gy).
    # Utilise le centroide pondere par "rougeur" (R - 0.5*(G+B)) pour localiser ou les lignes
    # rouges sont les plus intenses.
    H, W = img_gray.shape[:2]
    gx_round, gy_round = int(round(gx_float)), int(round(gy_float))
    x0 = max(0, gx_round - roi_radius); x1 = min(W, gx_round + roi_radius + 1)
    y0 = max(0, gy_round - roi_radius); y1 = min(H, gy_round + roi_radius + 1)
    roi = img_gray[y0:y1, x0:x1]
    if roi.size == 0:
        return None, None
    # On veut le centre de "rougeur" : les pixels les plus rouges (= les lignes de grille)
    # img_gray est en fait l'image RGB ici, on calcule rougeur
    # On va plutot prendre img_rgb directement
    return None, None  # placeholder, on fait mieux ci-dessous


def detect_center_red(img_rgb, gx_float, gy_float, roi_radius):
    # Centroide pondere par l'intensite rouge dans la ROI.
    # Plus le pixel est rouge (R eleve, G/B faibles), plus son poids est grand.
    H, W = img_rgb.shape[:2]
    gx_round, gy_round = int(round(gx_float)), int(round(gy_float))
    x0 = max(0, gx_round - roi_radius); x1 = min(W, gx_round + roi_radius + 1)
    y0 = max(0, gy_round - roi_radius); y1 = min(H, gy_round + roi_radius + 1)
    roi = img_rgb[y0:y1, x0:x1].astype(np.float32)
    if roi.size == 0:
        return None
    R = roi[..., 0]; G = roi[..., 1]; B = roi[..., 2]
    # Score de "rougeur" : R - moyenne(G,B), clippe a 0
    redness = np.clip(R - 0.5 * (G + B), 0, None)
    total = redness.sum()
    if total < 1e-6:
        return None
    yy, xx = np.indices(redness.shape)
    cx_local = (xx * redness).sum() / total
    cy_local = (yy * redness).sum() / total
    # Coordonnees absolues
    return float(x0 + cx_local), float(y0 + cy_local)


def main():
    print("=" * 60)
    print("MESURE QUANTITATIVE : decalage GT vs centre visuel")
    print("=" * 60)

    # On teste sur la meme image que le user
    p2_img_path = os.path.join(PROJECT_ROOT, "data", "output_augmentation", "images",
                               "ECG_033_436_p0_aug.webp")
    p2_npz_path = os.path.join(PROJECT_ROOT, "data", "output_augmentation", "npz",
                               "ECG_033_436_p0_aug.npz")

    img_rgb = np.array(Image.open(p2_img_path).convert("RGB"))
    H, W = img_rgb.shape[:2]
    data = load_unified_npz(p2_npz_path)
    pts = data.get(NPZ_KEY, np.empty((0, 2)))
    if len(pts):
        valid = (pts[:, 0] >= ROI_RADIUS) & (pts[:, 0] < W - ROI_RADIUS) & \
                (pts[:, 1] >= ROI_RADIUS) & (pts[:, 1] < H - ROI_RADIUS)
        pts = pts[valid]

    print(f"Image : {p2_img_path}")
    print(f"Taille : {W}x{H}  |  Points GT a evaluer : {len(pts)}")

    # Pour chaque point GT, trouver le vrai centre de l'intersection rouge
    offsets_x, offsets_y, offsets_norm = [], [], []
    detected_centers = []
    for gx, gy in pts:
        center = detect_center_red(img_rgb, gx, gy, ROI_RADIUS)
        if center is None:
            continue
        cx_real, cy_real = center
        ox = cx_real - gx
        oy = cy_real - gy
        offsets_x.append(ox)
        offsets_y.append(oy)
        offsets_norm.append(np.sqrt(ox * ox + oy * oy))
        detected_centers.append((gx, gy, cx_real, cy_real))

    offsets_x = np.asarray(offsets_x)
    offsets_y = np.asarray(offsets_y)
    offsets_norm = np.asarray(offsets_norm)

    print(f"\n[OFFSET GT -> centre visuel rouge]  (en pixels resolution native)")
    print(f"  Points evalues          : {len(offsets_x)}")
    print(f"\n  Offset X (vers la droite si > 0)")
    print(f"    Moyenne   : {offsets_x.mean():+.3f} px")
    print(f"    Mediane   : {np.median(offsets_x):+.3f} px")
    print(f"    Ecart-type: {offsets_x.std():.3f} px")
    print(f"\n  Offset Y (vers le bas si > 0)")
    print(f"    Moyenne   : {offsets_y.mean():+.3f} px")
    print(f"    Mediane   : {np.median(offsets_y):+.3f} px")
    print(f"    Ecart-type: {offsets_y.std():.3f} px")
    print(f"\n  Distance euclidienne |offset|")
    print(f"    Moyenne   : {offsets_norm.mean():.3f} px")
    print(f"    Mediane   : {np.median(offsets_norm):.3f} px")
    print(f"    P95       : {np.percentile(offsets_norm, 95):.3f} px")
    print(f"    Maximum   : {offsets_norm.max():.3f} px")

    # Interpretation
    print(f"\n[INTERPRETATION]")
    if abs(offsets_x.mean()) > 0.3 or abs(offsets_y.mean()) > 0.3:
        print(f"  -> DECALAGE SYSTEMATIQUE detecte :")
        print(f"     {offsets_x.mean():+.2f} px en X, {offsets_y.mean():+.2f} px en Y")
        print(f"     Les GT sont decalees de cet offset par rapport aux intersections visuelles.")
    else:
        print(f"  -> Pas de decalage systematique significatif.")
        print(f"     Les offsets visibles sont dus a la dispersion locale (ecart-type {offsets_norm.std():.2f}).")

    # Histogrammes
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(offsets_x, bins=80, color="steelblue", edgecolor="black", alpha=0.8)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1, label="Pas de decalage")
    axes[0].axvline(offsets_x.mean(), color="orange", linestyle="-",
                    label=f"Moyenne {offsets_x.mean():+.2f}")
    axes[0].set_title("Offset X (px)"); axes[0].set_xlabel("offset X"); axes[0].legend()

    axes[1].hist(offsets_y, bins=80, color="seagreen", edgecolor="black", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[1].axvline(offsets_y.mean(), color="orange", linestyle="-",
                    label=f"Moyenne {offsets_y.mean():+.2f}")
    axes[1].set_title("Offset Y (px)"); axes[1].set_xlabel("offset Y"); axes[1].legend()

    axes[2].scatter(offsets_x, offsets_y, s=4, alpha=0.4)
    axes[2].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[2].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[2].scatter([offsets_x.mean()], [offsets_y.mean()],
                    s=200, c="orange", marker="X", edgecolors="black",
                    linewidths=1.5, label=f"Moyenne ({offsets_x.mean():+.2f}, {offsets_y.mean():+.2f})")
    axes[2].set_title("Distribution 2D des offsets (px)")
    axes[2].set_xlabel("offset X"); axes[2].set_ylabel("offset Y")
    axes[2].set_aspect("equal"); axes[2].legend()

    fig.suptitle(f"Decalage GT NPZ vs centre visuel des intersections rouges - {os.path.basename(p2_img_path)}",
                 fontsize=12)
    plt.tight_layout()
    out_hist = os.path.join(OUT_DIR, "offset_distribution.png")
    plt.savefig(out_hist, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Histogrammes sauvegardes : {out_hist}")

    # Visualisation : afficher 6 zooms avec GT (blanc) ET centre visuel detecte (cyan)
    centrals = sorted(detected_centers, key=lambda p: abs(p[0] - W/2) + abs(p[1] - H/2))[:6]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for ax, (gx, gy, cx_real, cy_real) in zip(axes.ravel(), centrals):
        x0 = max(0, int(gx) - 20); x1 = min(W, int(gx) + 20)
        y0 = max(0, int(gy) - 20); y1 = min(H, int(gy) + 20)
        crop = img_rgb[y0:y1, x0:x1]
        ax.imshow(crop, interpolation="nearest")
        ax.scatter(gx - x0,      gy - y0,      s=300, c="white", marker="x",
                   linewidths=3, label=f"GT NPZ ({gx:.1f}, {gy:.1f})")
        ax.scatter(cx_real - x0, cy_real - y0, s=300, c="cyan",  marker="+",
                   linewidths=3, label=f"Centre visuel ({cx_real:.1f}, {cy_real:.1f})")
        offset = np.sqrt((cx_real - gx)**2 + (cy_real - gy)**2)
        ax.set_title(f"Offset = {offset:.2f} px", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.axis("off")
    fig.suptitle("Blanc (X) = GT depuis NPZ  |  Cyan (+) = centre visuel detecte (centroide rouge)",
                 fontsize=12)
    plt.tight_layout()
    out_zoom = os.path.join(OUT_DIR, "offset_zooms.png")
    plt.savefig(out_zoom, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[OK] Zooms compares sauvegardes : {out_zoom}")


if __name__ == "__main__":
    main()
