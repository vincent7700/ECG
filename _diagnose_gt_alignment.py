# Diagnostic d'alignement GT vs image visible
# Compare une image P1 (non-augmentee) et P2 (augmentee) avec leurs NPZ respectifs,
# pour determiner OU se trouve le decalage entre les coordonnees GT et les intersections visibles.
#
# Ne modifie AUCUN fichier existant. Ecrit uniquement des images de diagnostic dans
# stage/_diagnose_out/

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = r"C:\Users\v\Desktop\ECGPerturb-main"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from shared.npz_schema import load_unified_npz

# Image a tester (P1 et P2 doivent exister)
SAMPLE_STEM = "ECG_033_436_p0"   # P1
SAMPLE_STEM_P2 = SAMPLE_STEM + "_aug"  # P2

P1_IMG_DIR = os.path.join(PROJECT_ROOT, "data", "output_impression", "images")
P1_NPZ_DIR = os.path.join(PROJECT_ROOT, "data", "output_impression", "npz")
P2_IMG_DIR = os.path.join(PROJECT_ROOT, "data", "output_augmentation", "images")
P2_NPZ_DIR = os.path.join(PROJECT_ROOT, "data", "output_augmentation", "npz")

OUT_DIR = os.path.join(PROJECT_ROOT, "stage", "_diagnose_out")
os.makedirs(OUT_DIR, exist_ok=True)

NPZ_KEY = "grid_major_5mm"


def load_pair(img_path, npz_path):
    img = np.array(Image.open(img_path).convert("RGB"))
    H, W = img.shape[:2]
    data = load_unified_npz(npz_path)
    pts = data.get(NPZ_KEY, np.empty((0, 2)))
    if len(pts):
        valid = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
        pts = pts[valid]
    return img, pts, (W, H)


def pick_n_central_points(pts, W, H, n=4):
    # Selectionne n points proches du centre pour des zooms
    if len(pts) == 0:
        return []
    cx, cy = W / 2, H / 2
    dists = ((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    # Prendre des points repartis : le plus central + 3 autres bien repartis
    order = np.argsort(dists)
    chosen = [pts[order[0]]]
    # Ajouter 3 autres points pas trop proches des deja-pris
    for idx in order[1:]:
        candidate = pts[idx]
        too_close = False
        for c in chosen:
            if np.linalg.norm(candidate - c) < 150:
                too_close = True
                break
        if not too_close:
            chosen.append(candidate)
        if len(chosen) >= n:
            break
    return chosen


def render_diagnostic(img, pts, W, H, title, save_path, points_to_zoom):
    fig = plt.figure(figsize=(24, 12))

    # Subplot 1 (full image avec tous les GT en bleu) - colonne entiere a gauche
    ax_full = plt.subplot2grid((2, 4), (0, 0), rowspan=2)
    ax_full.imshow(img)
    if len(pts):
        ax_full.scatter(pts[:, 0], pts[:, 1], s=4, c="blue", alpha=0.6)
    # Marquer en jaune les points qu'on va zoomer
    for p in points_to_zoom:
        ax_full.scatter(p[0], p[1], s=120, c="yellow", marker="o",
                        edgecolors="black", linewidths=1.5)
    ax_full.set_title(f"{title}\n{len(pts)} points GT (bleus) - 4 points selectionnes (jaune)",
                      fontsize=11)
    ax_full.axis("off")

    # Subplot 2-5 : zooms (60px de cote) avec croix blanche au coord exacte
    ZOOM_RADIUS = 30
    for i, p in enumerate(points_to_zoom[:4]):
        row, col = i // 2, 1 + (i % 2)
        # Ligne 0 : col 1,2  | Ligne 1 : col 1,2  | Mais on a un 4-grid (2x2)
        # Reorganisation : subplot row=i//2, col=1+i%2
        ax = plt.subplot2grid((2, 4), (row, 1 + (i % 2)))
        gx, gy = p[0], p[1]
        x0 = max(0, int(gx) - ZOOM_RADIUS); x1 = min(W, int(gx) + ZOOM_RADIUS)
        y0 = max(0, int(gy) - ZOOM_RADIUS); y1 = min(H, int(gy) + ZOOM_RADIUS)
        crop = img[y0:y1, x0:x1]
        ax.imshow(crop, interpolation="nearest")
        ax.scatter(gx - x0, gy - y0, s=300, c="white", marker="x", linewidths=3)
        ax.scatter(gx - x0, gy - y0, s=30,  c="yellow", marker="o", alpha=0.7)
        ax.set_title(f"Zoom ({int(gx)}, {int(gy)})", fontsize=10)
        ax.axis("off")

    # Ajouter encore 2 zooms en bas (cols 2-3 ligne 1) si on n'a pas assez de place ?
    # Cette grille 2x4 a 8 cases : 1 prise par full image (2 lignes), reste 6 cases.
    # On utilise 4 pour les zooms. On laisse les 2 dernieres vides ou ajouter 2 zooms supplementaires.
    if len(points_to_zoom) >= 6:
        for i, p in enumerate(points_to_zoom[4:6]):
            ax = plt.subplot2grid((2, 4), (i, 3))
            gx, gy = p[0], p[1]
            x0 = max(0, int(gx) - ZOOM_RADIUS); x1 = min(W, int(gx) + ZOOM_RADIUS)
            y0 = max(0, int(gy) - ZOOM_RADIUS); y1 = min(H, int(gy) + ZOOM_RADIUS)
            crop = img[y0:y1, x0:x1]
            ax.imshow(crop, interpolation="nearest")
            ax.scatter(gx - x0, gy - y0, s=300, c="white", marker="x", linewidths=3)
            ax.scatter(gx - x0, gy - y0, s=30,  c="yellow", marker="o", alpha=0.7)
            ax.set_title(f"Zoom ({int(gx)}, {int(gy)})", fontsize=10)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[OK] Diagnostique sauvegarde : {save_path}")


def main():
    print("=" * 60)
    print("DIAGNOSTIC : alignement GT vs intersections visibles")
    print("=" * 60)

    # === Test sur P1 (non-augmentee) ===
    p1_img_path = os.path.join(P1_IMG_DIR, SAMPLE_STEM + ".webp")
    p1_npz_path = os.path.join(P1_NPZ_DIR, SAMPLE_STEM + ".npz")
    print(f"\n[P1] Image : {p1_img_path}")
    print(f"[P1] NPZ   : {p1_npz_path}")

    if os.path.exists(p1_img_path) and os.path.exists(p1_npz_path):
        img1, pts1, (W1, H1) = load_pair(p1_img_path, p1_npz_path)
        print(f"[P1] Taille : {W1}x{H1}  |  Points GT : {len(pts1)}")
        # On prend 6 points : 4 centraux + 2 plus excentres pour voir si le decalage varie
        zoom_pts = pick_n_central_points(pts1, W1, H1, n=6)
        out_p1 = os.path.join(OUT_DIR, f"{SAMPLE_STEM}_P1_diag.png")
        render_diagnostic(img1, pts1, W1, H1,
                          f"P1 (image source non-augmentee) - {SAMPLE_STEM}",
                          out_p1, zoom_pts)
    else:
        print("[P1] ABSENT")

    # === Test sur P2 (augmentee) ===
    p2_img_path = os.path.join(P2_IMG_DIR, SAMPLE_STEM_P2 + ".webp")
    p2_npz_path = os.path.join(P2_NPZ_DIR, SAMPLE_STEM_P2 + ".npz")
    print(f"\n[P2] Image : {p2_img_path}")
    print(f"[P2] NPZ   : {p2_npz_path}")

    if os.path.exists(p2_img_path) and os.path.exists(p2_npz_path):
        img2, pts2, (W2, H2) = load_pair(p2_img_path, p2_npz_path)
        print(f"[P2] Taille : {W2}x{H2}  |  Points GT : {len(pts2)}")
        zoom_pts = pick_n_central_points(pts2, W2, H2, n=6)
        out_p2 = os.path.join(OUT_DIR, f"{SAMPLE_STEM_P2}_P2_diag.png")
        render_diagnostic(img2, pts2, W2, H2,
                          f"P2 (image augmentee) - {SAMPLE_STEM_P2}",
                          out_p2, zoom_pts)
    else:
        print("[P2] ABSENT")

    # === Test SUPPLEMENTAIRE : verifier si le NPZ P2 a des coordonnees apres transformation,
    # ou si c'est juste les coords P1 reutilisees (qui serait un bug) ===
    if os.path.exists(p1_npz_path) and os.path.exists(p2_npz_path):
        d1 = load_unified_npz(p1_npz_path)
        d2 = load_unified_npz(p2_npz_path)
        pts_p1 = d1.get(NPZ_KEY, np.empty((0, 2)))
        pts_p2 = d2.get(NPZ_KEY, np.empty((0, 2)))
        print(f"\n[COMPARAISON NPZ] P1 vs P2 (grid_major_5mm)")
        print(f"  P1 : {len(pts_p1)} points")
        print(f"  P2 : {len(pts_p2)} points")
        if len(pts_p1) == len(pts_p2) and len(pts_p1) > 0:
            # Si les coordonnees sont IDENTIQUES, c'est tres suspect (augmentation pas propagee)
            diff = np.abs(pts_p1 - pts_p2).mean()
            print(f"  Diff moyenne (px) : {diff:.3f}")
            if diff < 1.0:
                print(f"  -> NPZ P1 et P2 ont des coordonnees QUASI IDENTIQUES !")
                print(f"     Cela suggere que l'augmentation n'a PAS propage la transformation au NPZ.")
            else:
                print(f"  -> NPZ P1 et P2 different (augmentation propagee).")
        else:
            print(f"  Tailles differentes : augmentation potentiellement non-propagee correctement.")

    print(f"\n[DONE] Inspecte les PNG dans : {OUT_DIR}")


if __name__ == "__main__":
    main()
