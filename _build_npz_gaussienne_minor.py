# -*- coding: utf-8 -*-
"""Cree training_npz_Gaussienne_grille_minor.ipynb depuis training_npz_Gaussienne.ipynb :
- cible = grille MINOR (grid_minor_1mm) au lieu de MAJOR (grid_major_5mm)
- _draw_points_mask VECTORISE (splat + flou gaussien) : indispensable car la minor a ~60k points
  par image (boucle Python = ~1s/image = 34min/epoch). Vectorise = ~87ms/image."""
import json, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SRC = "stage/training_npz_Gaussienne.ipynb"
DST = "stage/training_npz_Gaussienne_grille_minor.ipynb"
nb = json.load(open(SRC, encoding="utf-8"))

# --- corps vectorise de _draw_points_mask (remplace la boucle Python lente) ---
VEC_BODY = (
    "        # VECTORISE (rapide ~11x) : splat des points + flou gaussien (O(image), pas O(points)).\n"
    "        # Necessaire pour la grille minor (~60k points/image). Equivalent a la boucle (Dice 0.89),\n"
    "        # le splat arrondit au pixel mais l'ecart sub-pixel (<0.3px natif) est negligeable apres resize.\n"
    "        xs = np.clip(np.round(xs_f).astype(int), 0, W - 1)  # clip : round peut pousser <W a ==W (hors-borne)\n"
    "        ys = np.clip(np.round(ys_f).astype(int), 0, H - 1)\n"
    "        mask[ys, xs] = 1.0\n"
    "        k = int(round(3 * sigma)) * 2 + 1\n"
    "        mask = cv2.GaussianBlur(mask, (k, k), sigma)\n"
    "        _ref = np.zeros((k + 4, k + 4), np.float32); _ref[(k + 4) // 2, (k + 4) // 2] = 1.0\n"
    "        _peak1 = float(cv2.GaussianBlur(_ref, (k, k), sigma).max())  # pic d'une impulsion isolee -> normalise les pics a 1.0\n"
    "        mask = (mask / max(_peak1, 1e-8)).clip(0, 1)\n"
    "        return (mask * 255).clip(0, 255).astype(np.uint8)"
)
START = "        two_sigma2 = 2.0 * sigma * sigma"
END   = "return (mask * 255).clip(0, 255).astype(np.uint8)"

REPL = [
    # gros gain vitesse : ne pas charger les coordmaps (630ms/item) -> on ne veut que les points (2ms)
    ('load_unified(s["npz_path"])', 'load_unified(s["npz_path"], load_maps=False)'),
    ("grid_major_5mm", "grid_minor_1mm"),
    ("runs_npz_gaussienne", "runs_npz_gaussienne_minor"),
    ("5mm", "1mm"),
    ("grille major", "grille minor"),
    ("major", "minor"), ("Major", "Minor"),
]

n_vec = 0
for c in nb["cells"]:
    src = "".join(c["source"])
    # 1) vectorise _draw_points_mask
    if START in src and END in src:
        i0 = src.index(START); i1 = src.index(END) + len(END)
        src = src[:i0] + VEC_BODY + src[i1:]
        n_vec += 1
    # 2) cv2 dispo des les imports (le dessin vectorise en a besoin)
    if "import segmentation_models_pytorch as smp" in src and "import cv2" not in src:
        src = src.replace("import segmentation_models_pytorch as smp",
                          "import cv2\nimport segmentation_models_pytorch as smp")
    # 3) bascule major -> minor
    for a, b in REPL:
        src = src.replace(a, b)
    c["source"] = [src]
    if c["cell_type"] == "code":
        c["outputs"] = []; c["execution_count"] = None

intro = {"cell_type": "markdown", "metadata": {}, "source": [
    "# Détecteur de points — grille MINOR (1mm)\n", "\n",
    "Variante de `training_npz_Gaussienne` ciblant la **grille minor 1mm** (`grid_minor_1mm`) au lieu de la 5mm.\n", "\n",
    "**Deux changements vs la major :**\n",
    "1. `npz_key = grid_minor_1mm` (sortie : `runs_npz_gaussienne_minor/`).\n",
    "2. `_draw_points_mask` **vectorisé** (splat + flou gaussien) — obligatoire car la minor a ~**60 000 points/image** ; la boucle Python ferait ~1 s/image (~34 min/epoch rien que pour les masques). Vectorisé = ~87 ms/image.\n", "\n",
    "⚠️ **Densité** : heatmap bien plus dense (~11 % de couverture vs 0.45 % en major), pics de ~1-2 px très rapprochés à 1024². Le Dice sera plus haut (plus de foreground) sans que ça signifie « mieux ». Si les blobs fusionnent, baisser `gaussian_sigma`.\n",
]}
nb["cells"].insert(0, intro)

json.dump(nb, open(DST, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"[OK] ecrit {DST} - {len(nb['cells'])} cellules | _draw_points_mask vectorise dans {n_vec} cellule(s)")
txt = json.dumps(nb)
print("  grid_minor_1mm:", txt.count("grid_minor_1mm"), "| grid_major_5mm restant:", txt.count("grid_major_5mm"))
print("  GaussianBlur present:", "GaussianBlur" in txt, "| import cv2:", "import cv2" in txt)
