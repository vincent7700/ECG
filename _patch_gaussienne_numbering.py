# -*- coding: utf-8 -*-
"""Patche training_npz_Gaussienne.ipynb : (1) ajoute une cellule de visu de la gaussienne CIBLE
en hot+zoom, (2) numerote toutes les cellules (en-tete # CELLULE N). Preserve les sorties."""
import json, io, sys, re, shutil, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
NB = "stage/training_npz_Gaussienne.ipynb"
if not os.path.exists(NB + ".bak"):   # backup de l'ORIGINAL (ne pas ecraser au re-run)
    shutil.copy2(NB, NB + ".bak")
nb = json.load(open(NB, encoding="utf-8"))

# --- nouvelle cellule : visu gaussienne cible (hot + zoom) ---
ZOOM_CELL = {
    "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
    "source": [
        "# VISU : TOUS les globes (la heatmap CIBLE complete) + zoom (globes ronds bien separes)\n",
        "# + superposition sur l ECG (chaque globe = 1 intersection de la grille).\n",
        "%matplotlib inline\n",
        "import cv2\n",
        "SET_Z, INDEX_Z = \"val\", 0\n",
        "flist = val_images if SET_Z == \"val\" else train_images\n",
        "sf = flist[INDEX_Z]\n",
        "img_z = np.array(Image.open(os.path.join(image_dir, sf)).convert(\"RGB\"))\n",
        "Hn, Wn = img_z.shape[:2]\n",
        "dz = load_unified(os.path.join(cfg.npz_dir, sf.replace(\".webp\", \"\")), load_maps=False)\n",
        "pts_z = dz.get(cfg.npz_key, np.empty((0, 2))); pts_z = pts_z[np.isfinite(pts_z).all(1)]\n",
        "hm_z = ECGNpzDataset._draw_points_mask(pts_z, Hn, Wn, cfg.point_radius, cfg.gaussian_sigma).astype(np.float32) / 255.0\n",
        "# zone de zoom : centre de l image, +/- Zr px\n",
        "cyz, cxz = Hn // 2, Wn // 2; Zr = 200\n",
        "fig, ax = plt.subplots(1, 3, figsize=(24, 8))\n",
        "ax[0].imshow(hm_z, cmap=\"hot\", vmin=0, vmax=1)\n",
        "ax[0].set_title(f\"TOUS les globes (heatmap cible, {len(pts_z)} intersections)\"); ax[0].axis(\"off\")\n",
        "ax[0].add_patch(plt.Rectangle((cxz - Zr, cyz - Zr), 2 * Zr, 2 * Zr, fill=False, ec=\"lime\", lw=2))\n",
        "ax[1].imshow(hm_z[cyz - Zr:cyz + Zr, cxz - Zr:cxz + Zr], cmap=\"hot\", vmin=0, vmax=1)\n",
        "ax[1].set_title(\"Zoom : globes ronds bien separes\"); ax[1].axis(\"off\")\n",
        "ov = img_z.astype(np.float32).copy(); red = np.zeros_like(ov); red[..., 0] = hm_z * 255\n",
        "ovr = (0.6 * ov + 0.8 * red).clip(0, 255).astype(np.uint8)\n",
        "ax[2].imshow(ovr[cyz - Zr:cyz + Zr, cxz - Zr:cxz + Zr])\n",
        "ax[2].set_title(\"Zoom : globes superposes aux intersections\"); ax[2].axis(\"off\")\n",
        "plt.tight_layout(); plt.show()\n",
    ],
}

# NOTE : la visu globes est desormais FUSIONNEE dans la cellule "Visu 1 image"
# (fusion 9+10). On n'insere plus la cellule standalone si la version fusionnee existe.
_merged = any("Heatmap gaussienne CIBLE" in "".join(c["source"]) for c in nb["cells"])
if not _merged and not any("TOUS les globes" in "".join(c["source"]) for c in nb["cells"]):
    idx_after = next(i for i, c in enumerate(nb["cells"]) if 'Prediction (sigmoid)' in "".join(c["source"]))
    nb["cells"].insert(idx_after + 1, ZOOM_CELL)

# description par detection de contenu (ordre = priorite ; le specifique d'abord)
def desc(src):
    s = src
    if "os.environ" in s and "CUDA" in s: return "Config environnement GPU"
    if "class TrainConfig" in s or "@dataclass" in s: return "Configuration (TrainConfig)"
    if "class ECGNpzDataset" in s: return "Dataset + dessin gaussienne (_draw_points_mask)"
    if "def train(" in s or "BCEDiceLoss" in s: return "Loss, metriques & boucle d'entrainement"
    if "train(cfg)" in s and "def train(" not in s and len(s.strip()) < 150: return "Lancement de l'entrainement"
    if "Heatmap gaussienne CIBLE" in s: return "Visu complete : image + heatmap + zoom + erreurs (deviation) + zooms blobs"
    if "TOUS les globes" in s: return "VISU tous les globes (heatmap + zoom + superposition)"
    if "output_real" in s or "REAL_ROOT" in s: return "Inference sur image reelle"
    if "Run selectionne" in s or ("best_model.pth" in s and "load_state_dict" in s): return "Chargement du modele entraine"
    if "Prediction (sigmoid)" in s: return "Visu 1 image : GT vs prediction"
    if "MAX_MATCH_DIST" in s or "centroide pondere" in s.lower() or "DEVIATION" in s: return "Metriques point-based + deviation pixel"
    if "val_images" in s and "train_images" in s and "os.listdir" in s: return "Liste des images train/val"
    if "import segmentation_models_pytorch" in s: return "Imports"
    return "..."

# numerote (1-based), en retirant un eventuel ancien en-tete
HDR = re.compile(r"^# ={3,} CELLULE \d+ .*={3,}\n", re.M)
n = 0
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    n += 1
    s = "".join(c["source"])
    s = HDR.sub("", s, count=1)              # retire ancien en-tete si present
    s = s.lstrip("\n")
    header = f"# ===================== CELLULE {n} — {desc(s)} =====================\n"
    c["source"] = [header + s]

json.dump(nb, open(NB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"[OK] {NB} patche | {len(nb['cells'])} cellules | backup: {NB}.bak")
print("Cellules numerotees :")
k = 0
for c in nb["cells"]:
    if c["cell_type"] == "code":
        k += 1
        first = "".join(c["source"]).split("\n")[0]
        print("  ", first.replace("# ===================== ", "").replace(" =====================", ""))
