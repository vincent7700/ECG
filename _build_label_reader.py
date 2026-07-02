# -*- coding: utf-8 -*-
"""Construit stage/training_label_reader.ipynb : CNN lecteur de labels de derivation."""
import json, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

cells = []
def md(t): cells.append({"cell_type":"markdown","metadata":{},"source":t.splitlines(keepends=True)})
def code(t): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":t.splitlines(keepends=True)})

md(r"""# Lecteur de labels de dérivation (CNN)

**But** : lire l'**étiquette imprimée** d'une dérivation (« I », « V1 », « aVR »…) sur un petit crop, pour identifier chaque tracé **par son label** (pas par sa position ni sa forme d'onde).

**Pourquoi** : l'identité par label est **immunisée contre le surapprentissage** (on n'a que ~2 ECG sources). Le texte est varié (polices, nomenclatures, augmentations) → ça généralise, y compris au réel.

- **Entraînement** : pipeline synthétique (`output_augmentation`) — GT gratuite (`label_centers` + `label_names`). Train = ECG_031/032, Val = ECG_033.
- **Test** : PM Cardio réel (`output_real`) — qualitatif (pas de GT de position).
""")

code(r'''# ── Cellule 1 : imports + config ──
import os, sys, glob, io, time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A

PROJECT_ROOT = r"C:\Users\v\Desktop\ECGPerturb-main"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from shared.npz_schema import load_unified

%matplotlib inline

IMG_DIR  = os.path.join(PROJECT_ROOT, "data", "output_augmentation", "images")
LBL_DIR  = os.path.join(PROJECT_ROOT, "data", "output_augmentation", "labels")
REAL_DIR = os.path.join(PROJECT_ROOT, "data", "output_real")
CACHE_DIR= os.path.join(PROJECT_ROOT, "data", "training", "label_reader_cache")
OUT_DIR  = os.path.join(PROJECT_ROOT, "data", "training", "runs_label_reader")
os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD2IDX = {l: i for i, l in enumerate(LEADS)}

CROP_HALF = 44      # demi-cote du crop (px image native) -- assez large pour ne PAS tronquer "III"
MODEL_RES = 96      # taille d'entree du CNN -- 96 (vs 64) pour distinguer I/II/III (compter les barres) et R/L/F
TRAIN_W   = 3648    # largeur de reference des images d'entrainement -> sert a scaler le crop a l'inference reelle
TRAIN_SOURCES = ["ECG_031", "ECG_032"]
VAL_SOURCES   = ["ECG_033"]

BATCH = 128
EPOCHS = 25
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE, "| classes:", len(LEADS))
''')

code(r'''# ── Cellule 2 : extraction des crops de labels -> cache .npy (a lancer une fois) ──
def extract_crops(sources, tag, force=False):
    """Pour chaque image des `sources`, crope les 12 labels (label_centers) et
    enregistre X (N,RES,RES,3) uint8 + y (N,) dans le cache."""
    # le tag de cache encode crop_half + resolution -> changer CROP_HALF/MODEL_RES regenere
    # automatiquement (pas de melange de tailles entre runs).
    key = f"{tag}_h{CROP_HALF}_r{MODEL_RES}"
    fx = os.path.join(CACHE_DIR, f"{key}_X.npy"); fy = os.path.join(CACHE_DIR, f"{key}_y.npy")
    if not force and os.path.exists(fx) and os.path.exists(fy):
        X, y = np.load(fx), np.load(fy)
        print(f"[{key}] cache existant : {len(y)} crops")
        return X, y
    X, y = [], []
    files = [f for f in sorted(os.listdir(IMG_DIR))
             if f.endswith(".webp") and any(f.startswith(p) for p in sources)]
    print(f"[{tag}] {len(files)} images...")
    for n, fname in enumerate(files):
        stem = fname[:-5]
        try:
            d = load_unified(os.path.join(LBL_DIR, stem), load_maps=False)
            lc = d.get("label_centers"); ln = d.get("label_names")
            if lc is None or ln is None:
                continue
            img = np.array(Image.open(os.path.join(IMG_DIR, fname)).convert("RGB"))
        except Exception:
            continue
        H, W = img.shape[:2]
        for i in range(len(lc)):
            lead = str(ln[i])
            if lead not in LEAD2IDX:
                continue
            cx, cy = int(round(float(lc[i][0]))), int(round(float(lc[i][1])))
            x0, x1 = max(0, cx - CROP_HALF), min(W, cx + CROP_HALF)
            y0, y1 = max(0, cy - CROP_HALF), min(H, cy + CROP_HALF)
            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (MODEL_RES, MODEL_RES), interpolation=cv2.INTER_AREA)
            X.append(crop); y.append(LEAD2IDX[lead])
        if (n + 1) % 300 == 0:
            print(f"   {n+1}/{len(files)} images, {len(y)} crops")
    X = np.asarray(X, np.uint8); y = np.asarray(y, np.int64)
    np.save(fx, X); np.save(fy, y)
    print(f"[{key}] -> {len(y)} crops sauves")
    return X, y

train_X, train_y = extract_crops(TRAIN_SOURCES, "train")
val_X,   val_y   = extract_crops(VAL_SOURCES,   "val")
print("train:", train_X.shape, "| val:", val_X.shape)
print("repartition train par classe:", {LEADS[i]: int((train_y==i).sum()) for i in range(len(LEADS))})
''')

code(r'''# ── Cellule 3 : Dataset + DataLoaders ──
class LabelCropDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X, self.y = X, y
        # PAS de flip (le texte ne se reflechit pas). Rotation FAIBLE (6deg) : au-dela,
        # les barres verticales de I/II/III deviennent des traits obliques ambigus.
        self.tf = A.Compose([
            A.Rotate(limit=6, border_mode=cv2.BORDER_REPLICATE, p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.GaussNoise(p=0.2),
        ]) if augment else None

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        img = self.X[i]
        if self.tf is not None:
            img = self.tf(image=img)["image"]
        t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        return t, int(self.y[i])

train_ds = LabelCropDataset(train_X, train_y, augment=True)
val_ds   = LabelCropDataset(val_X,   val_y,   augment=False)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
print("train batches:", len(train_loader), "| val batches:", len(val_loader))
''')

code(r'''# ── Cellule 4 : modele (resnet18, 12 classes) ──
# resnet18 (sans pre-entrainement) discrimine bien mieux les glyphes proches
# (I/II/III = compter des barres ; aVR/aVL/aVF = derniere lettre) que le petit CNN.
from torchvision.models import resnet18

def build_model(n_classes=len(LEADS)):
    m = resnet18(weights=None, num_classes=n_classes)
    return m

model = build_model().to(DEVICE)
print("Backbone: resnet18 | parametres:", sum(p.numel() for p in model.parameters()))
''')

code(r'''# ── Cellule 5 : entrainement ──
import datetime
# pondération de classes : III/aVR/aVL/aVF sont structurellement plus rares
# (absentes des formats 4x2 et 6x1;6x1) -> on compense pour une accuracy/derivation equitable.
counts = np.bincount(train_y, minlength=len(LEADS)).astype(np.float32)
weights = (counts.sum() / (len(LEADS) * np.clip(counts, 1, None)))
class_w = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
print("poids de classe:", {LEADS[i]: round(float(weights[i]), 2) for i in range(len(LEADS))})
criterion = nn.CrossEntropyLoss(weight=class_w)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

run_dir = os.path.join(OUT_DIR, "run_latest"); os.makedirs(run_dir, exist_ok=True)
best_acc, hist = 0.0, {"train_acc": [], "val_acc": [], "val_loss": []}

def run_epoch(loader, train=True):
    model.train(train)
    tot, correct, loss_sum = 0, 0, 0.0
    torch.set_grad_enabled(train)
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb); loss = criterion(logits, yb)
        if train:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        loss_sum += loss.item() * len(yb)
        correct += (logits.argmax(1) == yb).sum().item(); tot += len(yb)
    torch.set_grad_enabled(True)
    if tot == 0:   # loader vide -> evite ZeroDivisionError (signale plutot un cache/extraction rate)
        return float("nan"), 0.0
    return loss_sum / tot, correct / tot

print(f"Entrainement {EPOCHS} epochs sur {len(train_y)} crops...")
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc = run_epoch(train_loader, True)
    va_loss, va_acc = run_epoch(val_loader, False)
    scheduler.step(va_acc)
    hist["train_acc"].append(tr_acc); hist["val_acc"].append(va_acc); hist["val_loss"].append(va_loss)
    flag = ""
    if va_acc > best_acc:
        best_acc = va_acc
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_acc": va_acc,
                    "leads": LEADS, "crop_half": CROP_HALF, "model_res": MODEL_RES, "train_w": TRAIN_W},
                   os.path.join(run_dir, "best_label_reader.pth"))
        flag = "  [BEST]"
    print(f"Epoch {epoch:2d}/{EPOCHS} | train acc {tr_acc:.3f} | val acc {va_acc:.3f} "
          f"loss {va_loss:.3f} | lr {optimizer.param_groups[0]['lr']:.1e} | {time.time()-t0:.0f}s{flag}")

print(f"\nMeilleure val accuracy : {best_acc:.4f}  -> {os.path.join(run_dir,'best_label_reader.pth')}")
''')

code(r'''# ── Cellule 6 : evaluation sur la val synthetique (matrice de confusion + acc/classe) ──
model.eval()
all_pred, all_true = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        all_pred.append(model(xb.to(DEVICE)).argmax(1).cpu().numpy()); all_true.append(yb.numpy())
all_pred = np.concatenate(all_pred); all_true = np.concatenate(all_true)

acc = (all_pred == all_true).mean()
print(f"Val accuracy globale : {acc:.4f}\n")
print("Accuracy par derivation :")
for i, l in enumerate(LEADS):
    m = all_true == i
    if m.sum(): print(f"   {l:4s}: {(all_pred[m]==i).mean():.3f}  (n={int(m.sum())})")

# matrice de confusion
C = np.zeros((len(LEADS), len(LEADS)), int)
for t, p in zip(all_true, all_pred): C[t, p] += 1
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(C, cmap="Blues")
ax.set_xticks(range(len(LEADS))); ax.set_xticklabels(LEADS, rotation=45)
ax.set_yticks(range(len(LEADS))); ax.set_yticklabels(LEADS)
ax.set_xlabel("Predit"); ax.set_ylabel("Vrai"); ax.set_title(f"Matrice de confusion (val) - acc {acc:.3f}")
for i in range(len(LEADS)):
    for j in range(len(LEADS)):
        if C[i, j]: ax.text(j, i, C[i, j], ha="center", va="center",
                            color="white" if C[i, j] > C.max()*0.5 else "black", fontsize=8)
plt.colorbar(im); plt.tight_layout(); plt.show()
''')

code(r'''# ── Cellule 7 : TEST sur PM Cardio (output_real) -- crop SCALE-AWARE, ciblage MANUEL ──
# Correction clef : le crop est mis a l'ECHELLE de l'image. Les images PM Cardio font
# 578..7483 px de large (entrainement = 3648 px). Sans mise a l'echelle, un crop fixe ne
# capte qu'un bout de caractere sur les grandes images -> predictions aleatoires.
#
# Le ciblage des labels est VOLONTAIREMENT manuel : la detection auto sur photo reelle
# n'est pas fiable (labels tres pales, layouts variables 3x4 / 6x2 / +rythme, faux positifs
# des traces/QR/fond -- teste et rejete). Deux etapes :
#   MODE="locate" -> affiche l'image + grille de coordonnees pour LIRE les (x,y) des labels
#   MODE="read"   -> remplis LABEL_CENTERS, le CNN lit chaque crop
# Astuce : pour une photo gondolee/perspective, dewarpe d'abord (image plate = coords nettes).
%matplotlib inline

ckpt = torch.load(os.path.join(OUT_DIR, "run_latest", "best_label_reader.pth"),
                  map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"]); model.eval()
TRAIN_W_CK = ckpt.get("train_w", 3648)
print(f"Modele (val_acc {ckpt['val_acc']:.3f}) | crop_half={ckpt['crop_half']} res={ckpt['model_res']} train_w={TRAIN_W_CK}")

# ----------------------- A AJUSTER -----------------------
REAL_SUBFOLDER = "augmentation_brightness_120"   # photo nette ; sinon digital_data_opacity_015, photos_bents...
REAL_INDEX     = 8             # img_17_page_0 (exemple pre-rempli ci-dessous)
MODE           = "read"        # "locate" : reperer les (x,y) ; "read" : lire LABEL_CENTERS
# Exemple pre-rempli pour img_17. Pour une AUTRE image : mets MODE="locate", relis les (x,y), recolle.
LABEL_CENTERS  = [(505,1190),(505,1535),(505,1880),     # I, II, III
                  (1368,1310),(1368,1555),(1368,1880),  # aVR, aVL, aVF
                  (2186,1235),(2186,1555),(2186,1880),   # V1, V2, V3
                  (3080,1310),(3080,1585),(3080,1890)]   # V4, V5, V6 (bord droit perspectif -> moins sur)
EXPECTED       = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
# ---------------------------------------------------------

real_files = sorted(glob.glob(os.path.join(REAL_DIR, REAL_SUBFOLDER, "*")))
real_files = [f for f in real_files if f.lower().endswith((".jpg",".jpeg",".png",".webp",".bmp"))]
if not real_files:
    raise FileNotFoundError(f"Aucune image dans {os.path.join(REAL_DIR, REAL_SUBFOLDER)} -- verifie REAL_SUBFOLDER")
img = np.array(Image.open(real_files[REAL_INDEX]).convert("RGB"))
H, W = img.shape[:2]
scale = W / float(TRAIN_W_CK)
half  = max(8, int(round(ckpt["crop_half"] * scale)))     # <-- crop a l'echelle de l'image
print(f"Image: {os.path.basename(real_files[REAL_INDEX])} ({W}x{H}) | echelle x{scale:.2f} -> crop {2*half}px (train {2*ckpt['crop_half']}px)")

def predict_crop(img, cx, cy):
    x0, x1 = max(0, cx-half), min(W, cx+half); y0, y1 = max(0, cy-half), min(H, cy+half)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:   # (cx,cy) hors image -> evite le crash cv2.resize, renvoie un placeholder
        return "?", 0.0, np.zeros((ckpt["model_res"], ckpt["model_res"], 3), np.uint8)
    cc = cv2.resize(crop, (ckpt["model_res"], ckpt["model_res"]), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(cc.astype(np.float32)/255).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = torch.softmax(model(t), 1)[0].cpu().numpy()
    k = int(prob.argmax())
    return LEADS[k], float(prob[k]), crop

if MODE == "locate" or not LABEL_CENTERS:
    # Affiche l'image + grille de coords. Lis les (x,y) des labels, mets-les dans
    # LABEL_CENTERS, passe MODE="read". Le carre rouge central montre la TAILLE du crop.
    step = int(round(200 * scale))
    fig, ax = plt.subplots(figsize=(16, 11)); ax.imshow(img)
    ax.add_patch(plt.Rectangle((W//2-half, H//2-half), 2*half, 2*half, fill=False, ec="red", lw=2))
    ax.set_xticks(np.arange(0, W, step)); ax.set_yticks(np.arange(0, H, step))
    ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
    ax.set_title("MODE locate : lis les (x,y) des labels -> LABEL_CENTERS, puis MODE='read'. Carre rouge = taille du crop.")
    plt.show()
    print("Repere les coordonnees ci-dessus, remplis LABEL_CENTERS et mets MODE='read'.")
else:
    centers  = LABEL_CENTERS
    expected = EXPECTED if len(EXPECTED) == len(centers) else [None]*len(centers)

    # (a) apercu : positions + boites de crop sur l'image
    disp = img.copy(); r = max(6, int(0.004*W))
    for (cx, cy), exp in zip(centers, expected):
        cv2.rectangle(disp, (cx-half, cy-half), (cx+half, cy+half), (255, 0, 0), max(2, r//3))
        if exp: cv2.putText(disp, exp, (cx-half, cy-half-8), cv2.FONT_HERSHEY_SIMPLEX, 0.9*scale, (255,0,0), 2)
    fig, ax = plt.subplots(figsize=(15, 11)); ax.imshow(disp); ax.axis("off")
    ax.set_title("Boites de crop (rouge). Si decalees, corrige LABEL_CENTERS.")
    plt.show()

    # (b) lecture des crops + comparaison a l'attendu (vert=ok, rouge=faux)
    n = len(centers); cols = min(6, n); rows = (n + cols - 1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.2*cols, 2.6*rows)); axes = np.array(axes).reshape(-1)
    ok = 0
    for ax, (cx, cy), exp in zip(axes, centers, expected):
        lead, conf, crop = predict_crop(img, cx, cy)
        match = (exp is None) or (lead == exp); ok += int(exp is not None and lead == exp)
        ax.imshow(crop); ax.axis("off")
        ax.set_title(f"{lead} ({conf:.2f})" + (f"\n[att: {exp}]" if exp else ""),
                     fontsize=10, color=("green" if match else "red"))
    for ax in axes[n:]: ax.axis("off")
    if any(e is not None for e in expected):
        plt.suptitle(f"PM Cardio {os.path.basename(real_files[REAL_INDEX])} -- corrects : {ok}/{n}", fontsize=13)
    plt.tight_layout(); plt.show()
''')

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
p = "stage/training_label_reader.ipynb"
json.dump(nb, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("[OK] ecrit", p, "-", len(cells), "cellules")
