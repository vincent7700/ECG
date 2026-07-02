# -*- coding: utf-8 -*-
"""Construit stage/training_label_detector.ipynb : detecteur de labels (heatmap de points)."""
import json, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

cells = []
def md(t): cells.append({"cell_type":"markdown","metadata":{},"source":t.splitlines(keepends=True)})
def code(t): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":t.splitlines(keepends=True)})

md(r"""# Détecteur de labels de dérivation (heatmap de points)

**But** : trouver **où** sont les ~12 étiquettes de dérivation sur une image ECG, quel que soit le layout (3×4, 6×2, +rythme…), la perspective ou la résolution. Le détecteur ne lit pas le texte — il sort une **heatmap class-agnostic** dont les pics = positions des labels. Le **lecteur** (`training_label_reader`) dira ensuite *quelle* dérivation à chaque pic.

**Pourquoi** : sur le réel, placer les crops à la main ne marche pas (layouts variables, perspective → boîtes « à côté de la plaque »). Le détecteur automatise la localisation. GT gratuite : `label_centers` du synthétique.

**Pipeline final** : `image → détecteur (pics) → crops centrés → lecteur → 12 dérivations`.

- Entraînement : `output_augmentation` (031/032 train, 033 val), cible = heatmap gaussienne aux `label_centers`.
- Sortie : 1 canal (présence de label), pics extraits par maximum local.
""")

code(r'''# ── Cellule 1 : imports + config + letterbox ──
import os, sys, glob, io, time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

PROJECT_ROOT = r"C:\Users\v\Desktop\ECGPerturb-main"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from shared.npz_schema import load_unified

%matplotlib inline

IMG_DIR  = os.path.join(PROJECT_ROOT, "data", "output_augmentation", "images")
LBL_DIR  = os.path.join(PROJECT_ROOT, "data", "output_augmentation", "labels")
MASK_DIR = os.path.join(PROJECT_ROOT, "data", "output_augmentation", "masks")   # mask_all_signals = negatifs durs
REAL_DIR = os.path.join(PROJECT_ROOT, "data", "output_real")
CACHE_DIR= os.path.join(PROJECT_ROOT, "data", "training", "label_detector_cache")
OUT_DIR  = os.path.join(PROJECT_ROOT, "data", "training", "runs_label_detector")
os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)

IN_W, IN_H = 1024, 736      # entree du U-Net (+ haute resolution -> pics + precis sur les petits labels des membres)
SIGMA = 2.0                 # pic gaussien plus serre -> localisation plus precise
MAXC  = 24                  # nb max de labels caches par image
TRAIN_SOURCES = ["ECG_031", "ECG_032"]
VAL_SOURCES   = ["ECG_033"]
BATCH = 6                   # baisse vs 768 (1024 = ~1.8x VRAM)
EPOCHS = 30
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def letterbox(img, out_w=IN_W, out_h=IN_H, pad=255):
    """Redimensionne en gardant le ratio + pad (centre). Retourne (canvas, s, ox, oy)."""
    h, w = img.shape[:2]
    s = min(out_w / w, out_h / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((out_h, out_w, 3), pad, np.uint8)
    ox, oy = (out_w - nw) // 2, (out_h - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas, s, ox, oy

print("Device:", DEVICE, "| entree", IN_W, "x", IN_H)
''')

code(r'''# ── Cellule 2 : cache (images letterboxées + centres en coords d'entrée) ──
def _gauss(sigma):
    r = int(round(3 * sigma)); ax = np.arange(-r, r + 1)
    xx, yy = np.meshgrid(ax, ax)
    return np.exp(-(xx**2 + yy**2) / (2 * sigma**2)).astype(np.float32), r
GK, GR = _gauss(SIGMA)

def build_cache(sources, tag, force=False):
    # cache = images letterboxees + centres + MASQUE SIGNAL letterboxe (negatifs durs)
    key = f"{tag}_{IN_W}x{IN_H}"
    fi = os.path.join(CACHE_DIR, f"{key}_imgs.npy"); fc = os.path.join(CACHE_DIR, f"{key}_cts.npy")
    fs = os.path.join(CACHE_DIR, f"{key}_sig.npy")
    if not force and os.path.exists(fi) and os.path.exists(fc) and os.path.exists(fs):
        imgs = np.load(fi, mmap_mode="r"); cts = np.load(fc); sigs = np.load(fs, mmap_mode="r")
        print(f"[{key}] cache : {len(imgs)} images (+ masques signal)"); return imgs, cts, sigs
    files = [f for f in sorted(os.listdir(IMG_DIR))
             if f.endswith(".webp") and any(f.startswith(p) for p in sources)]
    print(f"[{key}] {len(files)} images...")
    imgs = np.zeros((len(files), IN_H, IN_W, 3), np.uint8)
    cts  = np.full((len(files), MAXC, 2), np.nan, np.float32)
    sigs = np.zeros((len(files), IN_H, IN_W), np.uint8)
    keep = 0
    for fname in files:
        stem = fname[:-5]
        try:
            d = load_unified(os.path.join(LBL_DIR, stem), load_maps=False)
            lc = d.get("label_centers")
            img = np.array(Image.open(os.path.join(IMG_DIR, fname)).convert("RGB"))
        except Exception:
            continue
        if lc is None: continue
        canvas, s, ox, oy = letterbox(img)
        imgs[keep] = canvas
        for i, (x, y) in enumerate(lc[:MAXC]):
            cts[keep, i] = [float(x) * s + ox, float(y) * s + oy]
        # masque signal (mask_all_signals) letterboxe a l'identique -> negatifs durs (tracés)
        mp = os.path.join(MASK_DIR, stem, "mask_all_signals.png")
        if os.path.exists(mp):
            m = np.array(Image.open(mp).convert("L"))
            m3 = np.repeat(m[:, :, None], 3, axis=2)
            mc, _, _, _ = letterbox(m3, pad=0)
            sigs[keep] = (mc[:, :, 0] > 127).astype(np.uint8)
        keep += 1
        if keep % 300 == 0: print(f"   {keep}/{len(files)}")
    imgs, cts, sigs = imgs[:keep], cts[:keep], sigs[:keep]
    np.save(fi, imgs); np.save(fc, cts); np.save(fs, sigs)
    print(f"[{key}] -> {keep} images"); return imgs, cts, sigs

def make_heatmap(centers):
    hm = np.zeros((IN_H, IN_W), np.float32)
    for x, y in centers:
        if not np.isfinite(x): continue
        cx, cy = int(round(x)), int(round(y))
        x0, x1 = max(0, cx - GR), min(IN_W, cx + GR + 1)
        y0, y1 = max(0, cy - GR), min(IN_H, cy + GR + 1)
        if x0 >= x1 or y0 >= y1: continue
        gx0, gy0 = x0 - (cx - GR), y0 - (cy - GR)
        patch = GK[gy0:gy0 + (y1 - y0), gx0:gx0 + (x1 - x0)]
        hm[y0:y1, x0:x1] = np.maximum(hm[y0:y1, x0:x1], patch)
    return hm

train_imgs, train_cts, train_sigs = build_cache(TRAIN_SOURCES, "train")
val_imgs,   val_cts,   val_sigs   = build_cache(VAL_SOURCES,   "val")
print("train:", train_imgs.shape, "| val:", val_imgs.shape,
      "| labels/img moy:", round(float(np.isfinite(train_cts[:,:,0]).sum(1).mean()), 1),
      "| signal moy:", round(float(np.asarray(train_sigs).mean()), 4))
''')

code(r'''# ── Cellule 3 : Dataset + DataLoaders (+ aperçu d'un échantillon) ──
class DetDataset(Dataset):
    def __init__(self, imgs, cts, sigs, augment=False):
        self.imgs, self.cts, self.sigs, self.aug = imgs, cts, sigs, augment
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        img = np.asarray(self.imgs[i]).astype(np.float32) / 255.0
        if self.aug:  # photometrique seulement (la heatmap reste valide)
            img = np.clip(img * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.05, 0.05), 0, 1)
            if np.random.rand() < 0.3:
                img = np.clip(img + np.random.normal(0, 0.02, img.shape), 0, 1).astype(np.float32)
        hm = make_heatmap(self.cts[i])
        sig = np.asarray(self.sigs[i]).astype(np.float32)   # 1 sur tracé -> negatif dur
        x = torch.from_numpy(img).permute(2, 0, 1)
        return x, torch.from_numpy(hm).unsqueeze(0), torch.from_numpy(sig).unsqueeze(0)

train_ds = DetDataset(train_imgs, train_cts, train_sigs, augment=True)
val_ds   = DetDataset(val_imgs,   val_cts,   val_sigs,   augment=False)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)

xb, yb, sb = next(iter(train_loader))
print("batch img", tuple(xb.shape), "heatmap", tuple(yb.shape), "signal", tuple(sb.shape))
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].imshow(xb[0].permute(1,2,0).numpy()); ax[0].set_title("image (letterbox)"); ax[0].axis("off")
ov = xb[0].permute(1,2,0).numpy().copy()
ov[..., 0] = np.maximum(ov[..., 0], yb[0,0].numpy())   # rouge = heatmap GT (labels)
ov[..., 2] = np.maximum(ov[..., 2], sb[0,0].numpy())   # bleu = signal (negatifs durs)
ax[1].imshow(ov); ax[1].set_title("heatmap GT (rouge) + signal/negatifs (bleu)"); ax[1].axis("off"); plt.show()
''')

code(r'''# ── Cellule 4 : modèle (U-Net resnet34, 1 canal heatmap) ──
def build_model():
    try:   # poids imagenet (meilleur depart) ; repli aleatoire si pas de reseau
        return smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1, activation=None)
    except Exception as e:
        print("imagenet indispo (", e, ") -> init aleatoire")
        return smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1, activation=None)
model = build_model().to(DEVICE)
print("U-Net resnet34 | params:", sum(p.numel() for p in model.parameters()))
''')

code(r'''# ── Cellule 5 : entraînement (MSE sur heatmap sigmoïde) ──
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
# MSE PONDEREE : la heatmap est creuse (99% de zeros). Un MSE simple s'effondre (le modele
# sort ~0 partout). On pondere les pixels-pics x(1+POS_W*cible) pour forcer l'apprentissage.
POS_W = 80.0
NEG_W = 25.0   # NEGATIFS DURS : poids sur les pixels de TRACE (signal) hors-label -> apprend a NE PAS tirer sur les tracés (corrige les FP type QRS / colonne-fantome des longues bandes)
def weighted_mse(pred, tgt, sig=None):
    w = 1.0 + POS_W * tgt
    if sig is not None:
        w = w + NEG_W * sig * (tgt < 0.1).float()   # tracé ET pas de label -> negatif dur
    return (w * (pred - tgt) ** 2).mean()
import datetime, shutil
run_dir = os.path.join(OUT_DIR, "run_latest")
_old = os.path.join(run_dir, "best_label_detector.pth")
if os.path.exists(_old):   # SAUVEGARDE NON DESTRUCTIVE : archive l'ancien best avant ce run
    _arch = os.path.join(OUT_DIR, "archive"); os.makedirs(_arch, exist_ok=True)
    _dst = os.path.join(_arch, f"best_{datetime.datetime.now():%Y%m%d_%H%M%S}.pth")
    shutil.copy2(_old, _dst); print("ancien best archive ->", _dst)
os.makedirs(run_dir, exist_ok=True)
best_val = float("inf")

def run_epoch(loader, train=True):
    model.train(train); tot, loss_sum = 0, 0.0
    torch.set_grad_enabled(train)
    for xb, yb, sb in loader:
        xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
        pred = torch.sigmoid(model(xb)); loss = weighted_mse(pred, yb, sb)
        if train:
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # anti-divergence
            optimizer.step()
        loss_sum += loss.item() * len(xb); tot += len(xb)
    torch.set_grad_enabled(True)
    return loss_sum / max(tot, 1)

print(f"Entrainement {EPOCHS} epochs...")
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tr = run_epoch(train_loader, True); va = run_epoch(val_loader, False)
    scheduler.step(va); flag = ""
    if va < best_val:
        best_val = va
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": va,
                    "in_w": IN_W, "in_h": IN_H, "sigma": SIGMA},
                   os.path.join(run_dir, "best_label_detector.pth"))
        flag = "  [BEST]"
    print(f"Epoch {epoch:2d}/{EPOCHS} | train {tr:.5f} | val {va:.5f} | "
          f"lr {optimizer.param_groups[0]['lr']:.1e} | {time.time()-t0:.0f}s{flag}")
print(f"\nMeilleure val MSE : {best_val:.5f} -> {os.path.join(run_dir,'best_label_detector.pth')}")
''')

code(r'''# ── Cellule 6 : extraction de pics + éval détection (precision/recall) sur la val ──
def find_peaks(hm, thresh=0.3, min_dist=17):
    """Maxima locaux > thresh (NMS par dilatation). Retourne liste de (x,y)."""
    d = cv2.dilate(hm, np.ones((min_dist, min_dist), np.float32))
    mask = (hm >= d) & (hm > thresh)
    ys, xs = np.where(mask)
    return list(zip(xs.tolist(), ys.tolist()))

@torch.no_grad()
def predict_hm(img_tensor):
    return torch.sigmoid(model(img_tensor.unsqueeze(0).to(DEVICE)))[0, 0].cpu().numpy()

# recharge le MEILLEUR checkpoint (l'etat final en memoire peut avoir diverge -> heatmap diffuse)
_best = os.path.join(OUT_DIR, "run_latest", "best_label_detector.pth")
if os.path.exists(_best):
    model.load_state_dict(torch.load(_best, map_location=DEVICE, weights_only=False)["model_state_dict"])
    print("meilleur checkpoint recharge pour l'eval")
model.eval()
# metrique : un pic predit est "bon" s'il tombe a < TOL px d'un centre GT
TOL = 12; TP = FP = FN = 0
for i in range(min(len(val_ds), 200)):
    x, _ = val_ds[i]
    hm = predict_hm(x)
    peaks = find_peaks(hm)
    gt = val_cts[i][np.isfinite(val_cts[i][:, 0])]
    used = set()
    for px, py in peaks:
        dists = [np.hypot(px - gx, py - gy) for gx, gy in gt]
        j = int(np.argmin(dists)) if dists else -1
        if j >= 0 and dists[j] < TOL and j not in used: TP += 1; used.add(j)
        else: FP += 1
    FN += len(gt) - len(used)
prec = TP / max(TP + FP, 1); rec = TP / max(TP + FN, 1)
print(f"Detection (val, 200 img, TOL={TOL}px) : precision {prec:.3f} | recall {rec:.3f} | F1 {2*prec*rec/max(prec+rec,1e-9):.3f}")

# visualisation sur 3 images val
fig, axes = plt.subplots(3, 2, figsize=(12, 11))
for r in range(3):
    x, _ = val_ds[r]; hm = predict_hm(x); peaks = find_peaks(hm)
    base = x.permute(1, 2, 0).numpy()
    axes[r,0].imshow(hm, cmap="hot"); axes[r,0].set_title(f"heatmap predite ({len(peaks)} pics)"); axes[r,0].axis("off")
    vis = (base*255).astype(np.uint8).copy()
    for px, py in peaks: cv2.circle(vis, (px, py), 6, (255, 0, 0), 2)
    axes[r,1].imshow(vis); axes[r,1].set_title("pics sur image"); axes[r,1].axis("off")
plt.tight_layout(); plt.show()
''')

code(r'''# ── Cellule 7 : PIPELINE COMPLET sur PM Cardio (détecteur -> lecteur) ──
# Le detecteur localise les labels automatiquement (n'importe quel layout / perspective),
# puis le lecteur lit chaque crop. Plus aucune coordonnee a la main.
%matplotlib inline
from torchvision.models import resnet18

LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
READER_CKPT = os.path.join(PROJECT_ROOT, "data", "training", "runs_label_reader", "run_latest", "best_label_reader.pth")

det_ck = torch.load(os.path.join(OUT_DIR, "run_latest", "best_label_detector.pth"), map_location=DEVICE, weights_only=False)
model.load_state_dict(det_ck["model_state_dict"]); model.eval()
rd_ck = torch.load(READER_CKPT, map_location=DEVICE, weights_only=False)
reader = resnet18(weights=None, num_classes=12).to(DEVICE); reader.load_state_dict(rd_ck["model_state_dict"]); reader.eval()
print(f"detecteur (val_loss {det_ck['val_loss']:.5f}) + lecteur (val_acc {rd_ck['val_acc']:.3f}) charges")

# --- A AJUSTER ---
REAL_SUBFOLDER = "augmentation_brightness_120"
REAL_INDEX     = 8
THRESH         = 0.7      # seuil de detection des pics (0.7 propre ; monte vers 0.85 sur layouts a longues bandes)
READER_CONF_MIN = 0.90   # garde une detection seulement si le LECTEUR est sur (coupe les crops sur tracé)
# -----------------
rf = sorted(glob.glob(os.path.join(REAL_DIR, REAL_SUBFOLDER, "*")))
rf = [f for f in rf if f.lower().endswith((".jpg",".jpeg",".png",".webp",".bmp"))]
if not rf: raise FileNotFoundError(f"Aucune image dans {REAL_SUBFOLDER}")
img = np.array(Image.open(rf[REAL_INDEX]).convert("RGB")); H, W = img.shape[:2]

# 1) detection (sur image letterboxée) -> pics -> coords image native
canvas, s, ox, oy = letterbox(img)
xin = torch.from_numpy(canvas.astype(np.float32) / 255).permute(2, 0, 1)
hm = predict_hm(xin); peaks = find_peaks(hm, thresh=THRESH)
pts = [((px - ox) / s, (py - oy) / s, float(hm[py, px])) for px, py in peaks]   # +valeur du pic detecteur
print(f"{os.path.basename(rf[REAL_INDEX])} ({W}x{H}) -> {len(pts)} labels detectes")

# 2) lecture de chaque crop (scale-aware, comme le lecteur)
rscale = W / float(rd_ck.get("train_w", 3648)); half = max(8, int(round(rd_ck["crop_half"] * rscale))); R = rd_ck["model_res"]
def read_label(cx, cy):
    cx, cy = int(cx), int(cy); x0,x1 = max(0,cx-half),min(W,cx+half); y0,y1 = max(0,cy-half),min(H,cy+half)
    c = img[y0:y1, x0:x1]
    if c.size == 0: return "?", 0.0
    cc = cv2.resize(c, (R, R), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(cc.astype(np.float32)/255).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): p = torch.softmax(reader(t), 1)[0].cpu().numpy()
    return LEADS[int(p.argmax())], float(p.max())
reads_all = [(cx, cy, *read_label(cx, cy), hv) for cx, cy, hv in pts]   # (cx, cy, lead, conf, hm_val)
# FILTRE 1 : un crop sur tracé (pas de texte) donne une confiance lecteur basse -> coupe.
conf_ok = [r for r in reads_all if r[3] >= READER_CONF_MIN]
# FILTRE 2 (dedup) : detections a moins de DEDUP_R px -> on garde le pic DETECTEUR le plus fort.
DEDUP_R = 2 * half
reads = []
for r in sorted(conf_ok, key=lambda r: -r[4]):   # plus fort pic d'abord
    if all((r[0]-k[0])**2 + (r[1]-k[1])**2 > DEDUP_R**2 for k in reads):
        reads.append(r)
print(f"{len(pts)} pics -> {len(conf_ok)} apres conf>={READER_CONF_MIN} -> {len(reads)} apres dedup")

# 3) affichage : retenues (rouge) vs coupees (gris)
kept_xy = {(int(r[0]), int(r[1])) for r in reads}
disp = img.copy()
for cx, cy, lead, conf, hv in reads_all:
    cx, cy = int(cx), int(cy); keep = (cx, cy) in kept_xy
    col = (255, 0, 0) if keep else (150, 150, 150)
    cv2.rectangle(disp, (cx-half, cy-half), (cx+half, cy+half), col, 3 if keep else 1)
    if keep:
        cv2.putText(disp, f"{lead}", (cx-half, cy-half-8), cv2.FONT_HERSHEY_SIMPLEX, 0.9*rscale, (0,140,255), 3)
sp = 1400 / W
plt.figure(figsize=(16, 11)); plt.imshow(cv2.resize(disp, (int(W*sp), int(H*sp)))); plt.axis("off")
plt.title(f"Pipeline detecteur->lecteur : {len(reads)} labels retenus (gris = coupe)"); plt.show()
print("lectures retenues:", [f"{l}({c:.2f})" for _,_,l,c,_ in reads])
''')

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
p = "stage/training_label_detector.ipynb"
json.dump(nb, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("[OK] ecrit", p, "-", len(cells), "cellules")
