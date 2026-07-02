# -*- coding: utf-8 -*-
"""Construit stage/training_digitization.ipynb : pipeline complet
detecteur -> grille/assignation (ordre canonique) -> extraction forme d'onde (modele signal) -> 12 courbes nommees."""
import json, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

cells = []
def md(t): cells.append({"cell_type":"markdown","metadata":{},"source":t.splitlines(keepends=True)})
def code(t): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":t.splitlines(keepends=True)})

md(r"""# Digitization complète : image ECG → 12 courbes nommées

Enchaîne les 3 modèles + la géométrie :

```
image réelle
  → DÉTECTEUR de labels (heatmap)        → positions des étiquettes
  → LECTEUR (resnet18)                    → identité (ancre)
  → GRILLE + ORDRE CANONIQUE             → chaque cellule = une dérivation
  → MODÈLE SIGNAL (masque du tracé)      → forme d'onde par cellule
  → {I:[...], II:[...], ..., V6:[...]}    → 12 courbes nommées
```

- L'assignation contourne la faiblesse du lecteur sur les membres (ordre canonique colonne-major, offset ancré sur les lectures sûres).
- Les colonnes-fantômes (FP sur QRS des longues bandes) sont virées par le filtre de **population de colonne**.
- Marche sur les layouts standards (3×4, 6×2) ; partiel sur 6×2+1R.
""")

code(r'''# ── Cellule 1 : imports + config + chargement des 3 modèles ──
import os, sys, glob, io
import numpy as np, cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch
import segmentation_models_pytorch as smp
from torchvision.models import resnet18

PROJECT_ROOT = r"C:\Users\v\Desktop\ECGPerturb-main"
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
%matplotlib inline
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REAL_DIR = os.path.join(PROJECT_ROOT, "data", "output_real")
LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]   # ordre canonique colonne-major

DET_CKPT    = os.path.join(PROJECT_ROOT, "data","training","runs_label_detector","run_latest","best_label_detector.pth")
READER_CKPT = os.path.join(PROJECT_ROOT, "data","training","runs_label_reader","run_latest","best_label_reader.pth")
SIGNAL_CKPT = os.path.join(PROJECT_ROOT, "data","training","runs_signal","run_20260511_005414","checkpoints","best_model.pth")

det_ck = torch.load(DET_CKPT, map_location=DEVICE, weights_only=False)
DW, DH = det_ck.get("in_w",1024), det_ck.get("in_h",736)
detector = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1, activation=None).to(DEVICE)
detector.load_state_dict(det_ck["model_state_dict"]); detector.eval()

rd_ck = torch.load(READER_CKPT, map_location=DEVICE, weights_only=False)
reader = resnet18(weights=None, num_classes=12).to(DEVICE)
reader.load_state_dict(rd_ck["model_state_dict"]); reader.eval()

sig_ck = torch.load(SIGNAL_CKPT, map_location=DEVICE, weights_only=False)
signal = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1).to(DEVICE)
signal.load_state_dict(sig_ck["model_state_dict"]); signal.eval()

print(f"détecteur {DW}x{DH} (val_loss {det_ck.get('val_loss',-1):.5f}) | lecteur (val_acc {rd_ck.get('val_acc',-1):.3f}) | signal (dice {sig_ck.get('val_dice',-1):.3f})")

def letterbox(img, ow, oh, pad=255):
    h, w = img.shape[:2]; s = min(ow/w, oh/h); nw, nh = int(round(w*s)), int(round(h*s))
    c = np.full((oh, ow, 3), pad, np.uint8); ox, oy = (ow-nw)//2, (oh-nh)//2
    c[oy:oy+nh, ox:ox+nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return c, s, ox, oy
''')

code(r'''# ── Cellule 2 : assignation (détecteur + lecteur → grille → ordre canonique) ──
def cluster1d(vals, tol):
    order = sorted(range(len(vals)), key=lambda i: vals[i]); groups = [[order[0]]]
    for i in order[1:]:
        if vals[i]-vals[groups[-1][-1]] <= tol: groups[-1].append(i)
        else: groups.append([i])
    return [(float(np.mean([vals[i] for i in g])), g) for g in groups]

def detect_and_read(img, THRESH=0.5):
    H, W = img.shape[:2]
    cv0, s, ox, oy = letterbox(img, DW, DH)
    with torch.no_grad():
        hm = torch.sigmoid(detector(torch.from_numpy(cv0.astype(np.float32)/255).permute(2,0,1).unsqueeze(0).to(DEVICE)))[0,0].cpu().numpy()
    d = cv2.dilate(hm, np.ones((17,17),np.float32)); ys, xs = np.where((hm>=d)&(hm>THRESH))
    rsc = W/float(rd_ck.get("train_w",3648)); half = max(8, int(round(rd_ck["crop_half"]*rsc))); R = rd_ck["model_res"]
    out = []
    for px, py in zip(xs, ys):
        cx, cy = (px-ox)/s, (py-oy)/s; cxi, cyi = int(cx), int(cy)
        c = img[max(0,cyi-half):cyi+half, max(0,cxi-half):cxi+half]
        if c.size == 0: continue
        cc = cv2.resize(c,(R,R),interpolation=cv2.INTER_AREA)
        with torch.no_grad():
            pr = torch.softmax(reader(torch.from_numpy(cc.astype(np.float32)/255).permute(2,0,1).unsqueeze(0).to(DEVICE)),1)[0].cpu().numpy()
        out.append({"x":cx,"y":cy,"hm":float(hm[py,px]),"lead":LEADS[int(pr.argmax())],
                    "conf":float(pr.max()),"probs":pr.astype(np.float32)})   # softmax complet -> affectation
    return out, half

def assign(img, THRESH=0.5):
    H, W = img.shape[:2]
    dets, half = detect_and_read(img, THRESH)
    anchors = [d for d in dets if d["conf"]>=0.90]
    if len(anchors) < 3: return None
    # COLONNES : garde les colonnes PLEINES (>= moitié de la plus peuplée) -> vire les fantômes
    colc = cluster1d([d["x"] for d in anchors], tol=0.07*W); maxpop = max(len(g) for _,g in colc)
    colx = sorted(cx for cx,g in colc if len(g) >= max(2, 0.5*maxpop))
    if not colx: return None
    kept = [d for d in anchors if min(abs(d["x"]-cx) for cx in colx) < 0.07*W]
    for d in kept: d["col"] = int(np.argmin([abs(d["x"]-cx) for cx in colx]))
    # RANGEES : cluster global des y, validé par support multi-colonnes + grille régulière
    rowc = cluster1d([d["y"] for d in kept], tol=0.045*H)
    def strong(idxs): return len(colx)<2 or len({kept[i]["col"] for i in idxs})>=2
    strong_rows = sorted(cy for cy,idxs in rowc if strong(idxs))
    if len(strong_rows) >= 2:
        spacing = float(np.median(np.diff(strong_rows))); y0 = strong_rows[0]
        lo, hi = strong_rows[0]-0.5*spacing, strong_rows[-1]+0.5*spacing
        rowy = sorted(cy for cy,idxs in rowc
                      if strong(idxs) or (abs((cy-y0)-round((cy-y0)/spacing)*spacing)<0.35*spacing and lo<=cy<=hi))
    else:
        rowy = sorted(c for c,_ in rowc); spacing = (rowy[1]-rowy[0]) if len(rowy)>1 else H*0.15
    ncols, nrows = len(colx), len(rowy)
    cells = []
    for ci in range(ncols):
        for ri in range(nrows):
            cand = [d for d in kept if d["col"]==ci and abs(d["y"]-rowy[ri])<0.045*H]
            cells.append({"col":ci,"row":ri,"cx":colx[ci],"y":rowy[ri],
                          "det": max(cand,key=lambda d:d["hm"]) if cand else None})
    # IDENTITE ADAPTATIVE (aucun ordre figé) : affectation optimale cellule <-> dérivation
    # maximisant la proba du LECTEUR, sous contrainte "chaque dérivation au plus une fois"
    # (algorithme hongrois). S'adapte a n'importe quelle disposition ; la bijection corrige
    # les erreurs du lecteur (V sûrs -> membres déduits par élimination).
    from scipy.optimize import linear_sum_assignment
    P = np.full((len(cells), len(LEADS)), 1.0/len(LEADS), np.float32)   # cellules sans détection -> a priori uniforme
    for i, g in enumerate(cells):
        if g["det"] is not None: P[i] = g["det"]["probs"]
    cost = -np.log(P + 1e-6)
    rr, cc = linear_sum_assignment(cost)   # rr: cellules, cc: dérivations (>=12 cellules -> seules les 12 meilleures sont prises)
    for g in cells: g["canon"] = "?"
    for i, j in zip(rr, cc): cells[i]["canon"] = LEADS[j]
    return {"cells":cells,"half":half,"colx":colx,"rowy":rowy,"spacing":spacing,"ncols":ncols,"nrows":nrows}
''')

code(r'''# ── Cellule 3 : masque signal + extraction de la forme d'onde par cellule ──
def signal_mask(img, thresh=0.5):
    """Modele signal -> masque binaire du tracé, en résolution native."""
    H, W = img.shape[:2]
    sq = cv2.resize(img, (1024,1024), interpolation=cv2.INTER_LINEAR).astype(np.float32)/255
    with torch.no_grad():
        sm = torch.sigmoid(signal(torch.from_numpy(sq).permute(2,0,1).unsqueeze(0).to(DEVICE)))[0,0].cpu().numpy()
    return cv2.resize(sm, (W, H), interpolation=cv2.INTER_LINEAR)   # proba signal, taille native

def extract_waveforms(img, res, smask, thr=0.4):
    """Pour chaque cellule assignée, suit le tracé dans sa bande -> amplitude(x).
    amplitude = baseline(y de la rangée) - y du tracé  (vers le haut = positif)."""
    H, W = img.shape[:2]; colx = res["colx"]; rowy = res["rowy"]; sp = res["spacing"]; ncols = res["ncols"]
    waves = {}
    for cell in res["cells"]:
        lead = cell["canon"]
        if lead == "?" or lead in waves: continue
        ci = cell["col"]; ry = cell["y"]
        x0 = int(colx[ci])
        if ci+1 < ncols: x1 = int(colx[ci+1])
        elif ncols > 1:  x1 = int(colx[ci] + (colx[ci]-colx[ci-1]))   # derniere colonne : meme largeur (evite la marge blanche)
        else:            x1 = W
        x0 = max(0, x0); x1 = min(W, x1)
        yb0, yb1 = max(0, int(ry-0.45*sp)), min(H, int(ry+0.45*sp))
        band = smask[yb0:yb1, x0:x1]
        amp = np.full(x1-x0, np.nan, np.float32)
        for j in range(band.shape[1]):
            ys = np.where(band[:, j] > thr)[0]
            if len(ys): amp[j] = ry - (yb0 + ys.mean())   # haut = positif
        waves[lead] = amp
    return waves
''')

code(r'''# ── Cellule 4 : choisir une image, ASSIGNER, visualiser la grille ──
# --- A AJUSTER ---
REAL_SUBFOLDER = "augmentation_brightness_120"
REAL_INDEX     = 21        # index dans le dossier ; change pour une autre image
# -----------------
rf = sorted(glob.glob(os.path.join(REAL_DIR, REAL_SUBFOLDER, "*")))
rf = [f for f in rf if f.lower().endswith((".jpg",".jpeg",".png",".webp",".bmp"))]
img = np.array(Image.open(rf[REAL_INDEX]).convert("RGB")); H, W = img.shape[:2]
res = assign(img)
assert res is not None, "pas assez d'ancres pour reconstruire la grille"
half = res["half"]; ncols = res["ncols"]
print(f"{os.path.basename(rf[REAL_INDEX])} -> grille {res['ncols']}x{res['nrows']}, {len(res['cells'])} cellules")

disp = img.copy()
for g in res["cells"]:
    x, y = int(g["cx"]), int(g["y"]); det = g["det"]
    if det is None: col = (0,90,255)                          # bleu = extrapolé (position)
    elif det["lead"] == g["canon"]: col = (0,170,0)           # vert = lecteur d'accord
    else: col = (255,140,0)                                   # orange = désaccord (on garde canon)
    cv2.rectangle(disp, (x-half,y-half), (x+half,y+half), col, 4)
    cv2.putText(disp, g["canon"], (x-half, y-half-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 3)
sp = 1300/W
plt.figure(figsize=(15,11)); plt.imshow(cv2.resize(disp,(int(W*sp),int(H*sp)))); plt.axis("off")
plt.title("Assignation (vert=lecteur OK, orange=désaccord→canon, bleu=position)"); plt.show()
''')

code(r'''# ── Cellule 5 : EXTRAIRE les formes d'onde et tracer les 12 dérivations ──
smask = signal_mask(img)
waves = extract_waveforms(img, res, smask, thr=0.4)
print("dérivations extraites:", list(waves.keys()))

# 12 courbes nommées (dans l'ordre canonique)
order = [l for l in LEADS if l in waves]
n = len(order); ncol = 2; nrow = (n+ncol-1)//ncol
fig, axes = plt.subplots(nrow, ncol, figsize=(15, 1.6*nrow), squeeze=False)
for k, lead in enumerate(order):
    ax = axes[k//ncol][k%ncol]; w = waves[lead]
    ax.plot(w, lw=0.8, color="k"); ax.axhline(0, color="r", lw=0.4, alpha=0.5)
    ax.set_ylabel(lead, rotation=0, ha="right", va="center", fontsize=11, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
for k in range(n, nrow*ncol): axes[k//ncol][k%ncol].axis("off")
plt.suptitle(f"Digitization : 12 dérivations nommées — {os.path.basename(rf[REAL_INDEX])}", fontsize=13)
plt.tight_layout(); plt.show()

# export structuré (optionnel) : dict lead -> array, sauvegardé en NPZ
out_path = os.path.join(PROJECT_ROOT, "stage", f"digitized_{os.path.splitext(os.path.basename(rf[REAL_INDEX]))[0]}.npz")
np.savez_compressed(out_path, **{l: waves[l] for l in order})
print("exporté ->", out_path)
''')

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
                   "language_info": {"name":"python","version":"3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
p = "stage/training_digitization.ipynb"
json.dump(nb, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("[OK] ecrit", p, "-", len(cells), "cellules")
