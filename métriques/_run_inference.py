# Script d'inference pour les 3 modeles sur output_real
# Genere : training_curves.png + resume.txt + 15 visualisations 3-up par modele

import os, sys, json, glob, shutil
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {DEVICE}")

ROOT = r"c:\Users\v\Desktop\ECGPerturb-main"
METRIQUES_DIR = os.path.join(ROOT, "stage", "métriques")
OUTPUT_REAL = os.path.join(ROOT, "data", "output_real")

FOLDERS = ["augmentation_brightness_160", "augmentation_rotation_15", "photos_crumbles"]
SELECT_INDICES = [10, 30, 50, 70, 90]  # Indices alphabetiques

MODELS = {
    "training_npz_copie": {
        "run_dir": os.path.join(ROOT, "data", "training", "runs_npz", "run_20260518_164651"),
        "mode": "fullres",
        "img_size": 1024,
    },
    "training_npz_patches_256": {
        "run_dir": os.path.join(ROOT, "data", "training", "runs_npz_patches256", "run_20260520_170838"),
        "mode": "patches",
        "patch_size": 256,
        "tile_overlap": 32,
    },
    "training_grid_major": {
        "run_dir": os.path.join(ROOT, "data", "training", "runs", "run_20260519_162837"),
        "mode": "fullres",
        "img_size": 1024,
    },
}


def load_model(run_dir):
    ckpt_path = os.path.join(run_dir, "checkpoints", "best_model.pth")
    config_path = os.path.join(run_dir, "config.json")
    cfg = json.load(open(config_path))
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = smp.Unet(
        encoder_name=cfg.get("encoder_name", "resnet34"),
        encoder_weights=None,
        in_channels=cfg.get("in_channels", 3),
        classes=cfg.get("num_classes", 1),
        activation=None,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE).eval()
    return model, cfg, ckpt


def predict_fullres(model, img_pil, img_size):
    W_orig, H_orig = img_pil.size
    img_resized = img_pil.resize((img_size, img_size), Image.BILINEAR)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor)).squeeze().cpu().numpy()
    pred_full = cv2.resize(pred, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
    return pred_full


def predict_patches(model, img_pil, patch_size, overlap):
    W_orig, H_orig = img_pil.size
    img_np = np.array(img_pil, dtype=np.float32) / 255.0
    H, W = img_np.shape[:2]
    stride = patch_size - overlap
    pred_sum = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    ys = list(range(0, max(H - patch_size, 0) + 1, stride))
    xs = list(range(0, max(W - patch_size, 0) + 1, stride))
    if ys[-1] + patch_size < H:
        ys.append(H - patch_size)
    if xs[-1] + patch_size < W:
        xs.append(W - patch_size)
    for y in ys:
        for x in xs:
            tile = img_np[y:y + patch_size, x:x + patch_size]
            t = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
            with torch.no_grad():
                p = torch.sigmoid(model(t)).squeeze().cpu().numpy()
            pred_sum[y:y + patch_size, x:x + patch_size] += p
            count[y:y + patch_size, x:x + patch_size] += 1
    return pred_sum / np.maximum(count, 1)


def make_3up(img_pil, pred_full, save_path, title):
    pred_binary = (pred_full > 0.5).astype(np.uint8) * 255
    img_arr = np.array(img_pil)
    overlay = img_arr.copy().astype(np.float32)
    mask_bool = pred_binary > 127
    overlay[mask_bool] = 0.45 * overlay[mask_bool] + 0.55 * np.array([255, 0, 0], dtype=np.float32)
    overlay = overlay.clip(0, 255).astype(np.uint8)

    pos_pct = 100 * mask_bool.sum() / mask_bool.size

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle(f"{title}\nmax={pred_full.max():.3f} | pixels positifs (seuil 0.5): {pos_pct:.2f}%", fontsize=10)
    axes[0].imshow(img_arr); axes[0].set_title(f"Image originale ({img_arr.shape[1]}x{img_arr.shape[0]})"); axes[0].axis("off")
    axes[1].imshow(pred_binary, cmap="gray", vmin=0, vmax=255); axes[1].set_title("Mask binarise (seuil 0.5)"); axes[1].axis("off")
    axes[2].imshow(overlay); axes[2].set_title("Superposition (rouge = predictions)"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=90, bbox_inches="tight")
    plt.close()


def write_resume(run_dir, ckpt, save_path):
    history = json.load(open(os.path.join(run_dir, "history.json")))
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    val_dices = history.get("val_dice", [])
    val_ious = history.get("val_iou", [])
    val_losses = history.get("val_loss", [])
    best_epoch = val_dices.index(max(val_dices)) + 1 if val_dices else "N/A"
    best_dice = max(val_dices) if val_dices else "N/A"
    best_iou = val_ious[best_epoch - 1] if isinstance(best_epoch, int) and val_ious else "N/A"
    best_loss = val_losses[best_epoch - 1] if isinstance(best_epoch, int) and val_losses else "N/A"

    lines = [
        "=" * 60,
        f"RESUME ENTRAINEMENT - {os.path.basename(run_dir)}",
        "=" * 60,
        "",
        f"Best epoch        : {best_epoch}",
        f"Best Val Dice     : {best_dice:.4f}" if isinstance(best_dice, float) else f"Best Val Dice     : {best_dice}",
        f"Best Val IoU      : {best_iou:.4f}" if isinstance(best_iou, float) else f"Best Val IoU      : {best_iou}",
        f"Val Loss (best ep): {best_loss:.4f}" if isinstance(best_loss, float) else f"Val Loss (best ep): {best_loss}",
        "",
        f"Nombre d'epochs   : {len(val_dices)}",
        f"Checkpoint epoch  : {ckpt.get('epoch', 'N/A')}",
        f"Checkpoint Dice   : {ckpt.get('val_dice', 'N/A')}",
        "",
        "=" * 60,
        "HYPERPARAMETRES",
        "=" * 60,
        f"Encoder           : {cfg.get('encoder_name', 'N/A')}",
        f"Encoder weights   : {cfg.get('encoder_weights', 'N/A')}",
        f"Loss type         : {cfg.get('loss_type', 'N/A')}",
        f"BCE weight        : {cfg.get('bce_weight', 'N/A')}",
        f"Learning rate     : {cfg.get('learning_rate', 'N/A')}",
        f"Weight decay      : {cfg.get('weight_decay', 'N/A')}",
        f"Batch size        : {cfg.get('batch_size', 'N/A')}",
        f"Num epochs        : {cfg.get('num_epochs', 'N/A')}",
        f"Image size        : {cfg.get('img_height', cfg.get('patch_size', 'N/A'))}",
        f"Point radius      : {cfg.get('point_radius', 'N/A')}",
        f"NPZ key / target  : {cfg.get('npz_key', cfg.get('mask_type', 'N/A'))}",
        f"Scheduler patience: {cfg.get('scheduler_patience', 'N/A')}",
        f"Early stop pat.   : {cfg.get('early_stop_patience', 'N/A')}",
        f"Train sources     : {cfg.get('train_sources', 'N/A')}",
        f"Val sources       : {cfg.get('val_sources', 'N/A')}",
        f"Seed              : {cfg.get('seed', 'N/A')}",
        "",
    ]
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def get_target_images(folder):
    full_path = os.path.join(OUTPUT_REAL, folder)
    files = sorted([f for f in os.listdir(full_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))])
    selected = []
    for idx in SELECT_INDICES:
        if idx < len(files):
            selected.append(files[idx])
    return [os.path.join(full_path, f) for f in selected]


def process_model(model_name, model_cfg):
    print(f"\n{'=' * 60}\n[MODEL] {model_name}\n{'=' * 60}")
    run_dir = model_cfg["run_dir"]
    model_dir = os.path.join(METRIQUES_DIR, f"métriques {model_name}")

    # 1. Copy training_curves.png
    src_curves = os.path.join(run_dir, "training_curves.png")
    dst_curves = os.path.join(model_dir, "training_curves.png")
    if os.path.exists(src_curves):
        shutil.copy(src_curves, dst_curves)
        print(f"  [OK] training_curves.png copie")
    else:
        print(f"  [WARN] {src_curves} introuvable")

    # 2. Load model and write resume
    print(f"  [*] Chargement du modele...")
    model, cfg, ckpt = load_model(run_dir)
    print(f"      Epoch checkpoint: {ckpt.get('epoch', 'N/A')} | val_dice: {ckpt.get('val_dice', 'N/A')}")
    resume_path = os.path.join(model_dir, "resume.txt")
    write_resume(run_dir, ckpt, resume_path)
    print(f"  [OK] resume.txt ecrit")

    # 3. Inference on each folder
    out_examples = os.path.join(model_dir, "output_real_examples")
    for folder in FOLDERS:
        print(f"  [*] Inference sur {folder}...")
        images = get_target_images(folder)
        out_folder = os.path.join(out_examples, folder)
        os.makedirs(out_folder, exist_ok=True)
        for img_path in images:
            fname = os.path.basename(img_path)
            stem = os.path.splitext(fname)[0]
            img_pil = Image.open(img_path).convert("RGB")
            if model_cfg["mode"] == "fullres":
                pred = predict_fullres(model, img_pil, model_cfg["img_size"])
            else:  # patches
                pred = predict_patches(model, img_pil, model_cfg["patch_size"], model_cfg["tile_overlap"])
            save_path = os.path.join(out_folder, f"{stem}_prediction.png")
            title = f"[{model_name}] {folder} / {fname}"
            make_3up(img_pil, pred, save_path, title)
        print(f"      {len(images)} images traitees -> {out_folder}")

    # Free memory
    del model
    torch.cuda.empty_cache() if DEVICE == "cuda" else None


if __name__ == "__main__":
    for model_name, model_cfg in MODELS.items():
        process_model(model_name, model_cfg)
    print(f"\n{'=' * 60}\n[DONE] Tous les modeles traites.\n{'=' * 60}")
