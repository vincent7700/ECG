# -*- coding: utf-8 -*-
"""Lance l'entrainement du notebook minor (cellules execees, pas de duplication), cape a 8 epochs."""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.chdir(r"C:\Users\v\Desktop\ECGPerturb-main")

# Env GPU AVANT tout import torch — SANS CUDA_LAUNCH_BLOCKING (la cell 1 du notebook le met a 1
# pour debug, ce qui serialise le GPU et rend l'entrainement ~10x trop lent). On le force a 0.
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

nb = json.load(open("stage/training_npz_Gaussienne_grille_minor.ipynb", encoding="utf-8"))
ns = {}
# on SAUTE la cellule 1 (elle remet CUDA_LAUNCH_BLOCKING=1). 2=imports, 3=config, 4=dataset, 5=loss+train()
for i in [2, 3, 4, 5]:
    src = "".join(nb["cells"][i]["source"])
    src = "\n".join("" if l.strip().startswith("%") else l for l in src.split("\n"))  # retire magics
    exec(src, ns)

cfg = ns["cfg"]
cfg.num_epochs = 8                 # premier modele (cape)
cfg.log_images_every_n_epochs = 4  # moins de figures
print(f"[runner] cible={cfg.npz_key} | sortie={cfg.output_dir} | epochs={cfg.num_epochs}", flush=True)
ns["train"](cfg)
print("[runner] termine.", flush=True)
