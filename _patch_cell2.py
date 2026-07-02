import json

NB_PATH = r'c:/Users/v/Desktop/ECGPerturb-main/stage/training_intersection.ipynb'

NEW_CELL_SOURCE = r"""from dataclasses import dataclass, field

@dataclass
class TrainConfig:
    # Detection des points d'intersection 5mm via PNG masks (alignement garanti).

    project_root: str = r"C:\Users\v\Desktop\ECGPerturb-main\data"
    image_dir: str = ""
    mask_dir: str = ""
    output_dir: str = ""

    mask_type: str = "mask_grid_intersections.png"

    encoder_name: str = "resnet34"
    encoder_weights: str = "imagenet"
    in_channels: int = 3
    num_classes: int = 1

    img_height: int = 1024
    img_width:  int = 1024

    batch_size: int = 4
    num_epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay:  float = 1e-5
    num_workers: int = 0
    pin_memory:  bool = True

    loss_type: str = "bce_dice"
    bce_weight: float = 0.5
    scheduler_patience: int = 5
    scheduler_factor:  float = 0.5
    early_stop_patience: int = 15

    train_sources: list = field(default_factory=lambda: ["ECG_031", "ECG_032"])
    val_sources:   list = field(default_factory=lambda: ["ECG_033"])

    device: str = ""
    seed: int = 42
    save_every_n_epochs:       int = 10
    log_images_every_n_epochs: int = 5

    def __post_init__(self):
        if not self.image_dir:
            self.image_dir = os.path.join(self.project_root, "output_augmentation", "images")
        if not self.mask_dir:
            self.mask_dir = os.path.join(self.project_root, "output_augmentation", "masks")
        if not self.output_dir:
            self.output_dir = os.path.join(self.project_root, "training", "runs_intersection")

        # GPU obligatoire — pas de fallback CPU silencieux
        if not self.device:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA non disponible.\n"
                    "  - Verifie 'nvidia-smi' dans un terminal.\n"
                    "  - Si un autre kernel Jupyter tourne (ex: training_npz.ipynb),\n"
                    "    il tient le GPU. Ferme/Stop ce kernel et relance celui-ci.\n"
                    "  - Si besoin Restart Kernel sur ce notebook."
                )
            self.device = "cuda"

        # Diagnostic GPU
        print(f"[OK] GPU detecte : {torch.cuda.get_device_name(0)}")
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        vram_used  = torch.cuda.memory_allocated(0) / 1e9
        vram_resv  = torch.cuda.memory_reserved(0) / 1e9
        print(f"   VRAM total    : {vram_total:.1f} GB")
        print(f"   VRAM allouee  : {vram_used:.2f} GB")
        print(f"   VRAM reservee : {vram_resv:.2f} GB")
        print(f"   PyTorch CUDA  : {torch.version.cuda}")

cfg = TrainConfig()
"""

nb = json.load(open(NB_PATH, encoding='utf-8'))
nb['cells'][2]['source'] = NEW_CELL_SOURCE.splitlines(keepends=True)
nb['cells'][2]['outputs'] = []
nb['cells'][2]['execution_count'] = None
json.dump(nb, open(NB_PATH, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print('Cell 2 patched (CUDA enforced + diagnostic)')
