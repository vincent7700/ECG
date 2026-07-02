"""Identifie les NPZ cassés et supprime leurs artefacts P2 (NPZ + JSON + masks/)
pour que `python DataAugmentation/pipeline.py --resume --augmentation-workers 1`
régénère uniquement ceux-là."""

import os, sys, glob, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.npz_schema import load_unified_npz

ROOT = r'c:/Users/v/Desktop/ECGPerturb-main/data/output_augmentation'

print('Scan en cours...')
npz_files = sorted(glob.glob(os.path.join(ROOT, 'npz', '*.npz')))
print(f'{len(npz_files)} NPZ a verifier')

broken = []
for i, p in enumerate(npz_files):
    try:
        load_unified_npz(p)
    except Exception:
        broken.append(p)
    if (i + 1) % 500 == 0:
        print(f'  {i+1}/{len(npz_files)}, casses: {len(broken)}')

print(f'\nTotal casses : {len(broken)}')
if not broken:
    print('Rien a faire.')
    sys.exit(0)

confirm = input(f'\nSupprimer NPZ + JSON + masks/ pour ces {len(broken)} fichiers ? [y/N] ')
if confirm.lower() != 'y':
    print('Annule.')
    sys.exit(0)

def _try_remove(path):
    try:
        os.remove(path)
        return True
    except PermissionError as e:
        print(f'  [LOCKED] {os.path.basename(path)} : {e}')
        return False
    except FileNotFoundError:
        return True


def _try_rmtree(path):
    try:
        shutil.rmtree(path)
        return True
    except PermissionError as e:
        print(f'  [LOCKED dir] {os.path.basename(path)} : {e}')
        return False


removed = 0
locked = []
for npz_path in broken:
    stem = os.path.splitext(os.path.basename(npz_path))[0]

    if not _try_remove(npz_path):
        locked.append(stem)
        continue

    removed += 1
    js_path = os.path.join(ROOT, 'json', f'{stem}_metadata.json')
    _try_remove(js_path)

    for ext in ('.webp', '.png', '.jpg', '.jpeg'):
        img_path = os.path.join(ROOT, 'images', stem + ext)
        if os.path.exists(img_path):
            _try_remove(img_path)
            break

    masks_dir = os.path.join(ROOT, 'masks', stem)
    if os.path.isdir(masks_dir):
        _try_rmtree(masks_dir)

print(f'\n{removed} NPZ supprimes.')
if locked:
    print(f'\n[!] {len(locked)} fichiers verrouilles par un autre process (kernel Jupyter ouvert ?) :')
    for s in locked[:10]:
        print(f'  {s}')
    print('\n  -> Ferme/Restart les kernels Jupyter, puis relance ce script.')
else:
    print('\nRelance maintenant :')
    print('  python DataAugmentation/pipeline.py --resume --augmentation-workers 1')
    print('(workers=1 evite la race condition qui a cause la corruption)')
