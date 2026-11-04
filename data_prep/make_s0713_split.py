"""
Build an isolated single-session (2025-07-13) train/val split, grouped BY MASS VALUE
(no frame leakage), for the depth-potential pilot.

- Pools all 2025-07-13 frames from gt_json/{train,val,test}/*.json
- VAL = 4 interior, spread mass values (each >=45 frames, >=2 bursts).
- TRAIN = the other 10 masses (INCLUDING the extremes 164 g & 1160 g, so VAL is
  pure interpolation, not extrapolation).
- Writes a self-contained data root `data_s0713/` (COCO jsons + symlinked images +
  symlinked depth) so the existing data/ dir and train.py are untouched.
  Train with:  --dataset_path <abs>/data_s0713/   (split names stay train/val)

WARNING: do not re-run casually. It DELETES the whole data_s0713/ first (shutil.rmtree), including
the cross-session splits test_0328 / test_1201. Everything in there is links, so no raw data is lost,
but you must then re-create those with data_prep/make_s0713_test_session.py (2025-03-28, then 2023-12-01).
"""
import json, os, shutil

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # data_prep/ -> repo root
DATA = REPO + "/data"
OUT = REPO + "/data_s0713"
SESSION = "2025-07-13"
VAL_MASSES = {208, 490, 670, 940}          # interior + spread
SRC_SPLITS = ["train", "val", "test"]

def mass_g(file_name):
    return int(float(os.path.basename(file_name).split("_")[0]))

# ---- load source jsons, keep one for the COCO header (info/licenses/categories) ----
src = {}
for s in SRC_SPLITS:
    with open(os.path.join(DATA, "gt_json", s, s + ".json")) as f:
        src[s] = json.load(f)
template = src["train"]

# ---- pool 2025-07-13 frames, remember physical source split, reindex ids ----
new_images = {"train": [], "val": []}
new_annos = {"train": [], "val": []}
phys = {}  # (split) -> list of (basename, source_split)
next_img_id, next_ann_id = 1, 1

for s in SRC_SPLITS:
    ann_by_img = {}
    for a in src[s]["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)
    for im in src[s]["images"]:
        fn = im["file_name"]
        if SESSION not in fn:
            continue
        target = "val" if mass_g(fn) in VAL_MASSES else "train"
        new_id = next_img_id; next_img_id += 1
        im2 = dict(im)
        im2["id"] = new_id
        im2["file_name"] = os.path.basename(fn)   # basename only: depth lookup needs it
        new_images[target].append(im2)
        for a in ann_by_img.get(im["id"], []):
            a2 = dict(a)
            a2["id"] = next_ann_id; next_ann_id += 1
            a2["image_id"] = new_id
            new_annos[target].append(a2)
        phys.setdefault(target, []).append((os.path.basename(fn), s))

# ---- write fresh data_s0713/ ----
if os.path.exists(OUT):
    shutil.rmtree(OUT)
for split in ("train", "val"):
    os.makedirs(os.path.join(OUT, "gt_json", split), exist_ok=True)
    os.makedirs(os.path.join(OUT, "images", split), exist_ok=True)
    coco = {
        "info": template.get("info", {}),
        "licenses": template.get("licenses", []),
        "categories": template["categories"],
        "images": new_images[split],
        "annotations": new_annos[split],
    }
    with open(os.path.join(OUT, "gt_json", split, split + ".json"), "w") as f:
        json.dump(coco, f)
    # symlink each image from its real physical split folder
    for base, srcsplit in phys[split]:
        real = os.path.join(DATA, "images", srcsplit, base)
        link = os.path.join(OUT, "images", split, base)
        os.symlink(real, link)

# depth folder: one symlink (flat, keyed by basename, split-independent)
os.symlink(os.path.join(DATA, "depth"),
           os.path.join(OUT, "depth"))

# ---- report ----
from collections import Counter
for split in ("train", "val"):
    masses = Counter(mass_g(i["file_name"]) for i in new_images[split])
    print(f"{split}: {len(new_images[split])} frames, {len(masses)} masses {sorted(masses)}")
print("VAL masses:", sorted(VAL_MASSES))
print("Output:", OUT)
print("Train with:  --dataset_path", OUT + "/")
