"""
Build a self-contained dataset from all annotated captures, without the clusters below a minimum mass.

- Pools the frames of data/gt_json/{train,val,test} (sessions 2023-12-01, 2025-03-28, 2025-07-13)
- Drops every frame below --min_mass_g, together with its annotation and depth map
- Writes one split, "all", with REAL COPIES of the images and depth maps (no links), so the
  result does not depend on the old repo. The source files are only read, never changed.

Output (default data_min120/, not in git):
    gt_json/all/all.json    COCO annotations (image/annotation ids reindexed)
    images/all/<name>       RGB images
    depth/<name>.npy        depth maps in mm

"all" is not a train/val split yet: train.py needs those, and they are built from this folder separately.
Refuses to run if the output folder already exists.

    ./run.sh data_prep/make_min_mass_dataset.py
"""
import argparse, json, os, shutil
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # data_prep/ -> repo root
DATA = REPO + "/data"
SRC_SPLITS = ["train", "val", "test"]
SPLIT = "all"


def mass_g(file_name):
    return int(float(os.path.basename(file_name).split("_")[0]))


parser = argparse.ArgumentParser(description="Dataset from all captures, without clusters below a minimum mass")
parser.add_argument("--min_mass_g", type=int, default=120)
parser.add_argument("--out", default=REPO + "/data_min120")
args = parser.parse_args()

if os.path.exists(args.out):
    raise SystemExit(f"Output already exists: {args.out}")

# ---- load source jsons, keep one for the COCO header (info/licenses/categories) ----
src = {}
for s in SRC_SPLITS:
    with open(os.path.join(DATA, "gt_json", s, s + ".json")) as f:
        src[s] = json.load(f)
template = src["train"]

# ---- pool frames at or above the minimum mass, remember the source split, reindex ids ----
new_images, new_annos, phys, dropped = [], [], [], Counter()
next_img_id, next_ann_id = 1, 1

for s in SRC_SPLITS:
    ann_by_img = {}
    for a in src[s]["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)
    for im in src[s]["images"]:
        fn = im["file_name"]
        if mass_g(fn) < args.min_mass_g:
            dropped[mass_g(fn)] += 1
            continue
        new_id = next_img_id; next_img_id += 1
        im2 = dict(im)
        im2["id"] = new_id
        im2["file_name"] = os.path.basename(fn)   # basename only: depth lookup needs it
        new_images.append(im2)
        for a in ann_by_img.get(im["id"], []):
            a2 = dict(a)
            a2["id"] = next_ann_id; next_ann_id += 1
            a2["image_id"] = new_id
            new_annos.append(a2)
        phys.append((os.path.basename(fn), s))

# ---- every source file must exist before anything is written ----
sources = [(os.path.join(DATA, "images", s, base), os.path.join(DATA, "depth", os.path.splitext(base)[0] + ".npy"))
           for base, s in phys]
missing = [p for pair in sources for p in pair if not os.path.exists(p)]
if missing:
    raise SystemExit(f"{len(missing)} source files are missing, e.g. {missing[0]}")

# ---- write the new dataset ----
os.makedirs(os.path.join(args.out, "gt_json", SPLIT))
os.makedirs(os.path.join(args.out, "images", SPLIT))
os.makedirs(os.path.join(args.out, "depth"))
for i, (image_path, depth_path) in enumerate(sources):
    shutil.copy2(image_path, os.path.join(args.out, "images", SPLIT))
    shutil.copy2(depth_path, os.path.join(args.out, "depth"))
    if (i + 1) % 200 == 0:
        print(f"  copied {i + 1}/{len(sources)}")

coco = {
    "info": template.get("info", {}),
    "licenses": template.get("licenses", []),
    "categories": template["categories"],
    "images": new_images,
    "annotations": new_annos,
}
with open(os.path.join(args.out, "gt_json", SPLIT, SPLIT + ".json"), "w") as f:
    json.dump(coco, f)

# ---- report ----
masses = Counter(mass_g(i["file_name"]) for i in new_images)
print(f"Kept {len(new_images)} frames, {len(new_annos)} annotations, {len(masses)} masses "
      f"({min(masses)}-{max(masses)} g)")
print(f"Dropped {sum(dropped.values())} frames, {len(dropped)} masses: {dict(sorted(dropped.items()))}")
print("Output:", args.out)
