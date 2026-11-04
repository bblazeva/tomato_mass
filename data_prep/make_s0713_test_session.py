"""
Add a cross-session TEST split to data_s0713/ for evaluating the 2025-07-13-trained
model on a *different* session.

- Pools every frame of the chosen session from gt_json/{train,val,test}/*.json
- Writes data_s0713/gt_json/<SPLIT>/<SPLIT>.json (image/annotation ids reindexed)
- Symlinks the images into data_s0713/images/<SPLIT>/ (depth folder already symlinked)
- Leaves existing splits untouched.

Each session goes to its own labelled split so they coexist:
    2025-03-28 -> split "test_0328",   2023-12-01 -> split "test_1201"

Usage:
    ./run.sh data_prep/make_s0713_test_session.py [SESSION] [SPLIT_NAME]
      SESSION    default 2025-03-28
      SPLIT_NAME default "test_<MMDD>" derived from SESSION
Eval with:
    evaluate.py --run <run> --dataset_path ./data_s0713/ --splits test_1201
"""
import json, os, sys, shutil

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # data_prep/ -> repo root
DATA = REPO + "/data"
OUT = REPO + "/data_s0713"
SESSION = sys.argv[1] if len(sys.argv) > 1 else "2025-03-28"
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "test_" + SESSION[5:7] + SESSION[8:10]
SRC_SPLITS = ["train", "val", "test"]

src = {s: json.load(open(os.path.join(DATA, "gt_json", s, s + ".json"))) for s in SRC_SPLITS}
template = src["train"]

images, annos, phys = [], [], []
next_img_id, next_ann_id = 1, 1
for s in SRC_SPLITS:
    ann_by_img = {}
    for a in src[s]["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)
    for im in src[s]["images"]:
        fn = im["file_name"]
        if SESSION not in fn:
            continue
        new_id = next_img_id; next_img_id += 1
        im2 = dict(im); im2["id"] = new_id; im2["file_name"] = os.path.basename(fn)
        images.append(im2)
        for a in ann_by_img.get(im["id"], []):
            a2 = dict(a); a2["id"] = next_ann_id; next_ann_id += 1; a2["image_id"] = new_id
            annos.append(a2)
        phys.append((os.path.basename(fn), s))

# write labelled split (rebuild fresh)
json_dir = os.path.join(OUT, "gt_json", SPLIT)
img_dir = os.path.join(OUT, "images", SPLIT)
for p in (json_dir, img_dir):
    if os.path.exists(p):
        shutil.rmtree(p)
    os.makedirs(p)

json.dump({
    "info": template.get("info", {}),
    "licenses": template.get("licenses", []),
    "categories": template["categories"],
    "images": images,
    "annotations": annos,
}, open(os.path.join(json_dir, SPLIT + ".json"), "w"))

for base, srcsplit in phys:
    os.symlink(os.path.join(DATA, "images", srcsplit, base),
               os.path.join(img_dir, base))

from collections import Counter
masses = Counter(int(float(i["file_name"].split("_")[0])) for i in images)
print(f"SESSION {SESSION} -> split '{SPLIT}': {len(images)} frames, "
      f"{len(masses)} masses {sorted(masses)}")
print("Eval with: --dataset_path", OUT + "/  --split", SPLIT)
