"""
Build leave-one-session-out splits from data_min120/ (see make_min_mass_dataset.py).

Each split holds out one whole capture session as "val" and trains on the other sessions, so
the val session is completely unseen. This is separate from the k-fold splits in data_min120/folds/:
this script only writes inside data_min120/loso/ and refuses to run if that folder already exists.

Output (not in git). Images and depth are relative links to the dataset, no copies:
    loso/hold_<session>/gt_json/{train,val}/{train,val}.json
    loso/hold_<session>/images/{train,val}/<name>
    loso/hold_<session>/depth

Train one model per split:
    ./run.sh train.py --dataset_path ./data_min120/loso/hold_lab/ --experiment_name loso_lab_20k ...

    ./run.sh data_prep/make_session_splits.py
"""
import argparse, json, os, re
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # data_prep/ -> repo root
SESSIONS = {"lab": "2025-03-28", "home": "2025-07-13", "2023": "2023-12-01"}
FILE_NAME = re.compile(r"^(\d+)_(\d{4}-\d{2}-\d{2})-")


def cluster_of(file_name):
    """-> (mass_g, session), parsed from the file name."""
    match = FILE_NAME.match(os.path.basename(file_name))
    return int(match.group(1)), match.group(2)


parser = argparse.ArgumentParser(description="Build leave-one-session-out splits")
parser.add_argument("--dataset", default=REPO + "/data_min120")
args = parser.parse_args()

out_root = os.path.join(args.dataset, "loso")
if os.path.exists(out_root):
    raise SystemExit(f"Splits already exist: {out_root}")

with open(os.path.join(args.dataset, "gt_json", "all", "all.json")) as f:
    coco = json.load(f)

unknown = {cluster_of(i["file_name"])[1] for i in coco["images"]} - set(SESSIONS.values())
if unknown:
    raise SystemExit(f"Sessions missing from SESSIONS: {sorted(unknown)}")

ann_by_img = {}
for a in coco["annotations"]:
    ann_by_img.setdefault(a["image_id"], []).append(a)

print(f"{'held out':<10}{'train images':>13}{'train clusters':>16}{'train mass g':>14}"
      f"{'val images':>12}{'val clusters':>14}{'val mass g':>12}{'val outside train range':>25}")
for name, session in SESSIONS.items():
    split_dir = os.path.join(out_root, f"hold_{name}")
    sizes = {}
    for split in ("train", "val"):
        images = [i for i in coco["images"] if (cluster_of(i["file_name"])[1] == session) == (split == "val")]
        os.makedirs(os.path.join(split_dir, "gt_json", split))
        image_dir = os.path.join(split_dir, "images", split)
        os.makedirs(image_dir)
        with open(os.path.join(split_dir, "gt_json", split, split + ".json"), "w") as f:
            json.dump({
                "info": coco.get("info", {}),
                "licenses": coco.get("licenses", []),
                "categories": coco["categories"],
                "images": images,
                "annotations": [a for i in images for a in ann_by_img[i["id"]]],
            }, f)
        for i in images:
            source = os.path.join(args.dataset, "images", "all", i["file_name"])
            os.symlink(os.path.relpath(source, image_dir), os.path.join(image_dir, i["file_name"]))
        sizes[split] = Counter(cluster_of(i["file_name"]) for i in images)
    os.symlink(os.path.relpath(os.path.join(args.dataset, "depth"), split_dir), os.path.join(split_dir, "depth"))

    train_masses = [c[0] for c in sizes["train"]]
    val_masses = [c[0] for c in sizes["val"]]
    outside = sum(n for c, n in sizes["val"].items() if not min(train_masses) <= c[0] <= max(train_masses))
    print(f"{name:<10}{sum(sizes['train'].values()):>13}{len(sizes['train']):>16}"
          f"{f'{min(train_masses)}-{max(train_masses)}':>14}{sum(sizes['val'].values()):>12}{len(sizes['val']):>14}"
          f"{f'{min(val_masses)}-{max(val_masses)}':>12}{outside:>25}")
print("\nOutput:", out_root)
