"""
Build k cross-validation folds from data_min120/ (see make_min_mass_dataset.py).

Every cluster (one mass value within one session) goes to the val split of exactly one fold, and
all images of a cluster stay together, so near-duplicate frames never leak between train and val.

How clusters are assigned, separately for each session:
  1. Sort the session's clusters by mass and cut them into groups of k neighbours in mass.
  2. Give each fold one cluster of every full group, so each fold gets the whole mass range of that session.
     Within a group, the cluster with the most images goes to the fold with the fewest images so far.
  3. Left-over clusters go to the folds with the fewest clusters, then the fewest images.
The assignment has no random element: the same input always gives the same folds.

Output (inside data_min120/, not in git). Images and depth are relative links to the dataset, no copies:
    folds/fold_<i>/gt_json/{train,val}/{train,val}.json
    folds/fold_<i>/images/{train,val}/<name>
    folds/fold_<i>/depth
    folds/folds.csv       one row per cluster: fold, mass_g, session, n_images

Refuses to run if the folds already exist. Train one model per fold:
    ./run.sh train.py --dataset_path ./data_min120/folds/fold_0/ --experiment_name <name>_fold0 ...

    ./run.sh data_prep/make_kfold_splits.py
"""
import argparse, csv, json, os, re
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # data_prep/ -> repo root
SESSIONS = ["2025-07-13", "2025-03-28", "2023-12-01"]   # largest first
FILE_NAME = re.compile(r"^(\d+)_(\d{4}-\d{2}-\d{2})-")


def cluster_of(file_name):
    """-> (mass_g, session), parsed from the file name."""
    match = FILE_NAME.match(os.path.basename(file_name))
    return int(match.group(1)), match.group(2)


def assign_folds(n_images, k):
    """n_images: {cluster: number of images} -> {cluster: fold}."""
    assignment = {}
    fold_images = [0] * k
    fold_clusters = [0] * k

    def place(cluster, fold):
        assignment[cluster] = fold
        fold_images[fold] += n_images[cluster]
        fold_clusters[fold] += 1

    leftovers = []
    for session in SESSIONS:
        clusters = sorted(c for c in n_images if c[1] == session)
        n_full = len(clusters) // k * k
        for start in range(0, n_full, k):
            free = list(range(k))
            for cluster in sorted(clusters[start:start + k], key=lambda c: -n_images[c]):
                fold = min(free, key=lambda f: (fold_images[f], f))
                free.remove(fold)
                place(cluster, fold)
        leftovers += clusters[n_full:]

    for cluster in sorted(leftovers, key=lambda c: -n_images[c]):
        place(cluster, min(range(k), key=lambda f: (fold_clusters[f], fold_images[f], f)))
    return assignment


parser = argparse.ArgumentParser(description="Build k cross-validation folds, grouped by cluster")
parser.add_argument("--dataset", default=REPO + "/data_min120")
parser.add_argument("--k", type=int, default=5)
args = parser.parse_args()

out_root = os.path.join(args.dataset, "folds")
if os.path.exists(out_root):
    raise SystemExit(f"Folds already exist: {out_root}")

with open(os.path.join(args.dataset, "gt_json", "all", "all.json")) as f:
    coco = json.load(f)

unknown = {cluster_of(i["file_name"])[1] for i in coco["images"]} - set(SESSIONS)
if unknown:
    raise SystemExit(f"Sessions missing from SESSIONS: {sorted(unknown)}")

n_images = Counter(cluster_of(i["file_name"]) for i in coco["images"])
assignment = assign_folds(n_images, args.k)
ann_by_img = {}
for a in coco["annotations"]:
    ann_by_img.setdefault(a["image_id"], []).append(a)

# ---- write the folds ----
for fold in range(args.k):
    fold_dir = os.path.join(out_root, f"fold_{fold}")
    for split in ("train", "val"):
        images = [i for i in coco["images"]
                  if (assignment[cluster_of(i["file_name"])] == fold) == (split == "val")]
        os.makedirs(os.path.join(fold_dir, "gt_json", split))
        image_dir = os.path.join(fold_dir, "images", split)
        os.makedirs(image_dir)
        with open(os.path.join(fold_dir, "gt_json", split, split + ".json"), "w") as f:
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
    os.symlink(os.path.relpath(os.path.join(args.dataset, "depth"), fold_dir), os.path.join(fold_dir, "depth"))

with open(os.path.join(out_root, "folds.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["fold", "mass_g", "session", "n_images"])
    for cluster, fold in sorted(assignment.items(), key=lambda item: (item[1], item[0])):
        writer.writerow([fold, cluster[0], cluster[1], n_images[cluster]])

# ---- report the balance of each fold ----
print(f"{len(n_images)} clusters, {sum(n_images.values())} images, {args.k} folds\n")
header = "".join(f"{s:>12}" for s in SESSIONS)
print(f"{'val split':<9} {'clusters':>8} {'images':>7} {'mean g':>7}{header}   (clusters / images per session)")
for fold in range(args.k):
    clusters = [c for c, f in assignment.items() if f == fold]
    images = sum(n_images[c] for c in clusters)
    mean = sum(c[0] * n_images[c] for c in clusters) / images
    per_session = ""
    for s in SESSIONS:
        in_session = [c for c in clusters if c[1] == s]
        per_session += f"{len(in_session):>6} /{sum(n_images[c] for c in in_session):>4}"
    print(f"fold {fold:<4} {len(clusters):>8} {images:>7} {mean:>7.0f}{per_session}")

print("\nVal masses, and val images outside the mass range of that fold's training data:")
for fold in range(args.k):
    clusters = sorted(c for c, f in assignment.items() if f == fold)
    train_masses = [c[0] for c, f in assignment.items() if f != fold]
    outside = sum(n_images[c] for c in clusters if not min(train_masses) <= c[0] <= max(train_masses))
    print(f"fold {fold}: {[c[0] for c in clusters]}   train {min(train_masses)}-{max(train_masses)} g, outside: {outside}")
print("\nOutput:", out_root)
