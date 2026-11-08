"""
Build a small test dataset from the images in data/images/test_home_color (session 2025-06-13):
one 538 g cluster at several distances, against a plain wall. This cluster and session are in no
training split. Images without a depth map are skipped; depth maps without an image are ignored.

There are no annotations, so the json lists images only. evaluate.py can run on it (the mass comes from
the file name), except for --input_modes rgbd_flat, and detections cannot be scored against a GT box.
The source files are only read, never changed.

Output (default data_home_color/, not in git; real copies, no links):
    gt_json/test_home_color/test_home_color.json
    images/test_home_color/<name>.png
    depth/<name>.npy

Refuses to run if the output folder already exists.

    ./run.sh data_prep/make_home_color_test.py
    ./run.sh evaluate.py --run <run> --dataset_path ./data_home_color/ --splits test_home_color
"""
import argparse, json, os, shutil

import cv2

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # data_prep/ -> repo root
SOURCE_IMAGES = REPO + "/data/images/test_home_color"
SOURCE_DEPTH = os.path.join(os.path.dirname(os.path.realpath(REPO + "/data/images")), "test_home_depth")
SPLIT = "test_home_color"

parser = argparse.ArgumentParser(description="Test dataset from the test_home_color images")
parser.add_argument("--out", default=REPO + "/data_home_color")
args = parser.parse_args()

if os.path.exists(args.out):
    raise SystemExit(f"Output already exists: {args.out}")

with open(REPO + "/data/gt_json/train/train.json") as f:
    template = json.load(f)

names = sorted(n for n in os.listdir(SOURCE_IMAGES)
               if os.path.exists(os.path.join(SOURCE_DEPTH, os.path.splitext(n)[0] + ".npy")))

os.makedirs(os.path.join(args.out, "gt_json", SPLIT))
os.makedirs(os.path.join(args.out, "images", SPLIT))
os.makedirs(os.path.join(args.out, "depth"))

images = []
for i, name in enumerate(names, start=1):
    image_path = os.path.join(SOURCE_IMAGES, name)
    height, width = cv2.imread(image_path).shape[:2]
    images.append({"id": i, "file_name": name, "height": height, "width": width})
    shutil.copy2(image_path, os.path.join(args.out, "images", SPLIT))
    shutil.copy2(os.path.join(SOURCE_DEPTH, os.path.splitext(name)[0] + ".npy"), os.path.join(args.out, "depth"))

with open(os.path.join(args.out, "gt_json", SPLIT, SPLIT + ".json"), "w") as f:
    json.dump({"info": template.get("info", {}), "licenses": template.get("licenses", []),
               "categories": template["categories"], "images": images, "annotations": []}, f)

print(f"{len(images)} images -> {args.out}")
