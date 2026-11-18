import argparse
import os
import random
import shutil
from pathlib import Path

import yaml

TARGET_CLASSES = ["face_visible", "face_mask", "face_covered"]

RULES = [
    ("face_mask", ["incorrect"]),
    ("face_visible", ["no_mask", "nomask", "without", "unmask", "nothing", "bare", "shield"]),
    ("face_covered", ["balaclava", "ski", "scarf", "covering", "covered", "hood", "niqab", "burqa"]),
    ("face_mask", ["mask", "medic", "surgical", "n95"]),
    ("face_visible", ["face", "head"]),
]

DEFAULT_ROBOFLOW = [
    "university-8zqnr/face-hiding-detection",
    "kemaya-wales/face_masks_extended",
    "new-workspace-cpitq/mask_v2-oft71",
]

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def normalize(name):
    return str(name).lower().replace("-", "_").replace(" ", "_")


def map_class(name, overrides):
    if name in overrides:
        return overrides[name]
    n = normalize(name)
    for target, keywords in RULES:
        if any(k in n for k in keywords):
            return target
    return None


def read_names(data_yaml):
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    return dict(enumerate(names))


def find_split_dirs(src):
    found = {}
    for split, alias in SPLIT_ALIASES.items():
        for img_dir, lbl_dir in [
            (src / split / "images", src / split / "labels"),
            (src / "images" / split, src / "labels" / split),
        ]:
            if img_dir.is_dir():
                found.setdefault(alias, []).append((img_dir, lbl_dir))
    return found


def convert_line(line, remap):
    parts = line.split()
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))
    if cls not in remap:
        return None
    vals = [float(v) for v in parts[1:]]
    if len(vals) == 4:
        x, y, w, h = vals
    else:
        xs, ys = vals[0::2], vals[1::2]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        x, y, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None
    return f"{remap[cls]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"


def download_roboflow(slugs, dest, api_key):
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    paths = []
    for slug in slugs:
        parts = slug.strip("/").split("/")
        workspace, project = parts[0], parts[1]
        proj = rf.workspace(workspace).project(project)
        version = int(parts[2]) if len(parts) > 2 else max(int(v.version) for v in proj.versions())
        location = dest / f"{workspace}__{project}__v{version}"
        if not (location / "data.yaml").exists():
            print(f"Downloading {workspace}/{project} v{version}")
            proj.version(version).download("yolov8", location=str(location))
        paths.append(location)
    return paths


def prepare(sources, out, overrides=None, val_ratio=0.15, seed=0):
    overrides = overrides or {}
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    counts = {c: 0 for c in TARGET_CLASSES}
    totals = {"train": 0, "val": 0, "test": 0}

    for idx, src in enumerate(sources):
        src = Path(src)
        data_yaml = src / "data.yaml"
        if not data_yaml.exists():
            print(f"Skipping {src}: no data.yaml")
            continue
        names = read_names(data_yaml)
        remap = {}
        print(f"\n{src.name}")
        for cid, cname in names.items():
            target = map_class(cname, overrides)
            print(f"  {cname!r:35} -> {target}")
            if target:
                remap[cid] = TARGET_CLASSES.index(target)

        splits = find_split_dirs(src)
        has_val = "val" in splits
        for split, dirs in splits.items():
            for img_dir, lbl_dir in dirs:
                for img in img_dir.iterdir():
                    if img.suffix.lower() not in IMAGE_EXT:
                        continue
                    dst_split = split
                    if split == "train" and not has_val and rng.random() < val_ratio:
                        dst_split = "val"
                    lines = []
                    lbl = lbl_dir / (img.stem + ".txt")
                    if lbl.exists():
                        for line in lbl.read_text(encoding="utf-8").splitlines():
                            conv = convert_line(line, remap)
                            if conv:
                                lines.append(conv)
                                counts[TARGET_CLASSES[int(conv.split()[0])]] += 1
                    name = f"d{idx}_{img.stem}"
                    shutil.copy2(img, out / "images" / dst_split / (name + img.suffix.lower()))
                    (out / "labels" / dst_split / (name + ".txt")).write_text("\n".join(lines), encoding="utf-8")
                    totals[dst_split] += 1

    if totals["val"] == 0 and totals["train"] > 1:
        train_imgs = sorted((out / "images" / "train").iterdir())
        for img in rng.sample(train_imgs, max(1, int(len(train_imgs) * val_ratio))):
            lbl = out / "labels" / "train" / (img.stem + ".txt")
            shutil.move(str(img), str(out / "images" / "val" / img.name))
            shutil.move(str(lbl), str(out / "labels" / "val" / lbl.name))
            totals["train"] -= 1
            totals["val"] += 1

    data = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": dict(enumerate(TARGET_CLASSES)),
    }
    if totals["test"]:
        data["test"] = "images/test"
    with open(out / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"\nImages: {totals}")
    print(f"Boxes:  {counts}")
    print(f"Dataset ready: {out / 'data.yaml'}")
    return out / "data.yaml"


def build_dataset(local=None, roboflow=None, out="datasets/face_cover", map_file=None, api_key=None):
    sources = [Path(p) for p in (local or [])]
    slugs = DEFAULT_ROBOFLOW if roboflow is None and not sources else (roboflow or [])
    if slugs:
        key = api_key or os.getenv("ROBOFLOW_API_KEY")
        if not key:
            raise SystemExit(
                "Set ROBOFLOW_API_KEY (free key at https://app.roboflow.com/settings/api) "
                "or pass --local with datasets you already downloaded."
            )
        sources += download_roboflow(slugs, Path("datasets/raw"), key)
    overrides = {}
    if map_file:
        with open(map_file, "r", encoding="utf-8") as f:
            overrides = yaml.safe_load(f) or {}
    return prepare(sources, out, overrides)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", nargs="*", help="YOLO-format dataset folders that contain data.yaml")
    ap.add_argument("--roboflow", nargs="*", help="workspace/project[/version] slugs")
    ap.add_argument("--map", help="YAML file with {source_class_name: target_class_or_null}")
    ap.add_argument("--out", default="datasets/face_cover")
    args = ap.parse_args()
    build_dataset(args.local, args.roboflow, args.out, args.map)


if __name__ == "__main__":
    main()
