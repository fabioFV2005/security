import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

from prepare_dataset import build_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="Existing data.yaml. If omitted, the dataset is downloaded and prepared")
    ap.add_argument("--local", nargs="*", help="Extra local YOLO datasets to merge")
    ap.add_argument("--roboflow", nargs="*", help="Roboflow slugs to download instead of the defaults")
    ap.add_argument("--model", default="yolo26s.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    data = args.data
    if not data:
        data = str(build_dataset(args.local, args.roboflow))

    device = 0 if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, training on CPU will be very slow. See README.")

    runs = Path("runs").resolve()
    model = YOLO(args.model)
    model.train(
        data=data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch if device != "cpu" else 8,
        device=device,
        workers=args.workers,
        project=str(runs),
        name="face_cover",
        exist_ok=True,
        patience=25,
        cos_lr=True,
        close_mosaic=10,
        hsv_v=0.5,
        degrees=5,
        fliplr=0.5,
        mixup=0.1,
    )

    best = runs / "face_cover" / "weights" / "best.pt"
    target = Path("models/face_cover.pt")
    target.parent.mkdir(exist_ok=True)
    shutil.copy2(best, target)
    print(f"\nModel saved to {target}. Run: python main.py")

    metrics = YOLO(str(target)).val(data=data, imgsz=args.imgsz, device=device)
    print(f"mAP50: {metrics.box.map50:.3f}  mAP50-95: {metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
