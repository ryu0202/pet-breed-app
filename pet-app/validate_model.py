"""Run the trained model on the frozen validation split and save evaluation files."""
import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps

from evaluate import evaluate, save_results
from predict import load_model


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(checkpoint_path, root, manifest, output, split="val"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected_hash = checkpoint["metadata"]["manifest_sha256"][split]
    if sha256(manifest) != expected_hash:
        raise ValueError(f"{split} 목록이 학습 때 고정한 목록과 다릅니다.")
    with manifest.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows or any(row["split"] != split for row in rows):
        raise ValueError(f"{split} 분할 목록을 확인해 주세요.")
    model, classes, transform = load_model(checkpoint_path)
    names = [item["name"] for item in classes]
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("image_id가 중복됩니다.")
    # Stop on missing/changed files instead of silently evaluating a subset.
    for row in rows:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root) or row["status"] != "valid":
            raise ValueError("이미지 경로 또는 상태가 잘못됐습니다.")
        label = int(row["label"])
        if not 0 <= label < len(classes) or row["breed"] != names[label]:
            raise ValueError("품종과 라벨 번호가 모델과 일치하지 않습니다.")
        if sha256(path) != row["sha256"]:
            raise ValueError(f"이미지가 변경됐습니다: {path}")
    output.mkdir(parents=True, exist_ok=False)
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(rows), 16):
            batch = rows[start:start + 16]
            tensors = []
            for row in batch:
                with Image.open(root / row["path"]) as image:
                    tensors.append(transform(ImageOps.exif_transpose(image).convert("RGB")))
            scores, indices = model(torch.stack(tensors)).softmax(dim=1).max(dim=1)
            for row, index, score in zip(batch, indices.tolist(), scores.tolist()):
                predictions.append({"image_id": row["image_id"], "path": row["path"],
                                    "y_true": names[int(row["label"])], "y_pred": names[index],
                                    "model_score": score})
            print(f"평가 중: {len(predictions)}/{len(rows)}", flush=True)
    frame = pd.DataFrame(predictions)
    frame.to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
    (output / "classes.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    summary, per_class, matrix, errors = evaluate(frame, names)
    summary.update({"split": "validation" if split == "val" else "test", "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": sha256(checkpoint_path), "best_epoch": checkpoint["best_epoch"],
                    "manifest_sha256": expected_hash, "image_root": str(root)})
    save_results(summary, per_class, matrix, errors, output)
    print(f"Accuracy: {summary['accuracy']:.2%} | Macro-F1: {summary['macro_f1']:.4f} | Errors: {len(errors)}")
    print("저장 위치:", output)


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="검증 데이터 예측 및 성능 평가")
    parser.add_argument("--backup", required=True, type=Path, help="raw와 dataset이 들어 있는 백업 폴더")
    parser.add_argument("--checkpoint", type=Path, default=base / "models/best.pt")
    parser.add_argument("--output", type=Path, default=base / "results" / ("validation_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        run(args.checkpoint.resolve(), (args.backup / "raw").resolve(),
            (args.backup / f"dataset/{args.split}.csv").resolve(), args.output.resolve(), args.split)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"평가 실패: {error}\n")


if __name__ == "__main__":
    main()
