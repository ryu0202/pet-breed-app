"""Evaluate saved top-1 predictions; this script does not train or run a model."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def evaluate(frame, classes):
    if not isinstance(classes, list) or not classes or any(not isinstance(c, str) or not c.strip() for c in classes):
        raise ValueError("클래스 목록은 비어 있지 않은 문자열 배열이어야 합니다.")
    if len(set(classes)) != len(classes) or any(c != c.strip() for c in classes):
        raise ValueError("클래스 이름에 중복 또는 앞뒤 공백이 있습니다.")
    required = ["image_id", "y_true", "y_pred"]
    if any(name not in frame.columns for name in required):
        raise ValueError("CSV에는 image_id,y_true,y_pred 열이 필요합니다.")
    if frame.empty:
        raise ValueError("평가할 행이 없습니다.")
    frame = frame.copy()
    for column in required:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column}에 비어 있는 값이 있습니다.")
        frame[column] = frame[column].astype(str).str.strip()
    if frame.image_id.duplicated().any():
        raise ValueError("image_id가 중복됩니다. 이미지당 한 행만 입력하세요.")
    unknown = (set(frame.y_true) | set(frame.y_pred)) - set(classes)
    if unknown:
        raise ValueError(f"클래스 목록에 없는 라벨: {sorted(unknown)}")

    report = classification_report(frame.y_true, frame.y_pred, labels=classes, output_dict=True, zero_division=0)
    per_class = pd.DataFrame([
        {"class": name, **{key: report[name][key] for key in ["precision", "recall", "f1-score", "support"]}}
        for name in classes
    ])
    missing = per_class.loc[per_class.support == 0, "class"].tolist()
    summary = {
        "n_images": len(frame),
        "accuracy": float(accuracy_score(frame.y_true, frame.y_pred)),
        "macro_f1": float(f1_score(frame.y_true, frame.y_pred, labels=classes, average="macro", zero_division=0)),
        "class_order": classes,
        "missing_true_classes": missing,
        "macro_f1_policy": "All configured classes; undefined scores are 0.",
        "confusion_matrix_axes": "rows=true, columns=predicted",
    }
    matrix = confusion_matrix(frame.y_true, frame.y_pred, labels=classes)
    errors = frame.loc[frame.y_true != frame.y_pred].copy()
    return summary, per_class, matrix, errors


def save_results(summary, per_class, matrix, errors, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    per_class.to_csv(output_dir / "per_class.csv", index=False, encoding="utf-8-sig")
    errors.to_csv(output_dir / "errors.csv", index=False, encoding="utf-8-sig")
    classes = summary["class_order"]
    pd.DataFrame(matrix, index=pd.Index(classes, name="true_class"), columns=classes).to_csv(
        output_dir / "confusion_matrix.csv", encoding="utf-8-sig"
    )
    # English labels in the default configuration work without extra fonts.
    if any(any(ord(ch) > 127 for ch in label) for label in classes):
        from matplotlib import font_manager
        available = {font.name for font in font_manager.fontManager.ttflist}
        for name in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
            if name in available:
                plt.rcParams["font.family"] = name
                break
    fig, ax = plt.subplots(figsize=(11, 9))
    chart = ax.imshow(matrix, cmap="Blues", vmin=0)
    fig.colorbar(chart, ax=ax, label="Image count")
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes,
           yticklabels=classes, xlabel="Predicted breed", ylabel="True breed", title="Confusion matrix (counts)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for row in range(len(classes)):
        for col in range(len(classes)):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if matrix[row, col] > matrix.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="정답과 top-1 예측 CSV의 분류 성능을 평가합니다.")
    parser.add_argument("predictions", type=Path, help="image_id,y_true,y_pred CSV")
    parser.add_argument("--classes", type=Path, default=Path(__file__).with_name("classes.json"))
    parser.add_argument("--output", type=Path, required=True, help="결과 저장 폴더")
    parser.add_argument("--split", choices=["validation", "test", "example"], required=True)
    args = parser.parse_args()
    try:
        classes = json.loads(args.classes.read_text(encoding="utf-8-sig"))
        frame = pd.read_csv(args.predictions, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        summary, per_class, matrix, errors = evaluate(frame, classes)
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        parser.error(str(exc))
    summary.update({"split": args.split, "predictions_file": str(args.predictions.resolve())})
    save_results(summary, per_class, matrix, errors, args.output)
    if args.split == "example":
        print("테스트용 가상 데이터 결과입니다. 실제 모델 성능이 아닙니다.")
    print(f"Images: {summary['n_images']} | Accuracy: {summary['accuracy']:.2%} | Macro-F1: {summary['macro_f1']:.4f}")
    if summary["missing_true_classes"]:
        print("주의: 정답 표본이 없는 품종이 있습니다:", ", ".join(summary["missing_true_classes"]))
    print("결과 저장:", args.output.resolve())


if __name__ == "__main__":
    main()
