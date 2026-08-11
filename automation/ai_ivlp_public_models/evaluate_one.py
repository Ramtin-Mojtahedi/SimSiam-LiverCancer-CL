#!/usr/bin/env python3
"""Download and independently evaluate one official TIAToolbox Kather100K model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from huggingface_hub import HfApi
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from tiatoolbox import __version__ as tiatoolbox_version
from tiatoolbox import rcParam
from tiatoolbox.models.architecture import fetch_pretrained_weights
from tiatoolbox.models.engine.patch_predictor import PatchPredictor

LABEL_DICT = {
    "BACK": 0,
    "NORM": 1,
    "DEB": 2,
    "TUM": 3,
    "ADI": 4,
    "MUC": 5,
    "MUS": 6,
    "STR": 7,
    "LYM": 8,
}
ID_TO_LABEL = {value: key for key, value in LABEL_DICT.items()}
CLASS_NAMES = [ID_TO_LABEL[index] for index in range(len(ID_TO_LABEL))]
EXPECTED_COUNTS = {
    "BACK": 211,
    "NORM": 176,
    "DEB": 230,
    "TUM": 286,
    "ADI": 208,
    "MUC": 178,
    "MUS": 270,
    "STR": 209,
    "LYM": 232,
}
SOURCE_REPORTED_F1 = {
    "densenet121-kather100k": 0.993,
    "densenet161-kather100k": 0.992,
    "densenet169-kather100k": 0.992,
    "densenet201-kather100k": 0.991,
    "resnet18-kather100k": 0.990,
    "resnet34-kather100k": 0.991,
    "resnet50-kather100k": 0.989,
    "resnet101-kather100k": 0.989,
    "resnext50_32x4d-kather100k": 0.992,
    "resnext101_32x8d-kather100k": 0.991,
}
DATASET_URL = "https://tiatoolbox.dcs.warwick.ac.uk/datasets/kather100k-validation-sample.zip"
HF_REPO_ID = "TIACentre/TIAToolbox_pretrained_weights"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_with_retries(url: str, destination: Path, attempts: int = 5) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "AI-IVLP-reproducibility-package/1.0"},
            )
            partial = destination.with_suffix(destination.suffix + ".part")
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
            if partial.stat().st_size <= 0:
                raise RuntimeError("Downloaded file is empty.")
            partial.replace(destination)
            return
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"Unable to download {url}: {last_error}")


def locate_dataset_root(search_root: Path) -> Path:
    candidates = [
        path.parent
        for path in search_root.rglob("BACK")
        if path.is_dir() and all((path.parent / name).is_dir() for name in LABEL_DICT)
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one validation dataset root; found {candidates}")
    return candidates[0]


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = (prediction == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == 1.0:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def save_confusion_matrix(cm: np.ndarray, output_path: Path, normalized: bool) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="True label",
        xlabel="Predicted label",
        title="Normalized confusion matrix" if normalized else "Confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = float(np.nanmax(cm)) / 2.0 if cm.size else 0.0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            value = cm[row, col]
            label = f"{value:.2f}" if normalized else str(int(value))
            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=7,
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(SOURCE_REPORTED_F1))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=Path("_work"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    started = time.time()
    model_name: str = args.model
    model_output = args.output_root / model_name
    models_dir = model_output / "models"
    results_dir = model_output / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    work_dir = args.work_root / model_name
    work_dir.mkdir(parents=True, exist_ok=True)
    dataset_zip = work_dir / "kather100k-validation-sample.zip"
    extracted_dir = work_dir / "extracted"

    download_with_retries(DATASET_URL, dataset_zip)
    dataset_zip_sha256 = sha256_file(dataset_zip)
    if not extracted_dir.exists():
        extracted_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dataset_zip, "r") as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise RuntimeError(f"Corrupt dataset ZIP member: {bad_member}")
            archive.extractall(extracted_dir)

    dataset_root = locate_dataset_root(extracted_dir)
    patch_paths: list[Path] = []
    y_true: list[int] = []
    observed_counts: dict[str, int] = {}
    for class_name, class_id in LABEL_DICT.items():
        class_paths = sorted(dataset_root.joinpath(class_name).glob("*.tif"))
        observed_counts[class_name] = len(class_paths)
        patch_paths.extend(class_paths)
        y_true.extend([class_id] * len(class_paths))

    if observed_counts != EXPECTED_COUNTS:
        raise RuntimeError(
            "Validation sample counts do not match the official 2,000-patch sample. "
            f"Observed={observed_counts}; expected={EXPECTED_COUNTS}"
        )
    if len(patch_paths) != 2000:
        raise RuntimeError(f"Expected 2,000 validation patches, found {len(patch_paths)}")

    rcParam["torch_compile_mode"] = "disable"
    weight_path = Path(
        fetch_pretrained_weights(model_name, save_path=models_dir, overwrite=False)
    )
    if not weight_path.exists() or weight_path.stat().st_size < 1_000_000:
        raise RuntimeError(f"Checkpoint missing or unexpectedly small: {weight_path}")
    weight_sha256 = sha256_file(weight_path)

    hub_info = HfApi().model_info(HF_REPO_ID)
    hub_revision = hub_info.sha

    predictor = PatchPredictor(
        model=model_name,
        weights=weight_path,
        batch_size=args.batch_size,
        num_workers=2,
        device="cpu",
        verbose=False,
    )
    output = predictor.run(
        [str(path) for path in patch_paths],
        patch_mode=True,
        return_probabilities=True,
        device="cpu",
    )
    if not isinstance(output, dict):
        raise RuntimeError(f"PatchPredictor returned unexpected type: {type(output)!r}")

    probabilities = np.asarray(output["probabilities"], dtype=np.float64)
    predictions = np.asarray(output["predictions"], dtype=np.int64)
    y_array = np.asarray(y_true, dtype=np.int64)
    if probabilities.shape != (2000, 9):
        raise RuntimeError(f"Expected probability shape (2000, 9), got {probabilities.shape}")
    if predictions.shape != (2000,):
        raise RuntimeError(f"Expected prediction shape (2000,), got {predictions.shape}")
    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError("Model probabilities contain non-finite values.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-3):
        raise RuntimeError("Probability rows do not sum to approximately one.")

    y_binary = label_binarize(y_array, classes=np.arange(9))
    report = classification_report(
        y_array,
        predictions,
        labels=np.arange(9),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    raw_cm = confusion_matrix(y_array, predictions, labels=np.arange(9))
    norm_cm = confusion_matrix(y_array, predictions, labels=np.arange(9), normalize="true")

    multiclass_auroc = roc_auc_score(
        y_array,
        probabilities,
        labels=np.arange(9),
        average="macro",
        multi_class="ovr",
    )
    weighted_auroc = roc_auc_score(
        y_array,
        probabilities,
        labels=np.arange(9),
        average="weighted",
        multi_class="ovr",
    )
    macro_auprc = average_precision_score(y_binary, probabilities, average="macro")
    weighted_auprc = average_precision_score(y_binary, probabilities, average="weighted")
    tumor_binary = (y_array == LABEL_DICT["TUM"]).astype(int)
    tumor_auroc = roc_auc_score(tumor_binary, probabilities[:, LABEL_DICT["TUM"]])
    tumor_auprc = average_precision_score(tumor_binary, probabilities[:, LABEL_DICT["TUM"]])
    multiclass_brier = float(np.mean(np.sum((probabilities - y_binary) ** 2, axis=1)))

    metrics = {
        "model_name": model_name,
        "validation_dataset": "TIAToolbox Kather100K official 2,000-patch validation sample",
        "n_validation_patches": int(len(y_array)),
        "n_classes": 9,
        "accuracy": float(accuracy_score(y_array, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_array, predictions)),
        "macro_f1": float(f1_score(y_array, predictions, average="macro")),
        "weighted_f1": float(f1_score(y_array, predictions, average="weighted")),
        "macro_precision": float(precision_score(y_array, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_array, predictions, average="macro", zero_division=0)),
        "macro_ovr_auroc": float(multiclass_auroc),
        "weighted_ovr_auroc": float(weighted_auroc),
        "macro_auprc": float(macro_auprc),
        "weighted_auprc": float(weighted_auprc),
        "tumor_ovr_auroc": float(tumor_auroc),
        "tumor_ovr_auprc": float(tumor_auprc),
        "multiclass_log_loss": float(log_loss(y_array, probabilities, labels=np.arange(9))),
        "multiclass_brier_score": multiclass_brier,
        "expected_calibration_error_15_bins": expected_calibration_error(y_array, probabilities, bins=15),
        "source_reported_kather100k_f1": SOURCE_REPORTED_F1[model_name],
        "source_reported_metric_note": "Published/model-zoo figure supplied by TIAToolbox; not generated by this rerun.",
        "rerun_metric_note": "Metrics above were independently recomputed on TIAToolbox's official 2,000-patch Kather100K validation sample.",
        "weight_filename": weight_path.name,
        "weight_bytes": int(weight_path.stat().st_size),
        "weight_sha256": weight_sha256,
        "weight_source_repo": HF_REPO_ID,
        "weight_source_revision": hub_revision,
        "dataset_archive_url": DATASET_URL,
        "dataset_archive_bytes": int(dataset_zip.stat().st_size),
        "dataset_archive_sha256": dataset_zip_sha256,
        "tiatoolbox_version": tiatoolbox_version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "runtime_seconds": float(time.time() - started),
    }

    with (results_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    pd.DataFrame([metrics]).to_csv(results_dir / "metrics.csv", index=False)
    pd.DataFrame(raw_cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(results_dir / "confusion_matrix_counts.csv")
    pd.DataFrame(norm_cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(results_dir / "confusion_matrix_normalized.csv")
    save_confusion_matrix(raw_cm, results_dir / "confusion_matrix_counts.png", normalized=False)
    save_confusion_matrix(norm_cm, results_dir / "confusion_matrix_normalized.png", normalized=True)

    per_class_rows: list[dict[str, object]] = []
    for class_name in CLASS_NAMES:
        class_metrics = report[class_name]
        per_class_rows.append({
            "class_id": LABEL_DICT[class_name],
            "class_name": class_name,
            "precision": float(class_metrics["precision"]),
            "recall": float(class_metrics["recall"]),
            "f1": float(class_metrics["f1-score"]),
            "support": int(class_metrics["support"]),
        })
    pd.DataFrame(per_class_rows).to_csv(results_dir / "per_class_metrics.csv", index=False)

    prediction_rows: list[dict[str, object]] = []
    for path, true_id, pred_id, probs in zip(
        patch_paths, y_array.tolist(), predictions.tolist(), probabilities.tolist()
    ):
        row: dict[str, object] = {
            "file_name": path.name,
            "relative_path": str(path.relative_to(dataset_root)),
            "true_class_id": true_id,
            "true_class_name": ID_TO_LABEL[true_id],
            "predicted_class_id": pred_id,
            "predicted_class_name": ID_TO_LABEL[pred_id],
            "confidence": float(max(probs)),
            "correct": bool(true_id == pred_id),
        }
        for class_id, probability in enumerate(probs):
            row[f"prob_{ID_TO_LABEL[class_id]}"] = float(probability)
        prediction_rows.append(row)
    pd.DataFrame(prediction_rows).to_csv(results_dir / "predictions.csv", index=False)

    dataset_manifest = {
        "dataset_name": "TIAToolbox Kather100K official validation sample",
        "n_patches": 2000,
        "patch_size_pixels": [224, 224],
        "classes": LABEL_DICT,
        "class_counts": observed_counts,
        "archive_url": DATASET_URL,
        "archive_bytes": dataset_zip.stat().st_size,
        "archive_sha256": dataset_zip_sha256,
    }
    with (results_dir / "dataset_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_manifest, handle, indent=2, sort_keys=True)

    with (model_output / "CHECKSUMS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerow({
            "relative_path": str(weight_path.relative_to(model_output)),
            "bytes": weight_path.stat().st_size,
            "sha256": weight_sha256,
        })

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
