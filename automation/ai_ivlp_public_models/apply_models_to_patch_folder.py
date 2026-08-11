#!/usr/bin/env python3
"""Apply one or all packaged Kather100K models to a folder of image patches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tiatoolbox import rcParam
from tiatoolbox.models.engine.patch_predictor import PatchPredictor

MODELS = [
    "densenet121-kather100k",
    "densenet161-kather100k",
    "densenet169-kather100k",
    "densenet201-kather100k",
    "resnet18-kather100k",
    "resnet34-kather100k",
    "resnet50-kather100k",
    "resnet101-kather100k",
    "resnext50_32x4d-kather100k",
    "resnext101_32x8d-kather100k",
]
CLASS_NAMES = ["BACK", "NORM", "DEB", "TUM", "ADI", "MUC", "MUS", "STR", "LYM"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply packaged human Kather100K classifiers to pre-extracted image patches. "
            "Rat outputs require rat-specific validation."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", *MODELS],
        help="'all' runs every checkpoint and creates a probability-averaged ensemble.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".png", ".jpg", ".jpeg", ".tif", ".tiff"],
    )
    parser.add_argument(
        "--strict-224",
        action="store_true",
        help="Fail if any patch is not exactly 224x224 pixels.",
    )
    return parser.parse_args()


def find_images(root: Path, extensions: list[str]) -> list[Path]:
    normalized = {extension.lower() for extension in extensions}
    images = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in normalized
    )
    if not images:
        raise RuntimeError(f"No image patches found under {root}")
    return images


def validate_patch_sizes(paths: list[Path]) -> None:
    invalid: list[tuple[str, tuple[int, int]]] = []
    for path in paths:
        with Image.open(path) as image:
            if image.size != (224, 224):
                invalid.append((str(path), image.size))
                if len(invalid) >= 20:
                    break
    if invalid:
        raise RuntimeError(
            "Patch-mode input should be 224x224 pixels for these checkpoints. "
            f"Examples of invalid images: {invalid}"
        )


def run_model(
    model_name: str,
    paths: list[Path],
    models_dir: Path,
    output_dir: Path,
    input_root: Path,
    batch_size: int,
    device: str,
) -> np.ndarray:
    weight_path = models_dir / f"{model_name}.pth"
    if not weight_path.exists():
        raise FileNotFoundError(f"Missing packaged checkpoint: {weight_path}")

    predictor = PatchPredictor(
        model=model_name,
        weights=weight_path,
        batch_size=batch_size,
        num_workers=2,
        device=device,
        verbose=False,
    )
    output = predictor.run(
        [str(path) for path in paths],
        patch_mode=True,
        return_probabilities=True,
        device=device,
    )
    probabilities = np.asarray(output["probabilities"], dtype=np.float64)
    predictions = np.asarray(output["predictions"], dtype=np.int64)
    if probabilities.shape != (len(paths), len(CLASS_NAMES)):
        raise RuntimeError(f"Unexpected output shape for {model_name}: {probabilities.shape}")

    rows: list[dict[str, object]] = []
    for path, predicted_id, probs in zip(paths, predictions.tolist(), probabilities.tolist()):
        row: dict[str, object] = {
            "relative_path": path.relative_to(input_root).as_posix(),
            "predicted_class_id": int(predicted_id),
            "predicted_class_name": CLASS_NAMES[int(predicted_id)],
            "confidence": float(max(probs)),
        }
        for class_id, probability in enumerate(probs):
            row[f"prob_{CLASS_NAMES[class_id]}"] = float(probability)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / f"{model_name}_predictions.csv", index=False)
    return probabilities


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = find_images(args.input_dir, args.extensions)
    if args.strict_224:
        validate_patch_sizes(image_paths)

    rcParam["torch_compile_mode"] = "disable"
    selected_models = MODELS if args.model == "all" else [args.model]
    probability_stack: list[np.ndarray] = []
    for model_name in selected_models:
        print(f"Running {model_name} on {len(image_paths)} patches...")
        probability_stack.append(
            run_model(
                model_name=model_name,
                paths=image_paths,
                models_dir=args.models_dir,
                output_dir=args.output_dir,
                input_root=args.input_dir,
                batch_size=args.batch_size,
                device=args.device,
            )
        )

    if len(probability_stack) > 1:
        ensemble_probabilities = np.mean(np.stack(probability_stack, axis=0), axis=0)
        ensemble_predictions = ensemble_probabilities.argmax(axis=1)
        rows: list[dict[str, object]] = []
        for path, predicted_id, probs in zip(
            image_paths, ensemble_predictions.tolist(), ensemble_probabilities.tolist()
        ):
            row: dict[str, object] = {
                "relative_path": path.relative_to(args.input_dir).as_posix(),
                "predicted_class_id": int(predicted_id),
                "predicted_class_name": CLASS_NAMES[int(predicted_id)],
                "confidence": float(max(probs)),
                "ensemble_model_count": len(selected_models),
            }
            for class_id, probability in enumerate(probs):
                row[f"prob_{CLASS_NAMES[class_id]}"] = float(probability)
            rows.append(row)
        pd.DataFrame(rows).to_csv(
            args.output_dir / "ensemble_mean_probability_predictions.csv", index=False
        )

    run_manifest = {
        "input_dir": str(args.input_dir.resolve()),
        "image_count": len(image_paths),
        "models": selected_models,
        "classes": CLASS_NAMES,
        "interpretation_warning": (
            "These are human colorectal H&E tissue classifiers. Rat data require "
            "rat-specific calibration and held-out animal validation."
        ),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2), encoding="utf-8"
    )
    print(f"Saved predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
