#!/usr/bin/env python3
"""Apply one packaged Kather100K checkpoint to image tiles or whole-slide images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a packaged human Kather100K classifier on tiles or WSIs. "
            "Rat results require rat-specific validation."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        nargs="+",
        help="One or more slide/tile paths, for example .svs or tiled .tif.",
    )
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--no-auto-mask",
        action="store_true",
        help="Disable automatic tissue masking.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing TIAToolbox output files.",
    )
    args = parser.parse_args()

    inputs = [path.resolve() for path in args.input]
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Input files do not exist: {missing}")

    weight_path = args.models_dir / f"{args.model}.pth"
    if not weight_path.exists():
        raise FileNotFoundError(f"Missing packaged checkpoint: {weight_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rcParam["torch_compile_mode"] = "disable"
    predictor = PatchPredictor(
        model=args.model,
        weights=weight_path,
        batch_size=args.batch_size,
        num_workers=2,
        device=args.device,
        verbose=True,
    )
    output = predictor.run(
        images=inputs,
        patch_mode=False,
        return_probabilities=True,
        auto_get_mask=not args.no_auto_mask,
        save_dir=args.output_dir,
        overwrite=args.overwrite,
        device=args.device,
    )
    manifest = {
        "model": args.model,
        "checkpoint": str(weight_path.resolve()),
        "inputs": [str(path) for path in inputs],
        "output": str(output),
        "class_order": ["BACK", "NORM", "DEB", "TUM", "ADI", "MUC", "MUS", "STR", "LYM"],
        "source_task": "Human colorectal H&E tissue classification at 224x224 pixels and 0.5 MPP.",
        "interpretation_warning": (
            "Direct WSI predictions are transfer outputs, not validated rat tumour, "
            "SATB2/CDX2, lymphocyte or treatment-response measurements."
        ),
    }
    (args.output_dir / "wsi_run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
