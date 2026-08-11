#!/usr/bin/env python3
"""Assemble all verified model artifacts into one shareable ZIP."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

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


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fmt(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("Usage: package_results.py <downloaded-artifacts-dir> <source-dir> <output-dir>")
    artifact_root = Path(sys.argv[1]).resolve()
    source_dir = Path(sys.argv[2]).resolve()
    output_dir = Path(sys.argv[3]).resolve()
    package_name = "AI_IVLP_TIA_Kather100K_10Models_Validated"
    package_root = output_dir / package_name
    zip_path = output_dir / f"{package_name}.zip"

    if package_root.exists():
        shutil.rmtree(package_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (package_root / "models").mkdir(parents=True)
    (package_root / "results").mkdir(parents=True)
    (package_root / "inference").mkdir(parents=True)
    (package_root / "manifests").mkdir(parents=True)

    all_metrics: list[dict[str, object]] = []
    inventory: list[dict[str, object]] = []

    for model_name in MODELS:
        candidates = list(artifact_root.rglob(f"{model_name}/results/metrics.json"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one metrics.json for {model_name}; found {len(candidates)}: {candidates}")
        model_artifact_dir = candidates[0].parents[1]
        metrics = json.loads(candidates[0].read_text(encoding="utf-8"))
        if metrics.get("model_name") != model_name:
            raise RuntimeError(f"Model-name mismatch in {candidates[0]}")
        all_metrics.append(metrics)

        weight_candidates = list(model_artifact_dir.glob(f"models/{model_name}.pth"))
        if len(weight_candidates) != 1:
            raise RuntimeError(f"Expected one checkpoint for {model_name}; found {weight_candidates}")
        source_weight = weight_candidates[0]
        if source_weight.stat().st_size < 1_000_000:
            raise RuntimeError(f"Checkpoint is unexpectedly small: {source_weight}")
        actual_hash = sha256_file(source_weight)
        if actual_hash != metrics.get("weight_sha256"):
            raise RuntimeError(
                f"Checkpoint hash mismatch for {model_name}: {actual_hash} != {metrics.get('weight_sha256')}"
            )
        destination_weight = package_root / "models" / source_weight.name
        shutil.copy2(source_weight, destination_weight)

        destination_results = package_root / "results" / model_name
        shutil.copytree(model_artifact_dir / "results", destination_results)
        inventory.append({
            "model_name": model_name,
            "checkpoint_file": f"models/{source_weight.name}",
            "checkpoint_bytes": destination_weight.stat().st_size,
            "checkpoint_sha256": actual_hash,
            "weight_source_repo": metrics["weight_source_repo"],
            "weight_source_revision": metrics["weight_source_revision"],
            "source_reported_kather100k_f1": metrics["source_reported_kather100k_f1"],
            "rerun_accuracy": metrics["accuracy"],
            "rerun_macro_f1": metrics["macro_f1"],
            "rerun_macro_ovr_auroc": metrics["macro_ovr_auroc"],
            "rerun_tumor_ovr_auroc": metrics["tumor_ovr_auroc"],
            "rerun_tumor_ovr_auprc": metrics["tumor_ovr_auprc"],
        })

    if len({row["model_name"] for row in all_metrics}) != 10:
        raise RuntimeError("Package does not contain ten unique evaluated models.")

    sorted_metrics = sorted(
        all_metrics,
        key=lambda row: (float(row["macro_f1"]), float(row["accuracy"])),
        reverse=True,
    )
    benchmark_fields = [
        "model_name", "n_validation_patches", "accuracy", "balanced_accuracy",
        "macro_f1", "weighted_f1", "macro_precision", "macro_recall",
        "macro_ovr_auroc", "weighted_ovr_auroc", "macro_auprc", "weighted_auprc",
        "tumor_ovr_auroc", "tumor_ovr_auprc", "multiclass_log_loss",
        "multiclass_brier_score", "expected_calibration_error_15_bins",
        "source_reported_kather100k_f1", "runtime_seconds", "weight_bytes",
        "weight_sha256", "weight_source_revision", "dataset_archive_sha256",
        "tiatoolbox_version", "torch_version", "python_version",
    ]
    with (package_root / "results" / "overall_model_benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=benchmark_fields)
        writer.writeheader()
        for row in sorted_metrics:
            writer.writerow({field: row.get(field, "") for field in benchmark_fields})
    (package_root / "results" / "overall_model_benchmark.json").write_text(
        json.dumps(sorted_metrics, indent=2, sort_keys=True), encoding="utf-8"
    )

    inventory_fields = list(inventory[0].keys())
    with (package_root / "manifests" / "model_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=inventory_fields)
        writer.writeheader()
        writer.writerows(inventory)

    for file_name in [
        "apply_models_to_patch_folder.py",
        "apply_model_to_wsi.py",
        "verify_package.py",
        "requirements.txt",
    ]:
        source = source_dir / file_name
        if not source.exists():
            raise RuntimeError(f"Missing package source file: {source}")
        destination = package_root / "inference" / file_name if file_name.endswith(".py") else package_root / file_name
        shutil.copy2(source, destination)

    best = sorted_metrics[0]
    rows_md = "\n".join(
        "| {rank} | `{name}` | {accuracy} | {f1} | {auc} | {tum_auc} | {source_f1} |".format(
            rank=index,
            name=row["model_name"],
            accuracy=fmt(row["accuracy"]),
            f1=fmt(row["macro_f1"]),
            auc=fmt(row["macro_ovr_auroc"]),
            tum_auc=fmt(row["tumor_ovr_auroc"]),
            source_f1=fmt(row["source_reported_kather100k_f1"], 3),
        )
        for index, row in enumerate(sorted_metrics, start=1)
    )
    readme = f"""# AI-IVLP: 10 verified Kather100K pathology checkpoints

## What is in this ZIP

This package contains **10 actual pretrained `.pth` checkpoint files**, independent
rerun results, per-patch predictions, confusion matrices, cryptographic hashes, and
ready-to-run inference scripts.

The checkpoints were trained by the TIA Centre on the **human Kather100K/NCT colorectal
H&E tissue task**. They were downloaded from the official
`TIACentre/TIAToolbox_pretrained_weights` repository through TIAToolbox 2.1.3.

## Independent rerun performed for this package

Each model was rerun on TIAToolbox's official **2,000-patch Kather100K validation
sample**:

- Patch size: 224 x 224 pixels
- Resolution represented by the source task: 0.5 microns per pixel
- Classes: BACK, NORM, DEB, TUM, ADI, MUC, MUS, STR and LYM
- Exact class counts are recorded in each model's `dataset_manifest.json`
- Weight and dataset SHA-256 hashes are included

This is **not** the full CRC-VAL-HE-7K set and is **not rat validation**.

## Overall measured results

| Rank | Model | Accuracy | Macro-F1 | Macro OVR AUROC | Tumour OVR AUROC | Source-reported F1* |
|---:|---|---:|---:|---:|---:|---:|
{rows_md}

Best model on this independent rerun: **`{best['model_name']}`**
(accuracy {fmt(best['accuracy'])}, macro-F1 {fmt(best['macro_f1'])},
macro OVR AUROC {fmt(best['macro_ovr_auroc'])}).

\*The source-reported F1 column is copied from the TIAToolbox model documentation.
It is shown separately and was not generated by this rerun.

## Fast use on Alberto's validation data

Install the environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Apply all models and create a probability-averaged ensemble on a folder of image patches:

```bash
python inference/apply_models_to_patch_folder.py \\
  --input-dir /path/to/rat_HE_patches \\
  --models-dir models \\
  --output-dir rat_predictions \\
  --model all
```

Apply one model directly to whole-slide files:

```bash
python inference/apply_model_to_wsi.py \\
  --input /path/to/slide.svs \\
  --models-dir models \\
  --output-dir wsi_predictions \\
  --model {best['model_name']}
```

## Critical scientific interpretation

These models recognize **human colorectal H&E tissue patterns**. They can be used for
transfer learning, feature extraction, screening and initialization on rat H&E data.
They are not automatically validated for rat lung tissue, treatment response,
SATB2/CDX2 quantification or lymphocyte density.

For final rat validation:

1. Keep all tiles and serial stains from each rat in the same split.
2. Fine-tune/calibrate using rat H&E annotations.
3. Validate on completely unseen rats.
4. Validate SATB2 and CDX2 with stain-specific annotations.
5. Compare standard and hyperthermic chemo-IVLP at the rat level, not the tile level.

## Directory structure

- `models/`: ten actual pretrained `.pth` checkpoints
- `results/overall_model_benchmark.csv`: combined rerun metrics
- `results/<model>/`: predictions, metrics and confusion matrices
- `inference/`: patch-folder and WSI inference scripts
- `manifests/model_inventory.csv`: source revisions, sizes and SHA-256 hashes
- `CHECKSUMS_SHA256.txt`: integrity hashes for all package files

## Reproducibility status

The package builder fails unless all ten weight files exist, exceed a minimum size,
match their recorded hashes, and have complete independent evaluation outputs.
"""
    (package_root / "README_FIRST.md").write_text(readme, encoding="utf-8")

    citations = """# Sources, licenses and attribution

## Models and software

- TIAToolbox: BSD-3-Clause software developed by the TIA Centre, University of Warwick.
- Model repository: `TIACentre/TIAToolbox_pretrained_weights`.
- Model task: nine-class human colorectal H&E tissue classification on Kather100K/NCT.

## Dataset

- NCT-CRC-HE-100K / Kather100K: human colorectal H&E image patches.
- This package reruns the ten models on TIAToolbox's official 2,000-patch validation sample.
- Dataset files are not redistributed in this ZIP; only predictions, metrics and archive hashes are included.

## Key references

1. Pocock J, et al. TIAToolbox as an end-to-end library for advanced tissue image analytics. Communications Medicine. 2022.
2. Kather JN, et al. Predicting survival from colorectal cancer histology slides using deep learning. PLOS Medicine. 2019.
3. Kather JN, et al. Multi-class texture analysis in colorectal cancer histology. Scientific Reports. 2016.

Users must review and comply with the original software, model and dataset licenses
before redistribution or clinical/research deployment.
"""
    (package_root / "LICENSES_AND_CITATIONS.md").write_text(citations, encoding="utf-8")

    build_manifest = {
        "package_name": package_name,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_count": 10,
        "models": MODELS,
        "validation_dataset": "TIAToolbox Kather100K official 2,000-patch validation sample",
        "github_repository": os.environ.get("GITHUB_REPOSITORY"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_sha": os.environ.get("GITHUB_SHA"),
        "scientific_status": "Public human-CRC model validation package; not rat or clinical validation.",
    }
    (package_root / "manifests" / "build_manifest.json").write_text(
        json.dumps(build_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    checksum_lines: list[str] = []
    for path in sorted(package_root.rglob("*")):
        if path.is_file() and path.name != "CHECKSUMS_SHA256.txt":
            checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(package_root).as_posix()}")
    (package_root / "CHECKSUMS_SHA256.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
        allowZip64=True,
    ) as archive:
        for path in sorted(package_root.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=(Path(package_name) / path.relative_to(package_root)).as_posix())

    with zipfile.ZipFile(zip_path, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Final ZIP integrity failure: {bad_member}")
        checkpoints = [name for name in archive.namelist() if name.endswith(".pth")]
        if len(checkpoints) != 10:
            raise RuntimeError(f"Final ZIP must contain 10 checkpoints; found {len(checkpoints)}")

    zip_hash = sha256_file(zip_path)
    (output_dir / f"{package_name}.zip.sha256.txt").write_text(
        f"{zip_hash}  {zip_path.name}\n", encoding="utf-8"
    )
    summary = {
        "zip_path": str(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "zip_sha256": zip_hash,
        "model_count": len(MODELS),
        "best_rerun_model": best["model_name"],
        "best_rerun_macro_f1": best["macro_f1"],
        "best_rerun_accuracy": best["accuracy"],
    }
    (output_dir / "FINAL_PACKAGE_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
