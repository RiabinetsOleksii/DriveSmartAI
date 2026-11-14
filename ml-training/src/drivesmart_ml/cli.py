"""Command-line entry points for GTSRB preparation."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .audit import audit_image_tree, class_counts
from .config import DataConfig, ExperimentConfig
from .data import load_gtsrb, make_gtsrb_loaders
from .mapping import validate_class_mapping
from .split import records_from_training_tree, split_records, write_split_manifest
from .mobilenet_training import train_mobilenet
from .training import train


def download_dataset(root: Path) -> None:
    load_gtsrb(root, "train", download=True)
    load_gtsrb(root, "test", download=True)
    print(f"GTSRB downloaded to {root}")


def audit_dataset(root: Path, output: Path) -> None:
    audit = audit_image_tree(root)
    output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output / "image_audit.csv", index=False)
    counts = class_counts(audit)
    pd.DataFrame({"class_id": list(range(43)), "count": [counts.get(index, 0) for index in range(43)]}).to_csv(
        output / "class_distribution.csv", index=False
    )
    plt.figure(figsize=(14, 5))
    plt.bar(range(43), [counts.get(index, 0) for index in range(43)])
    plt.xlabel("GTSRB class ID")
    plt.ylabel("Image count")
    plt.tight_layout()
    plt.savefig(output / "class_distribution.png", dpi=150)
    plt.close()
    print(json.dumps({"files": len(audit), "corrupted": int(audit["error"].notna().sum()), "classes": len(counts)}))


def smoke_check(root: Path, split_manifest: Path, image_size: int, batch_size: int, num_workers: int) -> None:
    config = DataConfig(root=root, image_size=image_size, batch_size=batch_size, num_workers=num_workers)
    loaders = make_gtsrb_loaders(config, split_manifest)
    batch, labels = next(iter(loaders["train"]))
    if tuple(batch.shape[1:]) != (3, image_size, image_size):
        raise RuntimeError(f"unexpected tensor shape: {tuple(batch.shape)}")
    if not 0 <= int(labels.min()) or int(labels.max()) > 42:
        raise RuntimeError("labels are outside 0..42")
    if not batch.isfinite().all():
        raise RuntimeError("batch contains NaN or Inf")
    print(
        f"Smoke check passed: train={len(loaders['train'].dataset)}, "
        f"validation={len(loaders['validation'].dataset)}, test={len(loaders['test'].dataset)}, "
        f"batch={tuple(batch.shape)}, labels={int(labels.min())}..{int(labels.max())}"
    )


def create_split(root: Path, output: Path, validation_fraction: float, seed: int) -> None:
    records = records_from_training_tree(root)
    train, validation = split_records(records, validation_fraction=validation_fraction, seed=seed)
    write_split_manifest(train, validation, output, seed=seed, validation_fraction=validation_fraction)
    print(f"Split manifest written to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download")
    download.add_argument("--root", type=Path, default=Path("..") / "datasets" / "gtsrb")
    audit = subparsers.add_parser("audit")
    audit.add_argument("--root", type=Path, default=Path("..") / "datasets" / "gtsrb")
    audit.add_argument("--output", type=Path, default=Path("outputs") / "audit")
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--root", type=Path, default=Path("..") / "datasets" / "gtsrb")
    smoke.add_argument("--image-size", type=int, default=64)
    smoke.add_argument("--batch-size", type=int, default=8)
    smoke.add_argument("--num-workers", type=int, default=0)
    smoke.add_argument("--split-manifest", type=Path, default=Path("outputs") / "split.csv")
    split = subparsers.add_parser("split")
    split.add_argument("--root", type=Path, default=Path("..") / "datasets" / "gtsrb")
    split.add_argument("--output", type=Path, default=Path("outputs") / "split.csv")
    split.add_argument("--validation-fraction", type=float, default=0.2)
    split.add_argument("--seed", type=int, default=42)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--root", type=Path, default=Path("..") / "datasets" / "gtsrb")
    train_parser.add_argument("--split-manifest", type=Path, default=Path("outputs") / "split.csv")
    train_parser.add_argument("--output", type=Path, default=Path("outputs") / "custom-cnn")
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--epochs", type=int, default=15)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--patience", type=int, default=5)
    mobilenet_parser = subparsers.add_parser("train-mobilenet")
    mobilenet_parser.add_argument("--root", type=Path, default=Path("..") / "datasets" / "gtsrb")
    mobilenet_parser.add_argument("--split-manifest", type=Path, default=Path("outputs") / "split.csv")
    mobilenet_parser.add_argument("--output", type=Path, default=Path("outputs") / "mobilenetv3")
    mobilenet_parser.add_argument("--image-size", type=int, default=64)
    mobilenet_parser.add_argument("--batch-size", type=int, default=16)
    mobilenet_parser.add_argument("--epochs", type=int, default=15)
    mobilenet_parser.add_argument("--classifier-epochs", type=int, default=2)
    mobilenet_parser.add_argument("--fine-tune-blocks", type=int, default=3)
    mobilenet_parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-4)
    mobilenet_parser.add_argument("--num-workers", type=int, default=0)
    mobilenet_parser.add_argument("--seed", type=int, default=42)
    mobilenet_parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    validate_class_mapping()
    if args.command == "download":
        download_dataset(args.root)
    elif args.command == "audit":
        audit_dataset(args.root, args.output)
    elif args.command == "smoke":
        smoke_check(args.root, args.split_manifest, args.image_size, args.batch_size, args.num_workers)
    elif args.command == "split":
        create_split(args.root, args.output, args.validation_fraction, args.seed)
    elif args.command == "train":
        history, device = train(
            DataConfig(root=args.root, batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed),
            ExperimentConfig(epochs=args.epochs, seed=args.seed, early_stopping_patience=args.patience),
            args.split_manifest,
            args.output,
        )
        last = history[-1]
        print(
            f"Training complete on {device}: epochs={len(history)}, "
            f"validation_accuracy={last.validation_accuracy:.4f}, "
            f"validation_macro_f1={last.validation_macro_f1:.4f}"
        )
    else:
        history, device, model = train_mobilenet(
            DataConfig(
                root=args.root,
                image_size=args.image_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=args.seed,
            ),
            ExperimentConfig(epochs=args.epochs, seed=args.seed, early_stopping_patience=args.patience),
            args.split_manifest,
            args.output,
            classifier_epochs=args.classifier_epochs,
            fine_tune_blocks=args.fine_tune_blocks,
            fine_tune_learning_rate=args.fine_tune_learning_rate,
        )
        last = history[-1]
        print(
            f"MobileNetV3 training complete on {device}: epochs={len(history)}, "
            f"parameters={sum(parameter.numel() for parameter in model.parameters())}, "
            f"validation_accuracy={last.validation_accuracy:.4f}, "
            f"validation_macro_f1={last.validation_macro_f1:.4f}"
        )


if __name__ == "__main__":
    main()
