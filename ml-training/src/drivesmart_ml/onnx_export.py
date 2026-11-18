"""Export and verify the final CustomCNN checkpoint as ONNX."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .mapping import CLASS_MAPPING, validate_class_mapping
from .model import CustomCNN
from .preprocessing import PreprocessingContract


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_custom_cnn(
    checkpoint_path: Path,
    output_path: Path,
    parity_tensors: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    if parity_tensors <= 0:
        raise ValueError("parity_tensors must be positive")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError(f"invalid checkpoint format: {checkpoint_path}")
    if checkpoint.get("model_class") != "CustomCNN":
        raise ValueError(f"checkpoint is not a CustomCNN: {checkpoint.get('model_class')}")

    config = checkpoint.get("config", {})
    num_classes = int(config.get("experiment", {}).get("num_classes", 43))
    image_size = int(config.get("data", {}).get("image_size", 64))
    if num_classes != 43 or image_size != 64:
        raise ValueError(
            f"unexpected CustomCNN contract: num_classes={num_classes}, image_size={image_size}"
        )
    validate_class_mapping(checkpoint.get("class_mapping") or CLASS_MAPPING)

    model = CustomCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.cpu().eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.zeros(1, 3, 64, 64, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    if input_meta.shape != ["batch_size", 3, 64, 64]:
        raise RuntimeError(f"unexpected ONNX input shape: {input_meta.shape}")
    if output_meta.shape != ["batch_size", 43]:
        raise RuntimeError(f"unexpected ONNX output shape: {output_meta.shape}")

    generator = torch.Generator().manual_seed(seed)
    test_inputs = torch.randn(parity_tensors, 3, 64, 64, generator=generator)
    with torch.inference_mode():
        torch_outputs = model(test_inputs).cpu().numpy()
    onnx_outputs = session.run(
        [output_meta.name],
        {input_meta.name: test_inputs.cpu().numpy()},
    )[0]

    max_abs_diff = float(np.max(np.abs(torch_outputs - onnx_outputs)))
    mean_abs_diff = float(np.mean(np.abs(torch_outputs - onnx_outputs)))
    torch_top1 = np.argmax(torch_outputs, axis=1)
    onnx_top1 = np.argmax(onnx_outputs, axis=1)
    top1_match = bool(np.array_equal(torch_top1, onnx_top1))
    if not top1_match:
        raise RuntimeError("PyTorch and ONNX Runtime top-1 predictions differ")

    preprocessing = PreprocessingContract().to_dict()
    checkpoint_mapping = checkpoint.get("class_mapping", CLASS_MAPPING)
    class_mapping = {
        str(class_id): checkpoint_mapping[class_id]
        for class_id in range(43)
    }
    metadata = {
        "model": "CustomCNN",
        "onnx_file": output_path.name,
        "onnx_sha256": _sha256(output_path),
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "input": {
            "name": input_meta.name,
            "shape": [None, 3, 64, 64],
            "dtype": "float32",
        },
        "output": {
            "name": output_meta.name,
            "shape": [None, 43],
            "dtype": "float32",
            "semantics": "unnormalized logits; index = GTSRB class ID",
        },
        "preprocessing": preprocessing,
        "class_mapping": class_mapping,
        "onnx_runtime": {
            "provider": "CPUExecutionProvider",
            "opset": 17,
        },
    }
    metadata_path = output_path.with_name("best_cnn_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    comparison = {
        "device": "cpu",
        "test_tensors": int(parity_tensors),
        "seed": seed,
        "shape": [int(parity_tensors), 43],
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "top1_match": top1_match,
        "torch_top1": torch_top1.astype(int).tolist(),
        "onnx_top1": onnx_top1.astype(int).tolist(),
    }
    comparison_path = output_path.with_name("best_cnn_parity.json")
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    metadata["parity"] = comparison
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "input_shape": [None, 3, 64, 64],
        "output_shape": [None, 43],
        "comparison": comparison,
        "metadata_path": str(metadata_path),
        "comparison_path": str(comparison_path),
    }
