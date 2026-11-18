import json

import numpy as np
import onnx
import onnxruntime as ort
import torch

from drivesmart_ml.model import CustomCNN
from drivesmart_ml.onnx_export import export_custom_cnn


def test_export_cnn_creates_valid_onnx_and_metadata(tmp_path) -> None:
    model = CustomCNN()
    checkpoint_path = tmp_path / "best_cnn.pth"
    torch.save(
        {
            "model_class": "CustomCNN",
            "model_state_dict": model.state_dict(),
            "config": {
                "data": {"image_size": 64},
                "experiment": {"num_classes": 43},
            },
            "metrics": {},
            "class_mapping": {
                class_id: name
                for class_id, name in enumerate(
                    [f"class-{class_id}" for class_id in range(43)]
                )
            },
            "preprocessing": {},
        },
        checkpoint_path,
    )
    output_path = tmp_path / "best_cnn.onnx"

    result = export_custom_cnn(checkpoint_path, output_path, parity_tensors=3, seed=7)

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape == ["batch_size", 3, 64, 64]
    assert session.get_outputs()[0].shape == ["batch_size", 43]
    logits = session.run(None, {"input": np.random.rand(2, 3, 64, 64).astype(np.float32)})
    assert logits[0].shape == (2, 43)
    assert result["comparison"]["top1_match"] is True
    assert 0.0 <= result["comparison"]["max_abs_diff"] < 1e-4

    metadata_path = tmp_path / "best_cnn_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["input"]["shape"] == [None, 3, 64, 64]
    assert metadata["output"]["shape"] == [None, 43]
    assert metadata["preprocessing"]["color"] == "RGB"
    assert len(metadata["class_mapping"]) == 43
