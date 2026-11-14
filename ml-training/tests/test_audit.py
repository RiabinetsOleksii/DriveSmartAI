import pandas as pd
from PIL import Image

from drivesmart_ml.audit import audit_image_tree, class_counts


def test_audit_ignores_metadata_and_recognizes_training_and_test_classes(tmp_path) -> None:
    training = tmp_path / "GTSRB" / "Training" / "00007"
    test_images = tmp_path / "GTSRB" / "Final_Test" / "Images"
    training.mkdir(parents=True)
    test_images.mkdir(parents=True)
    Image.new("RGB", (4, 4), color="red").save(training / "00007_00000.ppm")
    Image.new("RGB", (4, 4), color="blue").save(test_images / "00000.ppm")
    pd.DataFrame(
        [{"Filename": "00000.ppm", "Width": 4, "Height": 4, "Roi.X1": 0, "Roi.Y1": 0, "Roi.X2": 4, "Roi.Y2": 4, "ClassId": 12}]
    ).to_csv(tmp_path / "GT-final_test.csv", sep=";", index=False)
    (tmp_path / "GTSRB" / "Training" / "GT-00007.csv").write_text("metadata", encoding="utf-8")
    (tmp_path / "archive.zip").write_bytes(b"not an image")

    audit = audit_image_tree(tmp_path)

    assert len(audit) == 2
    assert audit["error"].isna().all()
    assert class_counts(audit) == {7: 1, 12: 1}
