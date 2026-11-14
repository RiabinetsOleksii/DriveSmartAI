from drivesmart_ml.split import SampleRecord, gtsrb_group_id, split_records, write_split_manifest


def make_records() -> list[SampleRecord]:
    return [
        SampleRecord(
            path=f"{class_id:05d}/{track:05d}_{frame:05d}.ppm",
            class_id=class_id,
            group_id=f"{class_id}:{track}",
        )
        for class_id in range(43)
        for track in range(5)
        for frame in range(2)
    ]


def test_group_id_uses_track_not_frame() -> None:
    assert gtsrb_group_id("00014_00007.ppm", 14) == "14:00014"


def test_split_is_reproducible_and_has_no_image_or_group_leakage(tmp_path) -> None:
    records = make_records()
    train_a, validation_a = split_records(records, validation_fraction=0.2, seed=123)
    train_b, validation_b = split_records(records, validation_fraction=0.2, seed=123)
    assert train_a == train_b
    assert validation_a == validation_b
    assert {item.path for item in train_a}.isdisjoint(item.path for item in validation_a)
    assert {item.group_id for item in train_a}.isdisjoint(item.group_id for item in validation_a)
    assert {item.class_id for item in train_a} == set(range(43))
    assert {item.class_id for item in validation_a} == set(range(43))
    write_split_manifest(train_a, validation_a, tmp_path / "split.csv", seed=123, validation_fraction=0.2)
    assert (tmp_path / "split.csv").exists()
    assert (tmp_path / "split.json").exists()
