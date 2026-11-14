from drivesmart_ml.mapping import CLASS_MAPPING, validate_class_mapping


def test_class_mapping_has_all_gtsrb_ids() -> None:
    validate_class_mapping()
    assert set(CLASS_MAPPING) == set(range(43))
    assert len(set(CLASS_MAPPING.values())) == 43
