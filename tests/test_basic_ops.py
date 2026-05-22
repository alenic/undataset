import pytest

from undata import UNBBox, ODDataset, ODSample


def _sample(image_path: str, label_id: int = 0) -> ODSample:
    return ODSample(
        image_path=image_path,
        bbox=[UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=label_id)],
    )


def test_append_assigns_incremental_ids_and_len():
    ds = ODDataset()

    ds.append(_sample("b.jpg", 1)).append(_sample("a.jpg", 2))

    assert list(ds.sample.keys()) == [0, 1]
    assert len(ds) == 2


def test_delete_removes_existing_and_missing_raises():
    ds = ODDataset()
    ds.append(_sample("a.jpg"))

    ds.delete(0)
    assert len(ds) == 0

    with pytest.raises(IndexError):
        ds.delete(0)


def test_get_sample_returns_copy_by_default():
    ds = ODDataset()
    ds.append(_sample("a.jpg", 3))

    s = ds.get_sample(0)
    s.image_path = "changed.jpg"
    s.bbox[0].label_id = 9

    assert ds.sample[0].image_path == "a.jpg"
    assert ds.sample[0].bbox[0].label_id == 3


def test_get_sample_inplace_true_returns_reference():
    ds = ODDataset()
    ds.append(_sample("a.jpg", 3))

    s = ds.get_sample(0, inplace=True)
    s.image_path = "changed.jpg"

    assert ds.sample[0].image_path == "changed.jpg"


def test_items_and_iter_return_copies_by_default():
    ds = ODDataset()
    ds.append(_sample("a.jpg", 1)).append(_sample("b.jpg", 2))

    first_idx, first = next(ds.items())
    first.image_path = "mutated.jpg"
    assert ds.sample[first_idx].image_path != "mutated.jpg"

    sample_from_iter = next(iter(ds))
    sample_from_iter.image_path = "mutated2.jpg"
    assert ds.sample[0].image_path != "mutated2.jpg"


def test_getitem_and_len_work_and_missing_raises():
    ds = ODDataset()
    ds.append(_sample("a.jpg"))

    assert ds[0].image_path == "a.jpg"
    assert len(ds) == 1

    with pytest.raises(IndexError):
        _ = ds[1]


def test_constructor_normalizes_maps_and_image_paths_uses_rootdir():
    ds = ODDataset(rootdir="/data", labels_map=["cat", "dog"], tags_map={"0": "day"})
    ds.append(_sample("img1.jpg"))

    assert ds.labels_map == {0: "cat", 1: "dog"}
    assert ds.tags_map == {0: "day"}
    assert ds.image_paths() == ["/data/img1.jpg"]


def test_set_tags_map_with_list_and_set_rootdir():
    ds = ODDataset()
    ds.set_tags_map(["day", "night"])
    ds.set_rootdir("/tmp/images")

    assert ds.tags_map == {0: "day", 1: "night"}
    assert ds.rootdir == "/tmp/images"


def test_append_after_delete_keeps_monotonic_ids():
    ds = ODDataset()
    ds.append(_sample("a.jpg")).append(_sample("b.jpg"))

    ds.delete(1)
    ds.append(_sample("c.jpg"))

    assert list(ds.sample.keys()) == [0, 2]


def test_append_recomputes_next_id_when_initialized_with_sparse_ids():
    ds = ODDataset(sample={5: _sample("a.jpg"), 8: _sample("b.jpg")})

    ds.append(_sample("c.jpg"))

    assert 9 in ds.sample
    assert ds.sample[9].image_path == "c.jpg"


def test_append_rejects_non_unsample_values():
    ds = ODDataset()

    with pytest.raises(TypeError):
        ds.append({"image_path": "a.jpg"})  # type: ignore[arg-type]


def test_items_inplace_true_returns_live_samples():
    ds = ODDataset()
    ds.append(_sample("a.jpg", 1))

    _, s = next(ds.items(inplace=True))
    s.image_path = "changed.jpg"
    s.bbox[0].label_id = 99

    assert ds.sample[0].image_path == "changed.jpg"
    assert ds.sample[0].bbox[0].label_id == 99


def test_unsample_remove_labels_returns_copy_by_default():
    sample = ODSample(
        image_path="a.jpg",
        bbox=[
            UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=1),
            UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=2),
            UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=None),
        ],
    )

    filtered = sample.remove_labels([2])

    assert [bb.label_id for bb in filtered.bbox] == [1, None]
    assert [bb.label_id for bb in sample.bbox] == [1, 2, None]


def test_unsample_remove_labels_inplace_removes_matching_boxes():
    sample = ODSample(
        image_path="a.jpg",
        bbox=[
            UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=1),
            UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=2),
            UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=3),
        ],
    )

    sample.remove_labels([1, 3], inplace=True)

    assert [bb.label_id for bb in sample.bbox] == [2]


def test_undataset_remove_labels_returns_copy_by_default():
    ds = ODDataset(labels_map={1: "cat", 2: "dog"})
    ds.append(
        ODSample(
            image_path="a.jpg",
            bbox=[
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=1),
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=2),
            ],
        )
    )

    filtered = ds.remove_labels([2], keep_labels_map=True)

    print(filtered)

    assert [bb.label_id for bb in filtered.sample[0].bbox] == [1]
    assert [bb.label_id for bb in ds.sample[0].bbox] == [1, 2]


def test_undataset_remove_labels_inplace_updates_all_samples():
    ds = ODDataset()
    ds.append(
        ODSample(
            image_path="a.jpg",
            bbox=[
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=1),
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=2),
            ],
        )
    )
    ds.append(
        ODSample(
            image_path="b.jpg",
            bbox=[
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=2),
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=3),
            ],
        )
    )

    ds.remove_labels([2], keep_labels_map=True, inplace=True)
    assert [bb.label_id for bb in ds.sample[0].bbox] == [1]
    assert [bb.label_id for bb in ds.sample[1].bbox] == [3]


def test_check_labels_reports_coherent_mapping():
    ds = ODDataset(labels_map={1: "cat", 2: "dog"})
    ds.append(
        ODSample(
            image_path="a.jpg",
            bbox=[
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=1),
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=2),
            ],
        )
    )

    result = ds.check_labels()

    assert result == {
        "is_coherent": True,
        "used_label_ids": [1, 2],
        "unmapped_label_ids": [],
        "unused_label_map_ids": [],
    }


def test_check_labels_reports_unmapped_and_unused_ids():
    ds = ODDataset(labels_map={1: "cat", 3: "bird"})
    ds.append(
        ODSample(
            image_path="a.jpg",
            bbox=[
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=1),
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=2),
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=None),
            ],
        )
    )

    result = ds.check_labels()

    assert result == {
        "is_coherent": False,
        "used_label_ids": [1, 2],
        "unmapped_label_ids": [2],
        "unused_label_map_ids": [3],
    }


def test_merge_labels_returns_copy_by_default_and_reassigns_remaining_ids():
    ds = ODDataset(labels_map={10: "cat", 20: "dog", 30: "bird", 40: "horse"})
    ds.append(
        ODSample(
            image_path="a.jpg",
            bbox=[
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=20),
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=40),
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=30),
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=None),
            ],
        )
    )

    merged = ds.merge_labels({0: [20, 40]}, {0: "pet"})

    assert merged.labels_map == {0: "pet", 1: "cat", 2: "bird"}
    assert [bb.label_id for bb in merged.sample[0].bbox] == [0, 0, 2, None]

    assert ds.labels_map == {10: "cat", 20: "dog", 30: "bird", 40: "horse"}
    assert [bb.label_id for bb in ds.sample[0].bbox] == [20, 40, 30, None]


def test_merge_labels_inplace_updates_current_dataset():
    ds = ODDataset(labels_map={10: "cat", 20: "dog", 30: "bird", 40: "horse"})
    ds.append(
        ODSample(
            image_path="a.jpg",
            bbox=[
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=20),
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=40),
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=30),
            ],
        )
    )

    ds.merge_labels({0: [20, 40]}, {0: "pet"}, inplace=True)

    assert ds.labels_map == {0: "pet", 1: "cat", 2: "bird"}
    assert [bb.label_id for bb in ds.sample[0].bbox] == [0, 0, 2]


def test_merge_labels_rejects_overlapping_groups():
    ds = ODDataset(labels_map={0: "cat", 1: "dog", 2: "bird"})

    with pytest.raises(ValueError, match="appears in more than one merge group"):
        ds.merge_labels({0: [0, 1], 1: [1, 2]}, {0: "pet", 1: "other"})


def test_remap_labels_allows_new_target_ids_and_drops_labels():
    ds = ODDataset(labels_map={10: "cat", 20: "dog", 30: "bird"})
    ds.append(
        ODSample(
            image_path="a.jpg",
            bbox=[
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=10),
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=20),
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=30),
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=None),
            ],
        )
    )

    ds.remap_labels({10: 0, 20: 0, 30: None}, {0: "pet"})

    assert ds.labels_map == {0: "pet"}
    assert [bb.label_id for bb in ds.sample[0].bbox] == [0, 0, None]


def test_remap_labels_requires_new_map_to_cover_preserved_ids():
    ds = ODDataset(labels_map={5: "cat", 7: "dog"})

    with pytest.raises(
        ValueError,
        match="new_labels_map keys must exactly match the resulting label ids",
    ):
        ds.remap_labels({5: 0}, {0: "pet"})


def test_normalize_labels_map_returns_copy_by_default():
    ds = ODDataset(labels_map={10: "cat", 30: "bird", 99: "unused"})
    ds.append(
        ODSample(
            image_path="a.jpg",
            bbox=[
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=30),
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=10),
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=None),
            ],
        )
    )

    normalized = ds.normalize_labels_map()

    assert normalized.labels_map == {0: "cat", 1: "bird"}
    assert [bb.label_id for bb in normalized.sample[0].bbox] == [1, 0, None]

    assert ds.labels_map == {10: "cat", 30: "bird", 99: "unused"}
    assert [bb.label_id for bb in ds.sample[0].bbox] == [30, 10, None]


def test_normalize_labels_map_inplace_fills_unmapped_and_drops_unused_ids():
    ds = ODDataset(labels_map={4: "dog", 9: "unused"})
    ds.append(
        ODSample(
            image_path="a.jpg",
            bbox=[
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=7),
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=4),
            ],
        )
    )

    ds.normalize_labels_map(inplace=True)

    assert ds.labels_map == {0: "dog", 1: "7"}
    assert [bb.label_id for bb in ds.sample[0].bbox] == [1, 0]


def test_normalize_labels_map_without_existing_labels_map_builds_one():
    ds = ODDataset()
    ds.append(
        ODSample(
            image_path="a.jpg",
            bbox=[
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=5),
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=2),
            ],
        )
    )

    ds.normalize_labels_map(inplace=True)

    assert ds.labels_map == {0: "2", 1: "5"}
    assert [bb.label_id for bb in ds.sample[0].bbox] == [1, 0]


def test_normalize_labels_map_ascending_order_uses_label_names():
    ds = ODDataset(labels_map={10: "zebra", 30: "ant", 20: "dog"})
    ds.append(
        ODSample(
            image_path="a.jpg",
            bbox=[
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=10),
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=20),
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=30),
            ],
        )
    )

    ds.normalize_labels_map(ascending_order=True, inplace=True)

    assert ds.labels_map == {0: "ant", 1: "dog", 2: "zebra"}
    assert [bb.label_id for bb in ds.sample[0].bbox] == [2, 1, 0]


def test_normalize_labels_map_ascending_order_appends_unmapped_used_ids():
    ds = ODDataset(labels_map={10: "zebra", 30: "ant"})
    ds.append(
        ODSample(
            image_path="a.jpg",
            bbox=[
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=20),
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=10),
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=30),
                UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=5),
            ],
        )
    )

    ds.normalize_labels_map(ascending_order=True, inplace=True)

    assert ds.labels_map == {0: "ant", 1: "zebra", 2: "5", 3: "20"}
    assert [bb.label_id for bb in ds.sample[0].bbox] == [3, 1, 0, 2]
