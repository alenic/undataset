import pytest

from undata import UNBBox, UNDataset, UNSample


def _sample(image_path: str, label_id: int = 0) -> UNSample:
    return UNSample(
        image_path=image_path,
        bbox=[UNBBox(coords=[0, 0, 10, 10], format="xywh", label_id=label_id)],
    )


def test_append_assigns_incremental_ids_and_len():
    ds = UNDataset()

    ds.append(_sample("b.jpg", 1)).append(_sample("a.jpg", 2))

    assert list(ds.sample.keys()) == [0, 1]
    assert len(ds) == 2


def test_delete_removes_existing_and_missing_raises():
    ds = UNDataset()
    ds.append(_sample("a.jpg"))

    ds.delete(0)
    assert len(ds) == 0

    with pytest.raises(IndexError):
        ds.delete(0)


def test_get_sample_returns_copy_by_default():
    ds = UNDataset()
    ds.append(_sample("a.jpg", 3))

    s = ds.get_sample(0)
    s.image_path = "changed.jpg"
    s.bbox[0].label_id = 9

    assert ds.sample[0].image_path == "a.jpg"
    assert ds.sample[0].bbox[0].label_id == 3


def test_get_sample_inplace_true_returns_reference():
    ds = UNDataset()
    ds.append(_sample("a.jpg", 3))

    s = ds.get_sample(0, inplace=True)
    s.image_path = "changed.jpg"

    assert ds.sample[0].image_path == "changed.jpg"


def test_items_and_iter_return_copies_by_default():
    ds = UNDataset()
    ds.append(_sample("a.jpg", 1)).append(_sample("b.jpg", 2))

    first_idx, first = next(ds.items())
    first.image_path = "mutated.jpg"
    assert ds.sample[first_idx].image_path != "mutated.jpg"

    sample_from_iter = next(iter(ds))
    sample_from_iter.image_path = "mutated2.jpg"
    assert ds.sample[0].image_path != "mutated2.jpg"


def test_getitem_and_len_work_and_missing_raises():
    ds = UNDataset()
    ds.append(_sample("a.jpg"))

    assert ds[0].image_path == "a.jpg"
    assert len(ds) == 1

    with pytest.raises(IndexError):
        _ = ds[1]


def test_constructor_normalizes_maps_and_get_image_paths_uses_rootdir():
    ds = UNDataset(rootdir="/data", labels_map=["cat", "dog"], tags_map={"0": "day"})
    ds.append(_sample("img1.jpg"))

    assert ds.labels_map == {0: "cat", 1: "dog"}
    assert ds.tags_map == {0: "day"}
    assert ds.get_image_paths() == ["/data/img1.jpg"]


def test_set_tags_map_with_list_and_set_rootdir():
    ds = UNDataset()
    ds.set_tags_map(["day", "night"])
    ds.set_rootdir("/tmp/images")

    assert ds.tags_map == {0: "day", 1: "night"}
    assert ds.rootdir == "/tmp/images"


def test_append_after_delete_keeps_monotonic_ids():
    ds = UNDataset()
    ds.append(_sample("a.jpg")).append(_sample("b.jpg"))

    ds.delete(1)
    ds.append(_sample("c.jpg"))

    assert list(ds.sample.keys()) == [0, 2]


def test_append_recomputes_next_id_when_initialized_with_sparse_ids():
    ds = UNDataset(sample={5: _sample("a.jpg"), 8: _sample("b.jpg")})

    ds.append(_sample("c.jpg"))

    assert 9 in ds.sample
    assert ds.sample[9].image_path == "c.jpg"

def test_append_rejects_non_unsample_values():
    ds = UNDataset()

    with pytest.raises(TypeError):
        ds.append({"image_path": "a.jpg"})  # type: ignore[arg-type]


def test_items_inplace_true_returns_live_samples():
    ds = UNDataset()
    ds.append(_sample("a.jpg", 1))

    _, s = next(ds.items(inplace=True))
    s.image_path = "changed.jpg"
    s.bbox[0].label_id = 99

    assert ds.sample[0].image_path == "changed.jpg"
    assert ds.sample[0].bbox[0].label_id == 99
