import undata as ud

dataset = ud.UNDataset()

dataset.set_labels_map(["dog", "cat"])

dataset.rootdir = "/home/alenic/datasets/visdrone/"

sample = ud.UNSample(
    image_path="VisDrone2019-DET-test-dev/images/0000006_00159_d_0000001.jpg",
    bbox=[
        ud.UNBBox(
            coords=[0.1, 0.2, 0.3, 0.4],
            label_id=0,
            format="rel_xywh",
        ),
        ud.UNBBox(
            coords=[0.5, 0.6, 0.7, 0.8],
            label_id=1,
            format="rel_xywh",
        ),
    ],
)
dataset.add_sample(sample)


sample = ud.UNSample(
    image_path="VisDrone2019-DET-test-dev/images/0000006_00611_d_0000002.jpg",
    bbox=[
        ud.UNBBox(
            coords=[0.3, 0.22, 6.3, 6.4],
            label_id=4,
            format="rel_xywh",
        ),
    ],
)
dataset.add_sample(sample)


sample = ud.UNSample(
    image_path="VisDrone2019-DET-test-dev/images/0000006_07596_d_0000020.jpg", bbox=None
)
dataset.add_sample(sample)


# print(dataset)

dataset.compute_image_wh()
df = dataset.as_dataframe()
print(df)

# dataset.bbox_convert("yolo")
# df = dataset.as_dataframe()
# print(df)

dataset.export_to_yolo("/home/alenic/gitrepo/project-r/anns")
print(dataset.from_dataframe(df).model_dump_json(indent=2))
