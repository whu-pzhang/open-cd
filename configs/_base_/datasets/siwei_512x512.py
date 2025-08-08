# dataset settings
dataset_type = "Siwei_Dataset"
data_root = "data/crop512"

crop_size = (512, 512)
train_pipeline = [
    dict(type="MultiImgLoadImageFromFile"),
    dict(type="MultiImgMultiAnnLoadAnnotations"),
    # dict(
    #     type="MultiImgRandomResize",
    #     scale=(2048, 512),
    #     ratio_range=(0.5, 2.0),
    #     keep_ratio=True,
    # ),
    dict(type="MultiImgRandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="MultiImgRandomFlip", prob=0.5),
    dict(type="MultiImgRandomFlip", prob=0.5, direction="vertical"),
    dict(type="MultiImgPackSegInputs"),
]

test_pipeline = [
    dict(type="MultiImgLoadImageFromFile"),
    dict(type="MultiImgResize", scale=(512, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="MultiImgMultiAnnLoadAnnotations"),
    dict(type="MultiImgPackSegInputs"),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path_from="images_t0",
            img_path_to="images_t1",
            seg_map_path="label_binary",
            seg_map_path_from="masks_t0",
            seg_map_path_to="masks_t1",
        ),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path_from="images_t0",
            img_path_to="images_t1",
            seg_map_path="label_binary",
            seg_map_path_from="masks_t0",
            seg_map_path_to="masks_t1",
        ),
        pipeline=test_pipeline,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path_from="images_t0",
            img_path_to="images_t1",
            seg_map_path="label_binary",
            seg_map_path_from="masks_t0",
            seg_map_path_to="masks_t1",
        ),
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(type="SCDMetric", iou_metrics=["mFscore", "mIoU"], cal_sek=True)
test_evaluator = val_evaluator
