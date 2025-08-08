_base_ = [
    "../_base_/datasets/second_512x512.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_20k.py",
]

# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)
backbone_norm_cfg = dict(type="LN", requires_grad=True)
data_preprocessor = dict(
    type="DualInputSegDataPreProcessor",
    mean=[123.675, 116.28, 103.53] * 2,
    std=[58.395, 57.12, 57.375] * 2,
    bgr_to_rgb=True,
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32),
)
checkpoint_file = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth"  # noqa
model = dict(
    type="SiamEncoderMultiDecoder",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="mmseg.SwinTransformer",
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type="GELU"),
        norm_cfg=backbone_norm_cfg,
        init_cfg=dict(type="Pretrained", checkpoint=checkpoint_file),
    ),
    decode_head=dict(
        type="GeneralSCDHead",
        binary_cd_neck=dict(type="FeatureFusionNeck", policy="abs_diff"),
        binary_cd_head=dict(
            type="mmseg.UPerHead",
            in_channels=[v * 1 for v in [96, 192, 384, 768]],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=96,
            dropout_ratio=0.1,
            num_classes=2,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type="mmseg.CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
            ),
        ),
        semantic_cd_head=dict(
            type="mmseg.UPerHead",
            in_channels=[96, 192, 384, 768],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=192,
            dropout_ratio=0.1,
            num_classes=6,
            ignore_index=255,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type="mmseg.CrossEntropyLoss",
                use_sigmoid=False,
                loss_weight=1.0,
                avg_non_ignore=True,
            ),
        ),
    ),
    postprocess_pred_and_label="cover_semantic",
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)


# optimizer
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1000),
    dict(
        type="PolyLR",
        power=0.9,
        begin=1000,
        end=10000,
        eta_min=1e-5,
        by_epoch=False,
    ),
]


train_cfg = dict(type="IterBasedTrainLoop", max_iters=10000, val_interval=1000)
default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=1000, max_keep_ckpts=2),
    visualization=dict(
        type="CDVisualizationHook", interval=1, draw_on_from_to_img=False
    ),
)

visualizer = dict(type="CDLocalVisualizer", alpha=1.0)

# compile = True
