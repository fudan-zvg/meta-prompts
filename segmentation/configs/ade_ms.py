_base_ = [
    '_base_/models/upernet_r50.py', '_base_/datasets/ade20k.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_80k.py'
]
IMG_MEAN = [v * 255 for v in [0.5, 0.5, 0.5]]
IMG_VAR = [v * 255 for v in [0.5, 0.5, 0.5]]
img_norm_cfg = dict(mean=IMG_MEAN, std=IMG_VAR, to_rgb=True)

head_c=512 #ori512
in_c=150
model = dict(
    type='MetaPromptsSeg',
    sd_path='checkpoints/v1-5-pruned-emaonly.ckpt',
    refine_step=3,
    num_prompt=in_c,
    decode_head=dict(
        type='UPerHead',
        in_channels=[in_c, in_c, in_c, in_c],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=head_c,
        dropout_ratio=0.,
        num_classes=150,
        loss_decode=
        [
        dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=1.0),
        dict(type='LovaszLoss', reduction='none', loss_weight=1.0)
        ]
        ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=in_c,
        in_index=2,
        channels=head_c,
        num_convs=1,
        dropout_ratio=0.,
        num_classes=150,
        loss_decode=dict(type='CrossEntropyLoss', loss_name='loss_ce_aux', use_sigmoid=False, loss_weight=0.4)
        ),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341))
)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

lr_config = dict(policy='poly', power=1, min_lr=0.0, by_epoch=False,
                warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6)


optimizer = dict(type='AdamW', lr=0.00008, weight_decay=0.001,
        paramwise_cfg=dict(bypass_duplicate=True,
                            custom_keys={'unet': dict(lr_mult=0.1),
                                        'encoder_vq': dict(lr_mult=0.0),
                                        'text_encoder': dict(lr_mult=0.1),
                                        'norm': dict(decay_mult=0.)}))

data = dict(samples_per_gpu=1, workers_per_gpu=2)
