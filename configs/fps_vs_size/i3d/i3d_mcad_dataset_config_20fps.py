_base_ = [
    '../../_base_/models/i3d_r50.py', '../../_base_/schedules/sgd_100e.py',
    '../../_base_/default_runtime.py'
]

root_dir = "/home/ICTDOMAIN/d20125529/fps_vs_size/MCAD_FRAMES"

# dataset settings
dataset_type = 'RawframeDataset'
data_root = F"{root_dir}/train"
data_root_val = F"{root_dir}/test_20fps"
data_root_test = F"{root_dir}/test_20fps"
split = 1  # official train/test splits. valid numbers: 1, 2, 3
ann_file_train = F"{root_dir}/train_annotations.txt"
ann_file_val = F"{root_dir}/test_20fps_annotations.txt"
ann_file_test = F"{root_dir}/test_20fps_annotations.txt"


# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3d',
        pretrained2d=None,
        pretrained= None,
        depth=50,
        conv1_kernel=(5, 7, 7),
        conv1_stride_t=2,
        pool1_stride_t=2,
        conv_cfg=dict(type='Conv3d'),
        norm_eval=False,
        inflate=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
        zero_init_residual=False),
    cls_head=dict(
        type='I3DHead',
        num_classes=7,
        in_channels=2048,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'))


file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='SampleFrames', clip_len=16, frame_interval=2, num_clips=1, start_index=0),
    dict(type='RawFrameDecode', **file_client_args),
    # dict(type='Resize', scale=(-1, 256)),
    # dict(
    #     type='MultiScaleCrop',
    #     input_size=224,
    #     scales=(1, 0.8),
    #     random_crop=False,
    #     max_wh_scale_gap=0),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),
    # dict(type='Resize', scale=(-1, 256)),
    # dict(type='CenterCrop', crop_size=224),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=2,
        num_clips=10,
        test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),
    # dict(type='Resize', scale=(-1, 256)),
    # dict(type='ThreeCrop', crop_size=256),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(img=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=45, val_begin=1, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(checkpoint=dict(interval=5, max_keep_ckpts=5))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=240)