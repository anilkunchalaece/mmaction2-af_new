_base_ = [
    '../../_base_/models/slowfast_r50.py', '../../_base_/default_runtime.py'
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
        type='ResNet3dSlowFast',
        pretrained=None,
        resample_rate=8,  # tau
        speed_ratio=8,  # alpha
        channel_ratio=8,  # beta_inv
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            norm_eval=False),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            norm_eval=False)),
    cls_head=dict(
        type='SlowFastHead',
        in_channels=2304,  # 2048+256
        num_classes=18,
        spatial_type='avg',
        dropout_ratio=0.5,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'))

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='SampleFrames', clip_len=16, frame_interval=2, num_clips=1),
    dict(type='RawFrameDecode', **file_client_args),
    # dict(type='Resize', scale=(-1, 256)),
    # dict(type='RandomResizedCrop'),
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

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=1e-4),
    clip_grad=dict(max_norm=40, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=34,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=45,
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=45)
]

default_hooks = dict(
    checkpoint=dict(interval=4, max_keep_ckpts=3), logger=dict(interval=100))

load_from = "https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth"