_base_ = [
    '../../_base_/models/c3d_sports1m_pretrained.py',
    '../../_base_/default_runtime.py'
]

samples_per_cls = [11056,3540,5196,11968,3351,6669,8444]

# dataset settings
dataset_type = 'RawframeDataset'
data_root = '/home/ICTDOMAIN/d20125529/action_tracklet_parser/Tubelet_Dataset/train'
data_root_val = '/home/ICTDOMAIN/d20125529/action_tracklet_parser/Tubelet_Dataset/test'
data_root_test = '/home/ICTDOMAIN/d20125529/action_tracklet_parser/Tubelet_Dataset/test'
split = 1  # official train/test splits. valid numbers: 1, 2, 3
ann_file_train = '/home/ICTDOMAIN/d20125529/action_tracklet_parser/Tubelet_Dataset/train_annotation.txt'
ann_file_val = '/home/ICTDOMAIN/d20125529/action_tracklet_parser/Tubelet_Dataset/test_annotation.txt'
ann_file_test = '/home/ICTDOMAIN/d20125529/action_tracklet_parser/Tubelet_Dataset/test_annotation.txt'


# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='C3D',
        pretrained= None,
        style='pytorch',
        conv_cfg=dict(type='Conv3d'),
        norm_cfg=None,
        act_cfg=dict(type='ReLU'),
        dropout_ratio=0.5,
        init_std=0.005),
    cls_head=dict(
        type='I3DHead',
        num_classes=7, # changed the no of classes 101 (UCF-101) to 6 (KTH)
        in_channels=4096,
        spatial_type=None,
        dropout_ratio=0.5,
        init_std=0.01,
        average_clips='prob',
        loss_cls=dict(type='CBFocalLoss', samples_per_cls=samples_per_cls)),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[104, 117, 128],
        std=[1, 1, 1],
        format_shape='NCTHW'),
    train_cfg=None,
    test_cfg=None)


file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='SampleFrames', clip_len=16, frame_interval=1, num_clips=1, start_index=0),
    dict(type='RawFrameDecode', **file_client_args),
    # dict(type='Resize', scale=(-1, 128)),
    dict(type='Resize', scale=(112, 112), keep_ratio=False),
    # dict(type='RandomCrop', size=112),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=1,
        num_clips=1,
        test_mode=True,
        start_index=0),
    dict(type='RawFrameDecode', **file_client_args),
    # dict(type='Resize', scale=(-1, 128)),
    dict(type='Resize', scale=(112, 112), keep_ratio=False),
    # dict(type='CenterCrop', crop_size=112),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=1,
        num_clips=10,
        start_index=0,
        test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),
    # dict(type='Resize', scale=(-1, 128)),
    dict(type='Resize', scale=(112, 112), keep_ratio=False),
    # dict(type='CenterCrop', crop_size=112),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=10, # changing batchsize from 30 to 10
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=10, #changing batchsize from  30 to 10
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
        data_prefix=dict(img=data_root_test),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=45, val_begin=1, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=45,
        by_epoch=True,
        milestones=[20, 40],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005), # modified lr from 0.001 to 0.0005
    clip_grad=dict(max_norm=40, norm_type=2))

default_hooks = dict(checkpoint=dict(interval=5))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (30 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=240)

load_from = "https://download.openmmlab.com/mmaction/recognition/c3d/c3d_sports1m_pretrain_20201016-dcc47ddc.pth"
