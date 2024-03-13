_base_ = '../../recognition/timesformer/timesformer_spaceOnly_8xb8-8x32x1-15e_kinetics400-rgb.py'

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


model = dict(
            backbone=dict(attention_type='divided_space_time',
            pretrained=None),
            cls_head=dict(
                num_classes = 7,
                loss_cls=dict(type='CBFocalLoss', samples_per_cls=samples_per_cls)
            ))


file_client_args = dict(io_backend='disk')

train_pipeline = [
    # dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=8, frame_interval=32, num_clips=1),
    # dict(type='DecordDecode'),
    dict(type='RawFrameDecode', **file_client_args),
    # dict(type='RandomRescale', scale_range=(256, 320)),
    # dict(type='RandomCrop', size=224),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    # dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=32,
        num_clips=1,
        test_mode=True),
    # dict(type='DecordDecode'),
    dict(type='RawFrameDecode', **file_client_args),
    # dict(type='Resize', scale=(-1, 256)),
    # dict(type='CenterCrop', crop_size=224),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    # dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=32,
        num_clips=1,
        test_mode=True),
    # dict(type='DecordDecode'),
    dict(type='RawFrameDecode', **file_client_args),
    # dict(type='Resize', scale=(-1, 224)),
    # dict(type='ThreeCrop', crop_size=224),
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
    type='EpochBasedTrainLoop', max_epochs=10, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

load_from=None