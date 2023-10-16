#配置文件
_base_ = [
    '../../_base_/schedules/sgd_100e.py',
    '../../_base_/default_runtime.py'
]
#去掉'../../_base_/models/tsn_r50.py',试试
#load_from = '/home/shelter/shelterX/wrl_code/mmaction2/checkpoints/RepLKNet-31B_ImageNet-22K-to-1K_224.pth '
#load_from 参数会覆盖整个模型，包括分类头
# model settings
model = dict(
    type='Recognizer2D',#2d卷积识别模型
    backbone=dict(
        type='RepLKNet',#怎么定位到具体代码位置的？
        #pretrained2d=True,
        pretrained='/home/shelter/shelterX/wrl_code/mmaction2/checkpoints/RepLKNet-31B_ImageNet-22K-to-1K_224.pth',#预训练文件位置
        large_kernel_sizes=[31,29,27,13], 
        layers=[2,2,18,2], 
        channels=[128,256,512,1024],
        drop_path_rate=0.2, 
        small_kernel=5,
        num_classes=400, 
        use_checkpoint=True,
        small_kernel_merged=False),
    cls_head=dict(
        type='TSNHead',
        num_classes=400,
        in_channels=1024,#输出channels是多少？
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),  # consensus 模块设置
        dropout_ratio=0.4,  # dropout 层概率
        init_std=0.01), # 线性层初始化 std 值
        # 模型训练和测试的设置
    train_cfg=None,  # 训练 TSN 的超参配置
    test_cfg=dict(average_clips=None))  # 测试 TSN 的超参配置


# dataset settings
dataset_type = 'VideoDataset'
data_root = 'data/kinetics400/videos_train'
data_root_val = 'data/kinetics400/videos_val'
ann_file_train = 'data/kinetics400/kinetics400_train_list_videos.txt'
ann_file_val = 'data/kinetics400/kinetics400_val_list_videos.txt'
ann_file_test = 'data/kinetics400/kinetics400_val_list_videos.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='DecordDecode'),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=25,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='TenCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# 优化器
optimizer = dict(
        type='SGD',
        lr=4e-4 / 16 , #先这样试试看
        momentum=0.9, 
        weight_decay=0.0001)  # 从 0.01 改为 0.005

optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# 学习策略
lr_config = dict(policy='step', step=[20, 40]) # step 与 total_epoch 相适应
total_epochs = 50 # 从 100 改为 50
checkpoint_config = dict(interval=5)

# runtime settings
work_dir = './work_dirs/replknet2d_relplknet_video_kinetics400_rgb/'