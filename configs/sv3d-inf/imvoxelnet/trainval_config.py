work_dir = './work_dirs/ss3d_inf_imvoxelnet'

# version:
# ========
# pip list | grep -i mm
#   mmcv-full               1.3.8
#   mmdet                   2.14.0       ~/github/DAIR-V2X-Version/mmdetection
#   mmdet3d                 0.17.1       /mnt/itti-dev/mmdetection3d
#   mmsegmentation          0.14.1       ~/github/DAIR-V2X-Version/mmsegmentation
# pip list | grep -i torch
#   torch                   1.10.0+cu111
#   torchaudio              0.10.0
#   torchvision             0.11.0+cu111
# pip list | grep -i tensor
#   tensorboard             2.11.2
#   tensorboard-data-server 0.6.1
#   tensorboard-plugin-wit  1.8.1
# pip list | grep -i setup
#   setuptools              59.5.0
#
# path:
# =====
# /mnt/itti-dev$ tree -L 1
#   .
#   ├── DAIR-V2X
#   └── mmdetection3d

dataset_type = "KittiDataset"
# data_root = "../../../data/DAIR-V2X/single-infrastructure-side/"
data_root = "../DAIR-V2X/data/DAIR-V2X/single-infrastructure-side/"
class_names = ["Pedestrian", "Cyclist", "Car"]
input_modality = dict(use_lidar=False, use_camera=True)
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
voxel_size = [0.32, 0.32, 0.32]
n_voxels = [int((point_cloud_range[i + 3] - point_cloud_range[i]) / voxel_size[i]) for i in range(3)]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (1280, 384)
img_resize_scale = [(1173, 352), (1387, 416)]

ped_center = -0.6
cyc_center = -0.6
car_center = -1.78

anchor_range_ped = [
    point_cloud_range[0],
    point_cloud_range[1],
    ped_center,
    point_cloud_range[3] - voxel_size[0],
    point_cloud_range[4] - voxel_size[1],
    ped_center,
]
anchor_range_cyc = [
    point_cloud_range[0],
    point_cloud_range[1],
    cyc_center,
    point_cloud_range[3] - voxel_size[0],
    point_cloud_range[4] - voxel_size[1],
    cyc_center,
]
anchor_range_car = [
    point_cloud_range[0],
    point_cloud_range[1],
    car_center,
    point_cloud_range[3] - voxel_size[0],
    point_cloud_range[4] - voxel_size[1],
    car_center,
]

anchor_size_pred = [0.6, 0.8, 1.73]
anchor_size_cyc = [0.6, 1.76, 1.73]
anchor_size_car = [1.6, 3.9, 1.56]

model = dict(
    type="ImVoxelNet",
    pretrained="torchvision://resnet50",
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        style='pytorch'),
    neck=dict(
        type="FPN", 
        in_channels=[256, 512, 1024, 2048], 
        out_channels=64, 
        num_outs=4),
    neck_3d=dict(type="OutdoorImVoxelNeck", in_channels=64, out_channels=256),
    bbox_head=dict(
        type="Anchor3DHead",
        num_classes=len(class_names),
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        anchor_generator=dict(
            type="Anchor3DRangeGenerator",
            ranges=[anchor_range_ped, anchor_range_cyc, anchor_range_car],
            sizes=[anchor_size_pred, anchor_size_cyc, anchor_size_car],
            rotations=[0, 1.57],
            reshape_out=False,
        ),
        diff_rad_by_sin=True,
        bbox_coder=dict(type="DeltaXYZWLHRBBoxCoder"),
        loss_cls=dict(
            type="FocalLoss", 
            use_sigmoid=True, 
            gamma=2.0, 
            alpha=0.25, 
            loss_weight=1.0),
        loss_bbox=dict(type="SmoothL1Loss", beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.2),
    ),
    # n_voxels=(216, 248, 12),
    # voxel_size=(.64, .64, .64)
    n_voxels=n_voxels,
    # voxel_size=voxel_size,
    anchor_generator=dict(
        type='AlignedAnchor3DRangeGenerator',
        ranges=[[0, -39.68, -3.08, 69.12, 39.68, 0.76]],
        rotations=[.0]),

    train_cfg = dict(
        assigner=[
            dict(  # for Pedestrian
                type="MaxIoUAssigner",
                iou_calculator=dict(type="BboxOverlapsNearest3D"),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1,
            ),
            dict(  # for Cyclist
                type="MaxIoUAssigner",
                iou_calculator=dict(type="BboxOverlapsNearest3D"),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1,
            ),
            dict(  # for Car
                type="MaxIoUAssigner",
                iou_calculator=dict(type="BboxOverlapsNearest3D"),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1,
            ),
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg = dict(
        use_rotate_nms=True, 
        nms_across_levels=False, 
        nms_thr=0.01, 
        score_thr=0.2, 
        min_bbox_size=0, 
        nms_pre=100, 
        max_num=50
    )
)

train_pipeline = [
    dict(type='LoadAnnotations3D'),
    dict(type='LoadImageFromFile'),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='Resize',
        img_scale=img_resize_scale,
        keep_ratio=True,
        multiscale_mode='range'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['img'])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=3,
    train=dict(
        type="RepeatDataset",
        times=3,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + "kitti_infos_train.pkl",
            split="training",
            pts_prefix="velodyne_reduced",
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            pcd_limit_range=point_cloud_range,
            test_mode=False,
        ),
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "kitti_infos_val.pkl",
        split="training",
        pts_prefix="velodyne_reduced",
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        pcd_limit_range=point_cloud_range,
        test_mode=True,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "kitti_infos_val.pkl",
        split="training",
        pts_prefix="velodyne_reduced",
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        pcd_limit_range=point_cloud_range,
        test_mode=True,
    ),
)

optimizer = dict(
    type="AdamW",
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={"backbone": dict(lr_mult=0.1, decay_mult=1.0)}),
)
optimizer_config = dict(grad_clip=dict(max_norm=35.0, norm_type=2))
lr_config = dict(policy="step", step=[8, 11])
total_epochs = 12

checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(
    interval=50, 
    hooks=[dict(type="TextLoggerHook"), 
    dict(type="TensorboardLoggerHook")])
evaluation = dict(interval=1)
dist_params = dict(backend="nccl")
find_unused_parameters = True  # todo: fix number of FPN outputs
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]