import numpy as np
import mayavi.mlab as mlab
import pickle
import argparse
import math
from pypcd import pypcd
import json
from vis_utils import id_to_str, load_pkl

import logging
import time

import sys
import os
import os.path as osp

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), "../../"))
from v2x.models.model_utils.mmdet3d_utils import show_result_meshlab
from utils.setup_log import trace_logger, setup_log, pcolor

from mmdet3d.core import (
    Box3DMode,
    CameraInstance3DBoxes,
    DepthInstance3DBoxes,
    LiDARInstance3DBoxes,
    show_multi_modality_result,
    show_result,
    show_seg_result,
)

def plot_pred_fusion(args):
    logging.warning(f'plot_pred_fusion( {args} )')
    data_all = load_pkl(osp.join(args.path, "result", id_to_str(args.id) + ".pkl"))
    print(data_all.keys())
    pc = np.fromfile(args.pcd_path, dtype=np.float32).reshape([-1, 4])
    print(pc.shape)
    print(pcolor(f'{type(data_all["boxes_3d"])} {data_all["boxes_3d"].shape}', 'yellow'))
    show_result(pc[:, :3], None, data_all['boxes_3d'], out_dir='/tmp/dair_mesh', filename='name', show=True, snapshot=True)


def plot_pred_single(args):
    logging.warning(f'plot_pred_single( {args} )')
    path = args.path
    file = id_to_str(args.id) + ".pkl"

    data_all = load_pkl(osp.join(path, file))
    pc = np.fromfile(args.pcd_path, dtype=np.float32).reshape([-1, 4])
    print(pc.shape)
    print(pcolor(f'{type(data_all["raw_boxes_3d"])} {len(data_all["raw_boxes_3d"])}', 'yellow'))
    show_result(pc[:, :3], None, np.array(data_all['raw_boxes_3d']), out_dir='/tmp/dair_mesh', filename='name', show=True, snapshot=True)


def add_arguments(parser):
    parser.add_argument("--task", type=str, default="coop", choices=["fusion", "single", "pcd_label"])
    parser.add_argument("--path", type=str, default="./coop-mono_v50100")
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--pcd-path", type=str, default="./000029.bin", help="pcd path to visualize")
    parser.add_argument("--label-path", type=str, default="./000029.json", help="label path to visualize")


if __name__ == '__main__':
    setup_log(f'tools_visualize_vis_label_in_3d_meshlab.log')

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    if args.task == "fusion":
        '''

        data_root=/mnt/datax/xuelian-yang/DAIR-V2X/cache/vic-late-lidar
        id=3144
        pcd_path=/mnt/datax/xuelian-yang/DAIR-V2X/cache/tmps/tmp_v_003144.bin
        python tools/visualize/vis_label_in_3d_meshlab.py --task fusion --path ${data_root} --id ${id} --pcd-path ${pcd_path}

        '''
        plot_pred_fusion(args)

    elif args.task == 'single':
        '''
        # veh

        pcd_path=/mnt/datax/xuelian-yang/DAIR-V2X/cache/tmps/tmp_v_003144.bin
        id=3144
        path=/mnt/datax/xuelian-yang/DAIR-V2X/cache/vic-late-lidar/veh/lidar
        python tools/visualize/vis_label_in_3d_meshlab.py --task single --path ${path} --id ${id} --pcd-path ${pcd_path}

        # inf

        pcd_path=/mnt/datax/xuelian-yang/DAIR-V2X/cache/tmps/tmp_i_009362.bin
        id=9362
        path=/mnt/datax/xuelian-yang/DAIR-V2X/cache/vic-late-lidar/inf/lidar
        python tools/visualize/vis_label_in_3d_meshlab.py --task single --path ${path} --id ${id} --pcd-path ${pcd_path}
        '''

        plot_pred_single(args)
    pass
