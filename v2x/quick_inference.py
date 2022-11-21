import os.path as osp
import os
import numpy as np
import torch.nn as nn
import logging
import sys
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])


from utils.setup_log import trace_logger, pcolor

logger = logging.getLogger(__name__)



from base_model import BaseModel
from model_utils import (
    init_model,
    inference_detector,
    inference_mono_3d_detector,
    BBoxList,
    EuclidianMatcher,
    SpaceCompensator,
    TimeCompensator,
    BasicFuser,
)

def get_vic_i():
    config = {}
    config['cfg'] = '/mnt/datax/xuelian-yang/DAIR-V2X/configs/vic3d/late-fusion-pointcloud/pointpillars/trainval_config_i.py'
    config['weight'] = '/mnt/datax/xuelian-yang/DAIR-V2X/configs/vic3d/late-fusion-pointcloud/pointpillars/vic3d_latefusion_inf_pointpillars_596784ad6127866fcfb286301757c949.pth'
    config['data'] = ['/mnt/datax/xuelian-yang/DAIR-V2X/cache-v0/tmps/tmp_i_009362.bin', '/mnt/datax/xuelian-yang/DAIR-V2X/cache-v0/tmps/tmp_i_011320.bin' ]
    return config

def get_vic_v():
    config = {}
    config['cfg'] = '/mnt/datax/xuelian-yang/DAIR-V2X/configs/vic3d/late-fusion-pointcloud/pointpillars/trainval_config_v.py'
    config['weight'] = '/mnt/datax/xuelian-yang/DAIR-V2X/configs/vic3d/late-fusion-pointcloud/pointpillars/vic3d_latefusion_veh_pointpillars_a70fa05506bf3075583454f58b28177f.pth'
    config['data'] = ['/mnt/datax/xuelian-yang/DAIR-V2X/cache-v0/tmps/tmp_v_003144.bin', '/mnt/datax/xuelian-yang/DAIR-V2X/cache-v0/tmps/tmp_v_004105.bin']
    return config

def get_inf():
    config = {}
    config['cfg'] = '/mnt/datax/xuelian-yang/DAIR-V2X/configs/sv3d-inf/pointpillars/trainval_config.py'
    config['weight'] = '/mnt/datax/xuelian-yang/DAIR-V2X/configs/sv3d-inf/pointpillars/sv3d_inf_pointpillars_bef39ec99769ac03d0e6b2c5ff6a0ef7.pth'
    config['data'] = ['/mnt/datax/xuelian-yang/DAIR-V2X/cache/tmps/tmp_i_009362.bin']
    return config 

if __name__ == '__main__':
    '''
    cfg = '/mnt/datax/xuelian-yang/DAIR-V2X/configs/sv3d-inf/pointpillars/trainval_config.py'
    weight = '/mnt/datax/xuelian-yang/DAIR-V2X/configs/sv3d-inf/pointpillars/sv3d_inf_pointpillars_bef39ec99769ac03d0e6b2c5ff6a0ef7.pth'
    model = init_model(cfg, weight)
    pcd = '/mnt/datax/xuelian-yang/DAIR-V2X/data/DAIR-V2X/cooperative-vehicle-infrastructure/infrastructure-side/velodyne/000009.pcd'
    pcd = '/mnt/datax/xuelian-yang/DAIR-V2X/cache/tmps/tmp_i_009362.bin'
    result1, data1 = inference_detector(model, pcd)
    print(f'result1: {result1}')
    result2, data2 = inference_detector(model, pcd)
    print(f'result2: {result2}')
    '''
    cfg_list = { 'vic_v': get_vic_v(), 'vic_i': get_vic_i(), 'inf': get_inf() }
    # cfg_list = { 'vic_v': get_vic_v() }
    cfg_list = { 'inf': get_inf() }
    for k, v in cfg_list.items():
        model_1 = init_model(v['cfg'], v['weight'])
        model_2 = init_model(v['cfg'], v['weight'])
        for pcd in v['data']:
            res_1, data_1 = inference_detector(model_1, pcd)
            res_2, data_2 = inference_detector(model_2, pcd)
            text_1 = f'{res_1}'
            text_2 = f'{res_2}'
            if text_1 != text_2:
                print(pcolor(f'\n\n\ndiff with {k}, {pcd}: {text_1} vs {text_2}', 'yellow'))
    pass
