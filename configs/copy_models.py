# -*- coding: utf-8 -*-

import argparse
import datetime
import logging
import os
import os.path as osp
import platform
import subprocess
from termcolor import colored

isLinux   = (platform.system() == "Linux")
isWindows = (platform.system() == "Windows")

logger = logging.getLogger(__name__)


def setup_log():
    medium_format = (
        '[%(asctime)s] %(levelname)s : %(filename)s[%(lineno)d] %(funcName)s'
        ' >>> %(message)s'
    )

    dt_now = datetime.datetime.now()
    log_name = osp.basename(__file__).replace('.py', '.log')
    get_log_file = osp.abspath(osp.join(osp.dirname(__file__), log_name))
    logging.basicConfig(
        filename=get_log_file,
        filemode='w',
        level=logging.INFO,
        format=medium_format
    )
    logging.info('@{} created at {}'.format(get_log_file, dt_now))
    print(colored('@{} created at {}'.format(get_log_file, dt_now), 'magenta'))


def parse_args():
    parser = argparse.ArgumentParser(description='Copy pretrained models from github.com/AIR-THU/DARI-V2X')
    parser.add_argument('--root-code', default='/mnt/itti-dev/DAIR-V2X', help='path of repos')
    parser.add_argument('--root-model', default='/mnt/models/AIR-THU_DAIR-V2X', help='path of models')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    setup_log()
    root_code = args.root_code
    root_model = args.root_model

    logger.info(f'root_code:  {root_code}')
    logger.info(f'root_model: {root_model}')
    print(colored(f'root_code:  {root_code}', 'blue'))
    print(colored(f'root_model: {root_model}', 'blue'))

    model_list = [
        'configs/sv3d-inf/imvoxelnet/sv3d_inf_imvoxelnet_747eb31f1271684a4fb8fc0ba4a3447e.pth',
        'configs/sv3d-inf/mvxnet/sv3d_inf_mvxnet_c2271983b04b73e573486fcbc559c31e.pth',
        'configs/sv3d-inf/pointpillars/sv3d_inf_pointpillars_bef39ec99769ac03d0e6b2c5ff6a0ef7.pth',
        'configs/sv3d-inf/second/sv3d_inf_second_ff017764c45b607efbec1c238378a7ac.pth',
        'configs/sv3d-veh/imvoxelnet/sv3d_veh_imvoxelnet_a02d4975873885c2f979f1f83a692a9e.pth',
        'configs/sv3d-veh/mvxnet/sv3d_veh_mvxnet_bf0e32c42649ee90e03f937214356dbf.pth',
        'configs/sv3d-veh/pointpillars/sv3d_veh_pointpillars_0ad9e1bd3bb211316d4dd0ce19c7d009.pth',
        'configs/sv3d-veh/second/sv3d_veh_second_05d43cd37acd8a480a91c2a16f26e440.pth',
        'configs/vic3d/early-fusion-pointcloud/pointpillars/vic3d_earlyfusion_veh_pointpillars_67fe2b82320754481ef37f176b647e43.pth',
        'configs/vic3d/late-fusion-image/imvoxelnet/vic3d_latefusion_inf_imvoxelnet_973cefc0b2c14fee1b8775aa996ac779.pth',
        'configs/vic3d/late-fusion-image/imvoxelnet/vic3d_latefusion_veh_imvoxelnet_9d0ad4d4930c41d62839d45c06f86326.pth',
        'configs/vic3d/late-fusion-pointcloud/pointpillars/vic3d_latefusion_inf_pointpillars_596784ad6127866fcfb286301757c949.pth',
        'configs/vic3d/late-fusion-pointcloud/pointpillars/vic3d_latefusion_veh_pointpillars_a70fa05506bf3075583454f58b28177f.pth',
    ]
    if not osp.exists(root_code):
        logger.error(f'path not exist: {root_code}')
        raise ValueError

    if not osp.exists(root_model):
        logger.error(f'path not exist: {root_model}')
        raise ValueError

    for idx, item in enumerate(model_list):
        print(colored(f'process [{idx:2d}]: {item}'))
        src_path = osp.join(root_model, item)
        if not osp.exists(src_path):
            logger.warning(f'file not exist: {src_path}')
            continue
        dst_path = osp.join(root_code, item)
        if osp.exists(dst_path):
            logger.warning(f'file already exist: {dst_path}')
            continue
        dst_dir = osp.dirname(dst_path)
        if not osp.exists(dst_dir):
            logger.info(f'mkdir {dst_dir}')
            os.makedirs(dst_dir)
        logger.info(f'cp {src_path} {dst_path}')
        subprocess.call(['cp', src_path, dst_path])


if __name__ == "__main__":
    main()
