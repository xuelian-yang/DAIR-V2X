import os
import os.path as osp
import time
import logging
import pickle
import numpy as np
from setup_log import trace_logger, setup_log, pcolor

def config_path():
    path_info = {}
    path_info['root'] = '/mnt/datax/xuelian-yang/DAIR-V2X/cache-v0'
    path_info['bin'] = 'tmps'
    path_info['inf'] = 'vic-late-lidar/inf/lidar'
    path_info['veh'] = 'vic-late-lidar/veh/lidar'
    path_info['result'] = 'vic-late-lidar/result'
    path_info['save'] = 'obj'
    return path_info

def analysis_data(config):
    # bin files
    in_dir = osp.join(config['root'], config['bin'])
    ou_root = osp.join(config['root'], config['save'])

    if not osp.exists(ou_root):
        os.makedirs(ou_root)

    ou_bin = osp.join(ou_root, config['bin'])
    if not osp.exists(ou_bin):
        os.mkdir(ou_bin)

    if osp.exists(in_dir):
        for item in sorted(os.listdir(in_dir)):
            pcd_path = osp.join(in_dir, item)

            pcd = np.fromfile(pcd_path, dtype=np.float32)
            cloud = pcd.reshape(-1, 4)
            obj_name = osp.join(ou_bin, osp.basename(item).replace('.bin', '.obj'))
            with open(obj_name, 'wt') as f_ou:
                n = cloud.shape[0]
                for i  in range(n):
                    f_ou.write(f'v {cloud[i, 0]:.6f} {cloud[i, 1]:.6f} {cloud[i, 2]:.6f}\n')

    names = [config['inf'], config['veh'], config['result']]
    for str_dir in names:
        in_dir = osp.join(config['root'], str_dir)
        if not osp.exists(in_dir):
            continue
        ou_dir = osp.join(ou_root, str_dir.replace('/', '_'))
        if not osp.exists(ou_dir):
            os.mkdir(ou_dir)

        for item in sorted(os.listdir(in_dir)):
            with open(osp.join(in_dir, item), "rb") as f:
                data = pickle.load(f)
                print(f'{osp.join(in_dir, item)} {sorted(data.keys())}', end='')
                if 'points' in data.keys():
                    print(f' {len(data["points"])} {len(data["points"][0])}', end='')
                    if 'inf' in str_dir:
                        obj_name = osp.join(ou_dir, 'i_' + osp.basename(item).replace('.pkl', '.obj'))
                    else:
                        obj_name = osp.join(ou_dir, 'v_' + osp.basename(item).replace('.pkl', '.obj'))
                    with open(obj_name, 'wt') as f_ou:
                        for p in data["points"]:
                            f_ou.write(f'v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n')
                print()
                for i in data.keys():
                    if isinstance(data[i], np.ndarray):
                        print(pcolor(f'    {i:<20s} {type(data[i])} {len(data[i])}   {data[i]}', 'yellow'))
                    elif isinstance(data[i], str):
                        print(pcolor(f'    {i:<20s} {type(data[i])} {len(data[i])}   {data[i]}', 'yellow'))
                    else:
                        print(pcolor(f'    {i:<20s} {type(data[i])} {len(data[i])}', 'yellow'))


if __name__ == '__main__':
    setup_log('save_obj.log')
    time_beg_save_obj = time.time()

    config = config_path()
    analysis_data(config)

    time_end_save_obj = time.time()
    logging.warning(f'save_obj.py elapsed {time_end_save_obj - time_beg_save_obj:.6f} seconds')
    print(pcolor(f'save_obj.py elapsed {time_end_save_obj - time_beg_save_obj:.6f} seconds', 'yellow'))
