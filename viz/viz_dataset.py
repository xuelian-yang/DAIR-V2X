# -*- coding: utf-8 -*-

"""
References
----------
    Create a collapsible vertical widget
        https://github.com/isl-org/Open3D/blob/v0.16.0/examples/python/visualization/non_english.py#L186
    Docs » Build from source » ML Module
        http://www.open3d.org/docs/release/compilation.html#ml-module
    Docs » open3d.t » open3d.t.geometry » open3d.t.geometry.PointCloud
        http://www.open3d.org/docs/release/python_api/open3d.t.geometry.PointCloud.html#open3d-t-geometry-pointcloud
    Docs » open3d.visualization » open3d.visualization.gui
        http://www.open3d.org/docs/release/python_api/open3d.visualization.gui.html
    How to identify which OS Python is running on?
        https://stackoverflow.com/questions/1854/how-to-identify-which-os-python-is-running-on/1857#1857
    How to set min/max for colormap when rendering a pointcloud? #2545
        https://github.com/isl-org/Open3D/issues/2545#issuecomment-987119956
    look_at(center, eye, up)
        http://www.open3d.org/docs/release/python_api/open3d.visualization.rendering.Camera.html#open3d.visualization.rendering.Camera.look_at
    Materials
        https://github.com/isl-org/Open3D/tree/v0.16.0/cpp/open3d/visualization/gui/Materials
        https://github.com/isl-org/Open3D/blob/v0.16.0/cpp/open3d/visualization/visualizer/GuiSettingsModel.h#L59
    Multiple viewports per window
        https://github.com/isl-org/Open3D/issues/999#issuecomment-720839907
    Open3D example - video.py
        https://github.com/isl-org/Open3D/blob/master/examples/python/visualization/video.py
    Surface reconstruction
        http://www.open3d.org/docs/release/tutorial/geometry/surface_reconstruction.html
    Update geometry using open3d.visualization.rendering.Open3DScene
        https://github.com/isl-org/Open3D/issues/2869#issuecomment-761942166

pip install pyscreenshot
"""

import copy
import cv2
import datetime
import glob
import importlib.util
import json
import logging
import numpy as np
import os
import os.path as osp
import PIL
import platform
import pprint
import pyscreenshot
import sys

from termcolor import colored
import threading
import time

try:
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
except ImportError:
    raise ImportError('Please run "pip install open3d" to install open3d first.')

has_ml3d = importlib.util.find_spec('open3d._ml3d')
if has_ml3d:
    print(colored(f'>>> has _ml3d <<<', 'blue'))
    from open3d.ml.vis import Colormap
else:
    print(colored(f'>>> no _ml3d <<<', 'red'))


isLinux   = (platform.system() == "Linux")
isMacOS   = (platform.system() == "Darwin")
isWindows = (platform.system() == "Windows")

logger = logging.getLogger(__name__)
g_time_beg = time.time()

name2id = {
    "car": 2,
    "van": 2,
    "truck": 2,
    "bus": 2,
    "cyclist": 1,
    "tricyclist": 3,
    "motorcyclist": 3,
    "barrow": 3,
    "barrowlist": 3,
    "pedestrian": 0,
    "trafficcone": 3,
    "pedestrianignore": 3,
    "carignore": 3,
    "otherignore": 3,
    "unknowns_unmovable": 3,
    "unknowns_movable": 3,
    "unknown_unmovable": 3,
    "unknown_movable": 3,
}

superclass = {
    -1: "ignore",
    0: "pedestrian",
    1: "cyclist",
    2: "car",
    3: "ignore",
}

color_superclass = {
    0: (0, 0, 255),
    1: (0, 255, 0),
    2: (255, 0, 0),
    3: (0, 255, 255),
}


def pcolor(string, color, on_color=None, attrs=None):
    return colored(string, color, on_color, attrs)


def setup_log():
    medium_format = (
        '[%(asctime)s] %(levelname)s : %(filename)s[%(lineno)d] %(funcName)s'
        ' >>> %(message)s'
    )

    dt_now = datetime.datetime.now()
    log_name = __file__.replace('.py', '.log')
    get_log_file = osp.join(osp.dirname(__file__), log_name)
    logging.basicConfig(
        filename=get_log_file,
        filemode='w',
        level=logging.INFO,
        format=medium_format
    )
    logging.info('@{} created at {}'.format(get_log_file, dt_now))
    print(pcolor('@{} created at {}'.format(get_log_file, dt_now), 'magenta'))


class PathConfig:
    def __init__(self, path_cfg):
        self.image_paths    = path_cfg['image_paths']
        self.point_paths    = path_cfg['point_paths']
        self.intr_paths     = path_cfg['intr_paths']
        self.extr_v2c_paths = path_cfg['extr_v2c_paths']
        self.extr_v2w_paths = path_cfg['extr_v2w_paths']
        self.label2d_paths  = path_cfg['label2d_paths']
        self.label3d_paths  = path_cfg['label3d_paths']

        self.num = len(self.image_paths)
        assert len(self.point_paths)     == self.num \
            and len(self.intr_paths)     == self.num \
            and len(self.extr_v2c_paths) == self.num \
            and len(self.extr_v2w_paths) == self.num \
            and len(self.label2d_paths)  == self.num \
            and len(self.label3d_paths)  == self.num, 'Unmatched data pairs'

    def read_frame(self, k):
        assert 0 <= k < self.num, f'index {k} is out of range [0, {self.num})'

        def load_json(path, debug=False):
            with open(path, mode="r") as f:
                data = json.load(f)
            if debug:
                logging.info(f'load_json({path})')
            return data

        # o3d.geometry vs o3d.t.geometry
        # image = o3d.io.read_image(self.image_paths[k]) 
        # point = o3d.io.read_point_cloud(self.point_paths[k])
        image = o3d.t.io.read_image(self.image_paths[k])
        cloud_t_xyzi = o3d.t.io.read_point_cloud(self.point_paths[k]) # xyzi

        # o3d.core.Tensor [arange, diag, empty, eye, from_dlpack, from_numpy,  full, load, ones, zeros]
        cloud_t_xyzi.point.colors = o3d.core.Tensor.full(cloud_t_xyzi.point.positions.shape, 0.3, cloud_t_xyzi.point.positions.dtype, cloud_t_xyzi.point.positions.device)

        # open3d has open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii),
        #   but doesn't have open3d.t.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
        cloud_xyz    = o3d.io.read_point_cloud(self.point_paths[k])
        label2d      = load_json(self.label2d_paths[k])
        label3d      = load_json(self.label3d_paths[k])
        intr         = load_json(self.intr_paths[k])
        extr_v2c     = load_json(self.extr_v2c_paths[k])
        extr_v2w     = load_json(self.extr_v2w_paths[k])

        '''
        if isinstance(cloud_t_xyzi, o3d.t.geometry.PointCloud):
            cloud_t_xyzi.point['__visualization_scalar'] = cloud_t_xyzi.point.intensity
        '''
        return image, cloud_t_xyzi, label2d, label3d, intr, extr_v2c, extr_v2w, cloud_xyz

    def read_frames(self):
        frames = []
        for i in range(self.num):
            frames.append(self.read_frame(i))
        return frames


def get_path():
    if isLinux:
        str_root = '/mnt/datax/Datasets/DAIR-V2X-Extracted/cooperative-vehicle-infrastructure-example/infrastructure-side'
    else:
        str_root = 'D:/0-DAIR-V2X-Dataset/DAIR-V2X-C-Example/cooperative-vehicle-infrastructure-example/infrastructure-side'
    dict_name = {
        'image_paths':    'image',                        # jpg
        'point_paths':    'velodyne',                     # pcd
        'intr_paths':     'calib/camera_intrinsic',       # json
        'extr_v2c_paths': 'calib/virtuallidar_to_camera', # json
        'extr_v2w_paths': 'calib/virtuallidar_to_world',  # json
        'label2d_paths':  'label/camera',                 # json
        'label3d_paths':  'label/virtuallidar'            # json
        }
    path_cfg = {}
    for k, v in dict_name.items():
        str_dir = osp.join(str_root, v)
        path_cfg[k] = [osp.join(str_dir, i) for i in sorted(os.listdir(str_dir))]
    return PathConfig(path_cfg)


def draw_2d_image_label(image, label2d):
    cv_img = copy.deepcopy(image.as_tensor().numpy())
    for item in label2d:
        box = item['2d_box']
        x1 = int(box['xmin'])
        x2 = int(box['xmax'])
        y1 = int(box['ymin'])
        y2 = int(box['ymax'])
        cv2.rectangle(cv_img, (x1, y1), (x2, y2), color_superclass[name2id[item['type'].lower()]], 2)
    return o3d.t.geometry.Image(cv_img)


def draw_3d_pointcloud_label(label3d):
    line_set_col = []
    for label in label3d:
        obj_size = [
            float(label["3d_dimensions"]["l"]),
            float(label["3d_dimensions"]["w"]),
            float(label["3d_dimensions"]["h"]),
        ]
        yaw_lidar = float(label["rotation"])
        center_lidar = [
            float(label["3d_location"]["x"]),
            float(label["3d_location"]["y"]),
            float(label["3d_location"]["z"]),
        ]
        yaw = np.zeros(3)
        yaw[2] = yaw_lidar
        rot_mat  = o3d.geometry.get_rotation_matrix_from_xyz(yaw)
        box3d    = o3d.geometry.OrientedBoundingBox(center_lidar, rot_mat, obj_size)
        line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
        line_set.paint_uniform_color((1, 0, 0))
        line_set_col.append(line_set)
    return line_set_col


def calc_point_color(cloud, image, intr, extr_v2c):
    # 1. get point cloud in camera coordinate
    np_cloud = cloud.point.positions.numpy()
    if "Tr_velo_to_cam" in extr_v2c.keys():
        velo2cam = np.array(extr_v2c["Tr_velo_to_cam"]).reshape(3, 4)
        r_v2c = velo2cam[:, :3]
        t_v2c = velo2cam[:, 3].reshape(3, 1)
    else:
        r_v2c = np.array(extr_v2c["rotation"])
        t_v2c = np.array(extr_v2c["translation"])
    new_cloud = r_v2c * np.matrix(np_cloud).T + t_v2c # point cloud in camera coordinate

    # 2. project point cloud to image coordinate
    intr_mat = np.zeros([3, 4])
    intr_mat[:3, :3] = np.array(intr['cam_K']).reshape([3, 3], order="C")
    new_cloud       = np.array(new_cloud.T)
    points_num      = list(new_cloud.shape)[:-1]
    points_shape    = np.concatenate([points_num, [1]], axis=0)
    points_2d_shape = np.concatenate([points_num, [3]], axis=0)
    points_4 = np.concatenate((new_cloud, np.ones(points_shape)), axis=-1)
    point_2d = np.matmul(intr_mat, points_4.T)
    point_2d = point_2d.T.reshape(points_2d_shape)
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    # uv_origin = (point_2d_res - 1).round()
    uv_origin = point_2d_res.astype(np.int32)

    # 3. rgb pixel to point color
    np_image = image.as_tensor().numpy()
    new_img = np_image * 1.0 / 255.0
    h, w, _ = np_image.shape
    v_in = np.where((uv_origin[:, 0] >= 0) & (uv_origin[:, 0] < w) &
                    (uv_origin[:, 1] >= 0) & (uv_origin[:, 1] < h), True, False)

    cloud.point.colors  = o3d.core.Tensor.zeros(cloud.point.positions.shape, cloud.point.positions.dtype, cloud.point.positions.device)
    # cloud.point.colors[v_in] = np.array([0.9, 0.2, 0.3])

    valid_uv_origin = uv_origin[v_in]
    cloud.point.colors[v_in] = new_img[valid_uv_origin[:, 1], valid_uv_origin[:, 0], :]

    logger.debug(f'calc_point_cloud( \n'
                 f'  cloud={type(np_cloud)} {np_cloud.dtype} {np_cloud.shape}, \n'
                 f'  image={type(np_image)} {np_image.dtype} {np_image.shape}, \n)')
    logger.debug(f'intr:\n{intr}\n')
    logger.debug(f'extr_v2c:\n{extr_v2c}\n')
    logger.debug(f'  new_cloud={type(new_cloud)} {new_cloud.dtype} {new_cloud.shape}')
    logger.debug(f'  points_4 ={type(points_4)} {points_4.dtype} {points_4.shape}')
    logger.debug(f'  point_2d_res: {type(point_2d_res)} {point_2d_res.dtype} {point_2d_res.shape}')
    logger.debug(f'  intr_mat=\n{intr_mat}')
    logger.debug(f'==> {np.amin(np_image)} {np.amax(np_image)}')
    logger.debug(f'v_in: {type(v_in)} {v_in.shape} {v_in.dtype}')
    logger.debug(f'uv_origin: {type(uv_origin)} {uv_origin.shape} {uv_origin.dtype}')

    return cloud

class AppWindow:
    MENU_OPEN            = 11
    MENU_EXPORT          = 12
    MENU_QUIT            = 13
    MENU_SETTINGS        = 21
    MENU_RESET_VIEWPORT  = 22
    MENU_ABOUT           = 31
    MENU_SHOW_ANIMATION  = 41
    MENU_SHOW_POINTCLOUD = 42
    MENU_SHOW_COORDINATE = 43
    MENU_SHOW_LABEL3D    = 44
    MENU_DEBUG_VIEWPOINT = 51
    MENU_DEBUG_STEP      = 52
    MENU_DEMO_SCREENSHOT = 61

    def __init__(self, data):
        global g_time_beg

        # config variables
        self.config_show_animation  = True
        self.config_show_pointcloud = True
        self.config_show_coordinate = True
        self.config_show_label3d    = True
        self.config_demo_screenshot = False

        # Screenshots to GIF
        self.paths_screenshot = []

        self._init_data(data)
        self._init_ui()
        self._init_menu()

        self.is_done = False
        threading.Thread(target=self._update_thread).start()

    def _init_data(self, data):
        # data for visualization
        self.coord           = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
        self.images_raw      = []
        self.images_label2d  = []
        self.pcds_t_xyzi_raw = []
        self.pcds_t_xyzi     = []
        self.pcds_t_xyzz     = []
        self.pcds_t_xyzrgbi  = []
        self.boxes_label3d   = []
        self.mesh            = []

        label2d_type_set = set()
        label3d_type_set = set()
        xyz_range        = np.zeros((3, 2), dtype=np.float32)
        xyz_range[:, 0]  = 1000
        xyz_range[:, 1]  = -1000

        for framd_id, data_frame in enumerate(data.read_frames()):
            image, cloud_t_xyzi, label2d, label3d, intr, extr_v2c, extr_v2w, cloud_xyz = data_frame
            self.images_raw.append(image)
            self.images_label2d.append(draw_2d_image_label(image, label2d))

            min_val = np.min(cloud_t_xyzi.point.positions.numpy(), axis=0)
            max_val = np.max(cloud_t_xyzi.point.positions.numpy(), axis=0)
            # print(pcolor(f'min_val: {type(min_val)} {min_val.shape} {min_val.dtype}', 'red'))
            for i in range(min_val.shape[0]):
                if xyz_range[i, 0] > min_val[i]:
                    xyz_range[i, 0] = min_val[i]
                if xyz_range[i, 1] < max_val[i]:
                    xyz_range[i, 1] = max_val[i]

            self.pcds_t_xyzi_raw.append(cloud_t_xyzi)
            t_xyzi = cloud_t_xyzi.clone()
            t_xyzi.point['__visualization_scalar'] = t_xyzi.point.intensity
            self.pcds_t_xyzi.append(t_xyzi)
            t_xyzz = cloud_t_xyzi.clone()
            t_xyzz.point['__visualization_scalar'] = t_xyzz.point.positions.numpy()[:, 2]
            self.pcds_t_xyzz.append(t_xyzz)

            t_xyzrgbi = cloud_t_xyzi.clone()
            t_xyzrgbi = calc_point_color(t_xyzrgbi, image, intr, extr_v2c)
            self.pcds_t_xyzrgbi.append(t_xyzrgbi)

            self.boxes_label3d.append(draw_3d_pointcloud_label(label3d))

            # point cloud to mesh
            if framd_id < 3:
                cloud_xyz.estimate_normals()
                radii = [0.2]
                tri_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cloud_xyz, o3d.utility.DoubleVector(radii))
                self.mesh.append(tri_mesh)

            # === Debug ===
            if framd_id == 0:
                logger.info(f'\n=== intr ===     {type(intr)}\n{intr}')
                logger.info(f'\n=== extr_v2c === {type(extr_v2c)}\n{extr_v2c}')
                logger.info(f'\n=== extr_v2w === {type(extr_v2w)}\n{extr_v2w}')
                logger.info(f'\n=== label2d ===  {type(label2d)} {len(label2d)} {type(label2d[0])}\n{label2d[0]}')
                logger.info(f'\n=== label3d ===  {type(label3d)} {len(label3d)} {type(label3d[0])}\n{label3d[0]}')

                if isinstance(cloud_t_xyzi, o3d.geometry.PointCloud):
                    logger.info(f'o3d.geometry.PointCloud:')
                    logger.info(f'  > has_points:      {cloud_t_xyzi.has_points()}')
                    logger.info(f'  > has_normals:     {cloud_t_xyzi.has_normals()}')
                    logger.info(f'  > has_colors:      {cloud_t_xyzi.has_colors()}')
                    logger.info(f'  > has_covariances: {cloud_t_xyzi.has_covariances()}')

                if isinstance(cloud_t_xyzi, o3d.t.geometry.PointCloud):
                    logger.info(f'o3d.t.geometry.PointCloud:')
                    logger.info(f'  > point.point {type(cloud_t_xyzi.point)} {cloud_t_xyzi.point.primary_key} {str(cloud_t_xyzi.point)}')
                    logger.info(f'  > intensity:  {np.amin(cloud_t_xyzi.point.intensity.numpy())} {np.amax(cloud_t_xyzi.point.intensity.numpy())} {np.mean(cloud_t_xyzi.point.intensity.numpy())}')
                    logger.info(f'  - device:     {cloud_t_xyzi.device}')
                    logger.info(f'  - is_cpu:     {cloud_t_xyzi.is_cpu}')
                    logger.info(f'  - is_cuda:    {cloud_t_xyzi.is_cuda}')
                    logger.info(f'  - material:   {cloud_t_xyzi.material}')

                # === mesh ===
                # tri_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_normal, 1.03)
                cloud_xyz.estimate_normals()
                # tri_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_normal, depth=9)
                radii = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
                tri_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cloud_xyz, o3d.utility.DoubleVector(radii))

                logger.info(f'  I> vertices:  {np.asarray(tri_mesh.vertices).shape}')
                logger.info(f'  I> triangles: {np.asarray(tri_mesh.triangles).shape}')

            for item in label2d:
                label2d_type_set.add(superclass[name2id[item["type"].lower()]])
            for item in label3d:
                label3d_type_set.add(item["type"])

        print(pcolor(f'> label2d_type_set: {label2d_type_set}', 'yellow'))
        print(pcolor(f'> label3d_type_set: {label3d_type_set}', 'yellow'))
        print(pcolor(f'Scene BoundingBox:', 'red'))
        print(pcolor(f'  > x_range: [{xyz_range[0, 0]:8.3f}, {xyz_range[0, 1]:8.3f}]', 'blue'))
        print(pcolor(f'  > y_range: [{xyz_range[1, 0]:8.3f}, {xyz_range[1, 1]:8.3f}]', 'blue'))
        print(pcolor(f'  > z_range: [{xyz_range[2, 0]:8.3f}, {xyz_range[2, 1]:8.3f}]', 'blue'))
        print(pcolor(f'>> loading data of {len(self.pcds_t_xyzi)} frames elapsed {time.time() - g_time_beg:.3f} seconds', 'red'))

    def _init_ui(self):
        # main window
        self.window = gui.Application.instance.create_window("Camera-LiDAR-Visualization", 1920, 1080)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)

        # 3D SceneWidget Grid
        self.widget3d_clear = gui.SceneWidget()
        self.widget3d_clear.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.widget3d_clear)
        self.widget3d_clear.scene.set_background([0.2, 0.2, 0.2, 1.0])

        self.widget3d_top_left = gui.SceneWidget()
        self.widget3d_top_left.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.widget3d_top_left)

        self.widget3d_top_right = gui.SceneWidget()
        self.widget3d_top_right.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.widget3d_top_right)

        self.widget3d_bottom_left = gui.SceneWidget()
        self.widget3d_bottom_left.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.widget3d_bottom_left)

        self.widget3d_bottom_right = gui.SceneWidget()
        self.widget3d_bottom_right.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.widget3d_bottom_right)

        # visualization / gui / Materials
        #   https://github.com/isl-org/Open3D/tree/v0.16.0/cpp/open3d/visualization/gui/Materials
        self.lit = rendering.MaterialRecord()
        self.lit.shader = "defaultLit"
        self.unlit = rendering.MaterialRecord()
        self.unlit.shader = "defalutUnlit"
        self.unlit_line = rendering.MaterialRecord()
        self.unlit_line.shader = "unlitLine"
        self.unlit_line.line_width = 3

        # point cloud with intensity as colormap
        self.unlit_gradient = rendering.MaterialRecord()
        self.unlit_gradient.shader = "defaultLit"

        if has_ml3d:
            # colormap = Colormap.make_rainbow()
            colormap = Colormap.make_greyscale()
            colormap = list(rendering.Gradient.Point(pt.value, pt.color + [1.0]) for pt in colormap.points)

            self.unlit_gradient.shader = "unlitGradient" # https://github.com/isl-org/Open3D/blob/v0.16.0/cpp/open3d/visualization/gui/Materials/unlitGradient.mat
            self.unlit_gradient.scalar_min = 0.0
            self.unlit_gradient.scalar_max = 1.0
            self.unlit_gradient.gradient = rendering.Gradient(colormap)
            self.unlit_gradient.gradient.mode = rendering.Gradient.GRADIENT

        # point cloud with height as colormap
        self.unlit_height = rendering.MaterialRecord()
        self.unlit_height.shader = "defaultLit"

        if has_ml3d:
            colormap = Colormap([
                Colormap.Point(0.0, [0.0, 0.0, 0.0]),
                Colormap.Point(1.0, [1.0, 0.0, 0.0])
            ])
            colormap = list(rendering.Gradient.Point(pt.value, pt.color + [1.0]) for pt in colormap.points)

            self.unlit_height.shader = "unlitGradient" # https://github.com/isl-org/Open3D/blob/v0.16.0/cpp/open3d/visualization/gui/Materials/unlitGradient.mat
            self.unlit_height.scalar_min = -1.0
            self.unlit_height.scalar_max =  3.0
            self.unlit_height.gradient = rendering.Gradient(colormap)
            self.unlit_height.gradient.mode = rendering.Gradient.GRADIENT

        # set viewport
        self.widget3d_top_left.scene.show_axes(False)
        self.widget3d_bottom_left.scene.show_axes(False)

        self.widget3d_top_right.scene.add_geometry('coord', self.coord, self.lit)
        self.widget3d_top_right.scene.show_axes(False)

        self.widget3d_bottom_right.scene.add_geometry('coord', self.coord, self.lit)
        self.widget3d_bottom_right.scene.add_geometry('mesh', self.mesh[0], self.lit)
        # self.widget3d_bottom_right.scene.show_skybox(True)
        self.widget3d_bottom_right.scene.set_background([0.5, 0.5, 0.5, 1.0])
        self._on_menu_reset_viewport()

        # set background
        self.widget3d_top_left.scene.set_background([0.5, 0.5, 1.0, 1.0])
        self.widget3d_top_right.scene.set_background([0.5, 0.5, 1.0, 1.0])
        self.widget3d_bottom_left.scene.set_background([0.5, 0.5, 1.0, 1.0])
        self.widget3d_bottom_right.scene.set_background([0.5, 0.5, 1.0, 1.0])

        # debug:
        logger.info(f'  I> widget3d.bg_color:              {self.widget3d_top_left.scene.background_color}')
        logger.info(f'  I> widget3d_top_right.bg_color:    {self.widget3d_top_right.scene.background_color}')
        logger.info(f'  I> widget3d_bottom_left.bg_color:  {self.widget3d_bottom_left.scene.background_color}')
        logger.info(f'  I> widget3d_bottom_right.bg_color: {self.widget3d_bottom_right.scene.background_color}')

        # Right panel
        em     = self.window.theme.font_size
        margin = 0.5 * em
        self.panel = gui.Vert(0.5 * em, gui.Margins(margin))
        # self.panel.add_child(gui.Label("Raw Image"))
        # self.image_raw_widget = gui.ImageWidget(self.images_raw[0])
        # self.panel.add_child(self.image_raw_widget)

        # RGB + 2D Label
        gui_label = gui.Label("2D Labeled Image")
        gui_label.text_color = gui.Color(1.0, 0.5, 0.0)
        self.panel.add_child(gui_label)
        self.image_label_widget = gui.ImageWidget(self.images_label2d[0])
        self.panel.add_child(self.image_label_widget)

        # Config Panel
        self._fileedit = gui.TextEdit()
        filedlgbutton  = gui.Button("...")
        filedlgbutton.horizontal_padding_em = 0.5
        filedlgbutton.vertical_padding_em   = 0

        fileedit_layout = gui.Horiz()
        fileedit_layout.add_child(gui.Label("Model file"))
        fileedit_layout.add_child(self._fileedit)
        fileedit_layout.add_fixed(0.25 * em)
        fileedit_layout.add_child(filedlgbutton)

        collapse_v = gui.CollapsableVert("Config", 0.33 * em, gui.Margins(em, 0, 0, 0))
        collapse_v.add_child(fileedit_layout)

        self._progress = gui.ProgressBar()
        self._progress.value = 0
        prog_layout = gui.Horiz(em)
        prog_layout.add_child(gui.Label("Cloud to Mesh Progress..."))
        prog_layout.add_child(self._progress)
        collapse_v.add_child(prog_layout)

        # tabs for config
        tabs = gui.TabControl()
        tab_layout_radio = gui.RadioButton(gui.RadioButton.VERT)
        tab_layout_radio.set_items(['four widgets', 'TL widget', 'TR widget', 'BL widget', 'BR widget'])
        tab_layout_radio.set_on_selection_changed(self._on_layout_radio_choice)
        tabs.add_tab('Layout RadioButton', tab_layout_radio)

        tab_layout_combo = gui.Combobox()
        tab_layout_combo.add_item('four widgets')
        tab_layout_combo.add_item('TL widget')
        tab_layout_combo.add_item('TR widget')
        tab_layout_combo.add_item('BL widget')
        tab_layout_combo.add_item('BR widget')
        tab_layout_combo.set_on_selection_changed(self._on_layout_combo_choice)
        tabs.add_tab('Layout Combobox', tab_layout_combo)

        collapse_v.add_child(tabs)

        self.panel.add_child(collapse_v)
        self.window.add_child(self.panel)

    def _init_menu(self):
        # ---- Menu ----
        if gui.Application.instance.menubar is None:
            # file
            file_menu = gui.Menu()
            file_menu.add_item("Open...",                  AppWindow.MENU_OPEN)
            file_menu.add_item("Export Current Image...",  AppWindow.MENU_EXPORT)
            file_menu.add_separator()
            file_menu.add_item("Quit",                     AppWindow.MENU_QUIT)

            # settings
            settings_menu = gui.Menu()
            settings_menu.add_item("Lighting & Materials", AppWindow.MENU_SETTINGS)
            settings_menu.add_item("Reset Viewport",       AppWindow.MENU_RESET_VIEWPORT)
            settings_menu.set_checked(AppWindow.MENU_SETTINGS, True)

            # help
            help_menu = gui.Menu()
            help_menu.add_item("About",                    AppWindow.MENU_ABOUT)

            # show
            show_menu = gui.Menu()
            show_menu.add_item("Show Animation",           AppWindow.MENU_SHOW_ANIMATION)
            show_menu.add_separator()
            show_menu.add_item("Show PointCloud",          AppWindow.MENU_SHOW_POINTCLOUD)
            show_menu.add_item("Show Coordinate",          AppWindow.MENU_SHOW_COORDINATE)
            show_menu.add_item("Show Label3D",             AppWindow.MENU_SHOW_LABEL3D)
            show_menu.set_checked(AppWindow.MENU_SHOW_ANIMATION,  self.config_show_animation)
            show_menu.set_checked(AppWindow.MENU_SHOW_POINTCLOUD, self.config_show_pointcloud)
            show_menu.set_checked(AppWindow.MENU_SHOW_COORDINATE, self.config_show_coordinate)
            show_menu.set_checked(AppWindow.MENU_SHOW_LABEL3D,    self.config_show_label3d)

            # debug
            debug_menu = gui.Menu()
            debug_menu.add_item("Print Viewpoint",         AppWindow.MENU_DEBUG_VIEWPOINT)
            debug_menu.add_item("Step",                    AppWindow.MENU_DEBUG_STEP)

            # demo
            demo_menu = gui.Menu()
            demo_menu.add_item("Take Screenshot",          AppWindow.MENU_DEMO_SCREENSHOT)
            demo_menu.set_checked(AppWindow.MENU_DEMO_SCREENSHOT, self.config_demo_screenshot)

            # menubar
            menu = gui.Menu()
            menu.add_menu("File",     file_menu)
            menu.add_menu("Settings", settings_menu)
            menu.add_menu("Help",     help_menu)
            menu.add_menu("Show",     show_menu)
            menu.add_menu("Debug",    debug_menu)
            menu.add_menu("Demo",     demo_menu)
            gui.Application.instance.menubar = menu

        # connect the menu items to the window
        self.window.set_on_menu_item_activated(AppWindow.MENU_QUIT,            self._on_menu_quit)
        self.window.set_on_menu_item_activated(AppWindow.MENU_RESET_VIEWPORT,  self._on_menu_reset_viewport)
        self.window.set_on_menu_item_activated(AppWindow.MENU_SHOW_ANIMATION,  self._on_menu_show_animation)
        self.window.set_on_menu_item_activated(AppWindow.MENU_SHOW_COORDINATE, self._on_menu_show_coordinate)
        self.window.set_on_menu_item_activated(AppWindow.MENU_SHOW_POINTCLOUD, self._on_menu_show_pointcloud)
        self.window.set_on_menu_item_activated(AppWindow.MENU_SHOW_LABEL3D,    self._on_menu_show_label3d)
        self.window.set_on_menu_item_activated(AppWindow.MENU_DEBUG_VIEWPOINT, self._on_menu_debug_viewpoint)
        self.window.set_on_menu_item_activated(AppWindow.MENU_DEBUG_STEP,      self._on_menu_debug_step)
        self.window.set_on_menu_item_activated(AppWindow.MENU_DEMO_SCREENSHOT, self._on_menu_demo_screenshot)

    def _on_layout(self, layout_context):
        print(pcolor(f'layout_context: {type(layout_context)}', 'yellow'))
        r    = self.window.content_rect
        gap  = 3
        panel_width = int(r.width * 0.30)
        h_3d = (r.height - 3*gap) / 2
        w_3d = (r.width - 3*gap - panel_width) / 2

        # clear
        self.widget3d_clear.frame = gui.Rect(r.x, r.y, r.width-panel_width, r.height)

        # 3D
        self.widget3d_top_left.frame     = gui.Rect(r.x+gap, r.y+gap, w_3d, h_3d)
        self.widget3d_bottom_left.frame  = gui.Rect(r.x+gap, self.widget3d_top_left.frame.get_bottom()+gap, w_3d, h_3d)
        self.widget3d_top_right.frame    = gui.Rect(self.widget3d_top_left.frame.get_right()+gap, r.y+gap, w_3d, h_3d)
        self.widget3d_bottom_right.frame = gui.Rect(self.widget3d_top_left.frame.get_right()+gap, self.widget3d_top_left.frame.get_bottom()+gap, w_3d, h_3d)
        # 2D
        self.panel.frame                 = gui.Rect(r.width - panel_width, r.y, panel_width, r.height)

    def _on_layout_combo_choice(self, new_val, new_idx):
        if new_idx == 0:
            self._on_layout(self.window.content_rect)
        elif 1 <= new_idx <= 4:
            widgets = [None, self.widget3d_top_left, self.widget3d_top_right, self.widget3d_bottom_left, self.widget3d_bottom_right]

            r    = self.window.content_rect
            panel_width = int(r.width * 0.30)
            for i in range(1, len(widgets)):
                focus_widget3d = widgets[i]
                if i == new_idx:
                    focus_widget3d.frame = gui.Rect(r.x, r.y, r.width - panel_width, r.height)
                else:
                    focus_widget3d.frame = gui.Rect(r.x, r.y, 0, 0)
            self.panel.frame     = gui.Rect(r.width - panel_width, r.y, panel_width, r.height)
        else:
            raise ValueError(f'Unexpected value')

    def _on_layout_radio_choice(self, idx):
        self._on_layout_combo_choice('radio choice', idx)

    def _on_close(self):
        self.is_done = True
        return True

    def _on_menu_quit(self):
        self.is_done = True
        gui.Application.instance.quit()

    def _on_menu_reset_viewport(self):
        self.widget3d_top_left.scene.camera.look_at([70,0,0], [-30,0,50], [100,0,50]) # look_at(center, eye, up)
        self.widget3d_bottom_left.scene.camera.look_at([70,0,0], [-30,0,50], [100,0,50])

        self.widget3d_top_right.scene.camera.look_at([70,0,0], [-30,0,50], [100,0,50])
        # self.widget3d_top_right.setup_camera(75, self.widget3d_top_right.scene.bounding_box, (0, 0, 0))
        self.widget3d_bottom_right.scene.camera.look_at([70,0,0], [-30,0,50], [100,0,50])

    def _on_menu_show_animation(self):
        self.config_show_animation = not self.config_show_animation
        gui.Application.instance.menubar.set_checked(AppWindow.MENU_SHOW_ANIMATION, self.config_show_animation)

    def _on_menu_show_pointcloud(self):
        self.config_show_pointcloud = not self.config_show_pointcloud
        gui.Application.instance.menubar.set_checked(AppWindow.MENU_SHOW_POINTCLOUD, self.config_show_pointcloud)

    def _on_menu_show_coordinate(self):
        self.config_show_coordinate = not self.config_show_coordinate
        gui.Application.instance.menubar.set_checked(AppWindow.MENU_SHOW_COORDINATE, self.config_show_coordinate)

    def _on_menu_show_label3d(self):
        self.config_show_label3d = not self.config_show_label3d
        gui.Application.instance.menubar.set_checked(AppWindow.MENU_SHOW_LABEL3D, self.config_show_label3d)

    def _on_menu_debug_viewpoint(self):
        # todo: manually find the best viewpoint
        #       ref: https://github.com/isl-org/Open3D/issues/1483#issuecomment-582121615
        model_mat = self.widget3d_top_left.scene.camera.get_model_matrix()
        proj_mat = self.widget3d_top_left.scene.camera.get_projection_matrix()
        view_mat = self.widget3d_top_left.scene.camera.get_view_matrix()

        print(pcolor(f'=== model_mat ===', 'blue'))
        print(f'{model_mat}')
        print(pcolor(f'=== proj_mat ===', 'blue'))
        print(f'{proj_mat}')
        print(pcolor(f'=== view_mat ===', 'blue'))
        print(f'{view_mat}')

    def _on_menu_debug_step(self):
        print(pcolor(f"_on_menu_debug_step(..)", 'red'))

    def _on_menu_demo_screenshot(self):
        self.config_demo_screenshot = not self.config_demo_screenshot
        gui.Application.instance.menubar.set_checked(AppWindow.MENU_DEMO_SCREENSHOT, self.config_demo_screenshot)
        win_sz = self.window.content_rect
        print(pcolor(f'content_rect: {win_sz.x} {win_sz.y} {win_sz.width} {win_sz.height}', 'yellow'))

        if not self.config_demo_screenshot:
            self._create_gif()

    def _create_gif(self):
        gif_name = osp.join('/mnt/datax/temp', 'DAIR-V2X.gif')
        frames = []
        for item in self.paths_screenshot:
            new_frame = PIL.Image.open(item)
            frames.append(new_frame)
        print(pcolor(f'writing {gif_name}', 'red'))
        frames[0].save(gif_name, format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0, comment=b'DAIR V2X Visualization')
        self.paths_screenshot = []

    def _update_thread(self):
        idx = 0
        while not self.is_done:
            time.sleep(0.100)

            image_raw_frame   = self.images_raw[idx]
            image_label_frame = self.images_label2d[idx]
            pcd_t_xyzi        = self.pcds_t_xyzi[idx]
            pcd_t_xyzz        = self.pcds_t_xyzz[idx]
            pcd_t_xyzrgbi     = self.pcds_t_xyzrgbi[idx]
            boxes3d           = self.boxes_label3d[idx]

            if self.config_show_animation:
                idx += 1
                if idx >= len(self.images_raw):
                    idx = 0

            if self.config_demo_screenshot:
                win_sz = self.window.content_rect
                screen = pyscreenshot.grab(bbox=(win_sz.x, 0, win_sz.width, win_sz.height), childprocess=False)
                save_name = osp.join('/mnt/datax/temp', f'frame-{idx:03d}.png')
                print(pcolor(f'  write {save_name}', 'blue'))
                screen.save(save_name)
                if save_name not in self.paths_screenshot:
                    self.paths_screenshot.append(save_name)

            def update():
                # self.image_raw_widget.update_image(image_raw_frame)
                self.image_label_widget.update_image(image_label_frame)

                self.widget3d_top_left.scene.clear_geometry()
                self.widget3d_top_right.scene.clear_geometry()
                self.widget3d_bottom_left.scene.clear_geometry()
                self.widget3d_bottom_right.scene.clear_geometry()

                if self.config_show_pointcloud:
                    self.widget3d_top_left.scene.add_geometry('pointcloud', pcd_t_xyzi, self.unlit_gradient)
                    self.widget3d_top_right.scene.add_geometry('pointcloud', pcd_t_xyzrgbi, self.unlit)
                    # self.widget3d_bottom_left.scene.add_geometry('pointcloud', pcd_t_xyzrgbi, self.lit)
                    self.widget3d_bottom_left.scene.add_geometry('pointcloud', pcd_t_xyzz, self.unlit_height)

                if self.config_show_coordinate:
                    self.widget3d_top_left.scene.add_geometry('coord', self.coord, self.lit)
                    self.widget3d_top_right.scene.add_geometry('coord', self.coord, self.lit)
                    self.widget3d_bottom_left.scene.add_geometry('coord', self.coord, self.lit)
                    self.widget3d_bottom_right.scene.add_geometry('coord', self.coord, self.lit)

                if self.config_show_label3d:
                    for box_id, box_lineset in enumerate(boxes3d):
                        self.widget3d_top_left.scene.add_geometry(f'bbox-{box_id:03d}', box_lineset, self.unlit_line)
                        self.widget3d_top_right.scene.add_geometry(f'bbox-{box_id:03d}', box_lineset, self.unlit_line)
                        self.widget3d_bottom_left.scene.add_geometry(f'bbox-{box_id:03d}', box_lineset, self.unlit_line)

                self.widget3d_bottom_right.scene.add_geometry('mesh', self.mesh[0], self.lit)

            if not self.is_done:
                gui.Application.instance.post_to_main_thread(self.window, update)

def main():
    global g_time_beg
    g_time_beg = time.time()
    app = o3d.visualization.gui.Application.instance
    app.initialize()

    path = get_path()
    app_win = AppWindow(path)

    app.run()

def debug_analysis_pcd():
    """
    git clone --recursive git@github.com:klintan/pypcd.git
    cd pypcd
    python setup.py install
    pip install python-lzf -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
    """

    from pypcd import pypcd
    if isLinux:
        str_root = '/mnt/datax/Datasets/DAIR-V2X-Extracted/cooperative-vehicle-infrastructure-example'
    else:
        str_root = 'D:/0-DAIR-V2X-Dataset/DAIR-V2X-C-Example/cooperative-vehicle-infrastructure-example'

    inf_pcd_path = osp.join(str_root, 'infrastructure-side/velodyne/000009.pcd')
    veh_pcd_path = osp.join(str_root, 'vehicle-side/velodyne/015344.pcd')
    inf_pcd = pypcd.PointCloud.from_path(inf_pcd_path)
    veh_pcd = pypcd.PointCloud.from_path(veh_pcd_path)
    print(pcolor(f'inf_pcd: {type(inf_pcd)} {type(inf_pcd.pc_data)} {inf_pcd.pc_data.shape} {inf_pcd.fields}', 'cyan'))
    print(pcolor(f'veh_pcd: {type(veh_pcd)} {type(veh_pcd.pc_data)} {veh_pcd.pc_data.shape} {veh_pcd.fields}', 'blue'))


def debug_create_gif():
    path = '/mnt/datax/temp/demo-v1'
    save_fig = f'{path}/DAIR-V2X.gif'

    frames = []
    files = sorted(glob.glob(f'{path}/*.png'))
    for item in files:
        new_frame = PIL.Image.open(item)
        frames.append(new_frame)
    print(pcolor(f'writing {save_fig}', 'red'))
    frames[0].save(save_fig, format='GIF', append_images=frames[1:], save_all=True, duration=200, loop=0, comment=b'DAIR V2X Visualization')


if __name__ == "__main__":
    print(pcolor(f'sys.version:        {sys.version}', 'yellow'))
    print(pcolor(f'cv2.__version__:    {cv2.__version__}', 'yellow'))
    print(pcolor(f'open3d.__version__: {o3d.__version__}\n', 'yellow'))

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    setup_log()
    main()
    # debug_analysis_pcd()
    # debug_create_gif()
