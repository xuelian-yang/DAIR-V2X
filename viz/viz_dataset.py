# -*- coding: utf-8 -*-

"""
References
----------
    Create a collapsible vertical widget
        https://github.com/isl-org/Open3D/blob/v0.16.0/examples/python/visualization/non_english.py#L186
    Docs » open3d.visualization » open3d.visualization.gui
        Docs » open3d.visualization » open3d.visualization.gui
    How to set min/max for colormap when rendering a pointcloud? #2545
        https://github.com/isl-org/Open3D/issues/2545#issuecomment-987119956
    look_at(center, eye, up)
        http://www.open3d.org/docs/release/python_api/open3d.visualization.rendering.Camera.html#open3d.visualization.rendering.Camera.look_at
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
    raise ImportError(
        'Please run "pip install open3d" to install open3d first.')

has_ml3d = importlib.util.find_spec('open3d._ml3d')
if has_ml3d:
    print(colored(f'>>> has _ml3d <<<', 'red'))
    from open3d.ml.vis import Colormap
else:
    print(colored(f'>>> no _ml3d <<<', 'red'))

"""
Open3D-ML
  http://www.open3d.org/docs/release/compilation.html

mkdir build
cd build

cmake -DBUILD_CUDA_MODULE=ON \
      -DGLIBCXX_USE_CXX11_ABI=OFF \
      -DBUILD_PYTORCH_OPS=ON \
      -DBUILD_TENSORFLOW_OPS=ON \
      -DBUNDLE_OPEN3D_ML=ON \
      -DPython3_ROOT=/home/sigma/anaconda3/envs/o3d-ml/bin/python \
      -DOPEN3D_ML_ROOT=/mnt/datax/github-install/Open3D-ML \
      ..

# Install the python wheel with pip
make -j install-pip-package
"""

logger = logging.getLogger(__name__)
# https://stackoverflow.com/a/1857
isLinux = (platform.system() == "Linux")
isMacOS = (platform.system() == "Darwin")
isWindows = (platform.system() == "Windows")

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
    medium_format_new = (
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
        format=medium_format_new
    )
    logging.info('@{} created at {}'.format(get_log_file, dt_now))
    print('@{} created at {}'.format(get_log_file, dt_now))


class PathConfig:
    def __init__(self, path_cfg):
        self.image_paths = path_cfg['image_paths']
        self.point_paths = path_cfg['point_paths']
        self.intr_paths = path_cfg['intr_paths']
        self.extr_v2c_paths = path_cfg['extr_v2c_paths']
        self.extr_v2w_paths = path_cfg['extr_v2w_paths']
        self.label2d_paths = path_cfg['label2d_paths']
        self.label3d_paths = path_cfg['label3d_paths']

        self.num = len(self.image_paths)
        assert len(self.point_paths) == self.num \
            and len(self.intr_paths) == self.num \
            and len(self.extr_v2c_paths) == self.num \
            and len(self.extr_v2w_paths) == self.num \
            and len(self.label2d_paths) == self.num \
            and len(self.label3d_paths) == self.num

    def read_frame(self, k):
        assert 0 <= k < self.num

        def load_json(path, debug=False):
            with open(path, mode="r") as f:
                data = json.load(f)
            if debug:
                logging.info(f'load_json({path})')
            return data

        # image = o3d.io.read_image(self.image_paths[k]) 
        image = o3d.t.io.read_image(self.image_paths[k])
        # point = o3d.io.read_point_cloud(self.point_paths[k])
        point = o3d.t.io.read_point_cloud(self.point_paths[k])
        # point.point.colors = o3d.core.Tensor.zeros(point.point.positions.shape, point.point.positions.dtype, point.point.positions.device)
        point.point.colors = o3d.core.Tensor.full(point.point.positions.shape, 0.3, point.point.positions.dtype, point.point.positions.device)

        '''
        # http://www.open3d.org/docs/release/python_api/open3d.t.geometry.PointCloud.html#open3d-t-geometry-pointcloud
        device = o3d.core.Device("CPU:0")
        dtype = o3d.core.float32
        pcd = o3d.t.geometry.PointCloud(device)
        pcd.point.positions = o3d.core.Tensor([[0, 0, 0],
                                                [1, 1, 1],
                                                [2, 2, 2]], dtype, device)
        pcd.point.positions = o3d.core.Tensor([[0, 0, 0],
                                                [1, 1, 1],
                                                [2, 2, 2]], dtype, device)
        pcd.point.positions = o3d.core.Tensor([[0, 0, 0],
                                                [1, 1, 1],
                                                [2, 2, 2]], dtype, device)
        pcd.point.intensities = o3d.core.Tensor([0.3, 0.1, 0.4], dtype, device)
        pcd.point.labels = o3d.core.Tensor([3, 1, 4], o3d.core.int32, device)
        new_pcd = o3d.t.geometry.PointCloud(device)
        new_pcd.point.positions = point.point.positions
        new_pcd.point.intensity = point.point.intensity
        new_pcd.point.colors = o3d.core.Tensor.zeros(point.point.positions.shape, dtype, device)
        point = new_pcd
        '''

        point_normal = o3d.io.read_point_cloud(self.point_paths[k])
        label2d = load_json(self.label2d_paths[k])
        label3d = load_json(self.label3d_paths[k])
        intr = load_json(self.intr_paths[k])
        extr_v2c = load_json(self.extr_v2c_paths[k])
        extr_v2w = load_json(self.extr_v2w_paths[k])

        if isinstance(point, o3d.t.geometry.PointCloud):
            point.point['__visualization_scalar'] = point.point.intensity
        return image, point, label2d, label3d, intr, extr_v2c, extr_v2w, point_normal

    def read_frames(self):
        data = []
        for i in range(self.num):
            data.append(self.read_frame(i))
        return data


def get_path():
    if isLinux:
        str_root = '/mnt/datax/Datasets/DAIR-V2X-Extracted/cooperative-vehicle-infrastructure-example/infrastructure-side'
    else:
        str_root = 'D:/0-DAIR-V2X-Dataset/DAIR-V2X-C-Example/cooperative-vehicle-infrastructure-example/infrastructure-side'
    dict_name = {
        'image_paths': 'image',                           # jpg
        'point_paths': 'velodyne',                        # pcd
        'intr_paths': 'calib/camera_intrinsic',           # json
        'extr_v2c_paths': 'calib/virtuallidar_to_camera', # json
        'extr_v2w_paths': 'calib/virtuallidar_to_world',  # json
        'label2d_paths': 'label/camera',                  # json
        'label3d_paths': 'label/virtuallidar'             # json
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
        rot_mat = o3d.geometry.get_rotation_matrix_from_xyz(yaw)
        box3d = o3d.geometry.OrientedBoundingBox(center_lidar, rot_mat, obj_size)
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
    new_cloud = np.array(new_cloud.T)
    points_num = list(new_cloud.shape)[:-1]
    points_shape = np.concatenate([points_num, [1]], axis=0)
    points_2d_shape = np.concatenate([points_num, [3]], axis=0)
    points_4 = np.concatenate((new_cloud, np.ones(points_shape)), axis=-1)
    point_2d = np.matmul(intr_mat, points_4.T)
    point_2d = point_2d.T.reshape(points_2d_shape)
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
    # uv_origin = (point_2d_res - 1).round()
    uv_origin = point_2d_res.astype(np.int32)

    np_image = image.as_tensor().numpy()
    new_img = np_image * 1.0 / 255.0
    h, w, _ = np_image.shape
    v_in = np.where((uv_origin[:, 0] >= 0) & (uv_origin[:, 0] < w) &
                    (uv_origin[:, 1] >= 0) & (uv_origin[:, 1] < h), True, False)

    cloud.point.colors = o3d.core.Tensor.zeros(cloud.point.positions.shape, cloud.point.positions.dtype, cloud.point.positions.device)
    tensor_colors = o3d.core.Tensor.zeros(cloud.point.positions.shape, cloud.point.positions.dtype, cloud.point.positions.device)
    tensor_colors[v_in] = np.array([0.9, 0.2, 0.3])
    cloud.point.colors[v_in] = np.array([0.9, 0.2, 0.3])

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
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21
    MENU_ANIMATION = 31
    MENU_COORDINATE = 32
    MENU_VIEWPOINT = 33
    MENU_SCREENSHOT = 34

    def __init__(self, data):
        global g_time_beg

        # config variables
        self.config_animation = True
        self.config_coordinate = True
        self.config_screenshot = False

        # Screenshots to GIF
        self.paths_screenshot = []

        self._init_data(data)
        self._init_ui()
        self._init_menu()

        self.is_done = False
        threading.Thread(target=self._update_thread).start()

    def _init_data(self, data):
        # data for visualization
        self.rgb_images = []
        self.rgb_label2d_images = []
        self.pcd = []
        self.pcd_label3d = []
        self.mesh = []

        label2d_type_set = set()
        label3d_type_set = set()

        for framd_id, data_frame in enumerate(data.read_frames()):
            image, point, label2d, label3d, intr, extr_v2c, extr_v2w, point_normal = data_frame
            self.rgb_images.append(image)
            # self.pcd.append(point)
            point = calc_point_color(point, image, intr, extr_v2c)
            self.pcd.append(point)
            self.rgb_label2d_images.append(draw_2d_image_label(image, label2d))

            # point cloud to mesh
            if framd_id < 3:
                point_normal.estimate_normals()
                radii = [0.2]
                tri_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(point_normal, o3d.utility.DoubleVector(radii))
                self.mesh.append(tri_mesh)

            if framd_id == 0: # debug
                # mesh
                # tri_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_normal, 1.03)
                point_normal.estimate_normals()
                # tri_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_normal, depth=9)
                radii = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
                tri_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(point_normal, o3d.utility.DoubleVector(radii))

                logger.info(f'  I> vertices:  {np.asarray(tri_mesh.vertices).shape}')
                logger.info(f'  I> triangles: {np.asarray(tri_mesh.triangles).shape}')

                logger.info(f'\n=== intr === {type(intr)}\n{intr}')
                logger.info(f'\n=== extr_v2c === {type(extr_v2c)}\n{extr_v2c}')
                logger.info(f'\n=== extr_v2w === {type(extr_v2w)}\n{extr_v2w}')
                logger.info(f'{type(label3d)} {len(label3d)} {type(label3d[0])}\n{label3d[0]}')
                draw_3d_pointcloud_label(label3d)

                if isinstance(point, o3d.geometry.PointCloud):
                    logger.info(f'o3d.geometry.PointCloud:')
                    logger.info(f'  has_points:      {point.has_points()}')
                    logger.info(f'  has_normals:     {point.has_normals()}')
                    logger.info(f'  has_colors:      {point.has_colors()}')
                    logger.info(f'  has_covariances: {point.has_covariances()}')

                if isinstance(point, o3d.t.geometry.PointCloud):
                    logger.info(f'o3d.t.geometry.PointCloud:')
                    logger.info(f'  > point.point {type(point.point)} {point.point.primary_key} {str(point.point)}')
                    logger.info(f'  > intensity: {np.amin(point.point.intensity.numpy())} {np.amax(point.point.intensity.numpy())} {np.mean(point.point.intensity.numpy())}')
                    logger.info(f'  - device:   {point.device}')
                    logger.info(f'  - is_cpu:   {point.is_cpu}')
                    logger.info(f'  - is_cuda:  {point.is_cuda}')
                    logger.info(f'  - material: {point.material}')

                # fusion point cloud with image
                point = calc_point_color(point, image, intr, extr_v2c)
                self.pcd[0] = point

            for item in label2d:
                label2d_type_set.add(superclass[name2id[item["type"].lower()]])
            for item in label3d:
                label3d_type_set.add(item["type"])
            self.pcd_label3d.append(draw_3d_pointcloud_label(label3d))

        logger.info(f'> label2d_type_set: {label2d_type_set}')
        logger.info(f'> label3d_type_set: {label3d_type_set}')
        self.num = len(self.pcd)
        print(pcolor(f'>> loading data of {self.num} frames elapsed {time.time() - g_time_beg:.3f} seconds', 'red'))

        # 3D geometry
        self.coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
        self.geometry = o3d.geometry.PointCloud()

    def _init_ui(self):
        # main window
        self.window = gui.Application.instance.create_window("DAIR-V2X", 1920, 1080)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)

        # 3D SceneWidget Grid
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

        self.lit = rendering.MaterialRecord()
        self.lit.shader = "defaultLit"
        self.lit_line = rendering.MaterialRecord()
        self.lit_line.shader = "unlitLine"
        self.lit_line.line_width = 3

        self.lit_pc_rgb = rendering.MaterialRecord()
        self.lit_pc_rgb.shader = "defaultLit"

        self.lit_pc_intensity = rendering.MaterialRecord()
        self.lit_pc_intensity.shader = "defaultLit"

        if has_ml3d:
            # colormap = Colormap.make_rainbow()
            colormap = Colormap.make_greyscale()
            colormap = list(rendering.Gradient.Point(pt.value, pt.color + [1.0]) for pt in colormap.points)

            self.lit_pc_intensity.shader = "unlitGradient"
            self.lit_pc_intensity.scalar_min = 0.0
            self.lit_pc_intensity.scalar_max = 1.0
            self.lit_pc_intensity.gradient = rendering.Gradient(colormap)
            self.lit_pc_intensity.gradient.mode = rendering.Gradient.GRADIENT

        self.widget3d_top_left.scene.show_axes(False)
        self.widget3d_top_left.scene.camera.look_at([70,0,0], [-30,0,50], [100,0,50]) # look_at(center, eye, up)

        self.widget3d_top_right.scene.set_background([0.1, 0.1, 0.5, 0.5])
        self.widget3d_top_right.scene.add_geometry('coord', self.coord, self.lit)
        self.widget3d_top_right.setup_camera(75, self.widget3d_top_right.scene.bounding_box, (0, 0, 0))
        # self.widget3d_top_right.scene.show_axes(True)
        self.widget3d_top_right.scene.show_skybox(True)

        self.widget3d_bottom_left.scene.add_geometry('coord', self.coord, self.lit)
        # self.widget3d_bottom_left.setup_camera(75, self.widget3d_bottom_left.scene.bounding_box, (0, 0, 0))
        self.widget3d_bottom_left.scene.camera.look_at([70,0,0], [-30,0,50], [100,0,50]) # look_at(center, eye, up)

        self.widget3d_bottom_right.scene.add_geometry('coord', self.coord, self.lit)
        self.widget3d_bottom_right.setup_camera(75, self.widget3d_bottom_right.scene.bounding_box, (0, 0, 0))
        self.widget3d_bottom_right.scene.add_geometry('mesh', self.mesh[0], self.lit)
        self.widget3d_bottom_right.setup_camera(75, self.widget3d_bottom_right.scene.bounding_box, (0, 0, 0))
        # self.widget3d_bottom_right.scene.show_skybox(True)
        self.widget3d_bottom_right.scene.set_background([0.1, 0.1, 0.5, 1.0])
        self.widget3d_bottom_right.scene.camera.look_at([70,0,0], [-30,0,50], [100,0,50]) # look_at(center, eye, up)

        # debug:
        logger.info(f'  I> widget3d.bg_color:              {self.widget3d_top_left.scene.background_color}')
        logger.info(f'  I> widget3d_top_right.bg_color:    {self.widget3d_top_right.scene.background_color}')
        logger.info(f'  I> widget3d_bottom_left.bg_color:  {self.widget3d_bottom_left.scene.background_color}')
        logger.info(f'  I> widget3d_bottom_right.bg_color: {self.widget3d_bottom_right.scene.background_color}')

        # Right panel
        em = self.window.theme.font_size
        margin = 0.5 * em
        self.panel = gui.Vert(0.5 * em, gui.Margins(margin))
        self.panel.add_child(gui.Label("Color Image"))
        self.rgb_widget = gui.ImageWidget(self.rgb_images[0])
        self.panel.add_child(self.rgb_widget)

        # RGB + 2D Label
        self.panel.add_child(gui.Label("Color Image + Label2D"))
        self.rgb_label2d_widget = gui.ImageWidget(self.rgb_images[0])
        self.panel.add_child(self.rgb_label2d_widget)

        # Config Panel
        self._fileedit = gui.TextEdit()
        filedlgbutton = gui.Button("...")
        filedlgbutton.horizontal_padding_em = 0.5
        filedlgbutton.vertical_padding_em = 0

        fileedit_layout = gui.Horiz()
        fileedit_layout.add_child(gui.Label("Model file"))
        fileedit_layout.add_child(self._fileedit)
        fileedit_layout.add_fixed(0.25 * em)
        fileedit_layout.add_child(filedlgbutton)

        collapse = gui.CollapsableVert("Config", 0.33 * em, gui.Margins(em, 0, 0, 0))
        collapse.add_child(fileedit_layout)

        self._progress = gui.ProgressBar()
        self._progress.value = 0
        prog_layout = gui.Horiz(em)
        prog_layout.add_child(gui.Label("Cloud to Mesh Progress..."))
        prog_layout.add_child(self._progress)
        collapse.add_child(prog_layout)

        self.panel.add_child(collapse)
        self.window.add_child(self.panel)

    def _init_menu(self):
        # ---- Menu ----
        if gui.Application.instance.menubar is None:
            # file
            file_menu = gui.Menu()
            file_menu.add_item("Open...", AppWindow.MENU_OPEN)
            file_menu.add_item("Export Current Image...", AppWindow.MENU_EXPORT)
            file_menu.add_separator()
            file_menu.add_item("Quit", AppWindow.MENU_QUIT)
            # settings
            settings_menu = gui.Menu()
            settings_menu.add_item("Lighting & Materials",
                                   AppWindow.MENU_SHOW_SETTINGS)
            settings_menu.set_checked(AppWindow.MENU_SHOW_SETTINGS, True)
            # help
            help_menu = gui.Menu()
            help_menu.add_item("About", AppWindow.MENU_ABOUT)
            # debug
            debug_menu = gui.Menu()
            debug_menu.add_item("Enable Animation", AppWindow.MENU_ANIMATION)
            debug_menu.set_checked(AppWindow.MENU_ANIMATION, True)
            debug_menu.add_item("Show Coordinate", AppWindow.MENU_COORDINATE)
            debug_menu.set_checked(AppWindow.MENU_COORDINATE, True)
            debug_menu.add_separator()
            debug_menu.add_item("Show Viewpoint", AppWindow.MENU_VIEWPOINT)
            debug_menu.add_item("Take Screenshot", AppWindow.MENU_SCREENSHOT)

            # menubar
            menu = gui.Menu()
            menu.add_menu("File", file_menu)
            menu.add_menu("Settings", settings_menu)
            menu.add_menu("Help", help_menu)
            menu.add_menu("Debug", debug_menu)
            gui.Application.instance.menubar = menu

        # connect the menu items to the window
        self.window.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        self.window.set_on_menu_item_activated(AppWindow.MENU_ANIMATION, self._on_menu_animation)
        self.window.set_on_menu_item_activated(AppWindow.MENU_COORDINATE, self._on_menu_coordinate)
        self.window.set_on_menu_item_activated(AppWindow.MENU_VIEWPOINT, self._on_menu_viewpoint)
        self.window.set_on_menu_item_activated(AppWindow.MENU_SCREENSHOT, self._on_menu_screenshot)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        gap = 3
        panel_width = int(r.width * 0.30)
        h_3d = (r.height - 3*gap) / 2
        w_3d = (r.width - 3*gap - panel_width) / 2

        # 3D
        self.widget3d_top_left.frame              = gui.Rect(r.x+gap, r.y+gap, w_3d, h_3d)
        self.widget3d_bottom_left.frame  = gui.Rect(r.x+gap, self.widget3d_top_left.frame.get_bottom()+gap, w_3d, h_3d)
        self.widget3d_top_right.frame    = gui.Rect(self.widget3d_top_left.frame.get_right()+gap, r.y+gap, w_3d, h_3d)
        self.widget3d_bottom_right.frame = gui.Rect(self.widget3d_top_left.frame.get_right()+gap, self.widget3d_top_left.frame.get_bottom()+gap, w_3d, h_3d)
        # 2D
        self.panel.frame                 = gui.Rect(r.width - panel_width, r.y, panel_width, r.height)

    def _on_close(self):
        self.is_done = True
        return True

    def _on_menu_quit(self):
        self.is_done = True
        gui.Application.instance.quit()

    def _on_menu_animation(self):
        self.config_animation = not self.config_animation
        gui.Application.instance.menubar.set_checked(AppWindow.MENU_ANIMATION, self.config_animation)

    def _on_menu_coordinate(self):
        self.config_coordinate = not self.config_coordinate
        gui.Application.instance.menubar.set_checked(AppWindow.MENU_COORDINATE, self.config_coordinate)

    def _on_menu_viewpoint(self):
        # todo: manually find best viewpoint
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

    def _on_menu_screenshot(self):
        self.config_screenshot = not self.config_screenshot
        gui.Application.instance.menubar.set_checked(AppWindow.MENU_SCREENSHOT, self.config_screenshot)
        win_sz = self.window.content_rect
        print(pcolor(f'content_rect: {win_sz.x} {win_sz.y} {win_sz.width} {win_sz.height}', 'yellow'))

        if not self.config_screenshot:
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

            rgb_frame = self.rgb_images[idx]
            rgb_label2d_frame = self.rgb_label2d_images[idx]
            pcd = self.pcd[idx]
            pcd_label = self.pcd_label3d[idx]

            if self.config_animation:
                idx += 1
                if idx >= len(self.rgb_images):
                    idx = 0

            if self.config_screenshot:
                win_sz = self.window.content_rect
                screen = pyscreenshot.grab(bbox=(win_sz.x, 0, win_sz.width, win_sz.height), childprocess=False)
                save_name = osp.join('/mnt/datax/temp', f'frame-{idx:03d}.png')
                print(pcolor(f'  write {save_name}', 'blue'))
                screen.save(save_name)
                self.paths_screenshot.append(save_name)

            def update():
                self.rgb_widget.update_image(rgb_frame)
                self.rgb_label2d_widget.update_image(rgb_label2d_frame)

                # top left: point cloud with intensity
                self.widget3d_top_left.scene.clear_geometry()
                if self.config_coordinate:
                    self.widget3d_top_left.scene.add_geometry('coord', self.coord, self.lit)
                self.widget3d_top_left.scene.add_geometry('pointcloud', pcd, self.lit_pc_intensity)
                for box_id, box_lineset in enumerate(pcd_label):
                    self.widget3d_top_left.scene.add_geometry(f'bbox-{box_id:03d}', box_lineset, self.lit_line)

                # top right
                self.widget3d_top_right.scene.clear_geometry()
                self.widget3d_top_right.scene.add_geometry('coord', self.coord, self.lit)

                # bottom left: point cloud with rgb color from image
                self.widget3d_bottom_left.scene.clear_geometry()
                if self.config_coordinate:
                    self.widget3d_bottom_left.scene.add_geometry('coord', self.coord, self.lit)
                self.widget3d_bottom_left.scene.add_geometry('pointcloud', pcd, self.lit_pc_rgb)
                for box_id, box_lineset in enumerate(pcd_label):
                    self.widget3d_bottom_left.scene.add_geometry(f'bbox-{box_id:03d}', box_lineset, self.lit_line)

                # bottom right
                self.widget3d_bottom_right.scene.clear_geometry()
                self.widget3d_bottom_right.scene.add_geometry('coord', self.coord, self.lit)
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

def analysis_pcd():
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


if __name__ == "__main__":
    print(pcolor(f'sys.version:        {sys.version}', 'yellow'))
    print(pcolor(f'cv2.__version__:    {cv2.__version__}', 'yellow'))
    print(pcolor(f'open3d.__version__: {o3d.__version__}\n', 'yellow'))

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    setup_log()
    main()
    # analysis_pcd()
