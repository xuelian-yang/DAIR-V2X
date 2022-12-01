# -*- coding: utf-8 -*-

"""
References
----------
    look_at(center, eye, up)
        http://www.open3d.org/docs/release/python_api/open3d.visualization.rendering.Camera.html#open3d.visualization.rendering.Camera.look_at
    Update geometry using open3d.visualization.rendering.Open3DScene
        https://github.com/isl-org/Open3D/issues/2869#issuecomment-761942166
    Open3D example - video.py
        https://github.com/isl-org/Open3D/blob/master/examples/python/visualization/video.py

pip install pyscreenshot
"""

import copy
import cv2
import json
import logging
import numpy as np
import os
import os.path as osp
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
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (0, 255, 255),
}

def pcolor(string, color, on_color=None, attrs=None):
    return colored(string, color, on_color, attrs)


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

        image = o3d.t.io.read_image(self.image_paths[k])
        point = o3d.io.read_point_cloud(self.point_paths[k])
        label2d = load_json(self.label2d_paths[k])
        label3d = load_json(self.label3d_paths[k])
        intr = load_json(self.intr_paths[k])
        extr_v2c = load_json(self.extr_v2c_paths[k])
        extr_v2w = load_json(self.extr_v2w_paths[k])
        return image, point, label2d, label3d, intr, extr_v2c, extr_v2w

    def read_frames(self):
        data = []
        for i in range(self.num):
            data.append(self.read_frame(i))
        return data


def get_path():
    str_root = '/mnt/datax/Datasets/DAIR-V2X-Extracted/cooperative-vehicle-infrastructure-example/infrastructure-side'
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


class AppWindow:
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21
    MENU_VIEWPOINT = 31
    MENU_SCREENSHOT = 32

    def __init__(self, data):
        # config
        self.config_screenshot = False

        global g_time_beg

        self.rgb_images = []
        self.rgb_label2d_images = []
        self.pcd = []
        label2d_type_set = set()
        label3d_type_set = set()
        for framd_id, data_frame in enumerate(data.read_frames()):
            image, point, label2d, label3d, intr, extr_v2c, extr_v2w = data_frame
            self.rgb_images.append(image)
            self.pcd.append(point)
            self.rgb_label2d_images.append(draw_2d_image_label(image, label2d))
            if framd_id == 0:
                print(f'{type(label2d)} {len(label2d)} {type(label2d[0])}')
                # pprint.pprint(label2d[0])

            for item in label2d:
                label2d_type_set.add(superclass[name2id[item["type"].lower()]])
            for item in label3d:
                label3d_type_set.add(item["type"])

        print(pcolor(f'label2d_type_set: {label2d_type_set}', 'magenta'))
        print(pcolor(f'label3d_type_set: {label3d_type_set}', 'magenta'))
        self.num = len(self.pcd)
        print(pcolor(f'loading data of {self.num} frames elapsed {time.time() - g_time_beg:.3f} seconds', 'cyan'))

        self.geometry = o3d.geometry.PointCloud()

        self.window = gui.Application.instance.create_window("DAIR-V2X", 1920, 1080)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)

        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.widget3d)

        self.lit = rendering.MaterialRecord()
        self.lit.shader = "defaultLit"
        self.widget3d.scene.show_axes(True)
        self.widget3d.scene.camera.look_at([70,0,0], [0,0,150], [1,0,1]) # look_at(center, eye, up)

        em = self.window.theme.font_size
        margin = 0.5 * em
        self.panel = gui.Vert(0.5 * em, gui.Margins(margin))
        self.panel.add_child(gui.Label("Color image"))
        self.rgb_widget = gui.ImageWidget(self.rgb_images[0])
        self.panel.add_child(self.rgb_widget)

        # RGB + 2D Label
        self.panel.add_child(gui.Label("Color image + label2d"))
        self.rgb_label2d_widget = gui.ImageWidget(self.rgb_images[0])
        self.panel.add_child(self.rgb_label2d_widget)
        self.window.add_child(self.panel)

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
            debug_menu.add_item("Show viewpoint", AppWindow.MENU_VIEWPOINT)
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
        self.window.set_on_menu_item_activated(AppWindow.MENU_VIEWPOINT, self._on_menu_viewpoint)
        self.window.set_on_menu_item_activated(AppWindow.MENU_SCREENSHOT, self._on_menu_screenshot)

        self.is_done = False
        threading.Thread(target=self._update_thread).start()

    def _on_layout(self, layout_context):
        contentRect = self.window.content_rect
        panel_width = 35 * layout_context.theme.font_size
        self.widget3d.frame = gui.Rect(contentRect.x, contentRect.y,
                                       contentRect.width - panel_width,
                                       contentRect.height)
        self.panel.frame = gui.Rect(self.widget3d.frame.get_right(),
                                    contentRect.y, panel_width,
                                    contentRect.height)

    def _on_close(self):
        self.is_done = True
        return True

    def _on_menu_quit(self):
        self.is_done = True
        gui.Application.instance.quit()

    def _on_menu_viewpoint(self):
        # todo: manually find best viewpoint
        model_mat = self.widget3d.scene.camera.get_model_matrix()
        proj_mat = self.widget3d.scene.camera.get_projection_matrix()
        view_mat = self.widget3d.scene.camera.get_view_matrix()
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

    def _update_thread(self):
        idx = 0
        while not self.is_done:
            time.sleep(0.100)

            rgb_frame = self.rgb_images[idx]
            rgb_label2d_frame = self.rgb_label2d_images[idx]
            pcd = self.pcd[idx]
            idx += 1

            if idx >= len(self.rgb_images):
                idx = 0

            if self.config_screenshot:
                win_sz = self.window.content_rect
                screen = pyscreenshot.grab(bbox=(win_sz.x, 0, win_sz.width, win_sz.height), childprocess=False)
                save_name = osp.join('/mnt/datax/temp', f'frame-{idx:03d}.png')
                print(pcolor(f'  write {save_name}', 'blue'))
                screen.save(save_name)

            def update():
                self.rgb_widget.update_image(rgb_frame)
                self.rgb_label2d_widget.update_image(rgb_label2d_frame)
                self.widget3d.scene.clear_geometry()
                self.widget3d.scene.add_geometry('pointcloud', pcd, self.lit)

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


if __name__ == "__main__":
    print(pcolor(f'sys.version:        {sys.version}', 'yellow'))
    print(pcolor(f'open3d.__version__: {o3d.__version__}\n', 'yellow'))

    main()
