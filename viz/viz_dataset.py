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
"""

import os
import os.path as osp
import sys

import time
import threading

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
        image = o3d.io.read_image(self.image_paths[k])
        point = o3d.io.read_point_cloud(self.point_paths[k])
        return image, point

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


class VideoWindow:
    def __init__(self, data):
        self.rgb_images = []
        self.pcd = []
        for img, pcd in data.read_frames():
            self.rgb_images.append(img)
            self.pcd.append(pcd)
        self.num = len(self.pcd)

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
        self.widget3d.scene.camera.look_at([70,0,0], [0,0,150], [1,0,1])

        em = self.window.theme.font_size
        margin = 0.5 * em
        self.panel = gui.Vert(0.5 * em, gui.Margins(margin))
        self.panel.add_child(gui.Label("Color image"))
        self.rgb_widget = gui.ImageWidget(self.rgb_images[0])
        self.panel.add_child(self.rgb_widget)
        self.window.add_child(self.panel)

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

    def _update_thread(self):
        idx = 0
        while not self.is_done:
            time.sleep(0.100)

            rgb_frame = self.rgb_images[idx]
            pcd = self.pcd[idx]
            idx += 1

            if idx >= len(self.rgb_images):
                idx = 0

            def update():
                self.rgb_widget.update_image(rgb_frame)
                self.widget3d.scene.clear_geometry()
                self.widget3d.scene.add_geometry('pointcloud', pcd, self.lit)

            if not self.is_done:
                gui.Application.instance.post_to_main_thread(self.window, update)

if __name__ == "__main__":
    print(f'sys.version:        {sys.version}')
    print(f'open3d.__version__: {o3d.__version__}')

    app = o3d.visualization.gui.Application.instance
    app.initialize()

    path = get_path()
    win = VideoWindow(path)
    app.run()
