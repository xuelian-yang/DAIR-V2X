
# Contents

- [Contents](#contents)
- [conda / pip install](#conda--pip-install)
- [Env](#env)
- [Check Label](#check-label)
- [Export Video/GIF](#export-videogif)
- [Visualize 3D Point Cloud Label with Open3D](#visualize-3d-point-cloud-label-with-open3d)

<!-- ========== ========== ========== ========== ========== -->

# conda / pip install

```bash
pip install -U pip -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

# Env

```bash
# https://blog.csdn.net/m0_57273938/article/details/125537800
python==3.7.11
numpy==1.21.4
vedo==2022.01
vtk==8.1.2
opencv-python==4.1.1.26
matplotlib==3.4.3
open3d==0.14.1

# https://www.pudn.com/news/62fef9d1f97302478e73255b.html
conda create -n dair python=3.7
  # 官方要求python为3.7
conda activate dair
conda install pip
pip install -U pip -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

# Check Label

- wrong image label:
  - ./data/DAIR-V2X/cooperative-vehicle-infrastructure/vis_results_vehicle-side/000254.png
  - [x] 点云重复检测结果不一致 - PointPillars 模型需要重训练
  - [x] 路侧点云四维 xyzr，而车端点云三维 xyz，兼容计算新坐标系下点云坐标的代码需要调整

# Export Video/GIF

# Visualize 3D Point Cloud Label with Open3D

- [x] Visualize predicted vehicle cloud label with Open3D 
- ~~8 points to (x, y, z, x_size, y_size, z_size, yaw)~~

  ```bash
  # label_world2v.py
   (x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)

   ..code - block:: none


                        front z
                             /
                            /
              (x0, y0, z1) + -----------  + (x1, y0, z1)
                          /|            / |
                         / |           /  |
           (x0, y0, z0) + ----------- +   + (x1, y1, z1)
                        |  /      .   |  /
                        | / origin    | /
           (x0, y1, z0) + ----------- + -------> x right
                        |             (x1, y1, z0)
                        |
                        v
                   down y
  ```

- [x] Visualize 3D Label with Open3D (3d_dimensions + 3d_location + rotation)

  ```bash
  # single-infrastructure-side-example/label/virtuallidar/000000.json
  [
    {
      "type": "Cyclist",
      "occluded_state": "0",
      "truncated_state": "1",
      "alpha": "0",
      "2d_box": {
        "xmin": "1528.72644",
        "ymin": "289.33963",
        "xmax": "1597.9943839999999",
        "ymax": "365.06311"
      },
      "3d_dimensions": { "h": "1.762347", "w": "0.761443", "l": "1.649016" },
      "3d_location": {
        "x": "53.004549165490424",
        "y": "-19.429439598303667",
        "z": "-0.8957502917292395"
      },
      "rotation": "1.6607597119606952"
    },
    ...
  ]
  ```

<!-- End of File -->
