
# Contents

- [Contents](#contents)
- [conda / pip install](#conda--pip-install)
- [setup env](#setup-env)
- [build open3d with OPEN3D\_ML from source (Linux)](#build-open3d-with-open3d_ml-from-source-linux)
- [build open3d with OPEN3D\_ML from source (Windows)](#build-open3d-with-open3d_ml-from-source-windows)
- [Env](#env)
- [Check Label](#check-label)
- [Export Video/GIF](#export-videogif)
- [Visualize 3D Point Cloud Label with Open3D](#visualize-3d-point-cloud-label-with-open3d)
- [References](#references)

<!-- ========== ========== ========== ========== ========== -->

# conda / pip install

```bash
pip install -U pip -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

# setup env

```bash
conda create -n o3d python=3.8 -y
conda activate o3d
conda install pip
pip install -U pip
pip install open3d
pip install opencv-python
pip install pillow
pip install pyscreenshot
pip install termcolor

git clone --recursive git@github.com:xuelian-yang/DAIR-V2X.git
cd DAIR-V2X
git checkout demo

python viz/viz_dataset.py
```

# build open3d with OPEN3D_ML from source (Linux)

```bash
# http://www.open3d.org/docs/release/compilation.html
git clone --recursive git@github.com:isl-org/Open3D.git
cd Open3D
bash util/install_deps_ubuntu.sh

which python
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

make -j install-pip-package

python -c "import open3d"

python -c "import open3d.ml.torch as ml3d"
python -c "import open3d.ml.tf as ml3d"
```

# build open3d with OPEN3D_ML from source (Windows)

```bash
git clone --recursive git@github.com:isl-org/Open3D.git
cd Open3D
mkdir build
cd build

which python
cmake -G "Visual Studio 16 2019" \
      -A x64 \
      -DBUILD_CUDA_MODULE=ON \
      -DGLIBCXX_USE_CXX11_ABI=OFF \
      -DBUILD_PYTORCH_OPS=ON \
      -DBUILD_TENSORFLOW_OPS=ON \
      -DBUNDLE_OPEN3D_ML=ON \
      -DOPEN3D_ML_ROOT=https://github.com/isl-org/Open3D-ML.git \
      -DCMAKE_INSTALL_PREFIX="<open3d_install_directory>" \
      ..

cmake --build . --config Release --target ALL_BUILD
# :: Activate the virtualenv first
# :: Install pip package in the current python environment
cmake --build . --config Release --target install-pip-package

python -c "import open3d; print(open3d)"
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

- [x] draw 2d image label
- [x] draw 3d point cloud label
- [x] [visualization with intensity](https://github.com/isl-org/Open3D/issues/2545#issuecomment-987119956)
- [x] save screenshot sequences as gif
- [x] multi 3d scene widgets
- [x] [point cloud to mesh](http://www.open3d.org/docs/release/tutorial/geometry/surface_reconstruction.html)
- [x] [add collapsible widget](https://github.com/isl-org/Open3D/blob/4eef4b3061e1f76fd7d2669ac388d1d19ce3c53d/examples/python/visualization/non_english.py#L186)
- [x] point cloud with texture
- [ ] mesh with texture
- [ ] offline processing (cloud to mesh, compute texture, progress bar)
- optimize code:
  - [ ] remove point_normal and reuse point
  - [ ] manually find the best viewport

# References

- [open-v2x/lidar](https://github.com/open-v2x/lidar)
  - [gitee open-v2x lidar](https://gitee.com/open-v2x/lidar)

<!-- End of File -->
