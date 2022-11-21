
# Contents

- [Contents](#contents)
- [conda / pip install](#conda--pip-install)
- [Env](#env)
- [Check Label](#check-label)
- [Export Video/GIF](#export-videogif)

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

<!-- End of File -->
