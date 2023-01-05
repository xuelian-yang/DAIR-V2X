# -*- coding: utf-8 -*-

"""
References
----------
https://www.robosense.ai/en/resources
https://www.robosense.ai/en/rslidar/RS-LiDAR-M1
"""

import math
import numpy as np
from termcolor import colored
import time


if __name__ == '__main__':
    time_beg_test_m1 = time.time()
    theta_arr = [0., 0.2, 30., 45., 60., 90., 180.]
    dis = 150.0
    for theta_deg in theta_arr:
        theta_rad = theta_deg / 180.0 * np.pi
        cos_theta = math.cos(theta_rad)
        sin_theta = math.sin(theta_rad)
        print(colored(f'angle theta: {theta_deg:6.2f} in degree: {theta_rad:8.4f} in radian:', 'yellow'))
        print(colored(f'  - cos({theta_deg:4.2f})         = {cos_theta:8.4f}', 'cyan'))
        print(colored(f'  - sin({theta_deg:4.2f})         = {sin_theta:8.4f}', 'cyan'))
        print(colored(f'  - sin({theta_deg:4.2f}) * {dis:5.1f} = {dis*sin_theta:8.4f}', 'red'))

    time_end_test_m1 = time.time()
    print(colored(f'test_m1.py elapsed {time_end_test_m1 - time_beg_test_m1:.6f} seconds', 'yellow'))
