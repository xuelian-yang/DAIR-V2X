#!/bin/bash

time_sh_start=$(date +"%s.%N")

# ============================================================================ #
# Extract
# ============================================================================ #
export dair_data_home=/mnt/datax/Datasets/DAIR-V2X
export dair_exp_home=/mnt/datax/itti-data/DAIR-V2X
mkdir -p ${dair_exp_home}

# DAIR-V2X-C
cd ${dair_data_home}/DAIR-V2X-C
unzip \*.zip
mv ./cooperative-vehicle-infrastructure-infrastructure-side-image      ./cooperative-vehicle-infrastructure/infrastructure-side
mv ./cooperative-vehicle-infrastructure-infrastructure-side-velodyne   ./cooperative-vehicle-infrastructure/infrastructure-side
mv ./cooperative-vehicle-infrastructure-vehicle-side-image             ./cooperative-vehicle-infrastructure/vehicle-side
mv ./cooperative-vehicle-infrastructure-vehicle-side-velodyne          ./cooperative-vehicle-infrastructure/vehicle-side
cd ${dair_data_home}/DAIR-V2X-C/cooperative-vehicle-infrastructure/infrastructure-side
mv ./cooperative-vehicle-infrastructure-infrastructure-side-image      ./image
mv ./cooperative-vehicle-infrastructure-infrastructure-side-velodyne   ./velodyne
cd ${dair_data_home}/DAIR-V2X-C/cooperative-vehicle-infrastructure/vehicle-side
mv ./cooperative-vehicle-infrastructure-vehicle-side-image             ./image
mv ./cooperative-vehicle-infrastructure-vehicle-side-velodyne          ./velodyne

# DAIR-V2X-C-test_A
cd ${dair_data_home}/DAIR-V2X-C-test_A
unzip \*.zip
mv ./DAIR-V2X-C-test_A-1/cooperative-vehicle-infrastructure                               ./cooperative-vehicle-infrastructure_test_A
mv ./DAIR-V2X-C-test_A-2/cooperative-vehicle-infrastructure-infrastructure-side-image     ./cooperative-vehicle-infrastructure_test_A/infrastructure-side
mv ./DAIR-V2X-C-test_A-3/cooperative-vehicle-infrastructure-infrastructure-side-velodyne  ./cooperative-vehicle-infrastructure_test_A/infrastructure-side
mv ./DAIR-V2X-C-test_A-4/cooperative-vehicle-infrastructure-vehicle-side-image            ./cooperative-vehicle-infrastructure_test_A/vehicle-side
mv ./DAIR-V2X-C-test_A-5/cooperative-vehicle-infrastructure-vehicle-side-velodyne         ./cooperative-vehicle-infrastructure_test_A/vehicle-side
cd ${dair_data_home}/DAIR-V2X-C-test_A/cooperative-vehicle-infrastructure_test_A/infrastructure-side
mv ./cooperative-vehicle-infrastructure-infrastructure-side-image                         ./image
mv ./cooperative-vehicle-infrastructure-infrastructure-side-velodyne                      ./velodyne
cd ${dair_data_home}/DAIR-V2X-C-test_A/cooperative-vehicle-infrastructure_test_A/vehicle-side
mv ./cooperative-vehicle-infrastructure-vehicle-side-image                                ./image
mv ./cooperative-vehicle-infrastructure-vehicle-side-velodyne                             ./velodyne
cd ${dair_data_home}/DAIR-V2X-C-test_A
rm -d -rf DAIR-V2X-C-test_A-1
rm -d -rf DAIR-V2X-C-test_A-2
rm -d -rf DAIR-V2X-C-test_A-3
rm -d -rf DAIR-V2X-C-test_A-4
rm -d -rf DAIR-V2X-C-test_A-5

# DAIR-V2X-I
cd ${dair_data_home}/DAIR-V2X-I
unzip \*.zip
mv ./single-infrastructure-side-image            ./single-infrastructure-side
mv ./single-infrastructure-side-velodyne         ./single-infrastructure-side
cd ${dair_data_home}/DAIR-V2X-I/single-infrastructure-side
mv ./single-infrastructure-side-image            ./image
mv ./single-infrastructure-side-velodyne         ./velodyne

# DAIR-V2X-V
cd ${dair_data_home}/DAIR-V2X-V
unzip \*.zip
mv ./single-vehicle-side-image                   ./single-vehicle-side
mv ./single-vehicle-side-velodyne                ./single-vehicle-side
cd ${dair_data_home}/DAIR-V2X-V/single-vehicle-side
mv ./single-vehicle-side-image                   ./image
mv ./single-vehicle-side-velodyne                ./velodyne

# Rope3D
cd ${dair_data_home}/Rope3D
unzip \*.zip
mv ./training-depth_2       ./training
mv ./training-image_2a      ./training
mv ./training-image_2b      ./training
mv ./training-image_2c      ./training
mv ./training-image_2d      ./training
mv ./validation-depth_2     ./validation
mv ./validation-image_2     ./validation
cd ${dair_data_home}/Rope3D/training
mv ./training-depth_2       ./depth_2
mv ./training-image_2a      ./image_2a
mv ./training-image_2b      ./image_2b
mv ./training-image_2c      ./image_2c
mv ./training-image_2d      ./image_2d
cd ${dair_data_home}/Rope3D/validation
mv ./validation-depth_2     ./depth_2
mv ./validation-image_2     ./image_2
cd ${dair_data_home}

# move to destination
mv ${dair_data_home}/DAIR-V2X-C/cooperative-vehicle-infrastructure                ${dair_exp_home}
mv ${dair_data_home}/DAIR-V2X-C-test_A/cooperative-vehicle-infrastructure_test_A  ${dair_exp_home}
mv ${dair_data_home}/DAIR-V2X-I/single-infrastructure-side                        ${dair_exp_home}
mv ${dair_data_home}/DAIR-V2X-V/single-vehicle-side                               ${dair_exp_home}
mkdir -p ${dair_exp_home}/Rope3D
mv ${dair_data_home}/Rope3D/training                                              ${dair_exp_home}/Rope3D
mv ${dair_data_home}/Rope3D/validation                                            ${dair_exp_home}/Rope3D

# ============================================================================ #
# Create symlink
# ============================================================================ #
export dair_dev_root=/mnt/itti-dev/DAIR-V2X
export dair_dataset_path=/mnt/datax/itti-data/DAIR-V2X
cd ${dair_dev_root}
mkdir ./data/DAIR-V2X


ln -sf ${dair_dataset_path}/cooperative-vehicle-infrastructure          ${dair_dev_root}/data/DAIR-V2X
ln -sf ${dair_dataset_path}/cooperative-vehicle-infrastructure_test_A   ${dair_dev_root}/data/DAIR-V2X
ln -sf ${dair_dataset_path}/Rope3D                                      ${dair_dev_root}/data/DAIR-V2X
ln -sf ${dair_dataset_path}/single-infrastructure-side                  ${dair_dev_root}/data/DAIR-V2X
ln -sf ${dair_dataset_path}/single-vehicle-side                         ${dair_dev_root}/data/DAIR-V2X

time_sh_end=$(date +"%s.%N")
time_diff_sh=$(bc <<< "$time_sh_end - $time_sh_start")
echo "elapsed:  $time_diff_sh   seconds. ($time_sh_end - $time_sh_start)"
echo "done"
