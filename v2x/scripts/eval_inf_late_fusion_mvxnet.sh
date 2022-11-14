# DATA="../data/DAIR-V2X/cooperative-vehicle-infrastructure"
# OUTPUT="../cache/vic-late-lidar"
DATA="../data/DAIR-V2X/cooperative-vehicle-infrastructure"
OUTPUT="../cache/sv3d-inf-late-fusion"
rm -r $OUTPUT
rm -r ../cache
mkdir -p $OUTPUT/result
mkdir -p $OUTPUT/inf/lidar
mkdir -p $OUTPUT/veh/lidar

#INFRA_MODEL_PATH="../configs/vic3d/late-fusion-pointcloud/pointpillars"
#INFRA_CONFIG_NAME="trainval_config_i.py"
#INFRA_MODEL_NAME="vic3d_latefusion_inf_pointpillars_596784ad6127866fcfb286301757c949.pth"

#VEHICLE_MODEL_PATH="../configs/vic3d/late-fusion-pointcloud/pointpillars"
#VEHICLE_CONFIG_NAME="trainval_config_v.py"
#VEHICLE_MODEL_NAME="vic3d_latefusion_veh_pointpillars_a70fa05506bf3075583454f58b28177f.pth"

INFRA_MODEL_PATH="../configs/sv3d-inf/mvxnet"
INFRA_CONFIG_NAME="trainval_config.py"
INFRA_MODEL_NAME="sv3d_inf_mvxnet_c2271983b04b73e573486fcbc559c31e.pth"

# SPLIT_DATA_PATH="../data/split_datas/cooperative-split-data.json"
SPLIT_DATA_PATH="../data/split_datas/single-infrastructure-split-data.json"

: '
SUPPROTED_DATASETS = {
    "dair-v2x-v": DAIRV2XV,
    "dair-v2x-i": DAIRV2XI,
    "vic-sync": VICSyncDataset,
    "vic-async": VICAsyncDataset,
}

SUPPROTED_MODELS = {
    "single_side": SingleSide,
    "late_fusion": LateFusion,
    "early_fusion": EarlyFusion,
    "veh_only": VehOnly,
    "inf_only": InfOnly,
}
'

# bash scripts/eval_inf_late_fusion_mvxnet.sh 0 inf_only 0 0 100


# srun --gres=gpu:a100:1 --time=1-0:0:0 --job-name "dair-v2x" \
CUDA_VISIBLE_DEVICES=$1
FUSION_METHOD=$2
DELAY_K=$3
EXTEND_RANGE_START=$4
EXTEND_RANGE_END=$5
TIME_COMPENSATION=$6
python eval.py \
  --input $DATA \
  --output $OUTPUT \
  --model $FUSION_METHOD \
  --dataset dair-v2x-i \
  --k $DELAY_K \
  --split val \
  --split-data-path $SPLIT_DATA_PATH \
  --inf-config-path $INFRA_MODEL_PATH/$INFRA_CONFIG_NAME \
  --inf-model-path $INFRA_MODEL_PATH/$INFRA_MODEL_NAME \
  --device ${CUDA_VISIBLE_DEVICES} \
  --pred-class car \
  --sensortype lidar \
  --extended-range $EXTEND_RANGE_START -39.68 -3 $EXTEND_RANGE_END 39.68 1 \
  --overwrite-cache \
  $TIME_COMPENSATION