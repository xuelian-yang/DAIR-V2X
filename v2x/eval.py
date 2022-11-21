import sys
import os
import os.path as osp

sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

import argparse
import logging

logger = logging.getLogger(__name__)

from tqdm import tqdm
import numpy as np

from v2x_utils import range2box, id_to_str, Evaluator
from config import add_arguments
from dataset import SUPPROTED_DATASETS
from dataset.dataset_utils import save_pkl
from models import SUPPROTED_MODELS
from models.model_utils import Channel

import datetime
import time
sys.path.append(osp.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.setup_log import setup_log, git_info, pcolor


def eval_vic(args, dataset, model, evaluator):
    logging.warning(f'eval_vic(.., {type(dataset)}')
    idx = -1
    for VICFrame, label, filt in tqdm(dataset):
        idx += 1
        if idx > 1:
            break
        # if idx % 10 != 0:
        #     continue
        try:
            # print(pcolor(f'{dataset.data[idx][0]}', 'yellow'))
            veh_id = dataset.data[idx][0]["vehicle_pointcloud_path"].split("/")[-1].replace(".pcd", "")
        except Exception:
            veh_id = VICFrame["vehicle_pointcloud_path"].split("/")[-1].replace(".pcd", "")

        pred = model(
            VICFrame,
            filt,
            None if not hasattr(dataset, "prev_inf_frame") else dataset.prev_inf_frame,
        )

        evaluator.add_frame(pred, label)
        pipe.flush()
        pred["label"] = label["boxes_3d"]
        pred["veh_id"] = veh_id
        save_pkl(pred, osp.join(args.output, "result", pred["veh_id"] + ".pkl"))

    evaluator.print_ap("3d")
    evaluator.print_ap("bev")
    print("Average Communication Cost = %.2lf Bytes" % (pipe.average_bytes()))


def eval_single(args, dataset, model, evaluator):
    logging.warnging(f'eval_single(..)')
    idx = -1
    for frame, label, filt in tqdm(dataset):
        idx += 1
        if idx > 10:
            break
        pred = model(frame, filt)
        if args.sensortype == "camera":
            evaluator.add_frame(pred, label["camera"])
        elif args.sensortype == "lidar":
            evaluator.add_frame(pred, label["lidar"])
        save_pkl({"boxes_3d": label["lidar"]["boxes_3d"]}, osp.join(args.output, "result", frame.id["camera"] + ".pkl"))

    evaluator.print_ap("3d")
    evaluator.print_ap("bev")


def print_configs(args):
    for v in vars(args):
        logging.info(f'{v:<20s}: {vars(args)[v]}')


if __name__ == "__main__":
    # setup_log(f'v2x_eval_{datetime.datetime.now()}.log')
    setup_log(f'v2x_eval.log')
    time_beg_eval = time.time()

    parser = argparse.ArgumentParser(conflict_handler="resolve")
    add_arguments(parser)
    args, _ = parser.parse_known_args()
    logging.info("=== parse_known_args: ===")
    print_configs(args)

    # add model-specific arguments
    SUPPROTED_MODELS[args.model].add_arguments(parser)
    args = parser.parse_args()

    logging.warning("=== parse_args: ===")
    print_configs(args)

    if args.quiet:
        level = logging.ERROR
    elif args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level,
    )

    extended_range = range2box(np.array(args.extended_range))
    logger.info("loading dataset")

    dataset = SUPPROTED_DATASETS[args.dataset](
        args.input,
        args,
        split=args.split,
        sensortype=args.sensortype,
        extended_range=extended_range,
    )
    logging.warning(f'dataset: {type(dataset)}')

    logger.info("loading evaluator")
    evaluator = Evaluator(args.pred_classes)

    logger.info(f"loading model of [eval_single: {args.eval_single}]")
    if args.eval_single:
        # SingleSide  / LateFusionVeh
        logging.warning(f'eval_single')
        model = SUPPROTED_MODELS[args.model](args)
        eval_single(args, dataset, model, evaluator)
    else:
        # InfOnly / VehOnly / EarlyFusion / LateFusionInf / LateFusion
        pipe = Channel()
        model = SUPPROTED_MODELS[args.model](args, pipe)
        eval_vic(args, dataset, model, evaluator)

    time_end_eval = time.time()
    logging.warning(f'eval.py elapsed {time_end_eval - time_beg_eval:.6f} seconds')
    print(pcolor(f'eval.py elapsed {time_end_eval - time_beg_eval:.6f} seconds', 'yellow'))
