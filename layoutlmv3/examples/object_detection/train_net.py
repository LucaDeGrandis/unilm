#!/usr/bin/env python
# --------------------------------------------------------------------------------
# MPViT: Multi-Path Vision Transformer for Dense Prediction
# Copyright (c) 2022 Electronics and Telecommunications Research Institute (ETRI).
# All Rights Reserved.
# Written by Youngwan Lee
# --------------------------------------------------------------------------------

"""
Detection Training Script for MPViT.
"""

import os
import itertools

import torch

from typing import Any, Dict, List, Set

from detectron2.data import build_detection_train_loader
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.solver.build import maybe_add_gradient_clipping

from ditod import add_vit_config
from ditod import DetrDatasetMapper

from detectron2.data.datasets import register_coco_instances
import logging
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm
from detectron2.engine.defaults import create_ddp_model
import weakref
from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer
from ditod import MyDetectionCheckpointer, ICDAREvaluator
from ditod import MyTrainer

import wandb
from detectron2.engine import hooks
from detectron2.engine.hooks import HookBase


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # add_coat_config(cfg)
    add_vit_config(cfg)
    cfg.CUSTOM_DATASET_DATA_COCO_TRAIN = None
    cfg.CUSTOM_DATASET_DATA_COCO_TEST = None
    cfg.CUSTOM_DATASET_IMG_DIR_TRAIN = None
    cfg.CUSTOM_DATASET_IMG_DIR_TEST = None
    cfg.WANDB_PROJECT = None
    cfg.WANDB_RUN_NAME = None
    cfg.WANDB_RUN_ID = None
    cfg.WANDB_EVAL_PERIOD = None
    print(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


class wandb_EvalHook(HookBase):
    def __init__(self, eval_period, eval_after_train, wandb_cfg):
        self._period = eval_period
        self._eval_after_train = eval_after_train
        self.wandb_cfg = wandb_cfg

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if self._period > 0 and next_iter % self._period == 0:
            # do the last eval in after_train
            if next_iter != self.trainer.max_iter:
                # Add the metrics to wandb
                metrics = self.trainer.storage.__dict__["_latest_scalars"]
                metrics_filtered = {}
                for key, item in metrics.items():
                    if isinstance(key, str):
                        if ("bbox/AP" in key) or ("segm/AP" in key) or ("loss" in key):
                            metrics_filtered[key] = float(item[0])
                print("Logging to wandb ...")
                wandb.log(metrics_filtered, step=next_iter)
                print("Log created succesfully!")

    def after_train(self):
        next_iter = self.trainer.iter + 1
        # This condition is to prevent the eval from running after a failed training
        if self._eval_after_train and self.trainer.iter + 1 >= self.trainer.max_iter:
            metrics = self.trainer.storage.__dict__["_latest_scalars"]
            metrics_filtered = {}
            for key, item in metrics.items():
                if isinstance(key, str):
                    if ("bbox/AP" in key) or ("segm/AP" in key) or ("loss" in key):
                        metrics_filtered[key] = float(item[0])
            print("Logging to wandb ...")
            wandb.log(metrics_filtered, step=next_iter)
            print("Log created succesfully!")


def main(args):
    cfg = setup(args)

    """
    prepare the wandb logging information
    """

    if cfg.WANDB_PROJECT is not None:
        print('Logging to wandb project:', cfg.WANDB_PROJECT)
        
        assert (cfg.WANDB_RUN_NAME is not None) or (cfg.WANDB_RUN_ID is not None)

        if (cfg.WANDB_RUN_NAME is not None) and (cfg.WANDB_RUN_ID is None):
            print('Creating a new run in wandb.')
            print('Run name:', cfg.WANDB_RUN_NAME)
            wandb.init(project=cfg.WANDB_PROJECT, name=cfg.WANDB_RUN_NAME)
        elif (cfg.WANDB_RUN_ID is not None):
            print('Continuing id {cfg.WANDB_RUN_ID} in wandb.')
            print('Run name:', cfg.WANDB_RUN_NAME)
            if cfg.WANDB_RUN_NAME is not None:
                wandb.init(project=cfg.WANDB_PROJECT, id=cfg.WANDB_RUN_ID, name=cfg.WANDB_RUN_NAME)
            else:
                wandb.init(project=cfg.WANDB_PROJECT, id=cfg.WANDB_RUN_ID)
        if cfg.WANDB_EVAL_PERIOD is None:
            cfg.WANDB_EVAL_PERIOD = cfg.TEST.EVAL_PERIOD
    else:
        print('Logging to wandb is disabled. Set a name in cfg.WANDB_PROJECT to log to wandb.')
    
    """
    register publaynet first
    """
    register_coco_instances(
        "custom_dataset_train",
        {},
        cfg.CUSTOM_DATASET_DATA_COCO_TRAIN,
        cfg.CUSTOM_DATASET_IMG_DIR_TRAIN,
    )

    register_coco_instances(
        "custom_dataset_val",
        {},
        cfg.CUSTOM_DATASET_DATA_COCO_TEST,
        cfg.CUSTOM_DATASET_IMG_DIR_TEST,
    )

    if args.eval_only:
        model = MyTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = MyTrainer.test(cfg, model)
        return res

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.WANDB_PROJECT is not None:
        trainer.register_hooks([wandb_EvalHook(cfg.WANDB_EVAL_PERIOD, True, {})])  # add the wandb eval hook
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    args = parser.parse_args()
    print("Command Line Args:", args)

    if args.debug:
        import debugpy

        print("Enabling attach starts.")
        debugpy.listen(address=('0.0.0.0', 9310))
        debugpy.wait_for_client()
        print("Enabling attach ends.")

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
