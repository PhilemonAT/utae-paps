"""
Script for semantic inference with pre-trained models
Author: Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""
import argparse
import json
import os
import pprint
import pickle as pkl
import random

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from src import utils, model_utils
from src.dataset_extended import PASTIS_Climate_Dataset

import wandb

from train_fusion import iterate, overall_performance, save_results, prepare_output

parser = argparse.ArgumentParser()
# Model parameters
parser.add_argument(
    "--weight_folder",
    type=str,
    default="",
    help="Path to the main folder containing the pre-trained weights",
)
parser.add_argument(
    "--dataset_folder",
    default="",
    type=str,
    help="Path to the folder where the results are saved.",
)
parser.add_argument(
    "--climate_folder",
    default="",
    type=str,
    help="Path to the folder where the results are saved.",
)
parser.add_argument(
    "--cv_type",
    default="official",
    type=str,
)
parser.add_argument(
    "--fold",
    default=None,
    type=int,
)
parser.add_argument(
    "--res_dir",
    default="./inference_utae",
    type=str,
    help="Path to directory where results are written."
)
parser.add_argument(
    "--num_workers", default=8, type=int, help="Number of data loading workers"
)
parser.add_argument(
    "--device",
    default="cuda",
    type=str,
    help="Name of device to use for tensor computations (cuda/cpu)",
)
parser.add_argument(
    "--display_step",
    default=50,
    type=int,
    help="Interval in batches between display of training metrics",
)
parser.add_argument("--model_tag", type=str)
parser.add_argument("--config_tag", type=str)
parser.add_argument("--run_tag", type=str)
parser.add_argument("--experiment_name", type=str)

def main(config):
    experiment_name = config.experiment_name
    wandb.init(project="TEST", config=config, name=experiment_name,
               tags=[config.run_tag, config.model_tag, config.config_tag])
    wandb.config.update(vars(config))

    official_fold_sequence = [
        [[1, 2, 3], [4], [5]],
        [[2, 3, 4], [5], [1]],
        [[3, 4, 5], [1], [2]],
        [[4, 5, 1], [2], [3]],
        [[5, 1, 2], [3], [4]],
    ]

    region_fold_sequence = [
        [[1, 2], [3], [4]]
    ] * 10


    # Set all possible seeds
    random.seed(config.rdm_seed)
    np.random.seed(config.rdm_seed)
    torch.manual_seed(config.rdm_seed)
    torch.cuda.manual_seed(config.rdm_seed)
    torch.cuda.manual_seed_all(config.rdm_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device(config.device)
    prepare_output(config, cv_type=config.cv_type)

    # Choose relevant folds (different for official CV-folds vs. CV based on region-folds)
    if config.cv_type == "official":
        fold_sequence = (
            official_fold_sequence if config.fold is None else [official_fold_sequence[config.fold - 1]]
        )
    else:
        fold_sequence = (
            region_fold_sequence if config.fold is None else [region_fold_sequence[config.fold - 1]]
        )

    model = model_utils.get_model(config, mode="semantic")
    model = model.to(device)

    config.N_params = utils.get_ntrainparams(model)
    print(model)
    print("TOTAL TRAINABLE PARAMETERS :", config.N_params)

    for fold, (train_folds, val_fold, test_fold) in enumerate(fold_sequence):
        if config.fold is not None:
            fold = config.fold - 1

        # Dataset definition
        dt_args = dict(
            folder=config.dataset_folder, 
            climate_folder=config.climate_folder,
            norm=True,
            reference_date=config.ref_date,
            mono_date=config.mono_date,
            target="semantic",
            sats=["S2"],
            apply_noise=config.apply_noise,
            noise_std=config.noise_std    
        )

        if config.cv_type == 'regions':
            class_mapping = {
                0: 0,   
                1: 1,   
                2: 2,
                3: 3,
                4: 4,
                5: 5,
                6: 6,
                7: 19,  # Discard class 7
                8: 19,  # Discard class 8
                9: 19,  # Discard class 9
                10: 19, # Discard class 10
                11: 19, # Discard class 11
                12: 12, 
                13: 13,
                14: 14,
                15: 19, # Discard class 15
                16: 19, # Discard class 16
                17: 17,
                18: 19, # Discard class 18
                19: 19,
            }
        else:
            class_mapping = None
        
        dt_test = PASTIS_Climate_Dataset(**dt_args, folds=test_fold, cv_type=config.cv_type, class_mapping=class_mapping)

        collate_fn = lambda x: utils.pad_collate(x, pad_value=config.pad_value)
        test_loader = data.DataLoader(
            dt_test,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

        # Load weights
        if config.cv_type=='official':
            sd = torch.load(
                os.path.join(config.weight_folder, config.cv_type, "Fold_{}".format(fold+1), "model.pth.tar"),
                map_location=device,
            )
        else:
            sd = torch.load(
                os.path.join(config.weight_folder, config.cv_type, "Run_{}".format(fold+1), "model.pth.tar"),
                map_location=device
            )
        model.load_state_dict(sd["state_dict"])

        # Loss
        weights = torch.ones(config.num_classes, device=device).float()
        weights[config.ignore_index] = 0
        criterion = nn.CrossEntropyLoss(weight=weights)

        # Inference
        print("Testing . . .")
        model.eval()
        test_metrics, conf_mat, att_weights, dates_sat = iterate(
            model,
            data_loader=test_loader,
            criterion=criterion,
            config=config,
            optimizer=None,
            mode="test",
            return_att_climate=True,
            device=device,
        )
        print(
            "Loss {:.4f},  Acc {:.2f},  IoU {:.4f}".format(
                test_metrics["test_loss"],
                test_metrics["test_accuracy"],
                test_metrics["test_IoU"],
            )
        )
        save_results(fold + 1, test_metrics, conf_mat.cpu().numpy(), config, cv_type=config.cv_type)

        if config.cv_type=='official':
            pkl.dump(
                att_weights,
                open(
                    os.path.join(config.res_dir, config.cv_type, "Fold_{}".format(fold+1), "att_weights.pkl"), "wb"
                ),
            )
            pkl.dump(
                dates_sat,
                open(
                    os.path.join(config.res_dir, config.cv_type, "Fold_{}".format(fold+1), "dates_sat.pkl"), "wb"
                ),
            )
        else:
            pkl.dump(
                att_weights,
                open(
                    os.path.join(config.res_dir, config.cv_type, "Run_{}".format(fold+1), "att_weights.pkl"), "wb"
                ),
            )
            pkl.dump(
                dates_sat,
                open(
                    os.path.join(config.res_dir, config.cv_type, "Run_{}".format(fold+1), "dates_sat.pkl"), "wb"
                ),
            )

    if config.fold is None:
        overall_performance(config, cv_type=config.cv_type)

    wandb.finish()


if __name__ == "__main__":
    test_config = parser.parse_args()


    with open(os.path.join(test_config.weight_folder, test_config.cv_type, "conf.json")) as file:
        model_config = json.loads(file.read())

    config = {**model_config, **vars(test_config)}
    config = argparse.Namespace(**config)
    config.fold = test_config.fold

    pprint.pprint(config)
    main(config)
