import argparse
import json
import os
import pickle as pkl
import pprint
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchnet as tnt

import wandb

from src import utils, model_utils
from src.dataset_extended import PASTIS_Climate_Dataset
from src.learning.metrics import confusion_matrix_analysis
from src.learning.miou import IoU
from src.learning.weight_init import weight_init

parser = argparse.ArgumentParser(allow_abbrev=False)

# model params
parser.add_argument("--model", default="utae_fusion", type=str, help="Type of architecture to use. Can be one of: (utae/utae_ef/utae_mf/utae_encf/unet3d/fpn/convlstm/convgru/uconvlstm/buconvlstm)")
parser.add_argument("--input_dim", default=10, type=int)
parser.add_argument("--climate_input_dim", default=11, type=int)
parser.add_argument("--encoder_widths", default="[64,64,64,128]", type=str)
parser.add_argument("--decoder_widths", default="[32,32,64,128]", type=str)
parser.add_argument("--out_conv", default="[32, 32, 20]", type=str)
parser.add_argument("--str_conv_k", default=4, type=int)
parser.add_argument("--str_conv_s", default=2, type=int)
parser.add_argument("--str_conv_p", default=1, type=int)
parser.add_argument("--agg_mode", default="att_group", type=str)
parser.add_argument("--encoder_norm", default="group", type=str)
parser.add_argument("--n_head", default=16, type=int)
parser.add_argument("--d_model", default=256, type=int)
parser.add_argument("--d_k", default=4, type=int)
parser.add_argument("--encoder", default=False, type=bool)
parser.add_argument("--return_maps", default=False, type=bool)
parser.add_argument("--matching_type", default='causal', type=str)
parser.add_argument("--use_climate_mlp", default=False, type=bool)
parser.add_argument("--fusion_location", default=4, type=int)
parser.add_argument("--fusion_style", default="film", type=str)
parser.add_argument("--residual_film", default=False, type=bool)

# Set-up parameters
parser.add_argument("--dataset_folder", default="", type=str, help="Path to the dataset folder")
parser.add_argument("--climate_folder", default="", type=str, help="Path to the climate dataset folder")
parser.add_argument("--res_dir", default="./results", help="Path to the folder where the results should be stored")
parser.add_argument("--num_workers", default=8, type=int, help="Number of data loading workers")
parser.add_argument("--rdm_seed", default=1, type=int, help="Random seed")
parser.add_argument("--device", default="cuda", type=str, help="Device for tensor computations (cuda/cpu)")
parser.add_argument("--display_step", default=50, type=int, help="Interval in batches between display of training metrics")
parser.add_argument("--cache", dest="cache", action="store_true", help="If specified, the whole dataset is kept in RAM")
parser.add_argument("--apply_noise", default=True, type=bool, help="Apply Gaussian noise to the data as augmentation")
parser.add_argument("--noise_std", default=0.01, type=float, help="Standard deviation for Gaussian noise")

# Training parameters
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs per fold")
parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay for optimizer")
parser.add_argument("--type_scheduler", default=None, type=str, help="The type of learning rate scheduler to use")
parser.add_argument("--mono_date", default=None, type=str)
parser.add_argument("--ref_date", default="2018-09-01", type=str)
parser.add_argument("--cv_type", default="official", type=str, help="Type of cross-validation to use ('official' or 'regions')")
parser.add_argument("--fold", default=None, type=int, help="Specific fold or region to use")
parser.add_argument("--num_classes", default=20, type=int)
parser.add_argument("--ignore_index", default=-1, type=int)
parser.add_argument("--pad_value", default=-1000, type=float)
parser.add_argument("--padding_mode", default="reflect", type=str)
parser.add_argument("--val_every", default=1, type=int, help="Interval in epochs between two validation steps.")
parser.add_argument("--val_after", default=0, type=int, help="Do validation only after that many epochs.")

# W&B specific parameters
parser.add_argument("--experiment_name", default="default_experiment", type=str, help="W&B experiment name")
parser.add_argument("--run_tag", default="", type=str)
parser.add_argument("--model_tag", default="", type=str)
parser.add_argument("--config_tag", default="", type=str)

list_args = ["encoder_widths", "decoder_widths", "out_conv"]
parser.set_defaults(cache=False)

def iterate(model, data_loader, criterion, config, optimizer=None, 
            mode="train", return_att_climate=False, device=None):
    loss_meter = tnt.meter.AverageValueMeter()
    iou_meter = IoU(
        num_classes=config.num_classes,
        ignore_index=config.ignore_index,
        cm_device=config.device,
    )

    t_start = time.time()

    # Initalize list to store attention weights for whole epoch
    att_weights_epoch = []

    for i, batch in enumerate(data_loader):
        data_dict = batch
        if device is not None:
            data_dict = recursive_todevice(data_dict, device)
            
        input_sat, dates_sat = data_dict["input_satellite"]
        input_clim, dates_clim = data_dict["input_climate"]
        y = data_dict["target"]

        if mode != "train":
            with torch.no_grad():
                if return_att_climate:
                    out, att_weights = model(input_sat, dates_sat, input_clim, dates_clim, 
                                            batch_positions=dates_sat, return_att_clim=return_att_climate)
                    att_weights_epoch.append(att_weights[0]) # Only append first element (= layer)
                else:
                    out = model(input_sat, dates_sat, input_clim, dates_clim, batch_positions=dates_sat)
        else:
            optimizer.zero_grad()
            out = model(input_sat, dates_sat, input_clim, dates_clim, batch_positions=dates_sat)

        loss = criterion(out, y)
        
        if mode == "train":
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            pred = out.argmax(dim=1)
        iou_meter.add(pred, y)
        loss_meter.add(loss.item())

        if (i + 1) % config.display_step == 0:
            miou, acc = iou_meter.get_miou_acc()
            print(
                "Step [{}/{}], Loss: {:.4f}, Acc : {:.2f}, mIoU {:.2f}".format(
                    i + 1, len(data_loader), loss_meter.value()[0], acc, miou
                )
            )


    t_end = time.time()
    total_time = t_end - t_start
    print(f"Epoch time: {total_time:.1f}s")
    miou, acc = iou_meter.get_miou_acc()
    metrics = {
        f"{mode}_accuracy": acc,
        f"{mode}_loss": loss_meter.value()[0],
        f"{mode}_IoU": miou,
        f"{mode}_epoch_time": total_time,
    }

    wandb.log(metrics)

    if mode == "test":
        if return_att_climate:
            # Aggregate attention weights over batch dimension
            att_weights_epoch = torch.stack(att_weights_epoch)
            att_weights_avg = torch.mean(att_weights_epoch, dim=0)
            return metrics, iou_meter.conf_metric.value(), att_weights_avg
        else:
            return metrics, iou_meter.conf_metric.value()
    else:
        return metrics
    
def checkpoint(fold, log, config, cv_type="official"):
    if cv_type=="official":
        with open(
            os.path.join(config.res_dir,  cv_type, "Fold_{}".format(fold), "trainlog.json"), "w"
        ) as outfile:
            json.dump(log, outfile, indent=4)
    else:
        with open(
            os.path.join(config.res_dir,  cv_type, "Run_{}".format(fold), "trainlog.json"), "w"
        ) as outfile:
            json.dump(log, outfile, indent=4)

def save_results(fold, metrics, conf_mat, config, cv_type="official"):
    if cv_type=="official":
        with open(
            os.path.join(config.res_dir, cv_type, "Fold_{}".format(fold), "test_metrics.json"), "w"
        ) as outfile:
            json.dump(metrics, outfile, indent=4)
        pkl.dump(
            conf_mat,
            open(
                os.path.join(config.res_dir, cv_type, "Fold_{}".format(fold),  "conf_mat.pkl"), "wb"
            ),
        )
    else:
        with open(
            os.path.join(config.res_dir, cv_type, "Run_{}".format(fold), "test_metrics.json"), "w"
        ) as outfile:
            json.dump(metrics, outfile, indent=4)
        pkl.dump(
            conf_mat,
            open(
                os.path.join(config.res_dir, cv_type, "Run_{}".format(fold), "conf_mat.pkl"), "wb"
            ),
        )

def prepare_output(config, cv_type="official"):
    os.makedirs(config.res_dir, exist_ok=True)
    if cv_type=="official":
        for fold in range(1, 6):
            os.makedirs(os.path.join(config.res_dir, cv_type, "Fold_{}".format(fold)), exist_ok=True)
    else:
        for region in range(1, 11):
            os.makedirs(os.path.join(config.res_dir, cv_type, "Run_{}".format(region)), exist_ok=True)

def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_todevice(v, device) for k, v in x.items()}
    else:
        return [recursive_todevice(c, device) for c in x]

def overall_performance(config, cv_type="official"):
    cm = np.zeros((config.num_classes, config.num_classes))
    if cv_type=="official":
        for fold in range(1, 6):
            cm += pkl.load(
                open(
                    os.path.join(config.res_dir, cv_type, "Fold_{}".format(fold), "conf_mat.pkl"),
                    "rb",
                )
            )
    else:
        for region in range(1, 11):
            cm += pkl.load(
                open(
                    os.path.join(config.res_dir, cv_type, "Run_{}".format(region), "conf_mat.pkl"),
                    "rb",
                )
            )

    if config.ignore_index is not None:
        cm = np.delete(cm, config.ignore_index, axis=0)
        cm = np.delete(cm, config.ignore_index, axis=1)

    _, perf = confusion_matrix_analysis(cm)

    print("Overall performance:")
    print("Acc: {},  IoU: {}".format(perf["Accuracy"], perf["MACRO_IoU"]))

    with open(os.path.join(config.res_dir, cv_type, "overall.json"), "w") as file:
        file.write(json.dumps(perf, indent=4))

def main(config):
    experiment_name = config.experiment_name
    wandb.init(project="FINAL_EXPERIMENTS", config=config, name=experiment_name,
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

    for fold, (train_folds, val_fold, test_fold) in enumerate(fold_sequence):
        if config.fold is not None:
            fold = config.fold -1 

        # Dataset
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
        
        dt_train = PASTIS_Climate_Dataset(**dt_args, folds=train_folds, cv_type=config.cv_type, class_mapping=class_mapping, cache=config.cache)
        dt_val = PASTIS_Climate_Dataset(**dt_args, folds=val_fold, cv_type=config.cv_type, class_mapping=class_mapping, cache=config.cache)
        dt_test = PASTIS_Climate_Dataset(**dt_args, folds=test_fold, cv_type=config.cv_type, class_mapping=class_mapping)
                
        collate_fn = lambda x: utils.pad_collate(x, pad_value=config.pad_value)
        train_loader = data.DataLoader(
            dt_train,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

        val_loader = data.DataLoader(
            dt_val,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

        test_loader = data.DataLoader(
            dt_test,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
        
        print(
            "Train {}, Val {}, Test {}".format(len(dt_train), len(dt_val), len(dt_test))
        )


        # get U-TAE model
        model = model_utils.get_model(config, mode="semantic").to(device)
        
        config.N_params = utils.get_ntrainparams(model)
        with open(os.path.join(config.res_dir, config.cv_type, "conf.json"), "w") as file:
            file.write(json.dumps(vars(config), indent=4))

        print(model)
        print("TOTAL TRAINABLE PARAMETERS :", config.N_params)
        print("Trainable layers:")
        for name, p in model.named_parameters():
            if p.requires_grad:
                print(name)
        model.apply(weight_init)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        # Learning rate scheduler
        if config.type_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
        elif config.type_scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)

        weights = torch.ones(config.num_classes, device=device).float()
        weights[config.ignore_index] = 0
        criterion = nn.CrossEntropyLoss(weight=weights)

        # Training loop
        trainlog = {}
        best_mIoU = 0
        for epoch in range(1, config.epochs + 1):
            print(f"EPOCH {epoch}/{config.epochs}")

            model.train()
            train_metrics = iterate(model, train_loader, criterion, config, 
                                    optimizer, mode="train", device=device)
            
            if epoch % config.val_every == 0 and epoch > config.val_after:
                print("Validation . . . ")
                model.eval()
                val_metrics = iterate(model, val_loader, criterion, config, optimizer, mode="val", device=device)

                print(f"Loss {val_metrics['val_loss']:.4f}, Acc {val_metrics['val_accuracy']:.2f}, IoU {val_metrics['val_IoU']:.4f}")
                

                trainlog[epoch] = {**train_metrics, **val_metrics}
                checkpoint(fold + 1, trainlog, config, cv_type=config.cv_type)
                if val_metrics["val_IoU"] >= best_mIoU:
                    best_mIoU = val_metrics["val_IoU"]
                    if config.cv_type=='official':
                        torch.save(
                            {
                                "epoch": epoch,
                                "state_dict": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                            },
                            os.path.join(
                                config.res_dir, config.cv_type, "Fold_{}".format(fold + 1), "model.pth.tar"
                            ),
                        )
                    else:
                        torch.save(
                            {
                                "epoch": epoch,
                                "state_dict": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                            },
                            os.path.join(
                                config.res_dir, config.cv_type, "Run_{}".format(fold + 1), "model.pth.tar"
                            ),
                        )
            else:
                trainlog[epoch] = {**train_metrics}
                checkpoint(fold + 1, trainlog, config, cv_type=config.cv_type)
                
            if config.type_scheduler is not None:
                # Step the scheduler at end of epoch
                if config.type_scheduler == 'plateau':
                    # Expects a metric to observe
                    scheduler.step(val_metrics["val_loss"])
                else:
                    scheduler.step()

        print("Testing best epoch . . .")
        if config.cv_type=='official':
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        config.res_dir, config.cv_type, "Fold_{}".format(fold + 1), "model.pth.tar"
                    )
                )["state_dict"]
            )
        else:
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        config.res_dir, config.cv_type, "Run_{}".format(fold + 1), "model.pth.tar"
                    )
                )["state_dict"]
            )
        model.eval()

        test_metrics, conf_mat = iterate(
            model,
            data_loader=test_loader,
            criterion=criterion,
            config=config,
            optimizer=optimizer,
            mode="test",
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

    if config.fold is None:
        overall_performance(config, cv_type=config.cv_type)

    wandb.finish()


if __name__ == "__main__":
    config = parser.parse_args()
    for k, v in vars(config).items():
        if k in list_args and v is not None:
            v = v.replace("[", "")
            v = v.replace("]", "")
            config.__setattr__(k, list(map(int, v.split(","))))    
    
    assert config.num_classes == config.out_conv[-1]

    pprint.pprint(config)
    main(config)

