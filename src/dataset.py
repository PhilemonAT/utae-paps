import json
import os
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.utils.data as tdata


class PASTIS_Dataset(tdata.Dataset):
    def __init__(
        self,
        folder,
        norm=True,
        target="semantic",
        cache=False,
        mem16=False,
        cv_type="official",
        folds=None,
        reference_date="2018-09-01",
        class_mapping=None,
        mono_date=None,
        sats=["S2"],
    ):
        """
        Pytorch Dataset class to load samples from the PASTIS dataset, for semantic 
        segmentation
        The Dataset yields ((data, dates), target) tuples, where:
            - data contains the image time series
            - dates contains the date sequence of the observations expressed in number
              of days since a reference date
            - target is the semantic target
        Args:
            folder (str): Path to the dataset
            norm (bool): If true, images are standardised using pre-computed
                channel-wise means and standard deviations.
            reference_date (str, Format : 'YYYY-MM-DD'): Defines the reference date
                based on which all observation dates are expressed. Along with the image
                time series and the target tensor, this dataloader yields the sequence
                of observation dates (in terms of number of days since the reference
                date). This sequence of dates is used for instance for the positional
                encoding in attention based approaches.
            target (str): 'semantic'. Defines which type of target is
                returned by the dataloader.
                * If 'semantic' the target tensor is a tensor containing the class of
                  each pixel.
            cache (bool): If True, the loaded samples stay in RAM, default False.
            mem16 (bool): Additional argument for cache. If True, the image time
                series tensors are stored in half precision in RAM for efficiency.
                They are cast back to float32 when returned by __getitem__.
            cv_type (str): Defines the type of cross-validation split to use.
                * If 'official', uses the 5 official folds provided with the PASTIS 
                dataset.
                * If 'regions', uses a custom 4-region-based split
            folds (list, optional): List of ints specifying which of the 5 official
                folds or which of the 4 regions to load, respectively. By default 
                (when None is specified) all folds are loaded.
                Note: if cv_type = 'official', folds must be in [1,5] and if 
                cv_type = 'regions', folds must be in [1,4].
            class_mapping (dict, optional): Dictionary to define a mapping between the
                default 18 class nomenclature and another class grouping, optional.
            mono_date (int or str, optional): If provided only one date of the
                available time series is loaded. If argument is an int it defines the
                position of the date that is loaded. If it is a string, it should be
                in format 'YYYY-MM-DD' and the closest available date will be selected.
            sats (list): defines the satellites to use (only Sentinel-2 is available
                in v1.0)
        """
        super(PASTIS_Dataset, self).__init__()
        self.folder = folder
        self.norm = norm
        self.reference_date = datetime(*map(int, reference_date.split("-")))
        self.cache = cache
        self.mem16 = mem16
        self.mono_date = None
        if mono_date is not None:
            self.mono_date = (
                datetime(*map(int, mono_date.split("-")))
                if "-" in mono_date
                else int(mono_date)
            )
        self.memory = {}
        self.memory_dates = {}
        self.class_mapping = (
            np.vectorize(lambda x: class_mapping[x])
            if class_mapping is not None
            else class_mapping
        )
        self.target = target
        self.sats = sats

        # Get metadata
        print("Reading patch metadata . . .")
        self.meta_patch = gpd.read_file(os.path.join(folder, "metadata.geojson"))
        self.meta_patch.index = self.meta_patch["ID_PATCH"].astype(int)
        self.meta_patch.sort_index(inplace=True)

        # Add regions column to allow for 4-fold CV using the different regions
        self.meta_patch["Region"] = self.meta_patch["ID_PATCH"].apply(lambda x: int(str(x)[0]))

        self.date_tables = {s: None for s in sats}
        self.date_range = np.array(range(-200, 600))
        for s in sats:
            dates = self.meta_patch["dates-{}".format(s)]
            date_table = pd.DataFrame(
                index=self.meta_patch.index, columns=self.date_range, dtype=int
            )
            for pid, date_seq in dates.items():
                if type(date_seq) == str:
                    date_seq = json.loads(date_seq)
                d = pd.DataFrame().from_dict(date_seq, orient="index")
                d = d[0].apply(
                    lambda x: (
                        datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
                        - self.reference_date
                    ).days
                )
                date_table.loc[pid, d.values] = 1
            date_table = date_table.fillna(0)
            self.date_tables[s] = {
                index: np.array(list(d.values()))
                for index, d in date_table.to_dict(orient="index").items()
            }

        print("Done.")

        # Validate inputs for cv_type and folds
        assert cv_type in ["official", "regions"], "cv_type must be one of 'official' or 'regions'."
        if folds is not None:
            if cv_type == "official":
                assert all(fold in range(1, 6) for fold in folds), "If cv_type='official', folds must be in the range 1 to 5."
            elif cv_type == "regions":
                assert all(fold in range(1, 5) for fold in folds), "If cv_type='regions', folds must be in the range 1 to 4."

        # Select Fold samples (official PASTIS-folds or regions)
        if folds is not None:
            if cv_type=="official":
                self.meta_patch = pd.concat(
                    [self.meta_patch[self.meta_patch["Fold"] == f] for f in folds]
                )
            else:
                self.meta_patch = pd.concat(
                    [self.meta_patch[self.meta_patch["Region"] == f] for f in folds]
                )

        if self.meta_patch.empty:
            raise ValueError("The selected fold(s) or region(s) resulted in an empty dataset. "
                             "Please check the fold/region numbers.")

        self.len = self.meta_patch.shape[0]
        self.id_patches = self.meta_patch.index

        # Get normalisation values
        if norm:
            self.norm = {}
            for s in self.sats:
                if cv_type=="official":
                    with open(
                        os.path.join(folder, "NORM_{}_patch.json".format(s)), "r"
                    ) as file:
                        normvals = json.loads(file.read())
                    selected_folds = folds if folds is not None else range(1, 6)
                    means = [normvals["Fold_{}".format(f)]["mean"] for f in selected_folds]
                    stds = [normvals["Fold_{}".format(f)]["std"] for f in selected_folds]
                    self.norm[s] = np.stack(means).mean(axis=0), np.stack(stds).mean(axis=0)
                    self.norm[s] = (
                        torch.from_numpy(self.norm[s][0]).float(),
                        torch.from_numpy(self.norm[s][1]).float(),
                    )
                else: # cv_type = regions
                    with open(
                        os.path.join(folder, "NORM_regions_{}_patch.json".format(s)), "r"
                    ) as file:
                        normvals = json.loads(file.read())
                    selected_regions = folds if folds is not None else range(1, 5)
                    means = [normvals["Region_{}".format(f)]["mean"] for f in selected_regions]
                    stds = [normvals["Region_{}".format(f)]["std"] for f in selected_regions]
                    self.norm[s] = np.stack(means).mean(axis=0), np.stack(stds).mean(axis=0)
                    self.norm[s] = (
                        torch.from_numpy(self.norm[s][0]).float(),
                        torch.from_numpy(self.norm[s][1]).float(),
                    )
        else:
            self.norm = None
        print("Dataset ready.")

    def __len__(self):
        return self.len

    def get_dates(self, id_patch, sat):
        return self.date_range[np.where(self.date_tables[sat][id_patch] == 1)[0]]

    def __getitem__(self, item):
        id_patch = self.id_patches[item]

        # Retrieve and prepare satellite data
        if not self.cache or item not in self.memory.keys():
            data = {
                satellite: np.load(
                    os.path.join(
                        self.folder,
                        "DATA_{}".format(satellite),
                        "{}_{}.npy".format(satellite, id_patch),
                    )
                ).astype(np.float32)
                for satellite in self.sats
            }  # T x C x H x W arrays
            data = {s: torch.from_numpy(a) for s, a in data.items()}

            if self.norm is not None:
                data = {
                    s: (d - self.norm[s][0][None, :, None, None])
                    / self.norm[s][1][None, :, None, None]
                    for s, d in data.items()
                }

            if self.target == "semantic":
                target = np.load(
                    os.path.join(
                        self.folder, "ANNOTATIONS", "TARGET_{}.npy".format(id_patch)
                    )
                )
                target = torch.from_numpy(target[0].astype(int))

                if self.class_mapping is not None:
                    target = self.class_mapping(target)

            if self.cache:
                if self.mem16:
                    self.memory[item] = [{k: v.half() for k, v in data.items()}, target]
                else:
                    self.memory[item] = [data, target]

        else:
            data, target = self.memory[item]
            if self.mem16:
                data = {k: v.float() for k, v in data.items()}

        # Retrieve date sequences
        if not self.cache or id_patch not in self.memory_dates.keys():
            dates = {
                s: torch.from_numpy(self.get_dates(id_patch, s)) for s in self.sats
            }
            if self.cache:
                self.memory_dates[id_patch] = dates
        else:
            dates = self.memory_dates[id_patch]

        if self.mono_date is not None:
            if isinstance(self.mono_date, int):
                data = {s: data[s][self.mono_date].unsqueeze(0) for s in self.sats}
                dates = {s: dates[s][self.mono_date] for s in self.sats}
            else:
                mono_delta = (self.mono_date - self.reference_date).days
                mono_date = {
                    s: int((dates[s] - mono_delta).abs().argmin()) for s in self.sats
                }
                data = {s: data[s][mono_date[s]].unsqueeze(0) for s in self.sats}
                dates = {s: dates[s][mono_date[s]] for s in self.sats}

        if self.mem16:
            data = {k: v.float() for k, v in data.items()}

        if len(self.sats) == 1:
            data = data[self.sats[0]]
            dates = dates[self.sats[0]]

        return (data, dates), target


def prepare_dates(date_dict, reference_date):
    d = pd.DataFrame().from_dict(date_dict, orient="index")
    d = d[0].apply(
        lambda x: (
            datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
            - reference_date
        ).days
    )
    return d.values


def compute_norm_vals(folder, sat, cv_type):
    norm_vals = {}
    
    if cv_type=="official":
        for fold in range(1, 6):
            dt = PASTIS_Dataset(folder=folder, norm=False, cv_type="official", 
                                folds=[fold], sats=[sat])
            means = []
            stds = []
            for i, b in enumerate(dt):
                print("{}/{}".format(i, len(dt)), end="\r")
                data = b[0][0]  # T x C x H x W
                data = data.permute(1, 0, 2, 3).contiguous()  # C x B x T x H x W
                means.append(data.view(data.shape[0], -1).mean(dim=-1).numpy())
                stds.append(data.view(data.shape[0], -1).std(dim=-1).numpy())

            mean = np.stack(means).mean(axis=0).astype(float)
            std = np.stack(stds).mean(axis=0).astype(float)

            norm_vals["Fold_{}".format(fold)] = dict(mean=list(mean), std=list(std))

        with open(os.path.join(folder, "NORM_{}_patch.json".format(sat)), "w") as file:
            file.write(json.dumps(norm_vals, indent=4))
    
    elif cv_type=="regions":
        for fold in range(1, 5):
            dt = PASTIS_Dataset(folder=folder, norm=False, cv_type="regions", 
                                folds=[fold], sats=[sat])
            means = []
            stds = []
            for i, b in enumerate(dt):
                print("{}/{}".format(i, len(dt)), end="\r")
                data = b[0][0]  # T x C x H x W
                data = data.permute(1, 0, 2, 3).contiguous()  # C x B x T x H x W
                means.append(data.view(data.shape[0], -1).mean(dim=-1).numpy())
                stds.append(data.view(data.shape[0], -1).std(dim=-1).numpy())

            mean = np.stack(means).mean(axis=0).astype(float)
            std = np.stack(stds).mean(axis=0).astype(float)

            norm_vals["Region_{}".format(fold)] = dict(mean=list(mean), std=list(std))
        
        with open(os.path.join(folder, "NORM_regions_{}_patch.json".format(sat)), "w") as file:
            file.write(json.dumps(norm_vals, indent=4))
    else:
        raise ValueError("cv_type should be one of 'official' or 'regions'.")
    