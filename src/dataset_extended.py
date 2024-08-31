import os
import json
import torch
import torch.nn as nn
import torch.utils.data as tdata
import numpy as np
import pandas as pd
from datetime import datetime
import geopandas as gpd

class PASTIS_Climate_Dataset(tdata.Dataset):
    def __init__(
        self,
        folder,
        climate_folder,
        norm=True,
        target="semantic",
        fusion=None,
        cache=False,
        mem16=False,
        cv_type="official",
        folds=None,
        reference_date="2018-09-01",
        class_mapping=None,
        mono_date=None,
        sats=["S2"],
        apply_noise=True,
        noise_std=0.01,
        climate_input_dim=11,
        climate_apply_mlp=False,
        mlp_hidden_dim=64,
        climate_transformer_encoder=None
    ):
        super(PASTIS_Climate_Dataset, self).__init__()
        self.folder = folder
        self.climate_folder = climate_folder
        self.norm = norm
        self.reference_date = datetime(*map(int, reference_date.split("-")))
        self.cache = cache
        self.mem16 = mem16
        self.mono_date = mono_date
        self.memory = {}
        self.memory_dates = {}
        self.class_mapping = (
            np.vectorize(lambda x: class_mapping[x])
            if class_mapping is not None
            else class_mapping
        )
        self.target = target
        self.fusion = fusion
        self.sats = sats
        self.apply_noise = apply_noise
        self.noise_std = noise_std
        self.climate_apply_mlp = climate_apply_mlp

        if self.fusion == "early_match_dates":
            if climate_apply_mlp:
                print("Applying MLP to climate data")
                self.climate_mlp = ClimateMLP(input_dim=climate_input_dim, 
                                            output_dim=climate_input_dim, 
                                            hidden_dim=mlp_hidden_dim).to('cuda' if torch.cuda.is_available() else 'cpu')
        elif self.fusion == "early_weekly":
            if climate_transformer_encoder is not None:
                self.climate_transformer_encoder = climate_transformer_encoder
            else:
                raise ValueError("When using 'early_weekly' as fusion strategy, must specify climate_transformer_encoder.")

        # Get metadata
        print("Reading patch metadata . . .")
        self.meta_patch = gpd.read_file(os.path.join(folder, "metadata.geojson"))
        self.meta_patch.index = self.meta_patch["ID_PATCH"].astype(int)
        self.meta_patch.sort_index(inplace=True)
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

        # Load and normalize climate data
        print("Loading climate data . . .")
        self.climate_data = {}
        for var in os.listdir(self.climate_folder):
            if var.split('.')[1] != "csv":
                continue
            var_name = var.split('.')[0]
            df = pd.read_csv(os.path.join(self.climate_folder, var), index_col=0)
            if self.norm:
                mean = df.mean(axis=0)
                std = df.std(axis=0)
                df = (df - mean) / std
            self.climate_data[var_name] = df
        print("Climate data loaded.")

        # Get normalization values for satellite data
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

    def get_intervals_climate_data(self, dates_sat, climate_data):
        interval_climate_data = []

        # iterate over pairs of consecutive observation dates
        for i, current_date in enumerate(dates_sat):
            if i == 0:
                # for the first observation, use up to 7 days before the first date
                start_date = max(current_date - 7, 0)
            else:
                # use the interval between the previous and current observation date
                previous_date = dates_sat[i - 1]
                start_date = max(current_date - 7, previous_date)

            # extract climate data from start_date to current_date
            interval_data = climate_data[start_date:current_date]
            interval_climate_data.append(interval_data)

        return interval_climate_data

    def encode_climate_data(self, interval_climate_data):
        # convert interval data to tensors and handle padding
        sequences = [data.clone().detach().to('cuda' if torch.cuda.is_available() else 'cpu') 
                    if isinstance(data, torch.Tensor) 
                    else torch.tensor(data, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
                    for data in interval_climate_data]
        
        padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        
        # create embedding using the climate transformer encoder module
        climate_embedding = self.climate_transformer_encoder(padded_sequences)  # Output: (T x d_model)
        return climate_embedding.to('cpu')

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

        # Retrieve date sequences for satellite data
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

        # Load climate data
        climate_data = torch.stack(
            [torch.tensor(self.climate_data[var].loc[:, str(id_patch)].values).float()
             for var in self.climate_data], dim=1
        )

        # Calculate climate dates relative to reference_date
        climate_start_date = self.reference_date
        num_days = climate_data.size(0)
        climate_dates = [(climate_start_date + pd.Timedelta(days=int(i))) for i in range(num_days)]
        climate_dates = [(date - self.reference_date).days for date in climate_dates]
        climate_dates = torch.tensor(climate_dates, dtype=torch.int32)

        if self.apply_noise:
            climate_data += torch.normal(0, self.noise_std, size=climate_data.shape)
        
        if self.fusion is not None:
            sat_data, sat_dates = data, dates
            
            if self.fusion == "early_match_dates":
                if self.climate_apply_mlp:
                    climate_data = climate_data.to('cuda' if torch.cuda.is_available() else 'cpu')
                    climate_data = self.climate_mlp(climate_data)

                matched_data = []
                for t, date in enumerate(sat_dates):
                    sat_data_t = sat_data[t]       # (C x H x W) 
                    
                    clim_idx = torch.where(climate_dates == date)[0]

                    if len(clim_idx) > 0:
                        clim_vec = climate_data[clim_idx[0]] # (V)
                    else:
                        clim_vec = torch.zeros(climate_data.size(1)) # (V)
                    
                    clim_vec_expanded = clim_vec.unsqueeze(1).unsqueeze(2).expand(-1, 
                                                                                  sat_data.size(2), 
                                                                                  sat_data.size(3)) # (V x H x W)
                    # concatenate along channel dimension 
                    combined = torch.cat((sat_data_t, clim_vec_expanded), dim=0) # ((C+V) x H x W)
                    matched_data.append(combined)

                matched_data = torch.stack(matched_data, dim=0)  # (T x (C+V) x H x W)

                return (matched_data, sat_dates), target
            
            elif self.fusion == "early_weekly":
                # extract climate data intervals between satellite observation dates
                interval_climate_data = self.get_intervals_climate_data(sat_dates, climate_data)
                
                # encode climate data
                climate_embedding = self.encode_climate_data(interval_climate_data) # (T x d_model)

                # Fusion Step
                # expand climate embedding to match spatial dim. of satellite data
                climate_embedding = climate_embedding.unsqueeze(-1).unsqueeze(-1) # (T x d_model x 1 x 1)
                climate_embedding = climate_embedding.expand(-1, -1,
                                                             sat_data.size(2),
                                                             sat_data.size(3)) # (T x d_model x H x W)
                combined = torch.cat((sat_data, climate_embedding), dim=1) # (T x (C + d_model) x H x W)

                return (combined, sat_dates), target

        return {
            "input_satellite": (data, dates),
            "input_climate": (climate_data, climate_dates),
            "target": target
        }

class ClimateMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(ClimateMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)