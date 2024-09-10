import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
      
class ClimateTransformerEncoder(nn.Module):
    def __init__(self, 
                 climate_input_dim=11,
                 d_model=64,
                 nhead=4,
                 d_ffn=128,
                 num_layers=1):
        super(ClimateTransformerEncoder, self).__init__()
        
        self.climate_projection = nn.Linear(climate_input_dim, d_model)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        transformer_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ffn,
            batch_first=True
        )

        self.climate_transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=num_layers
        )
    
    def forward(self, climate_data):
        climate_data = self.climate_projection(climate_data) # (B x T' x d_model)
        batch_size = climate_data.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        
        climate_data = torch.cat((cls_token, climate_data), dim=1)
        climate_embedding = self.climate_transformer(climate_data)
        cls_embedding = climate_embedding[:, 0, :] # (B x d_model)

        return cls_embedding


class LateFusionModel(nn.Module):
    def __init__(self, 
                 utae_model, 
                 d_model=64, 
                 nhead=4, 
                 d_ffn=128, 
                 num_layers=1, 
                 num_classes=20, 
                 out_channels=128, 
                 climate_input_dim=11):
        super(LateFusionModel, self).__init__()
        
        self.utae_model = utae_model
        self.d_model = d_model

        # Use the ClimateTransformerEncoder for climate data encoding
        self.climate_transformer_encoder = ClimateTransformerEncoder(
            climate_input_dim=climate_input_dim, 
            d_model=d_model, 
            nhead=nhead, 
            d_ffn=d_ffn, 
            num_layers=num_layers
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.utae_model.decoder_widths[0] + d_model,  # adding the global embedding of climate data (with d_model channels)
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d( 
                in_channels=out_channels, 
                out_channels=num_classes,
                kernel_size=1
            ) 
        )
    
    def forward(self, input_sat, dates_sat, input_clim):
        # input_sat: tensor of satellite images (B x T x C x H x W)
        # dates_sat: tensor of satellite dates (B x T) (used as batch_positions)
        # input_clim: tensor of climate data (B x T' x V) 

        # Process satellite data
        satellite_features, maps = self.utae_model(input_sat, batch_positions=dates_sat)
    
        # Process climate data with the transformer encoder
        climate_embedding = self.climate_transformer_encoder(input_clim)  # (B x d_model)

        # Expand climate embedding to match spatial dimension
        climate_embedding = climate_embedding.unsqueeze(-1).unsqueeze(-1)  # (B x d_model x 1 x 1)
        climate_embedding = climate_embedding.expand(-1, -1, 
                                                     satellite_features.size(2), 
                                                     satellite_features.size(3))  # (B x d_model x H x W)

        # Concatenate satellite features and climate embedding
        combined_features = torch.cat((satellite_features, climate_embedding), dim=1)
        
        # Final convolution to generate the output
        output = self.final_conv(combined_features)

        return output


class EarlyFusionModel(nn.Module):
    def __init__(self,
                 utae_model,
                 climate_input_dim=11,
                 d_model=64,
                 fusion_strategy='match_dates',
                 mlp_hidden_dim=64,
                 pad_value=-1000):
        super(EarlyFusionModel, self).__init__()

        self.utae_model = utae_model 
        self.fusion_strategy = fusion_strategy
        self.pad_value = pad_value

        self.climate_mlp = nn.Sequential(
            nn.Linear(climate_input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, d_model)
        )

    def fuse_climate_data(self, sat_data, sat_dates, climate_data, climate_dates):
        fused_data = []

        for b in range(sat_data.size(0)): # Loop over the batch
            batch_sat_data = sat_data[b]
            batch_sat_dates = sat_dates[b]
            batch_climate_data = climate_data[b]
            batch_climate_dates = climate_dates[b]

            batch_fused = []

            for t, sat_date in enumerate(batch_sat_dates):

                if self.fusion_strategy == 'match_dates':
                    climate_indices = (batch_climate_dates == sat_date).nonzero(as_tuple=True)[0]
                    if len(climate_indices) > 0:
                        clim_vec = batch_climate_data[climate_indices[0]]
                    else:
                        # fill with pad value used for satellite data (-1000)
                        clim_vec = torch.full((batch_climate_data.size(1),), float(self.pad_value), device=sat_data.device)
                
                elif self.fusion_strategy == 'weekly':
                    clim_indices = (batch_climate_dates < sat_date) & (batch_climate_dates >= (sat_date - 7))
                    
                    if clim_indices.any():
                        clim_vec = batch_climate_data[clim_indices].mean(dim=0)
                    else:
                        clim_vec = torch.full((batch_climate_data.size(1),), float(self.pad_value), device=sat_data.device)
                else:
                    raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
            
                if hasattr(self, 'climate_mlp'):
                    clim_vec = self.climate_mlp(clim_vec)
                
                clim_vec_expanded = clim_vec.unsqueeze(-1).unsqueeze(-1).expand(-1, batch_sat_data.size(2), batch_sat_data.size(3))
                combined = torch.cat((batch_sat_data[t], clim_vec_expanded), dim=0) # ((C + d_model) x H x W)
                batch_fused.append(combined)

            batch_fused = torch.stack(batch_fused, dim=0) # (T x (C + d_model) x H x W)
            fused_data.append(batch_fused)
        
        fused_data = torch.stack(fused_data, dim=0) # (B x T x (C + d_model) x H x W)
        return fused_data

    def forward(self, input_sat, dates_sat, input_clim, dates_clim):
        
        fused_data = self.fuse_climate_data(input_sat, dates_sat, input_clim, dates_clim)

        output = self.utae_model(fused_data, batch_positions=dates_sat)

        return output