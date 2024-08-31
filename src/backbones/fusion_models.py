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




# class EarlyFusionModel(nn.Module):
#     def __init__(self, utae_model, climate_input_dim=11):
#         super(EarlyFusionModel, self).__init__()

#         self.utae_model = utae_model
#         self.climate_input_dim = climate_input_dim
        
#     def forward(self, input_sat, dates_sat, input_clim, dates_clim):
        
#         fused_data = []
        
#         for t in range(input_sat.size(1)): # loop over the temporal dimension T
            
#             sat_t = input_sat[:, t] # (B x C x H x W)

#             # create mask where dates are not zero (valid dates, otherwise date is padded)
#             valid_mask = dates_sat[:, t] != 0 # (B, )

#             clim_t_expanded = torch.zeros(input_sat.size(0), self.climate_input_dim, 
#                                           sat_t.size(2), sat_t.size(3), device=sat_t.device) # (B x V x H x W)
            
#             # for valid dates, find corresponding climate data
#             if valid_mask.any():
#                 clim_idx = torch.where(valid_mask, 
#                                        (dates_clim == dates_sat[:, t].unsqueeze(1)).nonzero(
#                                            as_tuple=True)[1], torch.tensor(-1, device=sat_t.device)
#                                            )
#                 valid_clim_idx = clim_idx[clim_idx >= 0]
#                 valid_sat_idx = torch.nonzero(valid_mask, as_tuple=True)[0]  # Indices of valid entries in the batch
#                 if len(valid_clim_idx) > 0:
#                     clim_t = input_clim[valid_sat_idx, valid_clim_idx] # (number of valid entries x V)
#                     clim_t_expanded[valid_sat_idx] = clim_t.unsqueeze(2).unsqueeze(3).expand(-1, -1, 
#                                                                                           sat_t.size(2), 
#                                                                                           sat_t.size(3))  # (B x V x H x W)
#             fused_t = torch.cat((sat_t, clim_t_expanded), dim=1) # (B x (C + V) x H x W)
#             fused_data.append(fused_t)
#         fused_data = torch.stack(fused_data, dim=1) # (B x T x (C + V) x H x W)
        
#         output = self.utae_model(fused_data, batch_positions=dates_sat)
        
#         return output