import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer

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

        self.climate_projection = nn.Linear(climate_input_dim, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        transformer_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ffn
        )
        self.climate_transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=num_layers
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.utae_model.decoder_widths[0] + d_model, # adding the global embedding of climate data (with d_model channels)
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, 
                      out_channels=num_classes,
                      kernel_size=1) 
        )
    
    def forward(self, input_sat, dates_sat, input_clim):
        # input_sat will be tensor of satellite images (B x T x C x H x W)
        # dates_sat will be tensor of satellite dates (B x T) (used as batch_positions)
        # input_clim will be tensor of climate data (B x T' x V) 

        # process satellite data
        satellite_features, maps = self.utae_model(input_sat, batch_positions=dates_sat)
    
        # prepare climate data
        climate_data = self.climate_projection(input_clim) # (B x T' x d_model)
        batch_size = climate_data.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        climate_data = torch.cat((cls_token, climate_data), dim=1) # (B x 1 + T' x d_model)
        
        # process climate data with transformer
        climate_embedding = self.climate_transformer(climate_data)
        climate_embedding = climate_embedding[:, 0, :] # extracting CLS token output (B x d_model)

        # expand climate embedding to match spatial dimension
        climate_embedding = climate_embedding.unsqueeze(-1).unsqueeze(-1) # (B x d_model x 1 x 1)
        climate_embedding = climate_embedding.expand(-1, -1, 
                                                     satellite_features.size(2), 
                                                     satellite_features.size(3)) # (B x d_model x H x W)

        combined_features = torch.cat((satellite_features, climate_embedding), dim = 1)
        output = self.final_conv(combined_features)

        return output
        


