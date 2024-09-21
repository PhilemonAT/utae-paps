import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
      
class ClimateTransformerEncoder(nn.Module):
    """
    Initializes the ClimateTransformerEncoder module.
    Args:
        climate_input_dim (int): Number of input features for climate data.
        d_model (int): Dimension of the model (hidden size).
        nhead (int): Number of attention heads.
        d_ffn (int): Dimension of the feedforward network.
        num_layers (int): Number of transformer encoder layers.
        use_cls_token (bool): Whether to use a CLS token for sequence classification.
    """
    def __init__(self, 
                 climate_input_dim=11,
                 d_model=64,
                 nhead=4,
                 d_ffn=128,
                 num_layers=1,
                 use_cls_token=True,
                 max_length=5000):
        super(ClimateTransformerEncoder, self).__init__()
        
        self.climate_projection = nn.Linear(climate_input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_length=max_length)
        self.use_cls_token = use_cls_token

        if use_cls_token:
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
    
    def forward(self, climate_data, mask=None):
        """
        Args:
            climate_data (Tensor): Input climate data of shape (B x T' x climate_input_dim).
            mask (Tensor, optional): Mask to be applied in the transformer encoder.

        Returns:
            Tensor: The output embedding of shape (B x d_model) if using CLS token, otherwise (B x T' x d_model).
        """
        climate_data = self.climate_projection(climate_data) # (B x T' x d_model)
        climate_data = self.positional_encoding(climate_data) # add positional encoding information
        batch_size, seq_len, _ = climate_data.size() 

        if self.use_cls_token:
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            climate_data = torch.cat((cls_token, climate_data), dim=1) # (B x (T'+1) x d_model)
            seq_len += 1

        # Apply transformer encoder
        if mask is not None:
            climate_embedding = self.climate_transformer(climate_data, mask=mask, 
                                                         is_causal=True)
        else:
            climate_embedding = self.climate_transformer(climate_data)

        # Return CLS token embedding if used, otherwise return full sequence embedding
        if self.use_cls_token:
            cls_embedding = climate_embedding[:, 0, :] # (B x d_model)
            return cls_embedding
        else:
            return climate_embedding # (B x T' x d_model)


class EarlyFusionModel(nn.Module):
    def __init__(self,
                 utae_model,
                 climate_input_dim=11,
                 d_model=64,
                 fusion_strategy='match_dates',
                 use_climate_mlp=False,
                 mlp_hidden_dim=64,
                 nhead_climate_transformer=4,
                 d_ffn_climate_transformer=128,
                 num_layers_climate_transformer=1):
        """
        Initializes the EarlyFusionModel, which integrates climate data into the satellite 
        data stream at an early stage before passing the combined data through the U-TAE model.

        This model supports different strategies for fusing climate data, such as matching 
        climate data with satellite observation dates, using weekly climate data, or employing 
        a causal approach. It can also use an MLP to preprocess climate data and FiLM layers 
        to modulate satellite features based on climate data.

        Args:
            utae_model (nn.Module): The U-TAE model instance used for processing satellite data. 
            climate_input_dim (int): Number of input features in the climate data.
            d_model (int): The dimensionality used for embeddings.
            fusion_strategy (str): Strategy for fusing climate data, can be:
                - 'match_dates': Use only climate data from the exact dates matching the satellite observations.
                - 'weekly': Use climate data from the whole week prior to each satellite observation.
                - 'causal': Use a causal transformer to incorporate climate data sequentially.
            use_climate_mlp (bool): If True, a small MLP processes the climate data before fusion to 
                                    align it with the satellite data's dimensions.
            mlp_hidden_dim (int): Hidden dimension size of the MLP used when `use_climate_mlp` is True.
        """
        super(EarlyFusionModel, self).__init__()
 
        self.utae_model = utae_model 
        self.fusion_strategy = fusion_strategy
        self.use_climate_mlp = use_climate_mlp
        self.pad_value = self.utae_model.pad_value

        if use_climate_mlp:
            self.climate_mlp = nn.Sequential(
                nn.Linear(climate_input_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, d_model)
            )


        if fusion_strategy == 'causal':
            assert use_climate_mlp == False, "Using climate MLP with causal fusion not implemented"

            self.causal_transformer_encoder = ClimateTransformerEncoder(
                climate_input_dim=climate_input_dim,
                d_model=d_model,
                nhead=nhead_climate_transformer,
                d_ffn=d_ffn_climate_transformer,
                num_layers=num_layers_climate_transformer,
                use_cls_token=False # not using CLS token; need per time step embeddings
            )
                    
    def transform_climate_data(self, sat_data, sat_dates, climate_data, climate_dates):
        """
        Fuses satellite and climate data based on the chosen fusion strategy.

        Args:
            sat_data (Tensor): Satellite data of shape (B x T x C x H x W).
            sat_dates (Tensor): Dates corresponding to the satellite data (B x T).
            climate_data (Tensor): Climate data of shape (B x T' x climate_input_dim).
            climate_dates (Tensor): Dates corresponding to the climate data (B x T').

        Returns:
            climate_matched (Tensor): Processed climate data of shape (B x T x V) or (B x T x d_model).
                                      When using 'causal' as fusion_strategy, the latter dimension is
                                      always d_model. Else, it's V, except if use_climate_mlp was True.
        """

        batch_size, num_sat_timesteps, _, height, width = sat_data.size()

        if self.fusion_strategy == 'causal':
            # Create causal mask for the entire sequence
            causal_mask = torch.triu(torch.ones((climate_dates.size(1), climate_dates.size(1)),
                                                device=climate_data.device), diagonal=1)
            
            # expand mask for batch and heads
            num_heads = self.causal_transformer_encoder.climate_transformer.layers[0].self_attn.num_heads
            causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1) # (B x T' x T')
            causal_mask = causal_mask.unsqueeze(1).expand(-1, num_heads, -1, -1) # (B x Heads x T' x T')
            causal_mask = causal_mask.reshape(batch_size * num_heads, climate_dates.size(1), climate_dates.size(1)) # (B*H, T', T')
            causal_mask = causal_mask.masked_fill(causal_mask==1, float('-inf'))          

            print("causal_mask has shape: ", causal_mask.shape)

            # compute climate embeddings for the entire sequence
            climate_embeddings = self.causal_transformer_encoder(climate_data, 
                                                                 mask=causal_mask
                                                                 ) # (B x T' x d_model)

            print("climate_embeddings have shape: ", climate_embeddings.shape)

            climate_matched = []

            for b in range(batch_size):
                batch_climate = []
                for t in range(num_sat_timesteps):
                    sat_date = sat_dates[b, t].item()

                    if sat_date != self.pad_value:
                        # Find the closest valid climate embedding before or at each satellite observation date
                        closest_climate_idx = (climate_dates[b] <= sat_dates[b, t].item()).nonzero(as_tuple=True)[0].max()
                        clim_vec = climate_embeddings[b, closest_climate_idx]  # (d_model,)
                    else:
                        # sat_date corresponds to a padded entry --> fill with pad_value
                        clim_vec = torch.full((climate_embeddings.size(-1),), float(self.pad_value), device=climate_data.device) 
                                       
                    batch_climate.append(clim_vec)
                
                climate_matched.append(torch.stack(batch_climate, dim=0)) # (T x d_model)
            
            climate_matched = torch.stack(climate_matched, dim=0) # (B x T x d_model)
            print("climate_matched has shape: ", climate_matched.shape)
            return climate_matched

        # other fusion strategies ('match_dates' or 'weekly')
        climate_matched = []

        for b in range(sat_data.size(0)): # Loop over the batch
            batch_sat_data = sat_data[b]
            batch_sat_dates = sat_dates[b]
            batch_climate_data = climate_data[b]
            batch_climate_dates = climate_dates[b]

            batch_climate = []

            for t, sat_date in enumerate(batch_sat_dates): # Loop over time steps

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
                    clim_vec = self.climate_mlp(clim_vec) # d_model
              
                batch_climate.append(clim_vec)

            batch_climate = torch.stack(batch_climate, dim=0) # (T x V) or (T x d_model) if MLP applied
            climate_matched.append(batch_climate)
        
        climate_matched = torch.stack(climate_matched, dim=0) # (B x T x V) or (B x T x d_model) if MLP applied
        return climate_matched

    def forward(self, input_sat, dates_sat, input_clim, dates_clim):
        """
        Forward pass through the EarlyFusionModel.

        Args:
            input_sat (Tensor): Input satellite data of shape (B x T x C x H x W).
            dates_sat (Tensor): Dates corresponding to the satellite data (B x T).
            input_clim (Tensor): Input climate data of shape (B x T' x climate_input_dim).
            dates_clim (Tensor): Dates corresponding to the climate data (B x T').

        Returns:
            Tensor: Output of the model after early fusion of satellite and climate data.
        """
        climate_matched = self.transform_climate_data(input_sat, dates_sat, input_clim, dates_clim)
        output = self.utae_model(input=input_sat, 
                                 climate_input=climate_matched, 
                                 batch_positions=dates_sat)

        # call UTAE here with input_sat as input images, dates_sat as batch_positions,
        #                     climate_matched as the corresponding transformed climate data 
        # Note: When using 'causal', climate_matched has size (B x T x d_model)
        #       When using 'match_dates' or 'weekly' and NOT using MLP, climate_matched has size (B x T x V)
        #       ---------------------------------------- USING MLP,     climate_matched has size (B x T x d_model)

        return output


class LateFusionModel(nn.Module):
    def __init__(self, 
                 utae_model,
                 d_model=64, 
                 nhead_climate_transformer=4, 
                 d_ffn_climate_transformer=128, 
                 num_layers_climate_transformer=1, 
                 out_conv=[32, 32, 20],
                 climate_input_dim=11,
                 use_FILM_late=False):
        """
        Initializes the LateFusionModel which combines satellite image time series and 
        climate data at a later stage in the network. The model uses a U-TAE model for 
        processing satellite data and a transformer encoder for climate data, followed by 
        a fusion of their features.

        Args:
            utae_model (nn.Module): The U-TAE model instance used for processing satellite data 
                (outputs satellite features and maps).
            d_model (int): Dimension of the transformer model for climate data.
            nhead (int): Number of attention heads in the transformer encoder.
            d_ffn (int): Dimension of the feedforward network inside the transformer encoder.
            num_layers (int): Number of layers in the transformer encoder.
            out_conv (int): Number of output channels for the final convolutional layers.
            climate_input_dim (int): Number of climate variables in the input data.
        """
        super(LateFusionModel, self).__init__()
        
        self.utae_model = utae_model
        self.d_model = d_model
        self.use_FILM_late = use_FILM_late

        # Use the ClimateTransformerEncoder for climate data encoding
        self.climate_transformer_encoder = ClimateTransformerEncoder(
            climate_input_dim=climate_input_dim, 
            d_model=d_model, 
            nhead=nhead_climate_transformer, 
            d_ffn=d_ffn_climate_transformer, 
            num_layers=num_layers_climate_transformer,
            use_cls_token=True
        )

        if use_FILM_late:
            input_dim = self.utae_model.decoder_widths[0]
            self.FILM_layer = FiLM(clim_vec_dim=d_model,
                                   sat_feature_dim=input_dim)
        else:
            # accounting for concatenating the global embedding of climate data (with d_model channels)
            input_dim = self.utae_model.decoder_widths[0] + d_model 

        self.final_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=input_dim,  
                out_channels=out_conv[0],
                kernel_size=3,
                padding=1,
                stride=1,
                padding_mode="reflect"
            ),
            nn.BatchNorm2d(out_conv[0]),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_conv[0],  
                out_channels=out_conv[1],
                kernel_size=3,
                padding=1,
                stride=1,
                padding_mode="reflect"
            ),
            nn.BatchNorm2d(out_conv[1]),
            nn.ReLU(),
            nn.Conv2d( 
                in_channels=out_conv[1], 
                out_channels=out_conv[2],
                kernel_size=3,
                padding=1,
                stride=1,
                padding_mode="reflect"
            )
        )
    
    def forward(self, input_sat, dates_sat, input_clim):
        """
        Forward pass through the LateFusionModel.

        In this model, satellite and climate data are processed separately using their 
        respective models (U-TAE for satellite data and a transformer encoder for climate data). 
        The climate embeddings are then spatially expanded and concatenated with the satellite 
        features before passing through the final convolutional layers for prediction.

        Args:
            input_sat (Tensor): Input satellite data of shape (B x T x C x H x W).
            dates_sat (Tensor): Dates corresponding to the satellite data of shape (B x T).
            input_clim (Tensor): Input climate data of shape (B x T' x V).

        Returns:
            Tensor: Output of the model after late fusion of satellite and climate data.
        """

        # Process satellite data
        satellite_features, _ = self.utae_model(input_sat, batch_positions=dates_sat)
    
        # Process climate data with the transformer encoder
        climate_embedding = self.climate_transformer_encoder(input_clim)  # (B x d_model)

        if self.use_FILM_late:
            combined_features = self.FILM_layer(satellite_features, climate_embedding)

        else:
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
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length = 5000):
        """
        Args:
            d_model:    dimension of embeddings
            max_length: max sequence length
        """
        super().__init__()

        pe = torch.zeros(max_length, d_model)

        k = torch.arange(0, max_length).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(k * div_term)
        pe[:, 1::2] = torch.cos(k * div_term)

        pe = pe.unsqueeze(0)
        
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: embeddings                     (B x T x d_model)
        
        Returns:
            embeddings + positional encodings (B x T x d_model)
        
        """
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)

        return x
    

class FiLM(nn.Module):
    def __init__(self,
                 clim_vec_dim,
                 sat_feature_dim,
                 hidden_dim=128):
        super(FiLM, self).__init__()
        """
        Initializes the FiLMLayer module, which applies Feature-wise Linear Modulation (FiLM)
        to modulate satellite features based on climate data. 

        FiLM, as described in the paper "FiLM: Visual Reasoning with a General Conditioning 
        Layer" by Perez et al. (2018), modulates neural network feature maps through an affine 
        transformation (scaling and shifting) using learned parameters (gamma and beta) that 
        are conditioned on some external input — in this case, the climate data.

        Args:
            clim_vec_dim (int): Dimension of the climate vector input.
            sat_feature_dim (int): Dimension of the satellite feature maps to be modulated.
            hidden_dim (int): Hidden dimension size for the MLP used to generate FiLM parameters.
        """
        
        self.mlp = nn.Sequential(
            nn.Linear(clim_vec_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * sat_feature_dim), # outputs both gamma and beta
            nn.Sigmoid()                                # since we are only working with standardized data
        )

    def forward(self, sat_features, clim_vec):
        """
        Applies FiLM modulation to satellite features using climate vector.

        The FiLM modulation scales and shifts the satellite features based on parameters 
        generated from the climate vector. This is achieved by computing gamma (scale) and beta 
        (shift) parameters from the climate vector through an MLP, then applying these 
        parameters to the satellite features.

        Args:
            sat_features (Tensor): Satellite feature maps of shape (B x decoder_widths[0] x H x W) 
            clim_vec (Tensor): Climate vector of shape (B x d_model)
            
        Returns:
            Tensor: Modulated satellite features of shape (B x decoder_widths[0] x H x W)
        """
        
        film_params = self.mlp(clim_vec)        # (B x 2*decoder_widths[0])
        
        # split along last dimension
        gamma, beta = film_params.chunk(2, dim=-1)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)   # (B x decoder_widths[0] x 1 x 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)     # (B x decoder_widths[0] x 1 x 1)

        modulated_features = gamma * sat_features + beta

        return modulated_features # (B x T x C x H x W)