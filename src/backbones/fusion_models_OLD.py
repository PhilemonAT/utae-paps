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
                 use_cls_token=True):
        super(ClimateTransformerEncoder, self).__init__()
        
        self.climate_projection = nn.Linear(climate_input_dim, d_model)
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
    
    def forward(self, climate_data, mask=None, is_causal=False):
        """
        Args:
            climate_data (Tensor): Input climate data of shape (B x T' x climate_input_dim).
            mask (Tensor, optional): Mask to be applied in the transformer encoder.
            is_causal (bool, optional): If True, applies a causal mask.

        Returns:
            Tensor: The output embedding of shape (B x d_model) if using CLS token, otherwise (B x T' x d_model).
        """
        climate_data = self.climate_projection(climate_data) # (B x T' x d_model)
        batch_size = climate_data.size(0)

        if self.use_cls_token:
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            climate_data = torch.cat((cls_token, climate_data), dim=1)

        # Apply transformer encoder
        if mask is not None:
            climate_embedding = self.climate_transformer(climate_data, mask=mask, is_causal=True)
        else:
            climate_embedding = self.climate_transformer(climate_data)

        if self.use_cls_token:
            cls_embedding = climate_embedding[:, 0, :] # (B x d_model)
            # output cls token embedding only (no temporal dimension)
            return cls_embedding
        else:
            # output full sequence embedding
            return climate_embedding # (B x T' x d_model)

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
            num_classes (int): Number of output classes for the final classification.
            out_channels (int): Number of output channels for the final convolutional layers.
            climate_input_dim (int): Number of climate variables in the input data.
        """
        super(LateFusionModel, self).__init__()
        
        self.utae_model = utae_model
        self.d_model = d_model

        # Use the ClimateTransformerEncoder for climate data encoding
        self.climate_transformer_encoder = ClimateTransformerEncoder(
            climate_input_dim=climate_input_dim, 
            d_model=d_model, 
            nhead=nhead, 
            d_ffn=d_ffn, 
            num_layers=num_layers,
            use_cls_token=True
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
                 use_climate_mlp=False,
                 mlp_hidden_dim=64,
                 use_film=False,
                 pad_value=-1000):
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
            use_film (bool): If True, applies Feature-wise Linear Modulation (FiLM) to fuse climate 
                             data with satellite data, modulating satellite features based on climate data.
            pad_value (float): Value used for padding missing data or aligning temporal sequences.
        """
        super(EarlyFusionModel, self).__init__()
 
        self.utae_model = utae_model 
        self.fusion_strategy = fusion_strategy
        self.use_climate_mlp = use_climate_mlp
        self.use_film = use_film
        self.pad_value = pad_value

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
                nhead=4,
                d_ffn=128,
                num_layers=1,
                use_cls_token=False # not using CLS token; need per time step embeddings
            )

            if use_film:
                self.FiLM = FiLMLayer(
                    clim_vec_dim=d_model,
                    sat_feature_dim=10
                )
        
        else:
            if use_film:
                self.FiLM = FiLMLayer(
                    clim_vec_dim=climate_input_dim,
                    sat_feature_dim=10
                )
            
    def fuse_climate_data(self, sat_data, sat_dates, climate_data, climate_dates):
        """
        Fuses satellite and climate data based on the chosen fusion strategy.

        Args:
            sat_data (Tensor): Satellite data of shape (B x T x C x H x W).
            sat_dates (Tensor): Dates corresponding to the satellite data (B x T).
            climate_data (Tensor): Climate data of shape (B x T' x climate_input_dim).
            climate_dates (Tensor): Dates corresponding to the climate data (B x T').

        Returns:
            Tensor: Fused data of shape (B x T x (C + d_model) x H x W) or (B x T x C x H x W) if FiLM is applied.
        """
        fused_data = []

        for b in range(sat_data.size(0)): # Loop over the batch
            batch_sat_data = sat_data[b]
            batch_sat_dates = sat_dates[b]
            batch_climate_data = climate_data[b]
            batch_climate_dates = climate_dates[b]

            batch_fused = []

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

                elif self.fusion_strategy == 'causal':
                    causal_mask = (batch_climate_dates.unsqueeze(0) < sat_date).unsqueeze(0) # (1 x 1 x T_clim)
                    
                    num_heads = self.causal_transformer_encoder.climate_transformer.layers[0].self_attn.num_heads 

                    # Expand mask to match batch * n_head and sequence dimensions correctly
                    causal_mask = causal_mask.expand(1 * num_heads, 
                                                     batch_climate_data.size(0), 
                                                     batch_climate_data.size(0))  # (B*n_heads, T_clim, T_clim)
                    
                    climate_embedding = self.causal_transformer_encoder(
                        batch_climate_data.unsqueeze(0), mask=causal_mask, is_causal=True
                    )[0] # (T_clim x d_model)

                    # use embedding corresponding to current time step
                    if sat_date != self.pad_value:
                        # if the current time step does not correspond to a padded entry
                        clim_vec = climate_embedding[t] # vector of size (d_model)
                    else:
                        # the current time step corresponds to a padded entry --> use padded clim_vec
                        clim_vec = torch.full((batch_climate_data.size(1),), float(self.pad_value), device=sat_data.device)

                else:
                    raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

                if not self.use_film:
                    if hasattr(self, 'climate_mlp'):
                        clim_vec = self.climate_mlp(clim_vec)
            
                    clim_vec_expanded = clim_vec.unsqueeze(-1).unsqueeze(-1).expand(-1, batch_sat_data.size(2), batch_sat_data.size(3))
                    combined = torch.cat((batch_sat_data[t], clim_vec_expanded), dim=0) # ((C + d_model) x H x W)
                
                else:
                    # Use feature-wise linear modulation to fuse climate and satellite data
                    pad_mask_t = (batch_sat_data[t] == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)

                    combined = self.FiLM(batch_sat_data[t], clim_vec, pad_mask_t) # (C x H x W)

                batch_fused.append(combined)

            batch_fused = torch.stack(batch_fused, dim=0) # (T x (C + d_model) x H x W) or (T x C x H x W) if FiLM applied
            fused_data.append(batch_fused)
        
        fused_data = torch.stack(fused_data, dim=0) # (B x T x (C + d_model) x H x W) or (B x T x C x H x W) if FiLM applied
        return fused_data

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
        fused_data = self.fuse_climate_data(input_sat, dates_sat, input_clim, dates_clim)
        output = self.utae_model(fused_data, batch_positions=dates_sat)

        return output
    
class FiLMLayer(nn.Module):
    def __init__(self,
                 clim_vec_dim,
                 sat_feature_dim=10,
                 hidden_dim=128):
        super(FiLMLayer, self).__init__()
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

    def forward(self, sat_features, clim_vec, pad_mask=None):
        """
        Applies FiLM modulation to satellite features using climate vector.

        The FiLM modulation scales and shifts the satellite features based on parameters 
        generated from the climate vector. This is achieved by computing gamma (scale) and beta 
        (shift) parameters from the climate vector through an MLP, then applying these 
        parameters to the satellite features.

        Args:
            sat_features (Tensor): Satellite feature maps of shape (C x H x W) or (B x C x H x W) if batched.
            clim_vec (Tensor): Climate vector of shape (clim_vec_dim) or (B x clim_vec_dim) if batched.
            pad_mask ....
            
        Returns:
            Tensor: Modulated satellite features of shape (C x H x W) or (B x C x H x W) if batched.
        """
        
        if pad_mask==True:
            return sat_features
        
        else:
            film_params = self.mlp(clim_vec)        # (2*feature_dim) or (B x 2*feature_dim) if batched input
            
            if film_params.dim() == 1:
                # split along first (and only) dimension 
                gamma, beta = film_params.chunk(2, dim=0)
            else:
                # split along second dimension (for batched input)
                gamma, beta = film_params.chunk(2, dim=1)

            gamma = gamma.unsqueeze(-1).unsqueeze(-1)   # (feature_dim, 1, 1) or (B x feature_dim x 1 x 1) if batched input
            beta = beta.unsqueeze(-1).unsqueeze(-1)     # (feature_dim, 1, 1) or (B x feature_dim x 1 x 1) if batched input

            modulated_features = gamma * sat_features + beta
            
            return modulated_features