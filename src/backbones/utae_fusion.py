"""
U-TAE Implementation
Author: Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""
import math
import torch
import torch.nn as nn

from src.backbones.convlstm import ConvLSTM, BConvLSTM
from src.backbones.ltae import LTAE2d
from torch.nn import TransformerEncoderLayer


class UTAE_Fusion(nn.Module):
    """
    UTAE-Early Fusion Model

    add parameter 'fusion_style' which can be in {"film", "concat"}. Can add others later if required

    """
    def __init__(
        self,
        input_dim,
        climate_input_dim,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        out_conv=[32, 20],
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        agg_mode="att_group",
        encoder_norm="group",
        n_head=16,
        d_model=256,
        d_k=4,
        encoder=False,
        return_maps=False,
        pad_value=-1000,
        padding_mode="reflect",
        matching_type='causal',
        use_climate_mlp=False,
        fusion_location=1,
        fusion_style="film",
        residual_film=False,
    ):
        """
        U-TAE architecture for spatio-temporal encoding of satellite image time series.
        Args:
            input_dim (int): Number of channels in the input images.
            encoder_widths (List[int]): List giving the number of channels of the successive encoder_widths of the convolutional encoder.
            This argument also defines the number of encoder_widths (i.e. the number of downsampling steps +1)
            in the architecture.
            The number of channels are given from top to bottom, i.e. from the highest to the lowest resolution.
            decoder_widths (List[int], optional): Same as encoder_widths but for the decoder. The order in which the number of
            channels should be given is also from top to bottom. If this argument is not specified the decoder
            will have the same configuration as the encoder.
            out_conv (List[int]): Number of channels of the successive convolutions for the
            str_conv_k (int): Kernel size of the strided up and down convolutions.
            str_conv_s (int): Stride of the strided up and down convolutions.
            str_conv_p (int): Padding of the strided up and down convolutions.
            agg_mode (str): Aggregation mode for the skip connections. Can either be:
                - att_group (default) : Attention weighted temporal average, using the same
                channel grouping strategy as in the LTAE. The attention masks are bilinearly
                resampled to the resolution of the skipped feature maps.
                - att_mean : Attention weighted temporal average,
                 using the average attention scores across heads for each date.
                - mean : Temporal average excluding padded dates.
            encoder_norm (str): Type of normalisation layer to use in the encoding branch. Can either be:
                - group : GroupNorm (default)
                - batch : BatchNorm
                - instance : InstanceNorm
            n_head (int): Number of heads in LTAE.
            d_model (int): Parameter of LTAE
            d_k (int): Key-Query space dimension
            encoder (bool): If true, the feature maps instead of the class scores are returned (default False)
            return_maps (bool): If true, the feature maps instead of the class scores are returned (default False)
            pad_value (float): Value used by the dataloader for temporal padding.
            padding_mode (str): Spatial padding strategy for convolutional layers (passed to nn.Conv2d).
            matching_type (str): Strategy for processing and matching climate data, can be:
                                - 'match_dates': Use only climate data from the exact dates matching the satellite observations.
                                - 'weekly': Use climate data from the whole week prior to each satellite observation.
                                - 'causal': Use a causal transformer to incorporate climate data sequentially.            
            use_climate_mlp (bool): If True, a small MLP processes the climate data before fusion to 
                                    align it with the satellite data's dimensions.
            climate_dim (int): The dimension of the climate data processed by the EarlyFusionModel
            fusion_location (int): One of {1, 2, 3, 4} corresponding to 
                                    - 1: early fusion (after the first conv block);
                                    - 2: fusion along the entire encoder;
                                    - 3: mid fusion (at the lowest resolution);
                                    - 4: late fusion (after the decoder, before class assignments)
            fusion_style (str): One of {"film", "concat"}. Default is film (feature-wise linear modulation)
            residual_film (bool): Whether to use residual connection in the FiLM layer
        """
        super(UTAE_Fusion, self).__init__()
        self.n_stages = len(encoder_widths)
        self.return_maps = return_maps
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths
        self.enc_dim = (
            decoder_widths[0] if decoder_widths is not None else encoder_widths[0]
        )
        self.stack_dim = (
            sum(decoder_widths) if decoder_widths is not None else sum(encoder_widths)
        )
        self.pad_value = pad_value
        self.encoder = encoder
        
        self.fusion_location = fusion_location
        self.fusion_style = fusion_style
        self.residual_film = residual_film

        # ------------------------------------------------------------------------------------------
        # check that only valid specifications provided
        # ------------------------------------------------------------------------------------------
        if not fusion_location in [1, 2, 3, 4]:
            raise NotImplementedError(f"fusion_location: {fusion_location} not valid")
        if not fusion_style in ["film", "concat"]:
            raise NotImplementedError(f"fusion_style: {fusion_style} not valid")
        assert not (fusion_style == "concat" and fusion_location==3), "Cannot use fusion style 'concat' with mid-fusion"
        assert not (fusion_style == "concat" and fusion_location==2), "Cannot use fusion style 'concat' with encoder-fusion"
        assert not (matching_type == 'causal' and use_climate_mlp), "Using climate MLP with causal fusion not implemented"

        if encoder:
            self.return_maps = True

        if decoder_widths is not None:
            assert len(encoder_widths) == len(decoder_widths)
            assert encoder_widths[-1] == decoder_widths[-1]
        else:
            decoder_widths = encoder_widths

        # ------------------------------------------------------------------------------------------
        # Initalize classes to process and match climate data to satellite data depending on 
        # (early) fusion strategy
        # ------------------------------------------------------------------------------------------
        if fusion_location in [1, 2, 3]:
            self.matchclimate = PrepareMatchedDataEarly(climate_input_dim=climate_input_dim,
                                                        matching_type=matching_type,
                                                        use_climate_mlp=use_climate_mlp,
                                                        pad_value=pad_value)
            self.climate_dim = self.matchclimate.climate_dim

        elif fusion_location==4:
            self.climate_transformer_encoder = ClimateTransformerEncoder(climate_input_dim=climate_input_dim,
                                                                         use_cls_token=True)
            self.climate_dim = self.climate_transformer_encoder.d_model

        # ------------------------------------------------------------------------------------------
        # Determine input_dim to in_conv based on fusion strategy applied
        # ------------------------------------------------------------------------------------------
        if (fusion_location==1): # early fusion
            if fusion_style=='concat':
                input_dim = input_dim + self.climate_dim

        # ------------------------------------------------------------------------------------------
        # Instantiate UTAE architecture
        # ------------------------------------------------------------------------------------------
        self.in_conv = ConvBlock(
            nkernels=[input_dim] + [encoder_widths[0], encoder_widths[0]],
            pad_value=pad_value,
            norm=encoder_norm,
            padding_mode=padding_mode,
        )
        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=encoder_widths[i],
                d_out=encoder_widths[i + 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                pad_value=pad_value,
                norm=encoder_norm,
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1)
        )
        self.up_blocks = nn.ModuleList(
            UpConvBlock(
                d_in=decoder_widths[i],
                d_out=decoder_widths[i - 1],
                d_skip=encoder_widths[i - 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                norm="batch",
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1, 0, -1)
        )
        self.temporal_encoder = LTAE2d(
            in_channels=encoder_widths[-1],
            d_model=d_model,
            n_head=n_head,
            mlp=[d_model, encoder_widths[-1]],
            return_att=True,
            d_k=d_k,
        )
        self.temporal_aggregator = Temporal_Aggregator(mode=agg_mode)

        # ------------------------------------------------------------------------------------------
        # Determine input dimension to out_conv based on fusion_strategy applied
        # ------------------------------------------------------------------------------------------
        nkernels_out = [decoder_widths[0]] + out_conv
        if (fusion_location==4): # late fusion
            if fusion_style=="concat":
                nkernels_out = [decoder_widths[0] + self.climate_dim] + out_conv
        
        # ------------------------------------------------------------------------------------------
        # Instantiate last convolutional block, which maps channels to number of classes
        # ------------------------------------------------------------------------------------------
        self.out_conv = ConvBlock(nkernels=nkernels_out, padding_mode=padding_mode)
        
        # ------------------------------------------------------------------------------------------
        # Initialize FiLM-Layer depending on fusion_style
        # ------------------------------------------------------------------------------------------
        if fusion_style=="film":
            assert self.climate_dim is not None, "input dimension to FiLM not known"
            if fusion_location==1: # Early Fusion
                self.FiLM_Layer = FiLM(clim_vec_dim=self.climate_dim,
                                       sat_feature_dim=encoder_widths[0],
                                       hidden_dim=2*encoder_widths[0])
            elif fusion_location==2: # Fusion along entire encoder
                self.film_layers = nn.ModuleList(
                    FiLM(clim_vec_dim=self.climate_dim,
                         sat_feature_dim=encoder_widths[i + 1],
                         hidden_dim=2*encoder_widths[0]) # constant hidden_dim
                    for i in range(self.n_stages - 1)
                )
            elif fusion_location==3: # Mid Fusion
                self.FiLM_Layer = FiLM(clim_vec_dim=self.climate_dim,
                                       sat_feature_dim=encoder_widths[-1],
                                       hidden_dim=2*encoder_widths[0])
            elif fusion_location==4: # Late Fusion
                self.FiLM_Layer = FiLM(clim_vec_dim=self.climate_dim,
                                       sat_feature_dim=decoder_widths[0])

    def forward(self, input, sat_dates, climate_input, climate_dates, batch_positions=None, return_att=False, return_film=False):
        """
        climate_input: original, untransformed climate data (B x T' x V)
        """
        pad_mask = (
            (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        )  # BxT pad mask

        if self.fusion_location in [1, 2, 3]:
            climate_matched = self.matchclimate.transform_climate_data(sat_data=input,
                                                                       sat_dates=sat_dates,
                                                                       climate_data=climate_input,
                                                                       climate_dates=climate_dates) # (B x T x V) or (B x T x d_model) if causal or MLP applied
        if self.fusion_location==1:
            if self.fusion_style=="film":
                # Apply feature-wise linear modulation after first conv block
                out = self.in_conv.smart_forward(input) # (B x T x C1 X H x W)
                out = self.FiLM_Layer(sat_features=out,
                                      clim_vec=climate_matched,
                                      residual=self.residual_film,
                                      pad_mask=pad_mask)
            else:
                # Fallback option is always concat
                _, _, _, H, W = input.size()
                clim_vec_expanded = climate_matched.unsqueeze(-1).unsqueeze(-1)   # (B x T x climate_dim x 1 x 1)
                clim_vec_expanded = clim_vec_expanded.expand(-1, -1, -1, H, W)  # (B x T x climate_dim x H x W)
                input = torch.cat((input, clim_vec_expanded), dim=2)            # (B x T x (C + climate_dim) x H x W)
                out = self.in_conv.smart_forward(input)                         # (B x T x C1 x H x W)
        else:
            out = self.in_conv.smart_forward(input)

        feature_maps = [out]

        # SPATIAL ENCODER
        
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i].smart_forward(feature_maps[-1])
            if self.fusion_location==2:
                out = self.film_layers[i](out, climate_matched, self.residual_film, pad_mask)
            feature_maps.append(out)
        
        if self.fusion_location==3:
            out = self.FiLM_Layer(sat_features=out,
                                  clim_vec=climate_matched,
                                  residual=self.residual_film,pad_mask=pad_mask)

        # TEMPORAL ENCODER
        out, att = self.temporal_encoder(
            feature_maps[-1], batch_positions=batch_positions, pad_mask=pad_mask
        )
        # SPATIAL DECODER
        if self.return_maps:
            maps = [out]
        for i in range(self.n_stages - 1):
            skip = self.temporal_aggregator(
                feature_maps[-(i + 2)], pad_mask=pad_mask, attn_mask=att
            )
            out = self.up_blocks[i](out, skip)
            if self.return_maps:
                maps.append(out)

        if self.fusion_location==4:
            climate_embedding = self.climate_transformer_encoder(climate_input) # (B x d_model)
            if self.fusion_style=="film":
                out = self.FiLM_Layer(sat_features=out, 
                                      clim_vec=climate_embedding, 
                                      residual=self.residual_film)
            else:
                climate_embedding = climate_embedding.unsqueeze(-1).unsqueeze(-1)  # (B x d_model x 1 x 1)
                climate_embedding = climate_embedding.expand(-1, -1, 
                                                             out.size(2), 
                                                             out.size(3))  # (B x d_model x H x W)
                # Concatenate satellite features and climate embedding
                out = torch.cat((out, climate_embedding), dim=1)

        
        if self.encoder:
            return out, maps
        else:
            out = self.out_conv(out)
            if return_att:
                return out, att
            if return_film:
                return out
            if self.return_maps:
                return out, maps
            else:
                return out

class TemporallySharedBlock(nn.Module):
    """
    Helper module for convolutional encoding blocks that are shared across a sequence.
    This module adds the self.smart_forward() method the the block.
    smart_forward will combine the batch and temporal dimension of an input tensor
    if it is 5-D and apply the shared convolutions to all the (batch x temp) positions.
    """

    def __init__(self, pad_value=None):
        super(TemporallySharedBlock, self).__init__()
        self.out_shape = None
        self.pad_value = pad_value

    def smart_forward(self, input):
        if len(input.shape) == 4:
            return self.forward(input)
        else:
            b, t, c, h, w = input.shape

            if self.pad_value is not None:
                dummy = torch.zeros(input.shape, device=input.device).float()
                self.out_shape = self.forward(dummy.view(b * t, c, h, w)).shape

            out = input.view(b * t, c, h, w)
            if self.pad_value is not None:
                pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
                if pad_mask.any():
                    temp = (
                        torch.ones(
                            self.out_shape, device=input.device, requires_grad=False
                        )
                        * self.pad_value
                    )
                    temp[~pad_mask] = self.forward(out[~pad_mask])
                    out = temp
                else:
                    out = self.forward(out)
            else:
                out = self.forward(out)
            _, c, h, w = out.shape
            out = out.view(b, t, c, h, w)
            return out


class ConvLayer(nn.Module):
    def __init__(
        self,
        nkernels,
        norm="batch",
        k=3,
        s=1,
        p=1,
        n_groups=4,
        last_relu=True,
        padding_mode="reflect",
    ):
        super(ConvLayer, self).__init__()
        layers = []
        if norm == "batch":
            nl = nn.BatchNorm2d
        elif norm == "instance":
            nl = nn.InstanceNorm2d
        elif norm == "group":
            nl = lambda num_feats: nn.GroupNorm(
                num_channels=num_feats,
                num_groups=n_groups,
            )
        else:
            nl = None
        for i in range(len(nkernels) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=nkernels[i],
                    out_channels=nkernels[i + 1],
                    kernel_size=k,
                    padding=p,
                    stride=s,
                    padding_mode=padding_mode,
                )
            )
            if nl is not None:
                layers.append(nl(nkernels[i + 1]))

            if last_relu:
                layers.append(nn.ReLU())
            elif i < len(nkernels) - 2:
                layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, input):
        return self.conv(input)


class ConvBlock(TemporallySharedBlock):
    def __init__(
        self,
        nkernels,
        pad_value=None,
        norm="batch",
        last_relu=True,
        padding_mode="reflect",
    ):
        super(ConvBlock, self).__init__(pad_value=pad_value)
        self.conv = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=last_relu,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        return self.conv(input)


class DownConvBlock(TemporallySharedBlock):
    def __init__(
        self,
        d_in,
        d_out,
        k,
        s,
        p,
        pad_value=None,
        norm="batch",
        padding_mode="reflect",
    ):
        super(DownConvBlock, self).__init__(pad_value=pad_value)
        self.down = ConvLayer(
            nkernels=[d_in, d_in],
            norm=norm,
            k=k,
            s=s,
            p=p,
            padding_mode=padding_mode,
        )
        self.conv1 = ConvLayer(
            nkernels=[d_in, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        out = self.down(input)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


class UpConvBlock(nn.Module):
    def __init__(
        self, d_in, d_out, k, s, p, norm="batch", d_skip=None, padding_mode="reflect"
    ):
        super(UpConvBlock, self).__init__()
        d = d_out if d_skip is None else d_skip
        self.skip_conv = nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1),
            nn.BatchNorm2d(d),
            nn.ReLU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=d_in, out_channels=d_out, kernel_size=k, stride=s, padding=p
            ),
            nn.BatchNorm2d(d_out),
            nn.ReLU(),
        )
        self.conv1 = ConvLayer(
            nkernels=[d_out + d, d_out], norm=norm, padding_mode=padding_mode
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode
        )

    def forward(self, input, skip):
        out = self.up(input)
        out = torch.cat([out, self.skip_conv(skip)], dim=1)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


class Temporal_Aggregator(nn.Module):
    def __init__(self, mode="mean"):
        super(Temporal_Aggregator, self).__init__()
        self.mode = mode

    def forward(self, x, pad_mask=None, attn_mask=None):
        if pad_mask is not None and pad_mask.any():
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)

                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)

                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                attn = attn * (~pad_mask).float()[None, :, :, None, None]

                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                attn = attn * (~pad_mask).float()[:, :, None, None]
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                out = x * (~pad_mask).float()[:, :, None, None, None]
                out = out.sum(dim=1) / (~pad_mask).sum(dim=1)[:, None, None, None]
                return out
        else:
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)
                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)
                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                return x.mean(dim=1)


class RecUNet(nn.Module):
    """Recurrent U-Net architecture. Similar to the U-TAE architecture but
    the L-TAE is replaced by a recurrent network
    and temporal averages are computed for the skip connections."""

    def __init__(
        self,
        input_dim,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        out_conv=[32, 20],
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        temporal="lstm",
        input_size=128,
        encoder_norm="group",
        hidden_dim=128,
        encoder=False,
        padding_mode="reflect",
        pad_value=0,
    ):
        super(RecUNet, self).__init__()
        self.n_stages = len(encoder_widths)
        self.temporal = temporal
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths
        self.enc_dim = (
            decoder_widths[0] if decoder_widths is not None else encoder_widths[0]
        )
        self.stack_dim = (
            sum(decoder_widths) if decoder_widths is not None else sum(encoder_widths)
        )
        self.pad_value = pad_value

        self.encoder = encoder
        if encoder:
            self.return_maps = True
        else:
            self.return_maps = False

        if decoder_widths is not None:
            assert len(encoder_widths) == len(decoder_widths)
            assert encoder_widths[-1] == decoder_widths[-1]
        else:
            decoder_widths = encoder_widths

        self.in_conv = ConvBlock(
            nkernels=[input_dim] + [encoder_widths[0], encoder_widths[0]],
            pad_value=pad_value,
            norm=encoder_norm,
        )

        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=encoder_widths[i],
                d_out=encoder_widths[i + 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                pad_value=pad_value,
                norm=encoder_norm,
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1)
        )
        self.up_blocks = nn.ModuleList(
            UpConvBlock(
                d_in=decoder_widths[i],
                d_out=decoder_widths[i - 1],
                d_skip=encoder_widths[i - 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                norm=encoder_norm,
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1, 0, -1)
        )
        self.temporal_aggregator = Temporal_Aggregator(mode="mean")

        if temporal == "mean":
            self.temporal_encoder = Temporal_Aggregator(mode="mean")
        elif temporal == "lstm":
            size = int(input_size / str_conv_s ** (self.n_stages - 1))
            self.temporal_encoder = ConvLSTM(
                input_dim=encoder_widths[-1],
                input_size=(size, size),
                hidden_dim=hidden_dim,
                kernel_size=(3, 3),
            )
            self.out_convlstm = nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=encoder_widths[-1],
                kernel_size=3,
                padding=1,
            )
        elif temporal == "blstm":
            size = int(input_size / str_conv_s ** (self.n_stages - 1))
            self.temporal_encoder = BConvLSTM(
                input_dim=encoder_widths[-1],
                input_size=(size, size),
                hidden_dim=hidden_dim,
                kernel_size=(3, 3),
            )
            self.out_convlstm = nn.Conv2d(
                in_channels=2 * hidden_dim,
                out_channels=encoder_widths[-1],
                kernel_size=3,
                padding=1,
            )
        elif temporal == "mono":
            self.temporal_encoder = None
        self.out_conv = ConvBlock(nkernels=[decoder_widths[0]] + out_conv, padding_mode=padding_mode)

    def forward(self, input, batch_positions=None):
        pad_mask = (
            (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        )  # BxT pad mask

        out = self.in_conv.smart_forward(input)

        feature_maps = [out]
        # ENCODER
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i].smart_forward(feature_maps[-1])
            feature_maps.append(out)

        # Temporal encoder
        if self.temporal == "mean":
            out = self.temporal_encoder(feature_maps[-1], pad_mask=pad_mask)
        elif self.temporal == "lstm":
            _, out = self.temporal_encoder(feature_maps[-1], pad_mask=pad_mask)
            out = out[0][1]  # take last cell state as embedding
            out = self.out_convlstm(out)
        elif self.temporal == "blstm":
            out = self.temporal_encoder(feature_maps[-1], pad_mask=pad_mask)
            out = self.out_convlstm(out)
        elif self.temporal == "mono":
            out = feature_maps[-1]

        if self.return_maps:
            maps = [out]
        for i in range(self.n_stages - 1):
            if self.temporal != "mono":
                skip = self.temporal_aggregator(
                    feature_maps[-(i + 2)], pad_mask=pad_mask
                )
            else:
                skip = feature_maps[-(i + 2)]
            out = self.up_blocks[i](out, skip)
            if self.return_maps:
                maps.append(out)

        if self.encoder:
            return out, maps
        else:
            out = self.out_conv(out)
            if self.return_maps:
                return out, maps
            else:
                return out


class PrepareMatchedDataEarly(nn.Module):
    def __init__(self,
                 climate_input_dim=11,
                 d_model=64,
                 matching_type='match_dates',
                 use_climate_mlp=False,
                 mlp_hidden_dim=64,
                 nhead_climate_transformer=4,
                 d_ffn_climate_transformer=128,
                 num_layers_climate_transformer=1,
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
            matching_type (str): Strategy for fusing climate data, can be:
                - 'match_dates': Use only climate data from the exact dates matching the satellite observations.
                - 'weekly': Use climate data from the whole week prior to each satellite observation.
                - 'causal': Use a causal transformer to incorporate climate data sequentially.
            use_climate_mlp (bool): If True, a small MLP processes the climate data before fusion to 
                                    align it with the satellite data's dimensions.
            mlp_hidden_dim (int): Hidden dimension size of the MLP used when `use_climate_mlp` is True.
        """
        super(PrepareMatchedDataEarly, self).__init__()
 
        self.matching_type = matching_type
        self.use_climate_mlp = use_climate_mlp
        self.pad_value = pad_value
        self.climate_dim = climate_input_dim # The processed dimension of the climate data, used later in UTAE

        if use_climate_mlp:
            self.climate_mlp = nn.Sequential(
                nn.Linear(climate_input_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, d_model)
            )
            self.climate_dim = d_model # if we processed climate data with MLP, climate dim. will be of size d_model

        if matching_type == 'causal':
            assert use_climate_mlp == False, "Using climate MLP with causal fusion not implemented"

            self.causal_transformer_encoder = ClimateTransformerEncoder(
                climate_input_dim=climate_input_dim,
                d_model=d_model,
                nhead=nhead_climate_transformer,
                d_ffn=d_ffn_climate_transformer,
                num_layers=num_layers_climate_transformer,
                use_cls_token=False # not using CLS token; need per time step embeddings
            )
            self.climate_dim = d_model  # if we used causal transformer for climate data, climate dim. will be of size d_model

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

        if self.matching_type == 'causal':
            # Create causal mask for the entire sequence
            causal_mask = torch.triu(torch.ones((climate_dates.size(1), climate_dates.size(1)),
                                                device=climate_data.device), diagonal=1)
            
            # expand mask for batch and heads
            num_heads = self.causal_transformer_encoder.climate_transformer.layers[0].self_attn.num_heads
            causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1) # (B x T' x T')
            causal_mask = causal_mask.unsqueeze(1).expand(-1, num_heads, -1, -1) # (B x Heads x T' x T')
            causal_mask = causal_mask.reshape(batch_size * num_heads, climate_dates.size(1), climate_dates.size(1)) # (B*H, T', T')
            causal_mask = causal_mask.masked_fill(causal_mask==1, float('-inf'))          

            # compute climate embeddings for the entire sequence
            climate_embeddings = self.causal_transformer_encoder(climate_data, 
                                                                 mask=causal_mask
                                                                 ) # (B x T' x d_model)

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
            return climate_matched

        # other matching types ('match_dates' or 'weekly')
        climate_matched = []

        for b in range(sat_data.size(0)): # Loop over the batch
            batch_sat_data = sat_data[b]
            batch_sat_dates = sat_dates[b]
            batch_climate_data = climate_data[b]
            batch_climate_dates = climate_dates[b]

            batch_climate = []

            for t, sat_date in enumerate(batch_sat_dates): # Loop over time steps

                if self.matching_type == 'match_dates':
                    climate_indices = (batch_climate_dates == sat_date).nonzero(as_tuple=True)[0]
                    if len(climate_indices) > 0:
                        clim_vec = batch_climate_data[climate_indices[0]]
                    else:
                        # fill with pad value used for satellite data (-1000)
                        clim_vec = torch.full((batch_climate_data.size(1),), float(self.pad_value), device=sat_data.device)
                
                elif self.matching_type == 'weekly':
                    clim_indices = (batch_climate_dates < sat_date) & (batch_climate_dates >= (sat_date - 7))
                    
                    if clim_indices.any():
                        clim_vec = batch_climate_data[clim_indices].mean(dim=0)
                    else:
                        clim_vec = torch.full((batch_climate_data.size(1),), float(self.pad_value), device=sat_data.device)

                else:
                    raise ValueError(f"Unknown fusion strategy: {self.matching_type}")

                if hasattr(self, 'climate_mlp'):
                    clim_vec = self.climate_mlp(clim_vec) # d_model
              
                batch_climate.append(clim_vec)

            batch_climate = torch.stack(batch_climate, dim=0) # (T x V) or (T x d_model) if MLP applied
            climate_matched.append(batch_climate)
        
        climate_matched = torch.stack(climate_matched, dim=0) # (B x T x V) or (B x T x d_model) if MLP applied
        return climate_matched


    # def forward(self, input_sat, dates_sat, input_clim, dates_clim):
    #     """
    #     Forward pass through the EarlyFusionModel.

    #     Args:
    #         input_sat (Tensor): Input satellite data of shape (B x T x C x H x W).
    #         dates_sat (Tensor): Dates corresponding to the satellite data (B x T).
    #         input_clim (Tensor): Input climate data of shape (B x T' x climate_input_dim).
    #         dates_clim (Tensor): Dates corresponding to the climate data (B x T').

    #     Returns:
    #         Tensor: Output of the model after early fusion of satellite and climate data.
    #     """
    #     climate_matched = self.transform_climate_data(input_sat, dates_sat, input_clim, dates_clim)
    #     output = self.utae_model(input=input_sat, 
    #                              climate_input=climate_matched, 
    #                              batch_positions=dates_sat)

    #     # call UTAE here with input_sat as input images, dates_sat as batch_positions,
    #     #                     climate_matched as the corresponding transformed climate data 
    #     # Note: When using 'causal', climate_matched has size (B x T x d_model)
    #     #       When using 'match_dates' or 'weekly' and NOT using MLP, climate_matched has size (B x T x V)
    #     #       ---------------------------------------- USING MLP,     climate_matched has size (B x T x d_model)

    #     return output


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
                 use_cls_token=False,
                 max_length=5000):
        super(ClimateTransformerEncoder, self).__init__()
        
        self.climate_projection = nn.Linear(climate_input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_length=max_length)
        self.use_cls_token = use_cls_token
        self.d_model = d_model

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

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
            nn.Linear(clim_vec_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * sat_feature_dim), # outputs both gamma and beta
        )

    def forward(self, sat_features, clim_vec, residual=False, pad_mask=None):
        """
        Applies FiLM modulation to satellite features using climate vector.

        The FiLM modulation scales and shifts the satellite features based on parameters 
        generated from the climate vector. This is achieved by computing gamma (scale) and beta 
        (shift) parameters from the climate vector through an MLP, then applying these 
        parameters to the satellite features.

        Args:
            sat_features (Tensor): Satellite feature maps of shape (B x T x C x H x W) 
            clim_vec (Tensor): Climate vector of shape (B x T x clim_vec_dim) 
            pad_mask ....
            
        Returns:
            Tensor: Modulated satellite features of shape (B x T x C x H x W)
        """
        film_params = self.mlp(clim_vec)        # (B x T x 2*C)
        
        # split along last dimension
        gamma, beta = film_params.chunk(2, dim=-1)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)   # (B x T x C x 1 x 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)     # (B x T x C x 1 x 1)

        if residual:
            modulated_features = sat_features + (gamma * sat_features + beta)
        else:
            modulated_features = gamma * sat_features + beta
        
        if pad_mask is not None:
            # For the padded entries, do not apply FiLM:
            padded_idx = (pad_mask==True).nonzero(as_tuple=True)
            modulated_features[padded_idx] = sat_features[padded_idx]

        return modulated_features # modulated features of shape (B x T x C x H x W)