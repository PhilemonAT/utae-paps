from src.backbones import utae_original, utae_fusion, unet3d, convlstm, convgru, fpn

def get_model(config, mode="semantic"):
    if mode == "semantic":
        if config.model == "utae":
            model = utae_original.UTAE(
                    input_dim=10,
                    encoder_widths=config.encoder_widths,
                    decoder_widths=config.decoder_widths,
                    out_conv=config.out_conv,
                    str_conv_k=config.str_conv_k,
                    str_conv_s=config.str_conv_s,
                    str_conv_p=config.str_conv_p,
                    agg_mode=config.agg_mode,
                    encoder_norm=config.encoder_norm,
                    n_head=config.n_head,
                    d_model=config.d_model,
                    d_k=config.d_k,
                    encoder=False,
                    return_maps=False,
                    pad_value=config.pad_value,
                    padding_mode=config.padding_mode,
                    last_relu=config.last_relu,
                    pos_type=config.pos_type
                )
        elif config.model == "utae_fusion":
            model = utae_fusion.UTAE_Fusion(
                input_dim=10,
                climate_input_dim=config.climate_input_dim,
                encoder_widths=config.encoder_widths,
                decoder_widths=config.decoder_widths,
                out_conv=config.out_conv,
                str_conv_k=config.str_conv_k,
                str_conv_s=config.str_conv_s,
                str_conv_p=config.str_conv_p,
                agg_mode=config.agg_mode,
                encoder_norm=config.encoder_norm,
                n_head=config.n_head,
                d_model=config.d_model,
                d_k=config.d_k,
                encoder=config.encoder,
                return_maps=config.return_maps,
                pad_value=config.pad_value,
                padding_mode=config.padding_mode,
                matching_type=config.matching_type,
                use_climate_mlp=config.use_climate_mlp,
                fusion_location=config.fusion_location,
                fusion_style=config.fusion_style,
                residual_film=config.residual_film
            )
        elif config.model == "unet3d":
            model = unet3d.UNet3D(
                in_channel=10, n_classes=config.num_classes, pad_value=config.pad_value
            )
        elif config.model == "fpn":
            model = fpn.FPNConvLSTM(
                input_dim=10,
                num_classes=config.num_classes,
                inconv=[32, 64],
                n_levels=4,
                n_channels=64,
                hidden_size=88,
                input_shape=(128, 128),
                mid_conv=True,
                pad_value=config.pad_value,
            )
        elif config.model == "convlstm":
            model = convlstm.ConvLSTM_Seg(
                num_classes=config.num_classes,
                input_size=(128, 128),
                input_dim=10,
                kernel_size=(3, 3),
                hidden_dim=160,
            )
        elif config.model == "convgru":
            model = convgru.ConvGRU_Seg(
                num_classes=config.num_classes,
                input_size=(128, 128),
                input_dim=10,
                kernel_size=(3, 3),
                hidden_dim=180,
            )
        elif config.model == "uconvlstm":
            model = utae_original.RecUNet(
                input_dim=10,
                encoder_widths=[64, 64, 64, 128],
                decoder_widths=[32, 32, 64, 128],
                out_conv=[32, 20],
                str_conv_k=4,
                str_conv_s=2,
                str_conv_p=1,
                temporal="lstm",
                input_size=128,
                encoder_norm="group",
                hidden_dim=64,
                encoder=False,
                padding_mode="zeros",
                pad_value=0,
            )
        elif config.model == "buconvlstm":
            model = utae_original.RecUNet(
                input_dim=10,
                encoder_widths=[64, 64, 64, 128],
                decoder_widths=[32, 32, 64, 128],
                out_conv=[32, 20],
                str_conv_k=4,
                str_conv_s=2,
                str_conv_p=1,
                temporal="lstm",
                input_size=128,
                encoder_norm="group",
                hidden_dim=30,
                encoder=False,
                padding_mode="zeros",
                pad_value=0,
            )
        return model
    elif mode == "panoptic":
        if config.backbone == "utae":
            model = utae_original.UTAE(
                input_dim=10,
                encoder_widths=config.encoder_widths,
                decoder_widths=config.decoder_widths,
                out_conv=config.out_conv,
                str_conv_k=config.str_conv_k,
                str_conv_s=config.str_conv_s,
                str_conv_p=config.str_conv_p,
                agg_mode=config.agg_mode,
                encoder_norm=config.encoder_norm,
                n_head=config.n_head,
                d_model=config.d_model,
                d_k=config.d_k,
                encoder=True,
                return_maps=False,
                pad_value=config.pad_value,
                padding_mode=config.padding_mode,
            )
        elif config.backbone == "uconvlstm":
            model = utae_original.RecUNet(
                input_dim=10,
                encoder_widths=[64, 64, 64, 128],
                decoder_widths=[32, 32, 64, 128],
                out_conv=[32, 20],
                str_conv_k=4,
                str_conv_s=2,
                str_conv_p=1,
                temporal="lstm",
                input_size=128,
                encoder_norm="group",
                hidden_dim=64,
                encoder=True,
                padding_mode="zeros",
                pad_value=0,
            )
        else:
            raise NotImplementedError

        model = paps.PaPs(
            encoder=model,
            num_classes=config.num_classes,
            shape_size=config.shape_size,
            mask_conv=config.mask_conv,
            min_confidence=config.min_confidence,
            min_remain=config.min_remain,
            mask_threshold=config.mask_threshold,
        )
        return model
    else:
        raise NotImplementedError
