# What network is used for which results in the report?

- `baseline.py` (by Lukas): training of all baselines
- `temporal_encoder_network.py` (by Lukas): training of the temporal encoder (without warp) and of the warp network (configurable in the network)
- all other networks are different flavours of the optical flow experiments (by Anna)
- `xception.py` (pretrained feature extractor): used by all networks as pretrained feature extractor
