# What network is used for which results in the report?

- `baseline.py` (by Lukas): training of all baselines
- `temporal_encoder_network.py` (by Lukas): training of the temporal encoder (without warp) and of the warp network (configurable in the network)
- `network_of_fc.py` (by Anna): training of the network using optical flow used for determination of parts of the optical flow representation computation strategy 
- `network_of_conv.py` (by Anna): training of a two branch network using optical flow and a joint convolution block
- `xception.py` (pretrained feature extractor): used by all networks as pretrained feature extractor

# What were the other networks used for?
- `network_just_of.py` (by Anna): Training of an experimental network only using optical flow instead of frames themselves (with expectedly bad results, which led to choosing the two branch models). Also used for trying different parameters for the Farneback method. 
- `temporal_encoder_small.py` (by Anna): Training of the small version of the temporal encoder without warp. --> See Presentation 2
- `warp_model_small.py` (by Anna): Training of the small version of the temporal encoder using warp. --> See Presentation 2