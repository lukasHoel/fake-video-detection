# General
Contains all runs that we have done throughout the project as tensorboard files.
To view the runs, simply start tensorboard via:

`tensorboard --logdir path/to/runs`

# Naming conventions:

- <b>Baseline (trained by Lukas):</b> results from the model `networks/baseline.py`
- <b>TemporalEncoderII (trained by Lukas):</b> results from the model `networks/temporal_encoder_network.py` either with our without warp
- <b>TE small(trained by Anna):</b> results from the model `networks/temporal_encoder_small.py` either with our without warp
- <b>Optical Flow(trained by Anna):</b> results from several optical flow models in the following subdirectries
    - <b>just\_of:</b> model network\_just\_of
    - <b>of\_fc\_config\_tests:</b> model network\_of\_fc
    - <b>of\_conv:</b> network\_of\_conv

Each run has a unique identifier as folder name. This identifier can also be used to load the corresponding model, if it was saved in `saved_results/models`.

# Matching of run-id's to numbers in the report
Here we provide a list, which runs contain results that are mentioned in our report. Other runs not mentioned in this list were done by us e.g. for hyperparameter tuning, debugging, etc. but are still left in the repository for future reference. Sometimes, we mention multiple run-id's per method, this is because we had to restart the jupyter notebook before doing testing sometimes.

## Tables 1 and 2 in report (runs by Lukas)
Here we list run-id's for training of the baseline and temporal encoder (with and without warp).
The models were trained on one single fake method + original videos with the configuration: 100 videos, 5 sequences, 10 frames, skip rate 5.

#### Deepfake
- Baseline: `2020-Jan-26_20-15-00_87e423d0-4078-11ea-b304-0242ac1c0002`
- Temporal Encoder: `2020-Jan-27_08-33-45_bb844366-40df-11ea-978f-0242ac1c0002`
- Temporal Encoder with Warp: `2020-Jan-27_13-15-16_0f33c94c-4107-11ea-b1ea-0242ac1c0002` and `2020-Jan-27_10-03-01_33ae4af6-40ec-11ea-ae77-0242ac1c0002`

#### NeuralTextures
- Baseline: `2020-Jan-26_18-42-52_a8c7549e-406b-11ea-b304-0242ac1c0002`
- Temporal Encoder: `2020-Jan-26_17-41-12_0b8741d8-4063-11ea-b0dc-0242ac1c0002` and `2020-Jan-26_18-32-32_36eecdda-406a-11ea-b713-0242ac1c0002`
- Temporal Encoder with Warp: `2020-Jan-27_15-11-57_5c758a8c-4117-11ea-b1ea-0242ac1c0002`

#### Face2Face
- Baseline: `2020-Jan-26_19-09-43_69439428-406f-11ea-b304-0242ac1c0002`
- Temporal Encoder: `2020-Jan-26_19-09-36_64b97d82-406f-11ea-9db4-0242ac1c0002`

#### FaceSwap
- Baseline: `2020-Jan-26_19-43-44_29c5b56a-4074-11ea-b304-0242ac1c0002`
- Temporal Encoder: `2020-Jan-26_20-48-58_464da536-407d-11ea-9db4-0242ac1c0002`

## Training on all fake methods simultaneously and training on 1000 videos (runs by Lukas)
In the ablation studies, we mentioned this kind of training. We made the following training runs for it.

#### Deepfake with 1000 videos, locally (small batch-size)
- Baseline: `2020-Jan-21_18-22-48_a5631e5c-3c72-11ea-ae0e-d0509936bc3f` and `2020-Jan-25_09-48-13_6c26b09e-3f4f-11ea-b90f-d0509936bc3f`
- Temporal Encoder: `2020-Jan-27_07-15-14_61d7bd4c-40cc-11ea-be9f-d0509936bc3f` and `2020-Jan-27_18-46-59_04cf7cb8-412d-11ea-9eb2-d0509936bc3f` and `2020-Jan-27_18-56-14_4f879646-412e-11ea-967e-d0509936bc3f`

#### All fake methods with 1000 videos, locally (small batch-size)
- Baseline: `2020-Jan-21_20-43-06_3e99b0ee-3c86-11ea-b918-d0509936bc3f` and `2020-Jan-25_10-30-50_60611f7a-3f55-11ea-ba0f-d0509936bc3f`
- Temporal Encoder: `2020-Jan-22_07-30-13_a59af72e-3ce0-11ea-b9eb-d0509936bc3f` and `2020-Jan-25_11-17-41_eb63b152-3f5b-11ea-8387-d0509936bc3f`

#### All fake methods with 100 videos, on Colab (small dataset size)
- Baseline: `2020-Jan-22_19-00-31_7624792e-3d49-11ea-a860-0242ac1c0002`and `2020-Jan-25_09-18-51_b3c8b0fe-3f53-11ea-97cc-0242ac1c0002`
- Temporal Encoder: `2020-Jan-21_19-11-12_ca36a744-3c81-11ea-9e95-0242ac1c0002` and `2020-Jan-25_14-57-03_f25deff8-3f82-11ea-b5d8-0242ac1c0002`

## Hyperparameter tuning from Table 3 (runs by Lukas)
Here we show the results of testing different values of the delta_t parameter and for the skip rate. We tested it on the temporal encoder model. The model was trained on one single fake method (deepfake) + original videos with the configuration: 100 videos, 5 sequences, 10 frames, skip rate 5. The different hyperparameters can be found in the `HPARAMS` tab in tensorboard. All runs for the hyperparameter tuning are listed below.

- `2020-Jan-02_13_50_54_e556662e-2d66-11ea-b9de-0242ac1c0002`
- `2020-Jan-03_18_06_37_c917ec42-2e53-11ea-a81c-0242ac1c0002`
- `2020-Jan-10_18_16_33_551d0f02-33d5-11ea-940c-0242ac1c0002`
- `2020-Jan-10_19_26_54_2902ec7a-33df-11ea-940c-0242ac1c0002`
- `2020-Jan-10_21_20_52_14aa5122-33ef-11ea-940c-0242ac1c0002`
- `2020-Jan-11_20_57_38_fffc4a7a-34b4-11ea-a605-0242ac1c0002`
- `2020-Jan-11_23_13_19_f47d3188-34c7-11ea-a605-0242ac1c0002`

