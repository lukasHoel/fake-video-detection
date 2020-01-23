Contains all saved models for the specific runs found in runs directory.

### Naming conventions:

- Baseline (trained by Lukas): baseline_<id>.pt
- TemporalEncoderII (trained by Lukas): temporal_encoder_2_<id>.pt

Each model has a unique identifier as suffix. This identifier can also be used to load the corresponding training/val/test run, if it was saved in runs directory.