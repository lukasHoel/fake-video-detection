Contains all runs for the specific model as tensorboard files.
To view the runs, simply start tensorboard via:

`tensorboard --logidr path/to/runs`

###Naming conventions:

- Baseline (trained by Lukas): networks/baseline.py
- TemporalEncoderII (trained by Lukas): networks/temporal_encoder_network.py

Each run has a unique identifier as folder name. This identifier can also be used to load the corresponding model, if it was saved in saved_results/models.