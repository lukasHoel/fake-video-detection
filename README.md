# ADL4CV - Advanced Deep Learning for Computer Vision Project
<b>Authors:</b> Anna Mittermair and Lukas Hoellein <br>
<b>Supervisor:</b> Maxim Maximov

(See files for a more specific matching of authorship to certain parts of the project. This might help in resolving any questions that might come up while studying our project code.)

## Low-Resolution Fake Video Detection

Existing  approaches  for  face  forgery  detection  test  a video frame-by-frame without using (temporal) connections between  the  frames.    This  makes  it  particularly  hard  to detect forgery when the videos are of low resolution (e.g. highly  compressed).    We  propose  a  new  way  to  sample the videos and incorporate them into networks that make use of the temporal context between frames.  We also propose architectures that make use of the optical flow between frames.  Our approaches improved the detection accuracy across the bench.

## Directory structure of this project:

- `networks:` contains all networks that we used for this project
- `notebooks:` contains the (very flexible, configurable) jupyter notebooks that we used for training/validation/testing. See Readme there for additional information.
- `runs:` contains saved results from our training/validation/test runs as tensorboard files. See Readme there for additional information.
- `saved_results:` contains saved models from our training/validation/test runs as .pt files. See Readme there for additional information
- `training:` contains our dataloaders and our solver used for training/validation/testing.
- `utils:` contains several utility python scripts, e.g. for dataset generation, optical flow calculation, documentation of commands, ...

## Existing Results (from our report and others)

All our existing results are saved as tensorboard files for transparency and can be accessed through the directory `runs`. See that directory for additional information.

## Inference

We saved all models that produced accuracy numbers mentioned in the report for transparency of our results. You can load these models and test them yourselves.
<br>Also see the directory `saved_results` for additional information as to how these models were saved.

1. Load pretrained models from our cloud storage: https://drive.google.com/drive/folders/1m_XR1HWRMkXv-pS2bUxo3hEHsMeJ3fxN?usp=sharing
2. Include our datasets into your google drive storage: https://drive.google.com/drive/folders/19b6ts-z3sW1D_WROVn_vggG_1dbxjbII?usp=sharing <br>
   Alternatively, you can download the video material from the official FaceForensics page and extract the sequences yourself using our extraction scripts in `utils`.
3. Start a corresponding jupyter notebook from `notebooks` to load the pretrained model. Follow the instructions in the notebook to perform inference with a dataset/model. <br> Also see the directory `notebooks` for additional information about using them.
4. All test results are automatically saved in a tensorboard file in the `runs` directory when using the notebooks.

## Training

Similar to inference, you can also train your own models.

1. Include our datasets into your google drive storage: https://drive.google.com/drive/folders/19b6ts-z3sW1D_WROVn_vggG_1dbxjbII?usp=sharing <br>
   Alternatively, you can download the video material from the official FaceForensics page and extract the sequences yourself using our extraction scripts in `utils`.
2. Start a corresponding jupyter notebook from `notebooks` to train a new model. Follow the instructions in the notebook to perform training. <br> Also see the directory `notebooks` for additional information about using them.
3. All training results are automatically saved in a tensorboard file in the `runs` directory when using the notebooks.

