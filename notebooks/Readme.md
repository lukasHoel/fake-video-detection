# How to use the jupyter notebooks

## 1. Make sure that you have our dataset (sequences of images) 
- either referenced in your google cloud storage: https://drive.google.com/drive/folders/19b6ts-z3sW1D_WROVn_vggG_1dbxjbII?usp=sharing
- or extracted on your own from the official FaceForensics videos with our scripts in `utils`

## 2. Select one of the jupyter notebooks for training/inference. 
- Baseline Training (by Lukas): Train/Test our baseline model
- Temporal Encoder Two Training (by Lukas): Train/Test the temporal encoder network (the suffix two can be ignored)
- Optical Flow Training (by Anna): Train/Test any of the optical flow networks. Just select the network in the notebook.

## 3. When doing inference make sure that you have a pretrained model available
- from our cloud storage: https://drive.google.com/drive/folders/1m_XR1HWRMkXv-pS2bUxo3hEHsMeJ3fxN?usp=sharing
- from your own pretrained models

