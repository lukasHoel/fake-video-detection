python main.py --inference --model FlowNet2 --save_flow --inference_dataset ImagesFromFolder --inference_dataset_root ~/001_870/0/ --resume FlowNet2_checkpoint.pth.tar

python -m flowiz work/inference/run.epoch-0-flow-field/*.flo -o ~/rectangle

