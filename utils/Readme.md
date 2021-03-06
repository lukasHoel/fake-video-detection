# Utils
With the scripts contained here, you can create your own dataset from the official FaceForensics dataset. It will create sequences of images and can be configured as described below. <br><br>
Additionally, this directory contains functions to create optical flow from images and contains preliminary testing results from using flownet2 as optical flow implementation.

## Author: Lukas Hoellein

### Usage of testset extraction (official splits):

    python create_splits.py -i "F:/Google Drive/FaceForensics_large"
                            -s sequences_299x299_10seq@10frames_skip_5_uniform
                            -c c40
                            -sd "F:/Google Drive/adl4cv/utils/splits"

    Detailed description:

    Splits a sequence_folder created by the "extract_images" script into train/val/test set according to the official
    FaceForensics train/val/test split found in https://github.com/ondyari/FaceForensics/tree/master/dataset/splits


### Usage of testset extraction (uniform split for less than 1000 videos):

	python create_testset.py -i C:\Users\admin\Desktop\FaceForensics
							 -o C:\Users\admin\Desktop\
							 -s sequences_299x299_skip_5_uniform
							 -m uniform
							 -c c40
							 -p 10

	Detailed description:
	
    Assumes equal amount of videos in all dataset directories!!
    Will move for all datasets including original (only selecting a few datasets is not possible due to ambiguities!)

    For all datasets: Recursively moves a uniform/random sampled subset of directories in 'input/dataset/compression/sequence' to
    'output/FaceForensics_Testset/dataset/compression/sequence'. The number of directories to be moved can be controlled
    by the percentage parameter.

    It is made sure that symmetric sequences (that is e.g. 000_003 and 003_000) are either both moved or none moved.
    If one of the two sequences got selected, the other one gets selected as well. It is made sure that the overall
    percentage is still met.

    :param input: root directory of the FaceForensics folder.
    :param output: root directory where the output folder "FaceForensics_Testset" shall be created
    :param sequence: from which sequence to extract into the testset (e.g. from sequences_299x299_skip_5_uniform)
    :param compression: from which compression level to extract into the testset (e.g. from c40)
    :param percentage: how many sequences to extract into the testset (e.g. 10%)
    :param sample_mode: how to extract (e.g. uniform: every i-th, random)
							 
							 
### Usage of sequence generation (with Face Cropping):

    requires installation of opencv (>3.x) and opencv-utils, e.g. pip install opencv-python, pip install opencv-utils

    python extract_images.py --data_path path/to/FaceForensics/ --dataset {FaceSwap, Deepfakes, Face2Face, original1, original2} --compression {c0, c23, c40}
    
    additional parameters:
        --num_sequences
        --frames_per_sequence
        --skip_frames
        --size
        --padding
        
    Detailed description:
    
    Extracts sequences from all videos in <data_path>/<dataset>/<compression>/videos into <data_path>/<dataset>/<compression>/sequences.
    Will use the FaceForensics file structure to identify all videos of dataset type specified.
    A sequence is a number of face-cropped images starting from a random/uniform frame number inside of the video with configurable number of
    frames to skip between two images.
    Videos will be saved in a subdirectory structure as follows:
    <root_of_dataset>
        <root_of_dataset>/videos
            <here lie all videos to extract from, e.g. foo.mp4, bar.mp4>
        <root_of_dataset>/sequences
            <foo>: subdirectory with the name of the original video (without datatype suffix e.g. without .mp4)
                <0>: number of sequence
                    0000.png: first picture in this sequence
                    0001.png: second picture in this sequence
                <1>
                    0000.png
                    0001.png
            <bar>
                <0>
                    0000.png
                    0001.png
                <1>
                    0000.png
                    0001.png

    :param data_path: path of FaceForensics root directory
    :param dataset: of which dataset to choose from. See DATASET_PATHS
    :param compression: of which compression level to choose from. See COMPRESSION
    :param num_sequences: how many sequences to extract. default: 5
    :param frames_per_sequence: how many frames each sequence shall contain. default: 10
    :param skip_frames: how many frames shall be skipped between two frames (to capture changes in expression) default: 5
    :param size: how big each frame shall be. Is considered to give both width and height, thus resulting image is quadratic. default: 128
    :param padding: how much padding shall be used around detected face crop in each direction (to capture all of the face) default: 30
    :param sample_mode: whether samples shall be selected uniform or random. default: uniform.
                        Uniform means deterministic selection of frame-numbers and random means random selection of frame-numbers
                        for the first frame in a sequence.

### Usage of old image extraction (from FaceForensics Github):
    python extract_compressed_videos.py --data_path path/to/FaceForensics/ --dataset {FaceSwap, Deepfakes, Face2Face, original1, original2} --compression {c0, c23, c40}
    
    This will extract images of the specified dataset in the specified compression.
    For 30 examples in the c40 compression, this will create ~10GB of images per dataset.
    
    specifying original1 will exract images from original_sequences/actors and original2 from original_sequences/youtube
    
    This program requires opencv installation, e.g. with conda install opencv
    
### Usage of downsampling:
    ./downsample.sh <base-path> e.g. ./downsample.sh manipulated_sequences/Deepfakes/c40/images/
    
    base-path shall be the directory in which the different videos/images lie, e.g.
        FaceForensics\manipulated_sequences\Deepfakes\c40\images\
            033_097
            123_456
            234_567
            345_678
            
    It will create a subdirectory "downsampled" in every subdirectory, e.g. in 033_097/downsampled
    Into that directory every image of e.g. 033_097 will be downsampled with height of 120 and corresponding width that keeps the aspect ratio
    e.g. the file 037_097/0000.png is later found in downsampled/0000_-1:120.png
    
    This program requires ffmpeg installation, e.g. sudo apt-get install ffmpeg
            
