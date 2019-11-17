Basic structure of the dataset:
Currently, we downloaded 30 example videos, but the FaceForensics dataset contains much more data.

original_sequences: Contains unmanipulated videos
    youtube
        c23
            videos
        c40
            videos:
            images: extracted images for corresponding videos 
                <subdir for one specific video, named just like the video file name, e.g. 033_097>
                    <all images as png numbered from 0000.png>
                    downsampled: images downsampled to have a height of 120 pixel and corresponding width, images are named like originals but with suffix _-1:120, e.g. 0000_-1:120.png
    actors
        <...same structure...>
manipulated_sequences
    <the following have the same structure as above, e.g. they have extracted images and downsamples>
    Deepfakes
    Face2Face
    FaceSwap

    <the following only have the videos as source>
    NeuralTextures
    DeepFakeDetection