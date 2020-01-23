### Author: Lukas Hoellein

I tested the flownet2 implementation here: https://github.com/NVIDIA/flownet2-pytorch
to calculate optical-flow and warp.

### Results
Hard to integrate in our codebase, calculates similar results (optically) as opencv implementation, crops images that are not a multiple of 64 in width/height. Since we use 299x299, we decided not to use it.

### Decision
Optical flow is a fixed input for us, so we do not require a neural network for it.
Instead, use standard OpenCV methods to calculate it directly.