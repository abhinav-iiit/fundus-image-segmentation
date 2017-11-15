# fundus-image-segmentation
* Tried to replicate results obtained by https://github.com/seva100/optic-nerve-cnn for cup segmentation on DrishtiGS dataset.
* data_testing folder contains 'masks' and 'images'. Folders masks and images contain cropped 512x512 ROI images and corresponding ground truth segmentations.
* test.py contains code for U-net model, input processing, loading of weights and generation of output.
* input folder contains resized and contrast adjusted ROI images for input to U-net.
* output folder contains segmentation maps obtained.
