# ml-support-code-in-python
Some functions for supporting machine learning, generally and caffe-specific

caffe - python layers for
    image read (file or lmdb)
    on-the-fly augmentation
    support for pixel-level segmentation (read mask as label, etc)
    controlling and reporting acc/loss of solver with solver.step

image processing
    image augmentation - including bounding boxes and pixel level
    acccuracy/precision/recall reporting for single label, multilabel , bounding box, and pixel level
    utilities e.g. read from anywhere (url/db/local file/img array)

