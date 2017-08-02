ml support code in python
=========================


Typical usage
often looks like this::

    #!/usr/bin/env python
    import ml_utils as mlu
    mlu.imutils.resize_keep_aspect(img_arr):


Some functions for supporting machine learning, generally and caffe-specific. Currently just a dump, ideally will clean up and add install.py, examples etc

caffe - python layers for
-------------------------
* image read (file or lmdb) - single / multi label, pixel level labels

* on-the-fly augmentation

* support for pixel-level segmentation (read mask as label, etc)

* controlling and reporting acc/loss of solver with solver.step

image processing
----------------

* image augmentation - including bounding boxes and pixel level

* acccuracy/precision/recall reporting for single label, multilabel , bounding box, and pixel level

* utilities e.g. read from anywhere (url/db/local file/img array)
    
* grabcut

read/write various file formats
-------------------------------

* lmdb, hd5

* yolo
    
* deepfashion

* tamara berg

* ILSRVC
    
* etc


tagging tools
-------------

* multilabel
    
* pixel level
