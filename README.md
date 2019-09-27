# Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation
This is a keras and tensorflow implementation of [**Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation**](https://arxiv.org/pdf/1901.02970.pdf), a **CVPR 2019 oral** paper. 

![NOCS Teaser](https://github.com/hughw19/NOCS_CVPR2019/raw/master/images/teaser.jpg)

The repository includes:

* Source code of NOCS.
* Training code
* Detection and evaluation code
* Pre-trained weights

# Requirements
* Python 3.5
* Tensorflow 1.14.0
* Keras 2.3.0
* cPickle

# Datasets
* CAMERA Dataset: [Training](http://download.cs.stanford.edu/orion/nocs/camera_train.zip)/[Test](http://download.cs.stanford.edu/orion/nocs/camera_val25K.zip)
* Real Dataset: [Training](http://download.cs.stanford.edu/orion/nocs/real_train.zip)/[Test](http://download.cs.stanford.edu/orion/nocs/real_test.zip)
* Ground truth pose annotation (for an easier evaluation): [Val&Real_test](http://download.cs.stanford.edu/orion/nocs/gts.zip)
* [Object Meshes](http://download.cs.stanford.edu/orion/nocs/obj_models.zip)

You can download the files and store them under data/.

NOTE: You are required to cite our paper if you use the dataset. The data is only for non-commercial use. Please reach out to us for other use cases.

## Pretrain weights
You can find the following checkpoints in this [download link](http://download.cs.stanford.edu/orion/nocs/ckpts.zip):
* NOCS RCNN jointly trained on CAMERA, Real & MS COCO with 32 bin classification setting
* NOCS RCNN jointly trained on CAMERA, Real & MS COCO with regression setting
* Mask RCNN pretrained on MS COCO dataset 

You can download the checkpoints and store them under logs/.

# Training
```
# Train a new model from pretrained COCO weight
python3 train.py
```

# Detection and Evaluation
```
# Detect using a checkpoint
python3 detect_eval.py --mode detect --ckpt_path=/logs/ckpt --draw

# Evaluate a checkpoint
python3 detect_eval.py --mode eval --ckpt_path=/output/ckpt 

```

