# Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation
This is a keras and tensorflow implementation of [**Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation**](https://arxiv.org/pdf/1901.02970.pdf), a CVPR 2019 oral paper. 

![NOCS Teaser](https://github.com/hughw19/NOCS_CVPR2019/raw/master/images/teaser.jpg)

The repository includes:

* Source code of NOCS.
* Training code
* Detection and evaluation code
* Pre-trained weights

# Datasets
* CAMERA Dataset: [Training](http://download.cs.stanford.edu/orion/nocs/camera_train.zip)/[Test](http://download.cs.stanford.edu/orion/nocs/camera_val25K.zip)
* Real Dataset: [Training](http://download.cs.stanford.edu/orion/nocs/real_train.zip)/[Test](http://download.cs.stanford.edu/orion/nocs/real_test.zip)

NOTE: You are required to cite our paper if you use the dataset. The data is only for non-commercial use. Please reach out to us for other use cases.


# Requirements
* Python 3.5
* Tensorflow 1.3.0
* Keras 2.1.5
* cPickle

# Training
```
# Train a new model from pretrained COCO weight
python3 train.py
```

# Detection and Evaluation
```
# Detect using a checkpoint
python3 detect_eval.py --mode detect --ckpt_path=/ckpts/ckpt 

# Evaluate a checkpoint
python3 detect_eval.py --mode eval --ckpt_path=/ckpts/ckpt 

```

