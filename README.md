# NOCS
# Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation
This is a keras and tensorflow implementation of Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation, a CVPR 2019 oral paper. 

The repository includes:

* Source code of NOCS.
* Training code
* Detection and evaluation code
* Pre-trained weights

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

