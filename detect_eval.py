"""
Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation
Detection and evaluation

Modified based on Mask R-CNN(https://github.com/matterport/Mask_RCNN)
Written by He Wang
"""

import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='detect', type=str, help="detect/eval")
parser.add_argument('--resnet', type=str)
parser.add_argument('--bin', type=int, default=0)
parser.add_argument('--use_delta', dest='use_delta', action='store_true')
parser.add_argument('--share_weight', dest='share_weight', action='store_true')
parser.add_argument('--coord_size', type=int, default=14)
parser.add_argument('--ckpt_path', type=str)
parser.add_argument('--data', type=str, help="val/real_val/real_test")
parser.add_argument('--gpu',  default='0', type=str)
parser.add_argument('--draw', dest='draw', action='store_true', help="training/inference")
parser.add_argument('--num_eval', type=int, default=-1)

parser.set_defaults(draw=False)
parser.set_defaults(use_delta=False)
parser.set_defaults(share_weight=False)

args = parser.parse_args()

mode = args.mode
data = args.data
n_bins = args.bin
coord_size = args.coord_size
resnet = 'resnet'+args.resnet
ckpt_path = args.ckpt_path
share_weight = args.share_weight
use_delta = args.use_delta
num_eval = args.num_eval

os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
print('Using GPU {}.'.format(args.gpu))

import sys
import datetime
import glob
import time
import numpy as np
from config import Config
import utils
import model as modellib
import visualize
from dataset import TOICOCODataset
import _pickle as cPickle


# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to COCO trained weights
COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")


class ScenesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "ShapeNetTOI"
    OBJ_MODEL_DIR = os.path.join(ROOT_DIR, 'data', 'ShapeNetTOIModels', 'ShapeNetTOIFinalModels')
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # background + 6 object categories
    MEAN_PIXEL = np.array([[ 120.66209412, 114.70348358, 105.81269836]])

    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640

    if IMAGE_MIN_DIM == 480 and IMAGE_MAX_DIM == 640:
        USE_SMALL_IMAGE = False
    elif IMAGE_MIN_DIM == 240 and IMAGE_MAX_DIM == 320:
        USE_SMALL_IMAGE = True
    else:
        print('Image resolution is wrong.')
        exit()

    RPN_ANCHOR_SCALES = (16, 32, 48, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 64

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50

    WEIGHT_DECAY = 0.0001
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    COORD_LOSS_SCALE = 1
    COORD_USE_BINS = (n_bins != 0)
    COORD_NUM_BINS = n_bins
    COORD_SHARE_WEIGHTS = share_weight
    COORD_REGRESS_LOSS   = 'Soft_L1'
    COORD_USE_DELTA = use_delta
    COORD_USE_UNET = False
    COORD_USE_FC = False

    COORD_USE_CASCADE_L1 = False
    COORD_SECOND_NORM = False

    COORD_POOL_SIZE = coord_size
    COORD_SHAPE = [2*coord_size, 2*coord_size]

    USE_BN = True
    if COORD_SHARE_WEIGHTS:
        USE_BN = False

    USE_RNN = False
    USE_SYMMETRY_LOSS = True
    if USE_RNN:
        IMAGES_PER_GPU = 1

    RESNET = resnet
    TRAINING_AUGMENTATION = False
    SOURCE_WEIGHT = [3, 1, 1] #'ShapeNetTOI', 'Real', 'coco'



class InferenceConfig(ScenesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


if __name__ == '__main__':

    config = ScenesConfig()
    config.display()

    # Training dataset
    toi_dir = os.path.join('data', 'shapenet_toi_330K')
    coco_dir = os.path.join('data', 'coco')
    #  real classes
    coco_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                  'bus', 'train', 'truck', 'boat', 'traffic light',
                  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                  'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                  'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                  'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                  'kite', 'baseball bat', 'baseball glove', 'skateboard',
                  'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                  'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                  'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                  'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                  'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                  'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                  'teddy bear', 'hair drier', 'toothbrush']

    
    synset_names = ['BG', #0
                    'bottle', #1
                    'bowl', #2
                    'camera', #3
                    'can',  #4
                    'laptop',#5
                    'mug'#6
                    ]


    class_map = {
        'bottle': 'bottle',
        'bowl':'bowl',
        'cup':'mug',
        'laptop': 'laptop',
    }


    coco_cls_ids = []
    for coco_cls in class_map:
        ind = coco_names.index(coco_cls)
        coco_cls_ids.append(ind)
    config.display()


    assert mode in ['detect', 'eval']
    if mode == 'detect':

        # Get path to saved weights
        # Either set a specific path or find last trained weights
        # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
        # model_path = model.find_last()[1]

        
        # model_name = 'ablation_study_toi_res101_softl1_noshared_augment_pool14'
        # checkpoint_name = 'mask_rcnn_shapenettoi_0285.h5'
        
        #model_name = 'ablation_study_toi_res101_softl1_noshared_augment_rot60'
        #checkpoint_name = 'mask_rcnn_shapenettoi_0275.h5'
        #model_name = 'real_training_res101_softl1_noshared_augment_coord28_rot60_camr+coco+real'
        #checkpoint_name = 'mask_rcnn_shapenettoi_0090.h5'

        # model_name = 'TCR_res50_softl1_noshared_augment_pool14/'
        # checkpoint_name = 'mask_rcnn_shapenettoi_0220.h5'
        
        # model_name = 'real_training_res50_softl1_noshared_augment_coord28_rot60_camr+coco+real'
        # checkpoint_name = 'mask_rcnn_shapenettoi_0125.h5'

        ckpt_parsing = ckpt_path.split('/')
        model_name = ckpt_parsing[-2]
        checkpoint_name = ckpt_parsing[-1]

        checkpoint_folder = checkpoint_name.split('.')[0]
        model_dir = './logs/'
        model_path = os.path.join(model_dir, model_name, checkpoint_name)
        
        inference_config = InferenceConfig()

        # Recreate the model in inference mode
        model = modellib.MaskRCNN(mode="inference",
                                  config=inference_config,
                                  model_dir=MODEL_DIR)

        gt_dir = os.path.join(toi_dir, 'gts', data)
        if data == 'train':
            dataset_train = TOICOCODataset(synset_names, 'train', config)
            dataset_train.load_scenes(toi_dir)
            dataset_train.load_real_scenes(toi_dir)
            dataset_train.prepare(class_map)
            dataset = dataset_train
        elif data == 'real_train':
            dataset_real_train = TOICOCODataset(synset_names, 'train', config)
            dataset_real_train.load_real_scenes(toi_dir)
            dataset_real_train.prepare(class_map)
            dataset = dataset_real_train
        elif data == 'coco_train':
            dataset_coco_train = TOICOCODataset(synset_names, 'train', config)
            dataset_coco_train.load_coco(coco_dir, "train", class_ids=coco_cls_ids)
            dataset_coco_train.prepare(class_map)
            dataset = dataset_coco_train
        elif data == 'val':
            dataset_val = TOICOCODataset(synset_names, 'val', config)
            dataset_val.load_scenes(toi_dir)
            dataset_val.prepare(class_map)
            dataset = dataset_val
        elif data == 'real_val':
            dataset_real_val = TOICOCODataset(synset_names, 'val', config)
            dataset_real_val.load_real_scenes(toi_dir)
            dataset_real_val.prepare(class_map)
            dataset = dataset_real_val
        elif data == 'coco_val':
            dataset_coco_val = TOICOCODataset(synset_names, 'val', config)
            dataset_coco_val.load_coco(coco_dir, "val", class_ids=coco_cls_ids)
            dataset_coco_val.prepare(class_map)
            dataset = dataset_coco_val
        elif data == 'real_test':
            dataset_real_test = TOICOCODataset(synset_names, 'test', config)
            dataset_real_test.load_real_scenes(toi_dir)
            dataset_real_test.prepare(class_map)
            dataset = dataset_real_test
        else:
            assert False, "Unknown data resource."

        

        # Load trained weights (fill in path to trained weights here)
        assert model_path != "", "Provide path to trained weights"
        print("Loading weights from ", model_path)
        model.load_weights(model_path, by_name=True)


        image_ids = dataset.image_ids
        #image_ids = range(4412, 4413)
        #image_ids = dataset.image_ids[:]
        save_per_images = 10
        now = datetime.datetime.now()
        save_dir = os.path.join('output', model_name, "{}_{}_{:%Y%m%dT%H%M}".format(
            checkpoint_folder.lower(), data, now))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        log_file = os.path.join(save_dir, 'error_log.txt')
        f_log = open(log_file, 'w')

        if data in ['real_val', 'real_train', 'real_test']:
            intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
        elif data in ['coco_val', 'coco_train']:
            intrinsics = None
        else: ## synthetic data
            intrinsics = np.array([[577.5, 0, 319.5], [0., 577.5, 239.5], [0., 0., 1.]])

        elapse_times = []
        if mode != 'eval':
            for i, image_id in enumerate(image_ids):

                print('*'*50)
                image_start = time.time()
                print('Image id: ', image_id)
                image_path = dataset.image_info[image_id]["path"]
                print(image_path)


                # record results
                result = {}

                # loading ground truth
                image = dataset.load_image(image_id)
                depth = dataset.load_depth(image_id)

                gt_mask, gt_coord, gt_class_ids, gt_scales, gt_domain_label = dataset.load_mask(image_id)
                gt_bbox = utils.extract_bboxes(gt_mask)
     
                result['image_id'] = image_id
                result['image_path'] = image_path

                result['gt_class_ids'] = gt_class_ids
                result['gt_bboxes'] = gt_bbox
                result['gt_RTs'] = None            
                result['gt_scales'] = gt_scales

                image_path_parsing = image_path.split('/')
                gt_pkl_path = os.path.join(gt_dir, 'results_{}_{}_{}.pkl'.format(data, image_path_parsing[-2], image_path_parsing[-1]))
                print(gt_pkl_path)
                if (os.path.exists(gt_pkl_path)):
                    with open(gt_pkl_path, 'rb') as f:
                        gt = cPickle.load(f)
                    result['gt_RTs'] = gt['gt_RTs']
                    if 'handle_visibility' in gt:
                        result['gt_handle_visibility'] = gt['handle_visibility']
                        assert len(gt['handle_visibility']) == len(gt_class_ids)
                        print('got handle visibiity.')
                    else: 
                        result['gt_handle_visibility'] = np.ones_like(gt_class_ids)
                else:
                    # align gt coord with depth to get RT
                    if not data in ['coco_val', 'coco_train']:
                        if len(gt_class_ids) == 0:
                            print('No gt instance exsits in this image.')

                        print('\nAligning ground truth...')
                        start = time.time()
                        result['gt_RTs'], _, error_message = utils.align(gt_class_ids, 
                                                                         gt_mask, 
                                                                         gt_coord, 
                                                                         depth, 
                                                                         intrinsics, 
                                                                         synset_names, 
                                                                         image_path,
                                                                         save_dir+'/'+'{}_{}_{}_gt_'.format(data, image_path_parsing[-2], image_path_parsing[-1]))
                        print('New alignment takes {:03f}s.'.format(time.time() - start))

                        if len(error_message):
                            f_log.write(error_message)

                    result['gt_handle_visibility'] = np.ones_like(gt_class_ids)

                ## detection
                start = time.time()
                detect_result = model.detect([image], verbose=0)
                r = detect_result[0]
                elapsed = time.time() - start
                
                print('\nDetection takes {:03f}s.'.format(elapsed))
                result['pred_class_ids'] = r['class_ids']
                result['pred_bboxes'] = r['rois']
                result['pred_RTs'] = None   
                result['pred_scores'] = r['scores']


                if not data in ['coco_val', 'coco_train']:
                    if len(r['class_ids']) == 0:
                        print('No instance is detected.')

                    print('Aligning predictions...')
                    start = time.time()
                    result['pred_RTs'], result['pred_scales'], error_message, elapses =  utils.align(r['class_ids'], 
                                                                                            r['masks'], 
                                                                                            r['coords'], 
                                                                                            depth, 
                                                                                            intrinsics, 
                                                                                            synset_names, 
                                                                                            image_path)
                                                                                            #save_dir+'/'+'{}_{}_{}_pred_'.format(data, image_path_parsing[-2], image_path_parsing[-1]))
                    print('New alignment takes {:03f}s.'.format(time.time() - start))
                    elapse_times += elapses
                    if len(error_message):
                        f_log.write(error_message)


                    if args.draw:
                        draw_rgb = False
                        utils.draw_detections(image, save_dir, data, image_path_parsing[-2]+'_'+image_path_parsing[-1], intrinsics, synset_names, draw_rgb,
                                              gt_bbox, gt_class_ids, gt_mask, gt_coord, result['gt_RTs'], gt_scales, result['gt_handle_visibility'],
                                              r['rois'], r['class_ids'], r['masks'], r['coords'], result['pred_RTs'], r['scores'], result['pred_scales'])
                
                else:
                    if args.draw:
                        draw_rgb = False
                        utils.draw_coco_detections(image, save_dir, data, image_path_parsing[-2]+'_'+image_path_parsing[-1], synset_names, draw_rgb,
                                                  gt_bbox, gt_class_ids, gt_mask,  
                                                  r['rois'], r['class_ids'], r['masks'], r['coords'], r['scores'])


                path_parse = image_path.split('/')
                image_short_path = '_'.join(path_parse[-3:])

                save_path = os.path.join(save_dir, 'results_{}.pkl'.format(image_short_path))
                with open(save_path, 'wb') as f:
                    cPickle.dump(result, f)
                print('Results of image {} has been saved to {}.'.format(image_short_path, save_path))

                
                elapsed = time.time() - image_start
                print('Takes {} to finish this image.'.format(elapsed))
                print('Alignment average time: ', np.mean(np.array(elapse_times)))
                print('\n')
            
            f_log.close()

        log_dir = save_dir


    else:
        #log_dir = 'output/ablation_study_toi_res101_softl1_noshared_augment_pool14/mask_rcnn_shapenettoi_0285_val_20181104T1818'
        log_dir = ckpt_path
        #log_dir = 'output/ablation_study_toi_res101_softl1_noshared_augment_coord56_rot60/mask_rcnn_shapenettoi_0125_val_20181106T1559'
        #log_dir = 'output/TCR_res50_softl1_noshared_augment_pool14_rot60/mask_rcnn_shapenettoi_0340_real_val_20181110T1404'
        #log_dir ='output/real_training_res101_softl1_noshared_augment_coord28_rot60_camr+coco+real/mask_rcnn_shapenettoi_0090_real_val_20181110T1336'


    result_pkl_list = glob.glob(os.path.join(log_dir, 'results_*.pkl'))
    result_pkl_list = sorted(result_pkl_list)[:num_eval]
    assert len(result_pkl_list)
    #print(result_pkl_list)

    final_results = []
    for pkl_path in result_pkl_list:
        with open(pkl_path, 'rb') as f:
            result = cPickle.load(f)
            if not 'gt_handle_visibility' in result:
                result['gt_handle_visibility'] = np.ones_like(result['gt_class_ids'])
                print('can\'t find gt_handle_visibility in the pkl.')
            else:
                assert len(result['gt_handle_visibility']) == len(result['gt_class_ids']), "{} {}".format(result['gt_handle_visibility'], result['gt_class_ids'])


        if type(result) is list:
            final_results += result
        elif type(result) is dict:
            final_results.append(result)
        else:
            assert False


    num_classes = len(synset_names)

    degree_thres = range(0, 31, 1) 
    shift_thres =  range(0, 31, 1)
    #utils.compute_coords_aps(final_results, synset_names, degree_thres, shift_thres)
    aps = utils.compute_degree_cm_mAP(final_results, synset_names, log_dir,
                                                                   degree_thresholds = [5, 10, 15],#range(0, 61, 1), 
                                                                   shift_thresholds= [5, 10, 15], #np.linspace(0, 1, 31)*15, 
                                                                   iou_3d_thresholds=np.linspace(0, 1, 101),
                                                                   iou_pose_thres=0.1,
                                                                   use_matches_for_pose=True)
    #aps = utils.compute_degree_cm_mAP(final_results, synset_names, degree_thresholds=range(0, 101, 1), shift_thresholds=None, log_dir=log_dir)
    #aps = utils.compute_degree_cm_mAP(final_results, synset_names, degree_thresholds=None, shift_thresholds=range(0, 31, 1), log_dir=log_dir)

    


    
