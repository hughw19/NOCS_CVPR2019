"""
Mask R-CNN
Common utility functions and classes.
Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import sys
import os
import cv2
import math
import random
import numpy as np
import tensorflow as tf
import scipy.misc
import skimage.color
import _pickle as cPickle
from ctypes import *
import copy
import ICP
import glob
import time


from aligning import estimateSimilarityTransform

sys.path.append('./cocoapi/PythonAPI')
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt



############################################################
#  Bounding Boxes
############################################################

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.
    Note: the areas are passed in rather than calculated here for
          efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps



def compute_overlaps_masks(masks1, masks2):
    '''Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    '''
    # flatten masks
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps


def compute_mean_l1_coord_diff(mask1, mask2, coord1, coord2, synset, cls_id):
    '''Computes IoU overlaps between two sets of masks.
    mask1, mask2: [Height, Width]
    coord1, coord2: [Height, Width, 3]
    
    '''
    # flatten masks
    num_pixels = mask1.shape[0] * mask1.shape[1]
    
    mask1 = np.reshape(mask1 > .5, (-1)).astype(np.float32)
    mask2 = np.reshape(mask2 > .5, (-1)).astype(np.float32)
    coord1 = np.reshape(coord1, (-1, 3)).astype(np.float32)
    coord2 = np.reshape(coord2, (-1, 3)).astype(np.float32)

    # intersections and union
    intersections = np.logical_and(mask1, mask2)
    num_pixel_intersection = len(np.where(intersections)[0])
    
    pts1 = coord1[intersections, :].transpose() - 0.5
    pts2 = coord2[intersections, :].transpose() - 0.5
    
    def rotation_y_matrix(theta):
        rotation_matrix =  \
                np.array([ np.cos(theta), 0,  np.sin(theta),
                             0,           1,  0,
                          -np.sin(theta), 0,  np.cos(theta)])
        rotation_matrix = np.reshape(rotation_matrix, (3, 3))
        return rotation_matrix
      
    if synset[cls_id] in ['bottle', 'bowl', 'can']:
        M = 20
        pts1_symmetry = np.zeros(pts1.shape+(M,))  ## shape: (3, N, 6)
        for i in range(M):
            rotated_pts1 = rotation_y_matrix(float(i)*np.float32(2*math.pi/M)) @ pts1
            pts1_symmetry[:, :, i] = rotated_pts1
        
        pts2_reshape = pts2.reshape([3, -1, 1])
        mean_dists = np.mean(np.linalg.norm(pts1_symmetry - pts2_reshape, axis=0), axis=0)
        mean_dist = np.amin(mean_dists)
    elif synset[cls_id] in ['phone']:
        pts1_symmetry = np.zeros(pts1.shape+(2,))
        
        for i in range(2):
            rotated_pts1 = rotation_y_matrix(float(i)*np.float32(2*math.pi/2)) @ pts1
            #print(rotated_pts1)
            pts1_symmetry[:, :, i] = rotated_pts1
        
        pts2_reshape = pts2.reshape([3, -1, 1])
        mean_dists = np.mean(np.linalg.norm(pts1_symmetry - pts2_reshape, axis=0), axis=0)
        mean_dist = np.amin(mean_dists)
    else:
        #print(synset[cls_id])
        diff = pts1 - pts2
        dist = np.linalg.norm(diff, axis=0)
        assert dist.shape[0] == num_pixel_intersection
        mean_dist = np.mean(dist)
    
    mean_l1_coord_diff = mean_dist
    #print(mean_l1_coord_diff, pts1.shape[0])
    
    return mean_l1_coord_diff




def compute_3d_iou(bbox_3d_1, bbox_3d_2, handle_visibility, class_name_1, class_name_2):
    '''Computes IoU overlaps between two 3d bboxes.
       bbox_3d_1, bbox_3d_1: [3, 8]
    '''
    # flatten masks
    def asymmetric_3d_iou(bbox_3d_1, bbox_3d_2):
        bbox_1_max = np.amax(bbox_3d_1, axis=0)
        bbox_1_min = np.amin(bbox_3d_1, axis=0)
        bbox_2_max = np.amax(bbox_3d_2, axis=0)
        bbox_2_min = np.amin(bbox_3d_2, axis=0)

        overlap_min = np.maximum(bbox_1_min, bbox_2_min)
        overlap_max = np.minimum(bbox_1_max, bbox_2_max)

        # intersections and union
        if np.amin(overlap_max - overlap_min) <0:
        	intersections = 0
        else:
        	intersections = np.prod(overlap_max - overlap_min)
        union = np.prod(bbox_1_max - bbox_1_min) + np.prod(bbox_2_max - bbox_2_min) - intersections
        overlaps = intersections / union
        return overlaps


    if bbox_3d_1 is None or bbox_3d_2 is None:
        return -1

    symmetry_flag = False
    if class_name_1 in ['bottle', 'bowl', 'can'] and class_name_1 == class_name_2:
        symmetry_flag = True
    if class_name_1 == 'mug' and class_name_1 == class_name_2 and handle_visibility==0:
        symmetry_flag = True

    if symmetry_flag:
        print('*'*10)
        n = 20
        theta = 2*math.pi/n

        y_rotation_matrix = np.array([[np.cos(theta), 0, np.sin(theta)],
                                      [0, 1, 0], 
                                      [-np.sin(theta), 0, np.cos(theta)]])

        max_iou = 0
        for i in range(n):
            bbox_center = np.mean(bbox_3d_1, -1, keepdims=True)
            bbox_3d_1 = y_rotation_matrix @ (bbox_3d_1 - bbox_center) + bbox_center
            max_iou = max(max_iou, asymmetric_3d_iou(bbox_3d_1, bbox_3d_2))
        return max_iou
    else:
        return asymmetric_3d_iou(bbox_3d_1, bbox_3d_2)



def compute_3d_iou_new(RT_1, RT_2, scales_1, scales_2, handle_visibility, class_name_1, class_name_2):
    '''Computes IoU overlaps between two 3d bboxes.
       bbox_3d_1, bbox_3d_1: [3, 8]
    '''
    # flatten masks
    def asymmetric_3d_iou(RT_1, RT_2, scales_1, scales_2):
        noc_cube_1 = get_3d_bbox(scales_1, 0)
        bbox_3d_1 = transform_coordinates_3d(noc_cube_1, RT_1)

        noc_cube_2 = get_3d_bbox(scales_2, 0)
        bbox_3d_2 = transform_coordinates_3d(noc_cube_2, RT_2)


        bbox_1_max = np.amax(bbox_3d_1, axis=0)
        bbox_1_min = np.amin(bbox_3d_1, axis=0)
        bbox_2_max = np.amax(bbox_3d_2, axis=0)
        bbox_2_min = np.amin(bbox_3d_2, axis=0)

        overlap_min = np.maximum(bbox_1_min, bbox_2_min)
        overlap_max = np.minimum(bbox_1_max, bbox_2_max)

        # intersections and union
        if np.amin(overlap_max - overlap_min) <0:
            intersections = 0
        else:
            intersections = np.prod(overlap_max - overlap_min)
        union = np.prod(bbox_1_max - bbox_1_min) + np.prod(bbox_2_max - bbox_2_min) - intersections
        overlaps = intersections / union
        return overlaps


    if RT_1 is None or RT_2 is None:
        return -1

    symmetry_flag = False
    if (class_name_1 in ['bottle', 'bowl', 'can'] and class_name_1 == class_name_2) or (class_name_1 == 'mug' and class_name_1 == class_name_2 and handle_visibility==0):
        print('*'*10)
    
        noc_cube_1 = get_3d_bbox(scales_1, 0)
        noc_cube_2 = get_3d_bbox(scales_2, 0)
        bbox_3d_2 = transform_coordinates_3d(noc_cube_2, RT_2)

        def y_rotation_matrix(theta):
            return np.array([[np.cos(theta), 0, np.sin(theta), 0],
                             [0, 1, 0 , 0], 
                             [-np.sin(theta), 0, np.cos(theta), 0],
                             [0, 0, 0 , 1]])

        n = 20
        max_iou = 0
        for i in range(n):
            rotated_RT_1 = RT_1@y_rotation_matrix(2*math.pi*i/float(n))
            max_iou = max(max_iou, 
                          asymmetric_3d_iou(rotated_RT_1, RT_2, scales_1, scales_2))
    else:
        max_iou = asymmetric_3d_iou(RT_1, RT_2, scales_1, scales_2)
    
    
    return max_iou
        



def compute_RT_distances(RT_1, RT_2):
    '''
    :param RT_1: [4, 4]. homogeneous affine transformation
    :param RT_2: [4, 4]. homogeneous affine transformation
    :return: theta: angle difference of R in degree, shift: l2 difference of T in centimeter
    '''
    #print(RT_1[3, :], RT_2[3, :])
    ## make sure the last row is [0, 0, 0, 1]
    if RT_1 is None or RT_2 is None:
        return -1

    try:
        assert np.array_equal(RT_1[3, :], RT_2[3, :])
        assert np.array_equal(RT_1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(RT_1[3, :], RT_2[3, :])



    R1 = RT_1[:3, :3]/np.cbrt(np.linalg.det(RT_1[:3, :3]))
    T1 = RT_1[:3, 3]

    R2 = RT_2[:3, :3]/np.cbrt(np.linalg.det(RT_2[:3, :3]))
    T2 = RT_2[:3, 3]

    R = R1 @ R2.transpose()
    theta = np.arccos((np.trace(R) - 1)/2) * 180/np.pi
    shift = np.linalg.norm(T1-T2) * 100
    # print(theta, shift)

    if theta < 5 and shift < 5:
        return 10 - theta - shift
    else:
        return -1


def compute_RT_degree_cm_symmetry(RT_1, RT_2, class_id, handle_visibility, synset_names):
    '''
    :param RT_1: [4, 4]. homogeneous affine transformation
    :param RT_2: [4, 4]. homogeneous affine transformation
    :return: theta: angle difference of R in degree, shift: l2 difference of T in centimeter


    synset_names = ['BG',  # 0
                    'bottle',  # 1
                    'bowl',  # 2
                    'camera',  # 3
                    'can',  # 4
                    'cap',  # 5
                    'phone',  # 6
                    'monitor',  # 7
                    'laptop',  # 8
                    'mug'  # 9
                    ]
    
    synset_names = ['BG',  # 0
                    'bottle',  # 1
                    'bowl',  # 2
                    'camera',  # 3
                    'can',  # 4
                    'laptop',  # 5
                    'mug'  # 6
                    ]
    '''

    ## make sure the last row is [0, 0, 0, 1]
    if RT_1 is None or RT_2 is None:
        return -1
    try:
        assert np.array_equal(RT_1[3, :], RT_2[3, :])
        assert np.array_equal(RT_1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(RT_1[3, :], RT_2[3, :])
        exit()

    R1 = RT_1[:3, :3] / np.cbrt(np.linalg.det(RT_1[:3, :3]))
    T1 = RT_1[:3, 3]

    R2 = RT_2[:3, :3] / np.cbrt(np.linalg.det(RT_2[:3, :3]))
    T2 = RT_2[:3, 3]

#     try:
#         assert np.abs(np.linalg.det(R1) - 1) < 0.01
#         assert np.abs(np.linalg.det(R2) - 1) < 0.01
#     except AssertionError:
#         print(np.linalg.det(R1), np.linalg.det(R2))

    if synset_names[class_id] in ['bottle', 'can', 'bowl']:  ## symmetric when rotating around y-axis
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    elif synset_names[class_id] == 'mug' and handle_visibility==0:  ## symmetric when rotating around y-axis
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    elif synset_names[class_id] in ['phone', 'eggbox', 'glue']:
        y_180_RT = np.diag([-1.0, 1.0, -1.0])
        R = R1 @ R2.transpose()
        R_rot = R1 @ y_180_RT @ R2.transpose()
        theta = min(np.arccos((np.trace(R) - 1) / 2),
                    np.arccos((np.trace(R_rot) - 1) / 2))
    else:
        R = R1 @ R2.transpose()
        theta = np.arccos((np.trace(R) - 1) / 2)

    theta *= 180 / np.pi
    shift = np.linalg.norm(T1 - T2) * 100
    result = np.array([theta, shift])

    return result


def compute_RT_projection_2d_symmetry(RT_1, RT_2, class_id, handle_visibility, mesh_vertices, intrinsics, synset_names, num_rotation=20):
    '''
    :param RT_1: [4, 4]. homogeneous affine transformation
    :param RT_2: [4, 4]. homogeneous affine transformation
    :param vertices: [3, N].
    :param intrinsics: [4, 4]
    :return: mean 2d projection distance in pixel

    synset_names = ['BG',  # 0
                    'bottle',  # 1
                    'bowl',  # 2
                    'camera',  # 3
                    'can',  # 4
                    'laptop',  # 5
                    'mug'  # 6
                    ]
    '''

    ## make sure the last row is [0, 0, 0, 1]
    if RT_1 is None or RT_2 is None:
        return -1
    try:
        assert np.array_equal(RT_1[3, :], RT_2[3, :])
        assert np.array_equal(RT_1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(RT_1[3, :], RT_2[3, :])
        exit()

    RT_1[:3, :3] = RT_1[:3, :3]/np.cbrt(np.linalg.det(RT_1[:3, :3]))
    R1 = RT_1[:3, :3]
    #T1 = RT_1[:3, 3]

    RT_2[:3, :3] = RT_2[:3, :3]/np.cbrt(np.linalg.det(RT_2[:3, :3]))
    R2 = RT_2[:3, :3]
    #T2 = RT_2[:3, 3]
    try:
        assert np.abs(np.linalg.det(R1) - 1) < 0.01
        assert np.abs(np.linalg.det(R2) - 1) < 0.01
    except AssertionError:
        print(np.linalg.det(R1), np.linalg.det(R2))
    
    # check the vertices are in meter unit
    vertices = np.copy(mesh_vertices)/1000
    assert np.amax(vertices) < 0.5, np.amax(vertices) 
    assert np.amax(vertices) > 0, np.amax(vertices)
    assert np.amin(vertices) < 0, np.amin(vertices)
    assert np.amin(vertices) > -0.5, np.amin(vertices)
 
    assert vertices.shape[0] == 3
    num_vertices = vertices.shape[1]
    
    coords_3d_1 = transform_coordinates_3d(vertices, RT_1)
    projected_1 = calculate_2d_projections(coords_3d_1, intrinsics)
    coords_3d_2 = transform_coordinates_3d(vertices, RT_2)
    projected_2 = calculate_2d_projections(coords_3d_2, intrinsics)
    
    # calculate reprojection 2d error
    dists = np.linalg.norm(projected_1 - projected_2, axis=1)
    assert len(dists) == num_vertices
    min_mean_dist = np.mean(dists)


    ## take care of symmetry categories

    # freely rotate around y axis
    if (synset_names[class_id] in ['bottle', 'can', 'bowl']) or (synset_names[class_id] == 'mug' and handle_visibility==0):  
        
        def y_rotation_matrix(theta):
            return np.array([[np.cos(theta), 0, np.sin(theta)],
                             [0, 1, 0], 
                             [-np.sin(theta), 0, np.cos(theta)]])
        for i in range(1, num_rotation):
            theta = 2*math.pi*i/float(num_rotation)
            coords_3d_2 = transform_coordinates_3d(y_rotation_matrix(theta)@vertices, RT_2)
            projected_2 = calculate_2d_projections(coords_3d_2, intrinsics)

            dists = np.linalg.norm(projected_1 - projected_2, axis=1)
            assert len(dists) == num_vertices
            min_mean_dist = min(min_mean_dist, np.mean(dists))

    # rotate 180 around y axis
    elif synset_names[class_id] in ['phone']:
        y_180_RT = np.diag([-1.0, 1.0, -1.0])
        
        coords_3d_2 = transform_coordinates_3d(y_180_RT@vertices, RT_2)
        projected_2 = calculate_2d_projections(coords_3d_2, intrinsics)
        dists = np.linalg.norm(projected_1 - projected_2, axis=1)
        assert len(dists) == num_vertices
        min_mean_dist = min(min_mean_dist, np.mean(dists))
        
    # rotate 180 around z axis
    elif synset_names[class_id] in ['eggbox', 'glue']:
        z_180_RT = np.diag([-1.0, -1.0, 1.0])
        coords_3d_2 = transform_coordinates_3d(z_180_RT@vertices, RT_2)
        projected_2 = calculate_2d_projections(coords_3d_2, intrinsics)
        dists = np.linalg.norm(projected_1 - projected_2, axis=1)
        assert len(dists) == num_vertices
        min_mean_dist = min(min_mean_dist, np.mean(dists))

    else: ## normal asymmetric objects        
        min_mean_dist = min_mean_dist

    
    return min_mean_dist


def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum suppression and returns indices of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indicies into ixs[1:], so add 1 to get
        # indicies into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indicies of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)]. Note that (y2, x2) is outside the box.
    deltas: [N, (dy, dx, log(dh), log(dw))]
    """
    boxes = boxes.astype(np.float32)
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= np.exp(deltas[:, 2])
    width *= np.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    return np.stack([y1, x1, y2, x2], axis=1)


def box_refinement_graph(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.log(gt_height / height)
    dw = tf.log(gt_width / width)

    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result


def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]. (y2, x2) is
    assumed to be outside the box.
    """
    box = box.astype(np.float32)
    gt_box = gt_box.astype(np.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_y = box[:, 0] + 0.5 * height
    center_x = box[:, 1] + 0.5 * width

    gt_height = gt_box[:, 2] - gt_box[:, 0]
    gt_width = gt_box[:, 3] - gt_box[:, 1]
    gt_center_y = gt_box[:, 0] + 0.5 * gt_height
    gt_center_x = gt_box[:, 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = np.log(gt_height / height)
    dw = np.log(gt_width / width)

    return np.stack([dy, dx, dh, dw], axis=1)


def get_3d_bbox(scale, shift = 0):
    """
    Input: 
        scale: [3] or scalar
        shift: [3] or scalar
    Return 
        bbox_3d: [3, N]

    """
    if hasattr(scale, "__iter__"):
        bbox_3d = np.array([[scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]]) + shift
    else:
        bbox_3d = np.array([[scale / 2, +scale / 2, scale / 2],
                  [scale / 2, +scale / 2, -scale / 2],
                  [-scale / 2, +scale / 2, scale / 2],
                  [-scale / 2, +scale / 2, -scale / 2],
                  [+scale / 2, -scale / 2, scale / 2],
                  [+scale / 2, -scale / 2, -scale / 2],
                  [-scale / 2, -scale / 2, scale / 2],
                  [-scale / 2, -scale / 2, -scale / 2]]) +shift

    bbox_3d = bbox_3d.transpose()
    return bbox_3d



def transform_coordinates_3d(coordinates, RT):
    """
    Input: 
        coordinates: [3, N]
        RT: [4, 4]
    Return 
        new_coordinates: [3, N]

    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = RT @ coordinates
    new_coordinates = new_coordinates[:3, :]/new_coordinates[3, :]
    return new_coordinates


def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Input: 
        coordinates: [3, N]
        intrinsics: [3, 3]
    Return 
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)

    return projected_coordinates

 

############################################################
#  IMAGE AUGMENTATION
############################################################

def calculate_rotation(image_size, angle):
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    return new_w, new_h, affine_mat



def rotate_image(image, new_w, new_h, affine_mat, interpolation=cv2.INTER_LINEAR):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=interpolation
    )

    return result


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

def rotate_and_crop(image, rotate_degree, interpolation):
    image_height, image_width = image.shape[0:2]


    new_w, new_h, affine_mat = calculate_rotation(image.shape[0:2][::-1], rotate_degree)
    image_rotated = rotate_image(image, new_w, new_h, affine_mat, interpolation)
    image_rotated_cropped = crop_around_center(
        image_rotated,
        *largest_rotated_rect(
            image_width,
            image_height,
            math.radians(rotate_degree)
        )
    )

    return image_rotated_cropped


def rotate_and_crop_images(image, masks, coords, rotate_degree):

    image_height, image_width = image.shape[0:2]
    new_w, new_h, affine_mat = calculate_rotation(image.shape[0:2][::-1], rotate_degree)

    image_rotated = rotate_image(image, new_w, new_h, affine_mat, cv2.INTER_LINEAR)
    mask_rotated = rotate_image(masks, new_w, new_h, affine_mat, cv2.INTER_NEAREST)
    
    rect = largest_rotated_rect(
            image_width,
            image_height,
            math.radians(rotate_degree)
        )

    image_rotated_cropped = crop_around_center(image_rotated, *rect)
    mask_rotated_cropped = crop_around_center(mask_rotated, *rect)

    image_rotated_cropped = cv2.resize(image_rotated_cropped, (image_width, image_height),interpolation=cv2.INTER_LINEAR)
    mask_rotated_cropped = cv2.resize(mask_rotated_cropped, (image_width, image_height), interpolation=cv2.INTER_NEAREST)

    if coords is not None:
        coord_rotated = rotate_image(coords, new_w, new_h, affine_mat, cv2.INTER_NEAREST)
        coord_rotated_cropped = crop_around_center(coord_rotated, *rect)
        coord_rotated_cropped = cv2.resize(coord_rotated_cropped, (image_width, image_height), interpolation=cv2.INTER_NEAREST)

        return image_rotated_cropped, mask_rotated_cropped, coord_rotated_cropped
    else:
        return image_rotated_cropped, mask_rotated_cropped


############################################################
#  Dataset
############################################################

class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:
    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...
    See COCODataset and ShapesDataset as examples.
    """
    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.
        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.d
        """
        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        #self.num_classes = len(self.class_info)
        self.num_classes = 0

        #self.class_ids = np.arange(self.num_classes)
        self.class_ids = []

        #self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.class_names = []


        #self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
        #                              for info, id in zip(self.class_info, self.class_ids)}
        self.class_from_source_map = {}

        for cls_info in self.class_info:
            source = cls_info["source"]
            if source == 'coco':
                map_key = "{}.{}".format(cls_info['source'], cls_info['id'])
                self.class_from_source_map[map_key] = self.class_names.index(class_map[cls_info["name"]])
            else:
                self.class_ids.append(self.num_classes)
                self.num_classes += 1
                self.class_names.append(cls_info["name"])

                map_key = "{}.{}".format(cls_info['source'], cls_info['id'])
                self.class_from_source_map[map_key] = self.class_ids[-1]


        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)


        # Mapping from source class and image IDs to internal IDs
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))


        '''
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)
        '''

        print(self.class_names)
        print(self.class_from_source_map)
        print(self.sources)
        #print(self.source_class_ids)



    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.
        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id] if source_class_id in self.class_from_source_map else None

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    def append_data(self, class_info, image_info):
        self.external_to_class_id = {}
        for i, c in enumerate(self.class_info):
            for ds, id in c["map"]:
                self.external_to_class_id[ds + str(id)] = i

        # Map external image IDs to internal ones.
        self.external_to_image_id = {}
        for i, info in enumerate(self.image_info):
            self.external_to_image_id[info["ds"] + str(info["id"])] = i

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's availble online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = scipy.misc.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids


def resize_image(image, min_dim=None, max_dim=None, padding=False):
    """
    Resizes an image keeping the aspect ratio.
    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim
    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        image = scipy.misc.imresize(
            image, (round(h * scale), round(w * scale)))
    # Need padding?
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding


def resize_mask(mask, scale, padding):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image, the mask, and the coordinate map are resized consistently.
    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    h, w = mask.shape[:2]
    # for instance mask
    if len(mask.shape) == 3:
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
        new_padding = padding
    # for coordinate map
    elif len(mask.shape) == 4:
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1, 1], order=0)
        new_padding = padding + [(0, 0)]
    else:
        assert False

    mask = np.pad(mask, new_padding, mode='constant', constant_values=0)

    return mask


def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to cut memory load.
    Mini-masks can then resized back to image scale using expand_masks()
    See inspect_data.ipynb notebook for more details.
    """
    # for instance mask
    if len(mask.shape)==3:
        mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
        for i in range(mask.shape[-1]):
            m = mask[:, :, i]
            y1, x1, y2, x2 = bbox[i][:4]
            m = m[y1:y2, x1:x2]*255
            m = scipy.misc.imresize(m.astype(np.uint8), mini_shape, interp='nearest')
            mini_mask[:, :, i] = np.where(m >= 128, 1, 0)

    # for coordinate map
    elif len(mask.shape)==4:
        assert mask.shape[-1] == 3 ## coordinate map

        mini_mask = np.zeros(mini_shape + mask.shape[-2:], dtype=np.float32)
        for i in range(mask.shape[-2]):
            m = mask[:, :, i, :]
            y1, x1, y2, x2 = bbox[i][:4]
            m = m[y1:y2, x1:x2, :]*255
            m = scipy.misc.imresize(m.astype(np.uint8), mini_shape+(mask.shape[-1],), interp='nearest')
            mini_mask[:, :, i, :] = m.astype(float)/255

    else:
        assert False
    return mini_mask


def expand_mask(bbox, mini_mask, image_shape):
    """Resizes mini masks back to image size. Reverses the change
    of minimize_mask().
    See inspect_data.ipynb notebook for more details.
    """
    # for instance mask
    if len(mini_mask.shape) == 3:
        mask = np.zeros(image_shape[:2] + (mini_mask.shape[-1],), dtype=bool)
        for i in range(mask.shape[-1]):
            m = mini_mask[:, :, i]
            y1, x1, y2, x2 = bbox[i][:4]
            h = y2 - y1
            w = x2 - x1
            m = scipy.misc.imresize(m.astype(float), (h, w), interp='bilinear')
            mask[y1:y2, x1:x2, i] = np.where(m >= 128, 1, 0)
    elif len(mini_mask.shape) == 4:
        assert mini_mask.shape[-1] == 3  ## coordinate map
        mask = np.zeros(image_shape[:2] + mini_mask.shape[-2:], dtype=np.float32)
        for i in range(mask.shape[-2]):
            m = mini_mask[:, :, i, :]
            y1, x1, y2, x2 = bbox[i][:4]
            h = y2 - y1
            w = x2 - x1
            m = scipy.misc.imresize(m.astype(float), (h, w, mini_mask.shape[-1]), interp='nearest')
            mask[y1:y2, x1:x2, i, :] = m

    return mask


# TODO: Build and use this function to reduce code duplication
def mold_mask(mask, config):
    pass


def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network into a format similar
    to it's original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.
    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    y1, x1, y2, x2 = bbox
    mask = scipy.misc.imresize(
        mask, (y2 - y1, x2 - x1), interp='bilinear').astype(np.float32) / 255.0
    mask = np.where(mask >= threshold, 1, 0).astype(np.uint8)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask

def unmold_coord(coord, bbox, image_shape):
    """Converts a mask generated by the neural network into a format similar
    to it's original shape.
    coord: [height, width, 3] of type float. A small, typically 28x28 mask.
    bbox: [y1, x1, y2, x2]. The box to fit the mask in.
    Returns a coordinate map with the same size as the original image.
    """
    y1, x1, y2, x2 = bbox

    #max_coord_x = np.amax(coord[:, :, 0])
    #max_coord_y = np.amax(coord[:, :, 1])
    #max_coord_z = np.amax(coord[:, :, 2])

    #print('before resize:')
    #print(max_coord_x, max_coord_y, max_coord_z)

    #coord = scipy.misc.imresize(
    #    coord, (y2 - y1, x2 - x1, 3), interp='nearest').astype(np.float32)/ 255.0
    #    #coord, (y2 - y1, x2 - x1, 3), interp='bilinear').astype(np.uint8)
    coord = cv2.resize(coord, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)

    #max_coord_x_resize = np.amax(coord[:, :, 0])
    #max_coord_y_resize = np.amax(coord[:, :, 1])
    #max_coord_z_resize = np.amax(coord[:, :, 2])

    #print('after resize:')
    #print(max_coord_x_resize, max_coord_y_resize, max_coord_z_resize)


    # Put the mask in the right location.
    full_coord= np.zeros(image_shape, dtype=np.float32)
    full_coord[y1:y2, x1:x2, :] = coord
    return full_coord

## for COCO
def annToRLE(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ann['segmentation']
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(segm['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, height, width)
    else:
        # rle
        rle = ann['segmentation']
    return rle

def annToMask(ann, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    rle = annToRLE(ann, height, width)
    m = maskUtils.decode(rle)
    return m


############################################################
#  Anchors
############################################################

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.
    Returns:
    anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)


############################################################
#  Evaluation
############################################################

def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.
    x: [rows, columns].
    """

    pre_shape = x.shape
    assert len(x.shape) == 2, x.shape
    new_x = x[~np.all(x == 0, axis=1)]
    post_shape = new_x.shape
    assert pre_shape[0] == post_shape[0]
    assert pre_shape[1] == post_shape[1]

    return new_x


def compute_matches(gt_boxes, gt_class_ids, gt_masks,
                    pred_boxes, pred_class_ids, pred_scores, pred_masks,
                    iou_threshold=0.5, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.
    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    if len(gt_class_ids)==0 or len(pred_class_ids) == 0:
        return -1 * np.ones([len(gt_class_ids)]), -1 * np.ones([len(pred_class_ids)]), None, np.zeros([0])

    pre_len = len(gt_boxes)
    gt_boxes = trim_zeros(gt_boxes)
    after_len = len(gt_boxes)
    assert pre_len == after_len
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]

    pre_len = len(pred_boxes)
    pred_boxes = trim_zeros(pred_boxes)
    after_len = len(pred_boxes)
    assert pre_len == after_len
    pred_scores = pred_scores[:pred_boxes.shape[0]]

    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    
    
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps, indices


def compute_3d_matches(gt_class_ids, gt_RTs, gt_scales, gt_handle_visibility, synset_names,
                       pred_boxes, pred_class_ids, pred_scores, pred_RTs, pred_scales,
                       iou_3d_thresholds, score_threshold=0):
    """Finds matches between prediction and ground truth instances.
    Returns:
        gt_matches: 2-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_matches: 2-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)
    indices = np.zeros(0)
    
    if num_pred:
        pred_boxes = trim_zeros(pred_boxes).copy()
        pred_scores = pred_scores[:pred_boxes.shape[0]].copy()

        # Sort predictions by score from high to low
        indices = np.argsort(pred_scores)[::-1]
        
        pred_boxes = pred_boxes[indices].copy()
        pred_class_ids = pred_class_ids[indices].copy()
        pred_scores = pred_scores[indices].copy()
        pred_scales = pred_scales[indices].copy()
        pred_RTs = pred_RTs[indices].copy()

        
    # pred_3d_bboxs = []
    # for i in range(num_pred):
    #     noc_cube = get_3d_bbox(pred_scales[i, :], 0)
    #     pred_bbox_3d = transform_coordinates_3d(noc_cube, pred_RTs[i])
    #     pred_3d_bboxs.append(pred_bbox_3d)

    # # compute 3d bbox for ground truths
    # # print('Compute gt bboxes...')
    # gt_3d_bboxs = []
    # for j in range(num_gt):
    #     noc_cube = get_3d_bbox(gt_scales[j], 0)
    #     gt_3d_bbox = transform_coordinates_3d(noc_cube, gt_RTs[j])
    #     gt_3d_bboxs.append(gt_3d_bbox)

    # Compute IoU overlaps [pred_bboxs gt_bboxs]
    #overlaps = [[0 for j in range(num_gt)] for i in range(num_pred)]
    overlaps = np.zeros((num_pred, num_gt), dtype=np.float32)
    for i in range(num_pred):
        for j in range(num_gt):
            #overlaps[i, j] = compute_3d_iou(pred_3d_bboxs[i], gt_3d_bboxs[j], gt_handle_visibility[j], 
            #    synset_names[pred_class_ids[i]], synset_names[gt_class_ids[j]])
            overlaps[i, j] = compute_3d_iou_new(pred_RTs[i], gt_RTs[j], pred_scales[i, :], gt_scales[j], gt_handle_visibility[j], synset_names[pred_class_ids[i]], synset_names[gt_class_ids[j]])

    # Loop through predictions and find matching ground truth boxes
    num_iou_3d_thres = len(iou_3d_thresholds)
    pred_matches = -1 * np.ones([num_iou_3d_thres, num_pred])
    gt_matches = -1 * np.ones([num_iou_3d_thres, num_gt])

    for s, iou_thres in enumerate(iou_3d_thresholds):
        for i in range(len(pred_boxes)):
            # Find best matching ground truth box
            # 1. Sort matches by score
            sorted_ixs = np.argsort(overlaps[i])[::-1]
            # 2. Remove low scores
            low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
            if low_score_idx.size > 0:
                sorted_ixs = sorted_ixs[:low_score_idx[0]]
            # 3. Find the match
            for j in sorted_ixs:
                # If ground truth box is already matched, go to next one
                #print('gt_match: ', gt_match[j])
                if gt_matches[s, j] > -1:
                    continue
                # If we reach IoU smaller than the threshold, end the loop
                iou = overlaps[i, j]
                #print('iou: ', iou)
                if iou < iou_thres:
                    break
                # Do we have a match?
                if not pred_class_ids[i] == gt_class_ids[j]:
                    continue

                if iou > iou_thres:
                    gt_matches[s, j] = i
                    pred_matches[s, i] = j
                    break

    return gt_matches, pred_matches, overlaps, indices


def compute_ap_from_matches_scores(pred_match, pred_scores, gt_match):
    # sort the scores from high to low
    # print(pred_match.shape, pred_scores.shape)
    assert pred_match.shape[0] == pred_scores.shape[0]

    score_indices = np.argsort(pred_scores)[::-1]
    pred_scores = pred_scores[score_indices]
    pred_match  = pred_match[score_indices]

    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    return ap


def compute_RT_overlaps(gt_class_ids, gt_RTs, gt_handle_visibility,
                        pred_class_ids, pred_RTs, 
                        synset_names):
    """Finds overlaps between prediction and ground truth instances.
    Returns:
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # print('num of gt instances: {}, num of pred instances: {}'.format(len(gt_class_ids), len(gt_class_ids)))
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)

    # Compute IoU overlaps [pred_bboxs gt_bboxs]
    #overlaps = [[0 for j in range(num_gt)] for i in range(num_pred)]
    overlaps = np.zeros((num_pred, num_gt, 2))

    for i in range(num_pred):
        for j in range(num_gt):
            overlaps[i, j, :] = compute_RT_degree_cm_symmetry(pred_RTs[i], 
                                                              gt_RTs[j], 
                                                              gt_class_ids[j], 
                                                              gt_handle_visibility[j],
                                                              synset_names)
            
    return overlaps


def compute_RT_projection_2d_overlaps(gt_class_ids, gt_RTs, gt_handle_visibility,
                                      pred_class_ids, pred_RTs, 
                                      meshes, intrinsics, synset_names):
    """Finds overlaps between prediction and ground truth instances.
    Returns:
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # print('num of gt instances: {}, num of pred instances: {}'.format(len(gt_class_ids), len(gt_class_ids)))
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)

    # Compute IoU overlaps [pred_bboxs gt_bboxs]
    #overlaps = [[0 for j in range(num_gt)] for i in range(num_pred)]
    overlaps = np.zeros((num_pred, num_gt))

    for i in range(num_pred):
        for j in range(num_gt):
            overlaps[i, j] = compute_RT_projection_2d_symmetry(pred_RTs[i], 
                                                               gt_RTs[j], 
                                                               gt_class_ids[j], 
                                                               gt_handle_visibility[j],
                                                               meshes[gt_class_ids[j]],
                                                               intrinsics,
                                                               synset_names)
            
    return overlaps


def compute_match_from_projection_2d_dist(overlaps, pred_class_ids, gt_class_ids, projection_thres_list):
    num_projection_thres = len(projection_thres_list)

    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)

    pred_matches = -1 * np.ones((num_projection_thres, num_pred))
    gt_matches = -1 * np.ones((num_projection_thres, num_gt))

    if num_pred == 0 or num_gt == 0:
        return gt_matches, pred_matches


    assert num_pred == overlaps.shape[0]
    assert num_gt == overlaps.shape[1]
    

                 
    for p, projection_thres in enumerate(projection_thres_list):
        for i in range(num_pred):
            # Find best matching ground truth box
            # 1. Sort matches by scores from low to high
            sorted_ixs = np.argsort(overlaps[i, :])
            # 2. Remove low scores
            # low_score_idx = np.where(sum_degree_shift >= 100)[0]
            # if low_score_idx.size > 0:
            #     sorted_ixs = sorted_ixs[:low_score_idx[0]]
            # 3. Find the match
            for j in sorted_ixs:
                # If ground truth box is already matched, go to next one
                #print(j, len(gt_match), len(pred_class_ids), len(gt_class_ids))
                if gt_matches[p, j] > -1 or pred_class_ids[i] != gt_class_ids[j]:
                    continue
                # If we reach IoU smaller than the threshold, end the loop
                if overlaps[i, j] > projection_thres:
                    continue

                gt_matches[p, j] = i
                pred_matches[p, i] = j
                break

    return gt_matches, pred_matches


def compute_match_from_degree_cm(overlaps, pred_class_ids, gt_class_ids, degree_thres_list, shift_thres_list):
    num_degree_thres = len(degree_thres_list)
    num_shift_thres = len(shift_thres_list)


    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)

    pred_matches = -1 * np.ones((num_degree_thres, num_shift_thres, num_pred))
    gt_matches = -1 * np.ones((num_degree_thres, num_shift_thres, num_gt))

    if num_pred == 0 or num_gt == 0:
        return gt_matches, pred_matches


    assert num_pred == overlaps.shape[0]
    assert num_gt == overlaps.shape[1]
    assert overlaps.shape[2] == 2
    

    for d, degree_thres in enumerate(degree_thres_list):                
        for s, shift_thres in enumerate(shift_thres_list):
            for i in range(num_pred):
                # Find best matching ground truth box
                # 1. Sort matches by scores from low to high
                sum_degree_shift = np.sum(overlaps[i, :, :], axis=-1)
                sorted_ixs = np.argsort(sum_degree_shift)
                # 2. Remove low scores
                # low_score_idx = np.where(sum_degree_shift >= 100)[0]
                # if low_score_idx.size > 0:
                #     sorted_ixs = sorted_ixs[:low_score_idx[0]]
                # 3. Find the match
                for j in sorted_ixs:
                    # If ground truth box is already matched, go to next one
                    #print(j, len(gt_match), len(pred_class_ids), len(gt_class_ids))
                    if gt_matches[d, s, j] > -1 or pred_class_ids[i] != gt_class_ids[j]:
                        continue
                    # If we reach IoU smaller than the threshold, end the loop
                    if overlaps[i, j, 0] > degree_thres or overlaps[i, j, 1] > shift_thres:
                        continue

                    gt_matches[d, s, j] = i
                    pred_matches[d, s, i] = j
                    break

    return gt_matches, pred_matches



def compute_degree_cm_mAP(final_results, synset_names, log_dir, degree_thresholds=[360], shift_thresholds=[100], iou_3d_thresholds=[0.1], iou_pose_thres=0.1, use_matches_for_pose=False):
    """Compute Average Precision at a set IoU threshold (default 0.5).
    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    
    num_classes = len(synset_names)
    degree_thres_list = list(degree_thresholds) + [360]
    num_degree_thres = len(degree_thres_list)

    shift_thres_list = list(shift_thresholds) + [100]
    num_shift_thres = len(shift_thres_list)

    iou_thres_list = list(iou_3d_thresholds)
    num_iou_thres = len(iou_thres_list)

    if use_matches_for_pose:
        assert iou_pose_thres in iou_thres_list

    iou_3d_aps = np.zeros((num_classes + 1, num_iou_thres))
    iou_pred_matches_all = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)]
    iou_pred_scores_all  = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)]
    iou_gt_matches_all   = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)]
    
    pose_aps = np.zeros((num_classes + 1, num_degree_thres, num_shift_thres))
    pose_pred_matches_all = [np.zeros((num_degree_thres, num_shift_thres, 0)) for _  in range(num_classes)]
    pose_gt_matches_all  = [np.zeros((num_degree_thres, num_shift_thres, 0)) for _  in range(num_classes)]
    pose_pred_scores_all = [np.zeros((num_degree_thres, num_shift_thres, 0)) for _  in range(num_classes)]

    # loop over results to gather pred matches and gt matches for iou and pose metrics
    progress = 0
    for progress, result in enumerate(final_results):
        print(progress, len(final_results))
        gt_class_ids = result['gt_class_ids'].astype(np.int32)
        gt_RTs = np.array(result['gt_RTs'])
        gt_scales = np.array(result['gt_scales'])
        gt_handle_visibility = result['gt_handle_visibility']
    
        pred_bboxes = np.array(result['pred_bboxes'])
        pred_class_ids = result['pred_class_ids']
        pred_scales = result['pred_scales']
        pred_scores = result['pred_scores']
        pred_RTs = np.array(result['pred_RTs'])
        #print(pred_bboxes.shape[0], pred_class_ids.shape[0], pred_scores.shape[0], pred_RTs.shape[0])

        if len(gt_class_ids) == 0 and len(pred_class_ids) == 0:
            continue


        for cls_id in range(1, num_classes):
            # get gt and predictions in this class
            cls_gt_class_ids = gt_class_ids[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros(0)
            cls_gt_scales = gt_scales[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros((0, 3))
            cls_gt_RTs = gt_RTs[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros((0, 4, 4))

            cls_pred_class_ids = pred_class_ids[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros(0)
            cls_pred_bboxes =  pred_bboxes[pred_class_ids==cls_id, :] if len(pred_class_ids) else np.zeros((0, 4))
            cls_pred_scores = pred_scores[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros(0)
            cls_pred_RTs = pred_RTs[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, 4, 4))
            cls_pred_scales = pred_scales[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, 3))


            # if len(cls_gt_class_ids) == 0 and len(cls_pred_class_ids) == 0:
            #     continue

            # elif len(cls_gt_class_ids) > 0 and len(cls_pred_class_ids) == 0:
            #     iou_gt_matches_all[cls_id] = np.concatenate((iou_gt_matches_all[cls_id], -1*np.ones((num_iou_thres, len(cls_gt_class_ids)))), axis=-1)
            #     if not use_matches_for_pose:
            #         pose_gt_matches_all[cls_id] = np.concatenate((pose_gt_matches_all[cls_id], -1*np.ones((num_degree_thres, num_shift_thres, len(cls_gt_class_ids)))), axis=-1)
            #     continue

            # elif len(cls_pred_class_ids)>0 and len(cls_gt_class_ids)==0:
            #     assert iou_pred_matches_all[cls_id].shape[1] == iou_pred_scores_all[cls_id].shape[1]

            #     iou_pred_matches_all[cls_id] = np.concatenate((iou_pred_matches_all[cls_id], -1*np.ones((num_iou_thres, len(cls_pred_class_ids)))),  axis=-1)
            #     cls_pred_scores_tile = np.tile(cls_pred_scores, (num_iou_thres, 1))
            #     iou_pred_scores_all[cls_id] = np.concatenate((iou_pred_scores_all[cls_id], cls_pred_scores_tile), axis=-1)

            #     assert iou_pred_matches_all[cls_id].shape[1] == iou_pred_scores_all[cls_id].shape[1]

            #     if not use_matches_for_pose:
            #         pose_pred_matches_all[cls_id] = np.concatenate((pose_pred_matches_all[cls_id], -1*np.ones((num_degree_thres, num_shift_thres, len(cls_pred_class_ids)))), axis=-1)
            #         cls_pred_scores_tile = np.tile(cls_pred_scores, (num_degree_thres, num_shift_thres, 1))
            #         pose_pred_scores_all[cls_id] = np.concatenate((pose_pred_scores_all[cls_id], cls_pred_scores_tile), axis=-1)
            #     continue



            # calculate the overlap between each gt instance and pred instance
            if synset_names[cls_id] != 'mug':
                cls_gt_handle_visibility = np.ones_like(cls_gt_class_ids)
            else:
                cls_gt_handle_visibility = gt_handle_visibility[gt_class_ids==cls_id] if len(gt_class_ids) else np.ones(0)


            iou_cls_gt_match, iou_cls_pred_match, _, iou_pred_indices = compute_3d_matches(cls_gt_class_ids, cls_gt_RTs, cls_gt_scales, cls_gt_handle_visibility, synset_names,
                                                                                           cls_pred_bboxes, cls_pred_class_ids, cls_pred_scores, cls_pred_RTs, cls_pred_scales,
                                                                                           iou_thres_list)
            if len(iou_pred_indices):
                cls_pred_class_ids = cls_pred_class_ids[iou_pred_indices]
                cls_pred_RTs = cls_pred_RTs[iou_pred_indices]
                cls_pred_scores = cls_pred_scores[iou_pred_indices]
                cls_pred_bboxes = cls_pred_bboxes[iou_pred_indices]


            iou_pred_matches_all[cls_id] = np.concatenate((iou_pred_matches_all[cls_id], iou_cls_pred_match), axis=-1)
            cls_pred_scores_tile = np.tile(cls_pred_scores, (num_iou_thres, 1))
            iou_pred_scores_all[cls_id] = np.concatenate((iou_pred_scores_all[cls_id], cls_pred_scores_tile), axis=-1)
            assert iou_pred_matches_all[cls_id].shape[1] == iou_pred_scores_all[cls_id].shape[1]
            iou_gt_matches_all[cls_id] = np.concatenate((iou_gt_matches_all[cls_id], iou_cls_gt_match), axis=-1)

            if use_matches_for_pose:
                thres_ind = list(iou_thres_list).index(iou_pose_thres)

                iou_thres_pred_match = iou_cls_pred_match[thres_ind, :]

                
                cls_pred_class_ids = cls_pred_class_ids[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros(0)
                cls_pred_RTs = cls_pred_RTs[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros((0, 4, 4))
                cls_pred_scores = cls_pred_scores[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros(0)
                cls_pred_bboxes = cls_pred_bboxes[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros((0, 4))


                iou_thres_gt_match = iou_cls_gt_match[thres_ind, :]
                cls_gt_class_ids = cls_gt_class_ids[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros(0)
                cls_gt_RTs = cls_gt_RTs[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros((0, 4, 4))
                cls_gt_handle_visibility = cls_gt_handle_visibility[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros(0)



            RT_overlaps = compute_RT_overlaps(cls_gt_class_ids, cls_gt_RTs, cls_gt_handle_visibility, 
                                              cls_pred_class_ids, cls_pred_RTs,
                                              synset_names)


            pose_cls_gt_match, pose_cls_pred_match = compute_match_from_degree_cm(RT_overlaps, 
                                                                                  cls_pred_class_ids, 
                                                                                  cls_gt_class_ids, 
                                                                                  degree_thres_list, 
                                                                                  shift_thres_list)
            

            pose_pred_matches_all[cls_id] = np.concatenate((pose_pred_matches_all[cls_id], pose_cls_pred_match), axis=-1)
            
            cls_pred_scores_tile = np.tile(cls_pred_scores, (num_degree_thres, num_shift_thres, 1))
            pose_pred_scores_all[cls_id]  = np.concatenate((pose_pred_scores_all[cls_id], cls_pred_scores_tile), axis=-1)
            assert pose_pred_scores_all[cls_id].shape[2] == pose_pred_matches_all[cls_id].shape[2], '{} vs. {}'.format(pose_pred_scores_all[cls_id].shape, pose_pred_matches_all[cls_id].shape)
            pose_gt_matches_all[cls_id] = np.concatenate((pose_gt_matches_all[cls_id], pose_cls_gt_match), axis=-1)

    
    
    # draw iou 3d AP vs. iou thresholds
    fig_iou = plt.figure()
    ax_iou = plt.subplot(111)
    plt.ylabel('AP')
    plt.ylim((0, 1))
    plt.xlabel('3D IoU thresholds')
    iou_output_path = os.path.join(log_dir, 'IoU_3D_AP_{}-{}.png'.format(iou_thres_list[0], iou_thres_list[-1]))
    iou_dict_pkl_path = os.path.join(log_dir, 'IoU_3D_AP_{}-{}.pkl'.format(iou_thres_list[0], iou_thres_list[-1]))

    iou_dict = {}
    iou_dict['thres_list'] = iou_thres_list
    for cls_id in range(1, num_classes):
        class_name = synset_names[cls_id]
        print(class_name)
        for s, iou_thres in enumerate(iou_thres_list):
            iou_3d_aps[cls_id, s] = compute_ap_from_matches_scores(iou_pred_matches_all[cls_id][s, :],
                                                                   iou_pred_scores_all[cls_id][s, :],
                                                                   iou_gt_matches_all[cls_id][s, :])    
        ax_iou.plot(iou_thres_list, iou_3d_aps[cls_id, :], label=class_name)
        
    iou_3d_aps[-1, :] = np.mean(iou_3d_aps[1:-1, :], axis=0)
    ax_iou.plot(iou_thres_list, iou_3d_aps[-1, :], label='mean')
    ax_iou.legend()
    fig_iou.savefig(iou_output_path)
    plt.close(fig_iou)

    iou_dict['aps'] = iou_3d_aps
    with open(iou_dict_pkl_path, 'wb') as f:
        cPickle.dump(iou_dict, f)
    

    # draw pose AP vs. thresholds
    if use_matches_for_pose:
        prefix='Pose_Only_'
    else:
        prefix='Pose_Detection_'


    pose_dict_pkl_path = os.path.join(log_dir, prefix+'AP_{}-{}degree_{}-{}cm.pkl'.format(degree_thres_list[0], degree_thres_list[-2], 
                                                                                          shift_thres_list[0], shift_thres_list[-2]))
    pose_dict = {}
    pose_dict['degree_thres'] = degree_thres_list
    pose_dict['shift_thres_list'] = shift_thres_list

    for i, degree_thres in enumerate(degree_thres_list):                
        for j, shift_thres in enumerate(shift_thres_list):
            # print(i, j)
            for cls_id in range(1, num_classes):
                cls_pose_pred_matches_all = pose_pred_matches_all[cls_id][i, j, :]
                cls_pose_gt_matches_all = pose_gt_matches_all[cls_id][i, j, :]
                cls_pose_pred_scores_all = pose_pred_scores_all[cls_id][i, j, :]

                pose_aps[cls_id, i, j] = compute_ap_from_matches_scores(cls_pose_pred_matches_all, 
                                                                        cls_pose_pred_scores_all, 
                                                                        cls_pose_gt_matches_all)

            pose_aps[-1, i, j] = np.mean(pose_aps[1:-1, i, j])
    
    pose_dict['aps'] = pose_aps
    with open(pose_dict_pkl_path, 'wb') as f:
        cPickle.dump(pose_dict, f)


    for cls_id in range(1, num_classes):
        class_name = synset_names[cls_id]
        print(class_name)
        # print(np.amin(aps[i, :, :]), np.amax(aps[i, :, :]))
    
        #ap_image = cv2.resize(pose_aps[cls_id, :, :]*255, (320, 320), interpolation = cv2.INTER_LINEAR)
        fig_iou = plt.figure()
        ax_iou = plt.subplot(111)
        plt.ylabel('Rotation thresholds/degree')
        plt.ylim((degree_thres_list[0], degree_thres_list[-2]))
        plt.xlabel('translation/cm')
        plt.xlim((shift_thres_list[0], shift_thres_list[-2]))
        plt.imshow(pose_aps[cls_id, :-1, :-1], cmap='jet', interpolation='bilinear')

        output_path = os.path.join(log_dir, prefix+'AP_{}_{}-{}degree_{}-{}cm.png'.format(class_name, 
                                                                                   degree_thres_list[0], degree_thres_list[-2], 
                                                                                   shift_thres_list[0], shift_thres_list[-2]))
        plt.colorbar()
        plt.savefig(output_path)
        plt.close(fig_iou)        
    
    #ap_mean_image = cv2.resize(pose_aps[-1, :, :]*255, (320, 320), interpolation = cv2.INTER_LINEAR) 
    
    fig_pose = plt.figure()
    ax_pose = plt.subplot(111)
    plt.ylabel('Rotation thresholds/degree')
    plt.ylim((degree_thres_list[0], degree_thres_list[-2]))
    plt.xlabel('translation/cm')
    plt.xlim((shift_thres_list[0], shift_thres_list[-2]))
    plt.imshow(pose_aps[-1, :-1, :-1], cmap='jet', interpolation='bilinear')
    output_path = os.path.join(log_dir, prefix+'mAP_{}-{}degree_{}-{}cm.png'.format(degree_thres_list[0], degree_thres_list[-2], 
                                                                             shift_thres_list[0], shift_thres_list[-2]))
    plt.colorbar()
    plt.savefig(output_path)
    plt.close(fig_pose)

    
    fig_rot = plt.figure()
    ax_rot = plt.subplot(111)
    plt.ylabel('AP')
    plt.ylim((0, 1.05))
    plt.xlabel('translation/cm')
    for cls_id in range(1, num_classes):
        class_name = synset_names[cls_id]
        print(class_name)
        ax_rot.plot(shift_thres_list[:-1], pose_aps[cls_id, -1, :-1], label=class_name)
    
    ax_rot.plot(shift_thres_list[:-1], pose_aps[-1, -1, :-1], label='mean')
    output_path = os.path.join(log_dir, prefix+'mAP_{}-{}cm.png'.format(shift_thres_list[0], shift_thres_list[-2]))
    ax_rot.legend()
    fig_rot.savefig(output_path)
    plt.close(fig_rot)

    fig_trans = plt.figure()
    ax_trans = plt.subplot(111)
    plt.ylabel('AP')
    plt.ylim((0, 1.05))

    plt.xlabel('Rotation/degree')
    for cls_id in range(1, num_classes):
        class_name = synset_names[cls_id]
        print(class_name)
        ax_trans.plot(degree_thres_list[:-1], pose_aps[cls_id, :-1, -1], label=class_name)

    ax_trans.plot(degree_thres_list[:-1], pose_aps[-1, :-1, -1], label='mean')
    output_path = os.path.join(log_dir, prefix+'mAP_{}-{}degree.png'.format(degree_thres_list[0], degree_thres_list[-2]))
    
    ax_trans.legend()
    fig_trans.savefig(output_path)
    plt.close(fig_trans)

    iou_aps = iou_3d_aps
    print('3D IoU at 25: {:.1f}'.format(iou_aps[-1, iou_thres_list.index(0.25)] * 100))
    print('3D IoU at 50: {:.1f}'.format(iou_aps[-1, iou_thres_list.index(0.5)] * 100))
    print('5 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_thres_list.index(5),shift_thres_list.index(5)] * 100))
    print('5 degree, 100cm: {:.1f}'.format(pose_aps[-1, degree_thres_list.index(5),shift_thres_list.index(100)] * 100))
    print('10 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_thres_list.index(10),shift_thres_list.index(5)] * 100))
    print('10 degree, 10cm: {:.1f}'.format(pose_aps[-1, degree_thres_list.index(10),shift_thres_list.index(10)] * 100))
    print('15 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_thres_list.index(15),shift_thres_list.index(5)] * 100))
    print('15 degree, 10cm: {:.1f}'.format(pose_aps[-1, degree_thres_list.index(15),shift_thres_list.index(10)] * 100))


    return iou_3d_aps, pose_aps


def compute_2D_projection_mAP(final_results, synset_names, log_dir, meshes, intrinsics, iou_3d_thresholds=[0.1], iou_pose_thres=0.1, projection_2d_thresholds=[5], use_matches_for_pose=False):
    """Compute Average Precision at a set IoU threshold (default 0.5).
    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    
    num_classes = len(synset_names)
    
    iou_thres_list = list(iou_3d_thresholds)
    num_iou_thres = len(iou_thres_list)

    projection_thres_list = list(projection_2d_thresholds)
    num_projection_thres = len(projection_thres_list)

    if use_matches_for_pose:
        assert iou_pose_thres in iou_thres_list

    iou_3d_aps = np.zeros((num_classes + 1, num_iou_thres))
    iou_pred_matches_all = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)]
    iou_pred_scores_all  = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)]
    iou_gt_matches_all   = [np.zeros((num_iou_thres, 0)) for _ in range(num_classes)]
    
    
    projection_2d_aps = np.zeros((num_classes + 1, num_projection_thres))
    projection_pred_matches_all = [np.zeros((num_projection_thres, 0)) for _ in range(num_classes)]
    projection_pred_scores_all  = [np.zeros((num_projection_thres, 0)) for _ in range(num_classes)]
    projection_gt_matches_all   = [np.zeros((num_projection_thres, 0)) for _ in range(num_classes)]


    # loop over results to gather pred matches and gt matches for iou and pose metrics
    progress = 0
    for progress, result in enumerate(final_results):
        print(progress, len(final_results))
        gt_class_ids = result['gt_class_ids'].astype(np.int32)
        gt_RTs = np.array(result['gt_RTs'])
        gt_scales = np.array(result['gt_scales'])
        gt_handle_visibility = result['gt_handle_visibility']
    
        pred_bboxes = np.array(result['pred_bboxes'])
        pred_class_ids = result['pred_class_ids']
        pred_scales = result['pred_scales']
        pred_scores = result['pred_scores']
        pred_RTs = np.array(result['pred_RTs'])
        #print(pred_bboxes.shape[0], pred_class_ids.shape[0], pred_scores.shape[0], pred_RTs.shape[0])

        if len(gt_class_ids) == 0 and len(pred_class_ids) == 0:
            continue


        for cls_id in range(1, num_classes):
            # get gt and predictions in this class
            cls_gt_class_ids = gt_class_ids[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros(0)
            cls_gt_scales = gt_scales[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros((0, 3))
            cls_gt_RTs = gt_RTs[gt_class_ids==cls_id] if len(gt_class_ids) else np.zeros((0, 4, 4))

            cls_pred_class_ids = pred_class_ids[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros(0)
            cls_pred_bboxes =  pred_bboxes[pred_class_ids==cls_id, :] if len(pred_class_ids) else np.zeros((0, 4))
            cls_pred_scores = pred_scores[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros(0)
            cls_pred_RTs = pred_RTs[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, 4, 4))
            cls_pred_scales = pred_scales[pred_class_ids==cls_id] if len(pred_class_ids) else np.zeros((0, 3))


            # calculate the overlap between each gt instance and pred instance
            if synset_names[cls_id] != 'mug':
                cls_gt_handle_visibility = np.ones_like(cls_gt_class_ids)
            else:
                cls_gt_handle_visibility = gt_handle_visibility[gt_class_ids==cls_id] if len(gt_class_ids) else np.ones(0)


            iou_cls_gt_match, iou_cls_pred_match, _, iou_pred_indices = compute_3d_matches(cls_gt_class_ids, cls_gt_RTs, cls_gt_scales, cls_gt_handle_visibility, synset_names,
                                                                                           cls_pred_bboxes, cls_pred_class_ids, cls_pred_scores, cls_pred_RTs, cls_pred_scales,
                                                                                           iou_thres_list)
            if len(iou_pred_indices):
                cls_pred_class_ids = cls_pred_class_ids[iou_pred_indices]
                cls_pred_RTs = cls_pred_RTs[iou_pred_indices]
                cls_pred_scores = cls_pred_scores[iou_pred_indices]
                cls_pred_bboxes = cls_pred_bboxes[iou_pred_indices]


            iou_pred_matches_all[cls_id] = np.concatenate((iou_pred_matches_all[cls_id], iou_cls_pred_match), axis=-1)
            cls_pred_scores_tile = np.tile(cls_pred_scores, (num_iou_thres, 1))
            iou_pred_scores_all[cls_id] = np.concatenate((iou_pred_scores_all[cls_id], cls_pred_scores_tile), axis=-1)
            assert iou_pred_matches_all[cls_id].shape[1] == iou_pred_scores_all[cls_id].shape[1]
            iou_gt_matches_all[cls_id] = np.concatenate((iou_gt_matches_all[cls_id], iou_cls_gt_match), axis=-1)

            if use_matches_for_pose:
                thres_ind = list(iou_thres_list).index(iou_pose_thres)

                iou_thres_pred_match = iou_cls_pred_match[thres_ind, :]

                
                cls_pred_class_ids = cls_pred_class_ids[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros(0)
                cls_pred_RTs = cls_pred_RTs[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros((0, 4, 4))
                cls_pred_scores = cls_pred_scores[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros(0)
                cls_pred_bboxes = cls_pred_bboxes[iou_thres_pred_match > -1] if len(iou_thres_pred_match) > 0 else np.zeros((0, 4))


                iou_thres_gt_match = iou_cls_gt_match[thres_ind, :]
                cls_gt_class_ids = cls_gt_class_ids[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros(0)
                cls_gt_RTs = cls_gt_RTs[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros((0, 4, 4))
                cls_gt_handle_visibility = cls_gt_handle_visibility[iou_thres_gt_match > -1] if len(iou_thres_gt_match) > 0 else np.zeros(0)



            RT_projection_2d_overlaps = compute_RT_projection_2d_overlaps(cls_gt_class_ids, cls_gt_RTs, cls_gt_handle_visibility,
                                                                          cls_pred_class_ids, cls_pred_RTs,
                                                                          meshes, intrinsics, synset_names)


            projection_cls_gt_match, projection_cls_pred_match = compute_match_from_projection_2d_dist(RT_projection_2d_overlaps, 
                                                                                                       cls_pred_class_ids, 
                                                                                                       cls_gt_class_ids, 
                                                                                                       projection_thres_list)
                            

            projection_pred_matches_all[cls_id] = np.concatenate((projection_pred_matches_all[cls_id], projection_cls_pred_match), axis=-1)
            
            cls_pred_scores_tile = np.tile(cls_pred_scores, (num_projection_thres, 1))
            projection_pred_scores_all[cls_id]  = np.concatenate((projection_pred_scores_all[cls_id], cls_pred_scores_tile), axis=-1)
            assert projection_pred_scores_all[cls_id].shape[1] == projection_pred_matches_all[cls_id].shape[1], '{} vs. {}'.format(projection_pred_scores_all[cls_id].shape, projection_pred_matches_all[cls_id].shape)
            projection_gt_matches_all[cls_id] = np.concatenate((projection_gt_matches_all[cls_id], projection_cls_gt_match), axis=-1)

    
    
    # draw iou 3d AP vs. iou thresholds
    fig_iou = plt.figure()
    ax_iou = plt.subplot(111)
    plt.ylabel('AP')
    plt.ylim((0, 1))
    plt.xlabel('3D IoU thresholds')
    iou_output_path = os.path.join(log_dir, 'IoU_3D_AP_{}-{}.png'.format(iou_thres_list[0], iou_thres_list[-1]))
    iou_dict_pkl_path = os.path.join(log_dir, 'IoU_3D_AP_{}-{}.pkl'.format(iou_thres_list[0], iou_thres_list[-1]))

    iou_dict = {}
    iou_dict['thres_list'] = iou_thres_list
    for cls_id in range(1, num_classes):
        class_name = synset_names[cls_id]
        print(class_name)
        for s, iou_thres in enumerate(iou_thres_list):
            iou_3d_aps[cls_id, s] = compute_ap_from_matches_scores(iou_pred_matches_all[cls_id][s, :],
                                                                   iou_pred_scores_all[cls_id][s, :],
                                                                   iou_gt_matches_all[cls_id][s, :])    
        ax_iou.plot(iou_thres_list, iou_3d_aps[cls_id, :], label=class_name)
        
    iou_3d_aps[-1, :] = np.mean(iou_3d_aps[1:-1, :], axis=0)
    ax_iou.plot(iou_thres_list, iou_3d_aps[-1, :], label='mean')
    ax_iou.legend()
    fig_iou.savefig(iou_output_path)
    plt.close(fig_iou)

    iou_dict['aps'] = iou_3d_aps
    with open(iou_dict_pkl_path, 'wb') as f:
        cPickle.dump(iou_dict, f)
    

    # draw pose AP vs. thresholds
    if use_matches_for_pose:
        prefix='Pose_Only_'
    else:
        prefix='Pose_Detection_'


    LINE_WIDTH=3.0
    AXES_TITLE_SIZE=28
    XTICK_SIZE=28
    YTICK_SIZE=28
    LEGEND_SIZE=24
    TITLE_SIZE=32

    # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=TITLE_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=AXES_TITLE_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=XTICK_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=YTICK_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=TITLE_SIZE)  # fontsize of the figure title

    plt.rcParams['font.family'] = 'Bahnschrift'


    projection_dict_pkl_path = os.path.join(log_dir, prefix+'Projection_2D_AP_{}-{}pixel.pkl'.format(projection_thres_list[0], projection_thres_list[-1]))
    pose_dict = {}
    pose_dict['projection_thres_list'] = projection_thres_list

    for i, projection_thres in enumerate(projection_thres_list):                
        for cls_id in range(1, num_classes):
            projection_2d_aps[cls_id, i] = compute_ap_from_matches_scores(projection_pred_matches_all[cls_id][i, :], 
                                                                             projection_pred_scores_all[cls_id][i, :], 
                                                                             projection_gt_matches_all[cls_id][i, :])


        projection_2d_aps[-1, i] = np.mean(projection_2d_aps[1:-1, i])
    
    pose_dict['aps'] = projection_2d_aps
    with open(projection_dict_pkl_path, 'wb') as f:
        cPickle.dump(pose_dict, f)

    plot_data_dir = 'data/linemod/plots'
    f, axarr = plt.subplots(2, 4, figsize=(20, 8), dpi=50)
    for i, cls_id in enumerate([1, 3, 4, 5, 6, 7, 8, 9]):
        
        class_name = synset_names[cls_id]

        axarr[i].set_title(class_name)
        axarr[i].set(xlabel='Pixel threshold', ylabel='AP %')
        axarr[i].set_xlim([0, 25])
        axarr[i].set_ylim([0, 101])
        axarr[i].plot(projection_thres_list, projection_2d_aps[cls_id, :]*100, label='Ours', linewidth=LINE_WIDTH)

        others_data_dir = plot_data_dir + '/' + class_name
        curve_data = glob.glob(others_data_dir, '*.txt')
        for curve_path in curve_data:
            curve_name = os.path.basename(curve_path)[:-4]
            if curve_name == 'bb8':
                curve_name = 'BB8'
            elif curve_name == 'posecnn':
                curve_name = 'PoseCNN'
            elif curve_name == 'tekin':
                curve_name = 'Tekin et al.'
            else:
                assert False, curve_name
            curve = np.loadtxt(curve_path)
            axarr[i].plot(curve[:, 0], curve[:, 1], label=curve_name, linewidth=LINE_WIDTH)

        axarr[i].legend(loc='lower right')



    for ax in axarr.flat:
        ax.label_outer()
        ax.set(ylim=(0, 101))
        ax.grid(True)       


    print(output_dir)
    plt.savefig(os.path.join(output_dir, 'result.png'))
    print('saved.')

    fig_pose = plt.figure()
    ax_pose = plt.subplot(111)
    plt.ylabel('AP')
    plt.ylim((0, 1.02))
    plt.xlim((projection_thres_list[0], projection_thres_list[-1]))
    plt.xlabel('Reprojection error thresholds/pixel')
    
    for cls_id in range(1, num_classes):
        class_name = synset_names[cls_id]
        print(class_name)
        ax_pose.plot(projection_thres_list, projection_2d_aps[cls_id, :], label=class_name)
    ax_pose.plot(projection_thres_list, projection_2d_aps[-1, :], label='mean')
    
    output_path = os.path.join(log_dir, prefix+'Projection_2D_AP_{}-{}pixel.png'.format(projection_thres_list[0], projection_thres_list[-1]))
    fig_pose.legend()
    fig_pose.savefig(output_path)
    plt.close(fig_pose)


    print('3D IoU at 25: {:.1f}'.format(iou_3d_aps[-1, iou_thres_list.index(0.25)] * 100))
    print('3D IoU at 50: {:.1f}'.format(iou_3d_aps[-1, iou_thres_list.index(0.5)] * 100))
    print('2D Projection at 2.5 pixel: {:.1f}'.format(projection_2d_aps[-1, projection_thres_list.index(2.5)] * 100))
    print('2D Projection at 5 pixel: {:.1f}'.format(projection_2d_aps[-1, projection_thres_list.index(5)] * 100))
    print('2D Projection at 10 pixel: {:.1f}'.format(projection_2d_aps[-1, projection_thres_list.index(10)] * 100))
    print('2D Projection at 15 pixel: {:.1f}'.format(projection_2d_aps[-1, projection_thres_list.index(15)] * 100))
    print('2D Projection at 20 pixel: {:.1f}'.format(projection_2d_aps[-1, projection_thres_list.index(20)] * 100))


    return iou_3d_aps, projection_2d_aps


def compute_coords_aps(final_results, synset_names, iou_thresholds, coord_thresholds):
    """Compute Average Precision at a set IoU threshold (default 0.5).
    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    
    num_classes = len(synset_names)
    num_iou_thres = len(iou_thresholds)
    num_coord_thres = len(coord_thresholds)
    
    mean_coord_dist_cls = {}
    
#     pred_matches_all = {}
#     pred_scores_all = {}
#     gt_matches_all = {}

    
    for cls_id in range(1, num_classes):
        mean_coord_dist_cls[cls_id] = []
#         pred_matches_all[cls_id] = [[[] for _ in range(num_shift_thres)] for _ in range(num_degree_thres)]
#         gt_matches_all[cls_id] = [[[] for _ in range(num_shift_thres)] for _ in range(num_degree_thres)]
#         pred_scores_all[cls_id] = [[[] for _ in range(num_shift_thres)] for _ in range(num_degree_thres)]

    progress = 0
    for progress, result in enumerate(final_results):
        print(progress, len(final_results))
        
        gt_class_ids = result['gt_class_ids'].astype(np.int32)
        gt_bboxes = np.array(result['gt_bboxes'])
        gt_masks = np.array(result['gt_masks'])
        gt_coords = np.array(result['gt_coords'])
        
        #print(gt_class_ids.shape, gt_bboxes.shape, gt_masks.shape, gt_coords.shape)
        #gt_scales = result['gt_scale']
        
        pred_class_ids = result['pred_class_ids'].astype(np.int32)
        pred_bboxes = np.array(result['pred_bboxes'])
        pred_masks = np.array(result['pred_masks'])
        pred_coords = np.array(result['pred_coords'])
        pred_scores = result['pred_scores']
        #print(pred_class_ids.shape, pred_bboxes.shape, pred_masks.shape, pred_coords.shape)
        

        
        
        for cls_id in range(1, num_classes):
            # get gt and predictions in this class
            if(len(gt_class_ids)):
                cls_gt_class_ids = gt_class_ids[gt_class_ids==cls_id] 
                cls_gt_bboxes = gt_bboxes[gt_class_ids==cls_id] 
                cls_gt_masks = gt_masks[..., gt_class_ids==cls_id]
                cls_gt_coords = gt_coords[..., gt_class_ids==cls_id, :]
            else:
                cls_gt_class_ids = []
                cls_gt_bboxes = []
                cls_gt_masks = []
                cls_gt_coords = []

            if(len(pred_class_ids)):
                cls_pred_class_ids = pred_class_ids[pred_class_ids==cls_id] 
                cls_pred_bboxes =  pred_bboxes[pred_class_ids==cls_id] 
                cls_pred_scores = pred_scores[pred_class_ids==cls_id]
                cls_pred_masks = pred_masks[:, :, pred_class_ids==cls_id]
                cls_pred_coords = pred_coords[:, :, pred_class_ids==cls_id, :]
            else:
                cls_pred_class_ids = []
                cls_pred_bboxes = []
                cls_pred_scores = []
                cls_pred_masks = []
                cls_pred_coords = []
            
            # calculate the overlap between each gt instance and pred instance
            
            gt_match, pred_match, overlaps, pred_indices = compute_matches(cls_gt_bboxes, cls_gt_class_ids, cls_gt_masks,
                                                             cls_pred_bboxes, cls_pred_class_ids, cls_pred_scores, 
                                                             cls_pred_masks, 0.5)
            
            if len(gt_match) and len(pred_match):
            
                cls_pred_masks_sorted = cls_pred_masks[..., pred_indices]
                cls_pred_coords_sorted = cls_pred_coords[..., pred_indices, :]

                for i in range(len(pred_match)):
                    if pred_match[i] > -1:
                        j = int(pred_match[i])
                        mean_coord_dist = compute_mean_l1_coord_diff(cls_pred_masks_sorted[..., i], 
                                                                     cls_gt_masks[..., j],
                                                                     cls_pred_coords_sorted[..., i, :], 
                                                                     cls_gt_coords[..., j, :],
                                                                     synset_names,
                                                                     cls_id)
                            #print(mean_coord_dist)

                        mean_coord_dist_cls[cls_id].append(mean_coord_dist)
                            #print(mean_coord_dist_cls[cls_id])

            

    for cls_id in range(1, num_classes):
        mean_coord_dist_cls[cls_id] = np.array(mean_coord_dist_cls[cls_id])
        print('mean coord dist of {} class: {}'.format(synset_names[cls_id], np.mean(mean_coord_dist_cls[cls_id])))


def compute_3d_matches_degree_cm(gt_class_ids, gt_RTs, gt_scales, 
                                 pred_boxes, pred_class_ids, pred_scores, pred_RTs, 
                                 synset_names,
                                 thres_shift = 5, 
                                 thres_degree=5, 
                                 bound_threshold=5):
    """Finds matches between prediction and ground truth instances.
    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    if len(pred_boxes):
        pre_len = len(pred_boxes)
        pred_boxes = trim_zeros(pred_boxes)
        after_len = len(pred_boxes)
        assert pre_len == after_len
        pred_scores = pred_scores[:pred_boxes.shape[0]]
        # Sort predictions by score from high to low
        indices = np.argsort(pred_scores)[::-1]
        pred_boxes = pred_boxes[indices]
        pred_class_ids = pred_class_ids[indices]
        pred_scores = pred_scores[indices]

    # print('num of gt instances: {}, num of pred instances: {}'.format(len(gt_class_ids), len(gt_class_ids)))
    num_pred = len(pred_boxes)
    num_gt = len(gt_class_ids)

    # Compute IoU overlaps [pred_bboxs gt_bboxs]
    #overlaps = [[0 for j in range(num_gt)] for i in range(num_pred)]
    overlaps = np.zeros((num_pred, num_gt), dtype=np.float32)

    for i in range(num_pred):
        for j in range(num_gt):
            overlaps[i, j] = compute_RT_upper_bound_symmetry(pred_RTs[i], 
                                                             gt_RTs[j], 
                                                             gt_class_ids[j], 
                                                             synset_names,
                                                             thres_shift=thres_shift, 
                                                             thres_theta=thres_degree)

    # Loop through predictions and find matching ground truth boxes
    # multi-level: 5, 10, 15, 20
    pred_match = -1 * np.ones(num_pred)
    gt_match = -1 * np.ones(num_gt)

    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by bound from low to high
        sorted_ixs = np.argsort(overlaps[i])
        # 2. Remove low scores
        # low_score_idx = np.where(overlaps[i, sorted_ixs] >= bound_threshold)[0]
        # if low_score_idx.size > 0:
        #     sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1 or pred_class_ids[i] != gt_class_ids[j]:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            min_bound  = overlaps[i, j]
            if min_bound > 1:
                break
            elif min_bound ==1.0:
                gt_match[j] = i
                pred_match[i] = j
                break
            else:
                assert False, 'Invalid overlaps {}'.format(min_bound)

    return gt_match, pred_match, pred_class_ids, pred_scores


def compute_ap(gt_boxes, gt_class_ids, gt_masks,
               pred_boxes, pred_class_ids, pred_scores, pred_masks,
               iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).
    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Get matches and overlaps
    gt_match, pred_match, overlaps = compute_matches(
        gt_boxes, gt_class_ids, gt_masks,
        pred_boxes, pred_class_ids, pred_scores, pred_masks,
        iou_threshold)

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps



def compute_ap_range(gt_box, gt_class_id, gt_mask,
                     pred_box, pred_class_id, pred_score, pred_mask,
                     iou_thresholds=None, verbose=1):
    """Compute AP over a range or IoU thresholds. Default range is 0.5-0.95."""
    # Default is 0.5 to 0.95 with increments of 0.05
    iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)

    # Compute AP over range of IoU thresholds
    AP = []
    for iou_threshold in iou_thresholds:
        ap, precisions, recalls, overlaps = \
            compute_ap(gt_box, gt_class_id, gt_mask,
                       pred_box, pred_class_id, pred_score, pred_mask,
                       iou_threshold=iou_threshold)
        if verbose:
            print("AP @{:.2f}:\t {:.3f}".format(iou_threshold, ap))
        AP.append(ap)
    AP = np.array(AP).mean()
    if verbose:
        print("AP @{:.2f}-{:.2f}:\t {:.3f}".format(
            iou_thresholds[0], iou_thresholds[-1], AP))
    return AP


def compute_recall(pred_boxes, gt_boxes, iou):
    """Compute the recall at the given IoU threshold. It's an indication
    of how many GT boxes were found by the given prediction boxes.
    pred_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    gt_boxes: [N, (y1, x1, y2, x2)] in image coordinates
    """
    # Measure overlaps
    overlaps = compute_overlaps(pred_boxes, gt_boxes)
    iou_max = np.max(overlaps, axis=1)
    iou_argmax = np.argmax(overlaps, axis=1)
    positive_ids = np.where(iou_max >= iou)[0]
    matched_gt_boxes = iou_argmax[positive_ids]

    recall = len(set(matched_gt_boxes)) / gt_boxes.shape[0]
    return recall, positive_ids


# ## Batch Slicing
# Some custom layers support a batch size of 1 only, and require a lot of work
# to support batches greater than 1. This function slices an input tensor
# across the batch dimension and feeds batches of size 1. Effectively,
# an easy way to support batches > 1 quickly with little code modification.
# In the long run, it's more efficient to modify the code to support large
# batches and getting rid of this function. Consider this a temporary solution
def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.
    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result


############################################################
#  Miscellaneous
############################################################
def load_mesh(mesh_path, is_save=False, is_normalized=False, is_flipped=False):
    with open(mesh_path, 'r') as f:
        lines = f.readlines()

    # mesh = []
    # for line in lines:
    #     if line[:2] == 'v ':
    #         vertex = line[2:].split(' ')
    #         vertex = vertex[:3]
    #         mesh += [[float(x) for x in vertex]]
    # mesh = np.array(mesh, dtype=np.float32)

    vertices = []
    faces = []
    for l in lines:
        l = l.strip()
        words = l.split(' ')
        if words[0] == 'v':
            vertices.append([float(words[1]), float(words[2]), float(words[3])])
        if words[0] == 'f':
            face_words = [x.split('/')[0] for x in words]
            faces.append([int(face_words[1])-1, int(face_words[2])-1, int(face_words[3])-1])


    vertices = np.array(vertices, dtype=np.float64)
    # flip mesh to unity rendering
    if is_flipped:
        vertices[:, 2] = -vertices[:, 2] 
    faces = np.array(faces, dtype=np.int32)
    
    if is_normalized:
        maxs = np.amax(vertices, axis=0)
        mins = np.amin(vertices, axis=0)
        diffs = maxs - mins
        assert diffs.shape[0] == 3
        vertices = vertices/np.linalg.norm(diffs)
    
    if is_save:
        np.savetxt(mesh_path.replace('.obj', '_vertices.txt'), X = vertices)

    return vertices, faces


def draw(img, imgpts, axes, color):
    imgpts = np.int32(imgpts).reshape(-1, 2)


    # draw ground layer in darker color
    color_ground = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    for i, j in zip([4, 5, 6, 7],[5, 7, 4, 6]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_ground, 3)


    # draw pillars in blue color
    color_pillar = (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6))
    for i, j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_pillar, 3)

    
    # finally, draw top layer in color
    for i, j in zip([0, 1, 2, 3],[1, 3, 0, 2]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, 3)


    # draw axes
    img = cv2.line(img, tuple(axes[0]), tuple(axes[1]), (0, 0, 255), 3)
    img = cv2.line(img, tuple(axes[0]), tuple(axes[3]), (255, 0, 0), 3)
    img = cv2.line(img, tuple(axes[0]), tuple(axes[2]), (0, 255, 0), 3) ## y last


    return img



def draw_coord_mask(image, class_ids, RTs, Vs, Fs, intrinsics, output_path, is_overlay=True):
    Rasterizer = cdll.LoadLibrary('./libRasterizer.so')
    # coords = np.copy(image).astype(np.float64)
    # image = cv2.imread(image_path)[:, :, :3]
    # image_1 = copy.deepcopy(image)
    # image_2 = copy.deepcopy(image[:, :, ::-1])

    # print(type(image_1), image_1.dtype)
    # print(type(image_2), image_2.dtype)

    # print(np.unique(np.equal(image_1, image_2)))

    if is_overlay:
        #coords = copy.deepcopy(image[:, :, (0, 1, 2)]).astype(np.float64)
        coords = copy.deepcopy(image[:,:,::-1]).astype(np.float64)
    else:
        coords = 255*np.ones_like(image, dtype=np.float64)
    masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    masks[:,:] = 255

    Rasterizer.InitializeZBuffer(image.shape[1], image.shape[0], image.shape[2])

    for ind, RT in enumerate(RTs):
        cls_id = class_ids[ind]
        vertices = Vs[ind].transpose()
        number_of_vertex = vertices.shape[1]
        vertices_homogeneous = np.vstack([vertices, np.ones((1, number_of_vertex))])

        transformed_vertices = RTs[ind] @ vertices_homogeneous
        transformed_vertices = transformed_vertices[:3, :] / transformed_vertices[3, :]
        projected_pixels = intrinsics @ transformed_vertices
        projected_pixels[2, :] = 1.0 / projected_pixels[2, :]
        projected_pixels[:2, :] = projected_pixels[:2, :] * projected_pixels[2, :]
        projected_pixels = projected_pixels.transpose().astype(np.float64)  # shape: N*2


        # for pixel in projected_pixels:
        #     c_x = int(pixel[0])
        #     c_y = int(pixel[1])
        #     cv2.circle(image, (c_x, c_y), 1, (0, 0, 255))

        # post processing
        coordinate = Vs[ind].copy()
        coordinate += 0.5 # Range 0-1
        coordinate[:, 2] = 1 - coordinate[:, 2] # left coordinate system in coord map

        # cv2 imwrite is in order of BGR
        coordinate = coordinate[:, (2, 1, 0)]
        coordinate = coordinate * 255

        F = Fs[ind];
        for j in range(F.shape[0]):
            #print(F[j, :])
            p0 = projected_pixels[F[j, 0]].copy()
            p1 = projected_pixels[F[j, 1]].copy()
            p2 = projected_pixels[F[j, 2]].copy()
            prop0 = coordinate[F[j, 0]].copy()
            prop1 = coordinate[F[j, 1]].copy()
            prop2 = coordinate[F[j, 2]].copy()

            Rasterizer.DrawTriangle(
                c_void_p(p0.ctypes.data),
                c_void_p(p1.ctypes.data),
                c_void_p(p2.ctypes.data),
                c_void_p(prop0.ctypes.data),
                c_void_p(prop1.ctypes.data),
                c_void_p(prop2.ctypes.data),
                c_void_p(coords.ctypes.data),
                ind + 1,
                c_void_p(masks.ctypes.data))

    mask_output_path = output_path+'_mask.png'
    cv2.imwrite(mask_output_path, masks)
    coord_output_path = output_path+'_coord.jpg'
    cv2.imwrite(coord_output_path, coords)


def draw_text(draw_image, bbox, text, draw_box=False):
    fontFace = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = 1
    thickness = 1
    

    retval, baseline = cv2.getTextSize(text, fontFace, fontScale, thickness)
    
    bbox_margin = 10
    text_margin = 10
    
    text_box_pos_tl = (min(bbox[1] + bbox_margin, 635 - retval[0] - 2* text_margin) , min(bbox[2] + bbox_margin, 475 - retval[1] - 2* text_margin)) 
    text_box_pos_br = (text_box_pos_tl[0] + retval[0] + 2* text_margin,  text_box_pos_tl[1] + retval[1] + 2* text_margin)

    # text_pose is the bottom-left corner of the text
    text_pos = (text_box_pos_tl[0] + text_margin, text_box_pos_br[1] - text_margin - 3)
    
    if draw_box:
        cv2.rectangle(draw_image, 
                      (bbox[1], bbox[0]),
                      (bbox[3], bbox[2]),
                      (255, 0, 0), 2)

    cv2.rectangle(draw_image, 
                  text_box_pos_tl,
                  text_box_pos_br,
                  (255,0,0), -1)
    
    cv2.rectangle(draw_image, 
                  text_box_pos_tl,
                  text_box_pos_br,
                  (0,0,0), 1)

    cv2.putText(draw_image, text, text_pos,
                fontFace, fontScale, (255,255,255), thickness)

    return draw_image


def draw_detections(image, save_dir, data_name, image_id, intrinsics, synset_names, draw_rgb_coord,
                    gt_bbox, gt_class_ids, gt_mask, gt_coord, gt_RTs, gt_scales, gt_handle_visibility,
                    pred_bbox, pred_class_ids, pred_mask, pred_coord, pred_RTs, pred_scores, pred_scales,
                    draw_gt=True, draw_pred=True, draw_tag=False):

    alpha = 0.5

    if draw_gt:
        output_path = os.path.join(save_dir, '{}_{}_coord_gt.png'.format(data_name, image_id))
        draw_image = image.copy()
        num_gt_instances = len(gt_class_ids)

        for i in range(num_gt_instances):
            mask = gt_mask[:, :, i]
            #mask = mask[:, :, np.newaxis]
            #mask = np.repeat(mask, 3, axis=-1)
            cind, rind = np.where(mask == 1)
            coord_data = gt_coord[:, :, i, :].copy()
            coord_data[:, :, 2] = 1 - coord_data[:, :, 2] # undo the z axis flipping to match original data        
            draw_image[cind, rind] = coord_data[cind, rind] * 255
            
        if draw_tag:
            for i in range(num_gt_instances):
                overlay = draw_image.copy()
                overlay = draw_text(overlay, gt_bbox[i], synset_names[gt_class_ids[i]], draw_box=True)
                cv2.addWeighted(overlay, alpha, draw_image, 1 - alpha, 0, draw_image)
            
        # #if draw_tag:
        # for i in range(num_gt_instances):
        #     print('a', synset_names[gt_class_ids[i]])
        #     if synset_names[gt_class_ids[i]] == 'camera':
        #         overlay = draw_image.copy()
        #         cv2.rectangle(overlay, 
        #               (gt_bbox[i][1], gt_bbox[i][0]),
        #               (gt_bbox[i][3], gt_bbox[i][2]),
        #               (255, 0, 0), 2)

        #         cv2.addWeighted(overlay, alpha, draw_image, 1 - alpha, 0, draw_image)

        
        cv2.imwrite(output_path, draw_image[:, :, ::-1])

        output_path = os.path.join(save_dir, '{}_{}_bbox_gt.png'.format(data_name, image_id))
        draw_image_bbox = image.copy()

        if gt_RTs is not None:
            for ind, RT in enumerate(gt_RTs):
                cls_id = gt_class_ids[ind]

                xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
                transformed_axes = transform_coordinates_3d(xyz_axis, RT)
                projected_axes = calculate_2d_projections(transformed_axes, intrinsics)


                bbox_3d = get_3d_bbox(gt_scales[ind], 0)
                transformed_bbox_3d = transform_coordinates_3d(bbox_3d, RT)
                projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
                draw_image_bbox = draw(draw_image_bbox, projected_bbox, projected_axes, (255, 0, 0))

        cv2.imwrite(output_path, draw_image_bbox[:, :, ::-1])


    if draw_pred:
        print('a'*50)
        # Vs, Fs = dataset.load_objs(image_id, is_normalized=True) ## scale is estimated in RT
        output_path   = os.path.join(save_dir, '{}_{}_coord_pred.png'.format(data_name, image_id))
        output_path_r = os.path.join(save_dir, '{}_{}_coord_pred_r.png'.format(data_name, image_id))
        output_path_g = os.path.join(save_dir, '{}_{}_coord_pred_g.png'.format(data_name, image_id))
        output_path_b = os.path.join(save_dir, '{}_{}_coord_pred_b.png'.format(data_name, image_id))
        # utils.draw_coord_mask(image, r['class_ids'], pred_RTs, Vs, Fs, intrinsics, output_path)
        draw_image = image.copy()
        if draw_rgb_coord:
            r_image = image.copy()
            g_image = image.copy()
            b_image = image.copy()

        
        num_pred_instances = len(pred_class_ids)    
        for i in range(num_pred_instances):
            
            mask = pred_mask[:, :, i]
            #mask = mask[:, :, np.newaxis]
            #mask = np.repeat(mask, 3, axis=-1)
            cind, rind = np.where(mask == 1)
            coord_data = pred_coord[:, :, i, :].copy()
            coord_data[:, :, 2] = 1 - coord_data[:, :, 2] # undo the z axis flipping to match original data
            draw_image[cind, rind] = coord_data[cind, rind] * 255
            if draw_rgb_coord:
                b_image[cind, rind, 2] = coord_data[cind, rind, 2] * 255
                b_image[cind, rind, 0:2] = 0

                g_image[cind, rind, 1] = coord_data[cind, rind, 1] * 255
                g_image[cind, rind, 0] = 0
                g_image[cind, rind, 2] = 0

                r_image[cind, rind, 0] = coord_data[cind, rind, 0] * 255
                r_image[cind, rind, 1:3] = 0

        if draw_tag:
            for i in range(num_pred_instances):
                overlay = draw_image.copy()
                text = synset_names[pred_class_ids[i]]+'({:.2f})'.format(pred_scores[i])
                overlay = draw_text(overlay, pred_bbox[i], text, draw_box=True)
                cv2.addWeighted(overlay, alpha, draw_image, 1 - alpha, 0, draw_image)

        cv2.imwrite(output_path, draw_image[:, :, ::-1])



        if draw_rgb_coord:
            cv2.imwrite(output_path_r, r_image[:, :, ::-1])
            cv2.imwrite(output_path_g, g_image[:, :, ::-1])
            cv2.imwrite(output_path_b, b_image[:, :, ::-1])
                        
        
        output_path = os.path.join(save_dir, '{}_{}_bbox_pred.png'.format(data_name, image_id))
        draw_image_bbox = image.copy()

        if gt_class_ids is not None:
            gt_match, pred_match, _, pred_indices = compute_matches(gt_bbox, gt_class_ids, gt_mask,
                                                                    pred_bbox, pred_class_ids, pred_scores, pred_mask,
                                                                    0.5)

            if len(pred_indices):
                pred_class_ids = pred_class_ids[pred_indices]
                pred_scores = pred_scores[pred_indices]        
                pred_RTs = pred_RTs[pred_indices]

        
        for ind in range(num_pred_instances):
            RT = pred_RTs[ind]
            cls_id = pred_class_ids[ind]
            
            if gt_class_ids is not None:## if gt exists, skip instances that fail to match
                gt_ind = int(pred_match[ind])
                if gt_ind == -1:
                    continue
            
            xyz_axis = 0.3*np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]).transpose()
            transformed_axes = transform_coordinates_3d(xyz_axis, RT)
            projected_axes = calculate_2d_projections(transformed_axes, intrinsics)


            bbox_3d = get_3d_bbox(pred_scales[ind, :], 0)
            transformed_bbox_3d = transform_coordinates_3d(bbox_3d, RT)
            projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
            draw_image_bbox = draw(draw_image_bbox, projected_bbox, projected_axes, (255, 0, 0))

        if draw_tag:
            if gt_class_ids is not None: ## if gt exists, draw rotation and translation error
                for ind in range(num_pred_instances):
                    gt_ind = int(pred_match[ind])
                    if gt_ind == -1:
                        continue

                    overlay = draw_image_bbox.copy()
                    RT = pred_RTs[ind]
                    gt_RT = gt_RTs[gt_ind]
                    cls_id = pred_class_ids[ind]
                    
                    degree, cm = compute_RT_degree_cm_symmetry(RT, gt_RT, cls_id, gt_handle_visibility, synset_names)
                    text = '{}({:.1f}, {:.1f})'.format(synset_names[cls_id], degree, cm)
                    overlay = draw_text(overlay, pred_bbox[ind], text)
                    cv2.addWeighted(overlay, alpha, draw_image_bbox, 1 - alpha, 0, draw_image_bbox)



        cv2.imwrite(output_path, draw_image_bbox[:, :, ::-1])



def draw_coco_detections(image, save_dir, data_name, image_id, synset_names, draw_rgb_coord,
                        gt_bbox, gt_class_ids, gt_mask, 
                        pred_bbox, pred_class_ids, pred_mask, pred_coord, pred_scores):

    output_path = os.path.join(save_dir, '{}_{:04d}_image.png'.format(data_name, image_id))
    cv2.imwrite(output_path, image[:, :, ::-1])

    
    output_path = os.path.join(save_dir, '{}_{:04d}_mask_gt.png'.format(data_name, image_id))
    draw_image = image.copy()
    for i in range(len(gt_class_ids)):
        mask = gt_mask[:, :, i]
        #mask = mask[:, :, np.newaxis]
        #mask = np.repeat(mask, 3, axis=-1)
        cind, rind = np.where(mask == 1)
        draw_image[cind, rind] = np.random.rand(3) * 255
        draw_image = draw_text(draw_image, gt_bbox[i], synset_names[gt_class_ids[i]], draw_box=True)
    cv2.imwrite(output_path, draw_image[:, :, ::-1])


    # Vs, Fs = dataset.load_objs(image_id, is_normalized=True) ## scale is estimated in RT
    output_path   = os.path.join(save_dir, '{}_{:04d}_coord_pred.png'.format(data_name, image_id))
    output_path_r = os.path.join(save_dir, '{}_{:04d}_coord_pred_r.png'.format(data_name, image_id))
    output_path_g = os.path.join(save_dir, '{}_{:04d}_coord_pred_g.png'.format(data_name, image_id))
    output_path_b = os.path.join(save_dir, '{}_{:04d}_coord_pred_b.png'.format(data_name, image_id))
    # utils.draw_coord_mask(image, r['class_ids'], pred_RTs, Vs, Fs, intrinsics, output_path)
    draw_image = image.copy()
    if draw_rgb_coord:
        r_image = image.copy()
        g_image = image.copy()
        b_image = image.copy()
    
    for i in range(len(pred_class_ids)):
        
        mask = pred_mask[:, :, i]
        #mask = mask[:, :, np.newaxis]
        #mask = np.repeat(mask, 3, axis=-1)
        cind, rind = np.where(mask == 1)
        coord_data = pred_coord[:, :, i, :].copy()
        coord_data[:, :, 2] = 1 - coord_data[:, :, 2] # undo the z axis flipping to match original data
        draw_image[cind, rind] = coord_data[cind, rind] * 255
        draw_image = draw_text(draw_image, pred_bbox[i], synset_names[pred_class_ids[i]], draw_box=True)
        if draw_rgb_coord:
            b_image[cind, rind, 2] = coord_data[cind, rind, 2] * 255
            b_image[cind, rind, 0:2] = 0

            g_image[cind, rind, 1] = coord_data[cind, rind, 1] * 255
            g_image[cind, rind, 0] = 0
            g_image[cind, rind, 2] = 0

            r_image[cind, rind, 0] = coord_data[cind, rind, 0] * 255
            r_image[cind, rind, 1:3] = 0

    cv2.imwrite(output_path, draw_image[:, :, ::-1])
    if draw_rgb_coord:
        cv2.imwrite(output_path_r, r_image[:, :, ::-1])
        cv2.imwrite(output_path_g, g_image[:, :, ::-1])
        cv2.imwrite(output_path_b, b_image[:, :, ::-1])
                    
    
    

def backproject(depth, intrinsics, instance_mask):
    intrinsics_inv = np.linalg.inv(intrinsics)
    image_shape = depth.shape
    width = image_shape[1]
    height = image_shape[0]

    x = np.arange(width)
    y = np.arange(height)

    #non_zero_mask = np.logical_and(depth > 0, depth < 5000)
    non_zero_mask = (depth > 0)
    final_instance_mask = np.logical_and(instance_mask, non_zero_mask)
    
    idxs = np.where(final_instance_mask)
    grid = np.array([idxs[1], idxs[0]])

    # shape: height * width
    # mesh_grid = np.meshgrid(x, y) #[height, width, 2]
    # mesh_grid = np.reshape(mesh_grid, [2, -1])
    length = grid.shape[1]
    ones = np.ones([1, length])
    uv_grid = np.concatenate((grid, ones), axis=0) # [3, num_pixel]

    xyz = intrinsics_inv @ uv_grid # [3, num_pixel]
    xyz = np.transpose(xyz) #[num_pixel, 3]

    z = depth[idxs[0], idxs[1]]

    # print(np.amax(z), np.amin(z))
    pts = xyz * z[:, np.newaxis]/xyz[:, -1:]
    pts[:, 0] = -pts[:, 0]
    pts[:, 1] = -pts[:, 1]

    return pts, idxs


def align(class_ids, masks, coords, depth, intrinsics, synset_names, image_path, save_path=None, if_norm=False, with_scale=True, verbose=False):
    num_instances = len(class_ids)
    error_messages = ''
    elapses = []
    if num_instances == 0:
        return np.zeros((0, 4, 4)), np.ones((0, 3)), error_messages, elapses

    RTs = np.zeros((num_instances, 4, 4))
    bbox_scales = np.ones((num_instances, 3))
    
    for i in range(num_instances):
        #class_name = synset_names[class_ids[i]]
        class_id = class_ids[i]
        mask = masks[:, :, i]
        coord = coords[:, :, i, :]
        abs_coord_pts = np.abs(coord[mask==1] - 0.5)
        bbox_scales[i, :] = 2*np.amax(abs_coord_pts, axis=0)

        pts, idxs = backproject(depth, intrinsics, mask)
        coord_pts = coord[idxs[0], idxs[1], :] - 0.5

        if if_norm:
            scale = np.linalg.norm(bbox_scales[i, :])
            bbox_scales[i, :] /= scale
            coord_pts /= scale

        
        try:
            start = time.time()
            
            scales, rotation, translation, outtransform = estimateSimilarityTransform(coord_pts, pts, False)

            aligned_RT = np.zeros((4, 4), dtype=np.float32) 
            if with_scale:
                aligned_RT[:3, :3] = np.diag(scales) / 1000 @ rotation.transpose()
            else:
                aligned_RT[:3, :3] = rotation.transpose()
            aligned_RT[:3, 3] = translation / 1000
            aligned_RT[3, 3] = 1
            
            if save_path is not None:
                coord_pts_rotated = aligned_RT[:3, :3] @ coord_pts.transpose() + aligned_RT[:3, 3:]
                coord_pts_rotated = coord_pts_rotated.transpose()
                np.savetxt(save_path+'_{}_{}_depth_pts.txt'.format(i, class_name), pts)
                np.savetxt(save_path+'_{}_{}_coord_pts.txt'.format(i, class_name), coord_pts)
                np.savetxt(save_path+'_{}_{}_coord_pts_aligned.txt'.format(i, class_name), coord_pts_rotated)

            if verbose:
                print('Mask ID: ', i)
                print('Scale: ', scales/1000)
                print('Rotation: ', rotation.transpose())
                print('Translation: ', translation/1000)

            elapsed = time.time() - start
            print('elapsed: ', elapsed)
            elapses.append(elapsed)
        

        except Exception as e:
            message = '[ Error ] aligning instance {} in {} fails. Message: {}.'.format(synset_names[class_id], image_path, str(e))
            print(message)
            error_messages += message + '\n'
            aligned_RT = np.identity(4, dtype=np.float32) 

        # print('Estimation takes {:03f}s.'.format(time.time() - start))
        # from camera world to computer vision frame
        z_180_RT = np.zeros((4, 4), dtype=np.float32)
        z_180_RT[:3, :3] = np.diag([-1, -1, 1])
        z_180_RT[3, 3] = 1

        RTs[i, :, :] = z_180_RT @ aligned_RT 

    return RTs, bbox_scales, error_messages, elapses


def align_ICP(class_ids, masks, depth, intrinsics, synset_names, image_path, save_path=None, if_norm=False, verbose=False):
    num_instances = len(class_ids)
    error_messages = ''

    print(intrinsics)
    if num_instances == 0:
        return np.zeros((0, 4, 4)), np.ones((0, 3)), error_messages

    RTs = np.zeros((num_instances, 4, 4))
    bbox_scales = np.ones((num_instances, 3))

    def rotation_y_matrix(theta):
            rotation_matrix =  \
                    np.array([ np.cos(theta), 0,  np.sin(theta),
                                 0,           1,  0,
                              -np.sin(theta), 0,  np.cos(theta)])
            rotation_matrix = np.reshape(rotation_matrix, (3, 3))
            return rotation_matrix

    for i in range(num_instances):
        #class_name = synset_names[class_ids[i]]
        class_id = class_ids[i]
        mask = masks[:, :, i]
        # coord = coords[:, :, i, :]
        # abs_coord_pts = np.abs(coord[mask==1] - 0.5)
        # bbox_scales[i, :] = 2*np.amax(abs_coord_pts, axis=0)

        depth_pts, idxs = backproject(depth, intrinsics, mask)
        # coord_pts = coord[idxs[0], idxs[1], :] - 0.5

        # if if_norm:
        #     scale = np.linalg.norm(bbox_scales[i, :])
        #     bbox_scales[i, :] /= scale
        #     coord_pts /= scale
        
        glob_paths = '/home/hewang/Projects/CoordRCNN/data/pts/real_test/{}*.txt'.format(synset_names[class_ids[i]])
        print(glob_paths)
        shape_paths = glob.glob(glob_paths)
        print(shape_paths)


        random.shuffle(shape_paths)
        shape_pts = np.loadtxt(shape_paths[0])
        print(shape_pts.shape)
        bbox_scales[i, :] = np.array([1, 1, 1])
        


        # depth_pts = rotation_y_matrix(0.01)@shape_pts.transpose() + 0.01
        # depth_pts = depth_pts.transpose()

        #try:
        if True:
            scales, rotation, translation = ICP.doICP(shape_pts,
                                                      depth_pts, 
                                                      threshold=5, isViz=False)

            scales = np.amax(shape_pts, axis=0) - np.amin(shape_pts, axis=0) 
            # scales, rotation, translation, outtransform = estimateSimilarityTransform(coord_pts, pts, False)
            #scales = np.diag(bbox_scales[i, :])
            aligned_RT = np.zeros((4, 4), dtype=np.float32) 
            aligned_RT[:3, :3] = np.diag(scales)@rotation#.transpose()
            aligned_RT[:3, 3] = translation / 1000
            aligned_RT[3, 3] = 1
            
            #bbox_scales[i, :] = scales@bbox_scales[i, :]/1000

            if save_path is not None:
                coord_pts_rotated = aligned_RT[:3, :3] @ coord_pts.transpose() + aligned_RT[:3, 3:]
                coord_pts_rotated = coord_pts_rotated.transpose()
                np.savetxt(save_path+'_{}_{}_depth_pts.txt'.format(i, class_name), pts)
                np.savetxt(save_path+'_{}_{}_coord_pts.txt'.format(i, class_name), coord_pts)
                np.savetxt(save_path+'_{}_{}_coord_pts_aligned.txt'.format(i, class_name), coord_pts_rotated)

            if verbose:
                print('Mask ID: ', i)
                print('Scale: ', scales)
                print('Rotation: ', rotation)
                print('Translation: ', translation)
        

        # except Exception as e:
        #     message = '[ Error ] aligning instance {} in {} fails. Message: {}.'.format(synset_names[class_id], image_path, str(e))
        #     print(message)
        #     error_messages += message + '\n'
        #     aligned_RT = np.identity(4, dtype=np.float32) 

        # print('Estimation takes {:03f}s.'.format(time.time() - start))
        # from camera world to computer vision frame
        z_180_RT = np.zeros((4, 4), dtype=np.float32)
        z_180_RT[:3, :3] = np.diag([-1, -1, 1])
        z_180_RT[3, 3] = 1

        RTs[i, :, :] = z_180_RT @ aligned_RT 

    return RTs, bbox_scales, error_messages
