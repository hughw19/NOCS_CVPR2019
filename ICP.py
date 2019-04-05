'''
Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation
Implementation of ICP

Written by Srinath Sridhar
'''

from open3d import *
import numpy as np
import copy
from matplotlib.mlab import PCA
import random

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    draw_geometries([source_temp, target_temp])

def doICP(model_pts_np, pred_pts_np, SigmaFactor=5, threshold=1000, isViz=False):
    model_pts = PointCloud()
    model_pts.points = Vector3dVector(model_pts_np)
    model_pts_orig = model_pts
    model_pts_orig_np = np.asarray(model_pts_orig.points)
    pred_pts = PointCloud()
    pred_pts.points = Vector3dVector(pred_pts_np)

    # First use PCA to compute the dimensions of target
    model_pca = PCA(model_pts_np)
    pred_pca = PCA(pred_pts_np)
    # print('Model mean:', model_pca.mu)
    # print('Model sigma:', model_pca.sigma)
    # print('Prediction mean:', pred_pca.mu)
    # print('Prediction sigma:', pred_pca.sigma)
    #
    # Scale model and move its center to prediction center
    ScaleFactor = np.array([1, 1, 1])
 

    #ScaleFactor = pred_pca.sigma * SigmaFactor * SigmaFactor # OUTPUT 1
    PredCenter = pred_pca.mu
    for i in range(0, 3):
        # model_pts_np[:, i] = (model_pts_np[:, i] - model_pca.mu[i]) * ScaleFactor[i] * Factor + pred_pca.mu[i]
        model_pts_np[:, i] = (model_pts_np[:, i]) * ScaleFactor[i] + PredCenter[i] # Scale and then move model origin to origin of prediction depth point cloud
        model_pts_orig_np[:, i] = (model_pts_orig_np[:, i]) * ScaleFactor[i]

    model_pts = PointCloud()
    model_pts.points = Vector3dVector(model_pts_np)
    model_pts_orig = PointCloud()
    model_pts_orig.points = Vector3dVector(model_pts_orig_np)
    # model_pca = PCA(model_pts_np)
    # print('Model mean (after transform):', model_pca.mu)
    # print('Model sigma (after transform):', model_pca.sigma)

    trans_init = np.identity(4)
    print('Apply point-to-point ICP')
    reg_p2p = registration_icp(model_pts, pred_pts, threshold, trans_init,
            TransformationEstimationPointToPoint())
    print(reg_p2p)
    # print('Transformation is:')
    # print(reg_p2p.transformation) # OUTPUT 2, 3
    if isViz:
        draw_registration_result(model_pts, pred_pts, reg_p2p.transformation)

    FinalTrans = reg_p2p.transformation.copy()
    # print('Before:\n', FinalTrans)
    FinalTrans[:3, 3] = FinalTrans[:3, 3] + FinalTrans[:3, :3] @ PredCenter
    # print('After:\n', FinalTrans)
    Translation = FinalTrans[:3, 3]
    Rotation = FinalTrans[:3, :3]
    # print('PredCenter:', PredCenter)
    # print('ICP Addition:', reg_p2p.transformation[:3, 3])
    # print('Final Translation:', Translation)

    if isViz:
        draw_registration_result(model_pts_orig, pred_pts, FinalTrans)

    return ScaleFactor, Rotation, Translation


if __name__ == '__main__':
    ScaleFactor, Rotation, Translation = doICP(np.loadtxt('./pts/bowl.txt'), np.loadtxt('./pts/bowl_pred.txt'), SigmaFactor=5, threshold=100, isViz=False)
    print('ScaleFactor:', ScaleFactor)
    print('Rotation:', Rotation)
    print('Translation:', Translation)
