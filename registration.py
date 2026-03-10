import open3d as o3d
import numpy as np


def preprocess(pcd, voxel):

    pcd_down = pcd.voxel_down_sample(voxel)

    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel * 2,
            max_nn=30
        )
    )

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel * 5,
            max_nn=100
        )
    )

    return pcd_down, fpfh


def register(source, target):

    voxel_size = 0.02

    # preprocess clouds
    source_down, source_fpfh = preprocess(source, voxel_size)
    target_down, target_fpfh = preprocess(target, voxel_size)

    # global registration
    distance_threshold = voxel_size * 3

    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 1000)
    )

    # estimate normals for ICP
    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2,
            max_nn=30
        )
    )

    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2,
            max_nn=30
        )
    )

    # ICP refinement
    result_icp = o3d.pipelines.registration.registration_icp(
        source,
        target,
        voxel_size * 0.4,
        result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    return result_icp.transformation 