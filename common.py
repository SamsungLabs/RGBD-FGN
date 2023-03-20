import numpy as np
import math
import argparse

def depth_to_pc(depth, meta, rgb=None):
	_, _, fx, fy, cx, cy = meta['intrinsic']

	mask = np.where((depth > 0) & (depth < 2.0))
	x, y = mask[1], mask[0]

	normalized_x = (x.astype(np.float32) - cx)
	normalized_y = (y.astype(np.float32) - cy)

	points_x = normalized_x * depth[y, x] / fx
	points_y = normalized_y * depth[y, x] / fy
	points_z = depth[y, x]

	if rgb is not None:
		rgb = rgb[y,x,:]

	points = np.vstack((points_x, points_y, points_z)).T # (N, 3)

	return points, rgb

def depth_to_pc_downsampling(depth, meta, rgb, seg=None):
	# 255 to 0 & table to 0
	if seg is not None:
		seg = np.where((seg == 255) | (seg == (meta['table_seg_id'] % 256)), 0, seg)
	_, _, fx, fy, cx, cy = meta['intrinsic']
	
	N_x = 160 # origin: 640
	N_y = 120 # origin: 480
	interval_x = math.floor(640/N_x)
	interval_y = math.floor(480/N_y)
	xmap, ymap = np.meshgrid(np.arange(N_x)*interval_x, np.arange(N_y)*interval_y)

	depth_downsize = depth[ymap,xmap] # (N_y, N_x)
	if seg is not None:
		seg_downsize = seg[ymap,xmap] # (N_y, N_x)
		seg_downsize = np.where(seg_downsize > 0, 1., 0.)
		occupancy = seg_downsize.reshape([-1, ]) # (N_y*N_x, ) (row major)

	points_z = depth_downsize # (N_y, N_x)
	points_x = (xmap - cx) / fx * np.abs(points_z) # (N_y, N_x)
	points_y = (ymap - cy) / fy * np.abs(points_z) # (N_y, N_x)
	points = np.stack([points_x, points_y, points_z], axis=-1) # (N_y, N_x, 3)
	points = points.reshape([-1, 3]) # (N_y*N_x, 3) (row major)

	rgb = rgb[ymap,xmap,:] # (N_y, N_x, 3)
	rgb = rgb.reshape([-1, 3]) # (N_y*N_x, 3) (row major)

	if seg is not None:
		return points, rgb, occupancy
	else:
		return points, rgb

def find_gt(_tree, pc, grasp_approach_direction, grasp_lateral_direction, grasp_upper_direction, grasp_position_distance, grip_width, grasp_collision, grasp_quality, cam_pose, radius):

    cam_pose_inv = inverse_transform(cam_pose)  # (4, 4)

    grasp_approach_direction_body = grasp_approach_direction @ np.transpose(
        cam_pose_inv[:3, :3])  # (N, 3)
    grasp_lateral_direction_body = grasp_lateral_direction @ np.transpose(
        cam_pose_inv[:3, :3])  # (N, 3)
    grasp_upper_direction_body = grasp_upper_direction @ np.transpose(
        cam_pose_inv[:3, :3])  # (N, 3)

    homog_mat = np.ones((pc.shape[0], 1))  # (Np, 1)
    pc_global = np.hstack((pc, homog_mat)) @ np.transpose(
        cam_pose)  # (Np, 4)
    pc_global = pc_global[:, :3]  # (Np, 3)

    closest_distance, closest_idx = _tree.query(pc_global, k=1, p=2)  

    ind = np.where(closest_distance < radius)
    for i in ind[0]:
        res = _tree.query_ball_point(pc_global[i,:], closest_distance[i]+0.0001)
        if np.sum(1-grasp_collision[res]) > 0:
            reference_value = np.multiply(grasp_quality[res], 1-grasp_collision[res])
        else:
            reference_value = grasp_quality[res]
        ind2 = np.argmax(reference_value)
        closest_idx[i] = res[ind2]

    grasp_approach_direction_body_gt = np.take(
        grasp_approach_direction_body, closest_idx, axis=0)  # (Np, 3)
    grasp_lateral_direction_body_gt = np.take(
        grasp_lateral_direction_body, closest_idx, axis=0)  # (Np, 3)
    grasp_upper_direction_body_gt = np.take(
        grasp_upper_direction_body, closest_idx, axis=0)  # (Np, 3)

    # set camera direction to -y in camera frame
    ind = np.where(grasp_upper_direction_body_gt[:,1] > 0.)
    grasp_lateral_direction_body_gt[ind] = -grasp_lateral_direction_body_gt[ind] 

    grasp_position_distance_gt =  np.take(
        grasp_position_distance, closest_idx, axis=0)  # (Np, )
    grip_width_gt = np.take(
        grip_width, closest_idx, axis=0)  # (Np, )
    grasp_score_gt = np.less(
        closest_distance, radius).astype(np.float32)  # (Np, )
    grasp_collision_gt =  np.take(
        grasp_collision, closest_idx, axis=0)  # (Np, )

    return grasp_approach_direction_body_gt, grasp_lateral_direction_body_gt, grasp_position_distance_gt, grip_width_gt, grasp_score_gt, grasp_collision_gt

def inverse_transform(trans):
    rot = np.transpose(trans[:3, :3])
    t = -np.matmul(rot, trans[:3, 3])
    output = np.zeros((4, 4), dtype=np.float32)
    output[3][3] = 1.
    output[:3, :3] = rot
    output[:3, 3] = t

    return output

def farthest_point_sampling(points, num_samples):
    indices = np.zeros(num_samples, dtype=int)
    distances = np.full(points.shape[0], np.inf)

    indices[0] = np.random.choice(points.shape[0])
    distances[indices[0]] = 0

    for i in range(1, num_samples):
        last_point = points[indices[i-1], :]
        dist = np.linalg.norm(points - last_point, axis=1)
        distances = np.minimum(distances, dist)
        indices[i] = np.argmax(distances)

    return indices
