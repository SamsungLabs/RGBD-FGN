import numpy as np
from grasp_network import GraspNetwork
from PIL import Image
from visualization import Visualizer
from common import depth_to_pc_downsampling
import matplotlib.pyplot as plt

if __name__ == '__main__':

	""" Load model"""
	model = GraspNetwork()
	model.load_weights('./logs/checkpoint').expect_partial()

	""" Load rgb, depth, intrinsic, minmax value for grasp depth """
	rgb_image = np.array(Image.open('1.jpg')) # (480, 640, 3)
	depth_image = np.array(np.load('1.npy'))/1000 # (480, 640) unit = (m)

	intrinsic = np.array([[608.5, 0.0, 320.0],
						 [0.0, 608.5, 241.1],
						 [0.0, 0.0, 1.0]]) # (3, 3)
	meta = dict(intrinsic=[640,480,intrinsic[0,0],intrinsic[1,1],intrinsic[0,2],intrinsic[1,2]])

	""" Depth to point cloud """
	depth_image = np.where(depth_image > 1.5, 0., depth_image)
	pc_i, rgb_i = depth_to_pc_downsampling(depth_image, meta, rgb_image)

	""" Inference """
	grasp_position_distance_max = 0.0572
	inputs = dict(pc=pc_i[np.newaxis,:,:],
					rgb=rgb_image[np.newaxis,:,:,:].astype(np.uint8))
	pred = model(inputs, False)
	grasp_approach_direction_pred = pred['grasp_approach_direction_pred'].numpy().squeeze()
	grasp_lateral_direction_pred = pred['grasp_lateral_direction_pred'].numpy().squeeze()
	grasp_position_distance_pred = pred['grasp_position_distance_pred'].numpy().squeeze() * grasp_position_distance_max
	grasp_score_pred = pred['grasp_score_pred'].numpy().squeeze()
	grasp_collision_pred = pred['grasp_collision_pred'].numpy().squeeze()
	grip_width_pred = pred['grip_width_pred'].numpy().squeeze() * 0.115
	grip_width_pred = np.maximum(grip_width_pred, 0.115/10) # (N, ), gripper maximum distance = 0.115
	occupancy_pred = pred['occupancy_pred'].numpy().squeeze()

	score_threshold = 0.1
	collision_threshold = 0.5
	occupancy_threshold = 0.99

	idx1 = np.where((grasp_score_pred > score_threshold) & (grasp_collision_pred < collision_threshold) & (occupancy_pred > occupancy_threshold) & (pc_i[:,2] > 0.))

	grasp_position_wrt_cam = pc_i[idx1] + np.multiply(grasp_position_distance_pred[idx1][:,np.newaxis],grasp_approach_direction_pred[idx1])
	grasp_poses_pred = np.zeros((grasp_approach_direction_pred[idx1].shape[0],4,4))
	for i in range(grasp_approach_direction_pred[idx1].shape[0]):
		grasp_poses_pred[i,:3,1] = np.transpose(grasp_lateral_direction_pred[idx1][i,:])
		grasp_poses_pred[i,:3,0] = np.transpose(np.cross(grasp_lateral_direction_pred[idx1][i,:],grasp_approach_direction_pred[idx1][i,:]))
		grasp_poses_pred[i,:3,2] = np.transpose(grasp_approach_direction_pred[idx1][i,:])
		grasp_poses_pred[i,:3,3] = np.transpose(grasp_position_wrt_cam[i,:])
		grasp_poses_pred[i,3,3] = 1.0

	grasp_collision_pred_plot = grasp_collision_pred[idx1]
	grip_width_pred_plot = grip_width_pred[idx1]
	grasp_score_pred_plot = grasp_score_pred[idx1]

	sorted_score_idx = np.argsort(grasp_score_pred_plot)[::-1]
	num_grasps = min(300, sorted_score_idx.shape[0])
	idx2 = sorted_score_idx[:num_grasps]

	""" option1: visualize using open3d"""
	vis = Visualizer()
	vis.update_scene(
		pc_i,
		rgb_i,
		grasp_poses_pred[idx2,:,:],
		grip_width_pred_plot[idx2]
	)

	while True:
		vis.update_render()

	
	""" option2: visualize using matplotlib.pyplot"""
	# N_x = 160 # origin: 640
	# N_y = 120 # origin: 480
	
	# plt.figure(1)
	# plt.subplot(2,2,1)
	# plt.imshow(grasp_score_pred.reshape(N_y,N_x))
	# plt.subplot(2,2,2)
	# plt.imshow(occupancy_pred.reshape(N_y,N_x))
	# plt.subplot(2,2,3)
	# plt.imshow(rgb_image)
	# plt.subplot(2,2,4)
	# plt.imshow(depth_image)
	# plt.show()

