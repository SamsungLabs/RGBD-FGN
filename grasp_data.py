import tensorflow as tf
import numpy as np
import os

class GraspData():
	def __init__(self, batch_size, data_path):
		self.batch_size = batch_size
		self.data_path = data_path
		self._build()

	def _build(self):
		input_data = np.load(os.path.join(self.data_path,'input_data.npz'))
		self.pc = input_data['pc'] # (N_data, N, 3)
		self.rgb = input_data['rgb'] # (N_data, 480, 640, 3)
		
		output_data = np.load(os.path.join(self.data_path,'output_data.npz'))
		param = np.load(os.path.join(self.data_path,'position_distance_minmax.npz'))
		self.grasp_approach_direction_gt = output_data['grasp_approach_direction_gt_wrt_cam'] # (N_data, N, 3)
		self.grasp_lateral_direction_gt = output_data['grasp_lateral_direction_gt_wrt_cam'] # (N_data, N, 3)
		self.grasp_position_distance_gt = output_data['grasp_position_distance_gt']/param['grasp_position_distance_max'] # (N_data, N), 0-1
		self.grip_width_gt = output_data['grip_width_gt']/0.115 # (N_data, N)
		self.grasp_score_gt = output_data['grasp_score_gt'] # (N_data, N)
		self.grasp_collision_gt = output_data['grasp_collision_gt'] # (N_data, N)
		self.pos_grasp_nums = np.maximum(np.sum(self.grasp_score_gt, axis=1),1.0) # (N_data, )
		self.occupancy_gt = output_data['occupancy_gt'] # (N_data, N)

	def _data_generator(self):
		for i in range(self.pc.shape[0]):
			inputs = dict(pc=tf.constant(self.pc[i,:,:], tf.float32),
							rgb=tf.constant(self.rgb[i,:,:,:], tf.uint8))
			outputs = dict(grasp_approach_direction_gt=tf.constant(self.grasp_approach_direction_gt[i,:,:], tf.float32),
							grasp_lateral_direction_gt=tf.constant(self.grasp_lateral_direction_gt[i,:,:], tf.float32),
							grasp_position_distance_gt=tf.constant(self.grasp_position_distance_gt[i,:], tf.float32),
							grip_width_gt=tf.constant(self.grip_width_gt[i,:], tf.float32),
							grasp_score_gt=tf.constant(self.grasp_score_gt[i,:], tf.float32),
							grasp_collision_gt=tf.constant(self.grasp_collision_gt[i,:], tf.float32),
							pos_grasp_nums=tf.constant(self.pos_grasp_nums[i], tf.float32),
							occupancy_gt=tf.constant(self.occupancy_gt[i,:], tf.float32))

			yield inputs, outputs
				
	def get_dataset(self):
		output_types = (dict(pc=tf.float32,
								rgb=tf.uint8),
						dict(grasp_approach_direction_gt=tf.float32,
								grasp_lateral_direction_gt=tf.float32,
								grasp_position_distance_gt=tf.float32,
								grip_width_gt=tf.float32,
								grasp_score_gt=tf.float32,
								grasp_collision_gt=tf.float32,
								pos_grasp_nums=tf.float32,
								occupancy_gt=tf.float32))
		dataset = tf.data.Dataset.from_generator(self._data_generator,
                                              output_types=output_types)
		dataset = dataset.batch(self.batch_size)

		return dataset