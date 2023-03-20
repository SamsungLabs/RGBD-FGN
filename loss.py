import tensorflow as tf
from tensorflow.keras.losses import Loss

class GraspLoss(Loss):
	def __init__(self):
		super(GraspLoss, self).__init__()

	def __call__(self, y_true, y_pred):
		grasp_orientation_loss = self._grasp_orientation_loss(y_true, y_pred)
		grasp_position_distance_loss = self._grasp_position_distance_loss(y_true, y_pred)
		grasp_score_loss = self._grasp_score_loss(y_true, y_pred)
		grasp_collision_loss = self._grasp_collision_loss(y_true, y_pred)
		grip_width_loss = self._grip_width_loss(y_true, y_pred)
		occupancy_loss = self._occupancy_loss(y_true, y_pred)
		total_loss = grasp_orientation_loss + grasp_position_distance_loss + grasp_score_loss \
						+ grasp_collision_loss + grip_width_loss + occupancy_loss
		losses = dict(total_loss=total_loss,
						grasp_orientation_loss=grasp_orientation_loss,
						grasp_position_distance_loss=grasp_position_distance_loss,
						grasp_score_loss=grasp_score_loss,
						grasp_collision_loss=grasp_collision_loss,
						grip_width_loss=grip_width_loss,
						occupancy_loss=occupancy_loss)

		return losses
		
	def _grasp_orientation_loss(self, y_true, y_pred):
		pos_grasp_nums = y_true['pos_grasp_nums'] # (b, )
		grasp_approach_direction_gt = y_true['grasp_approach_direction_gt'] # (b, N, 3)
		grasp_approach_direction_pred = y_pred['grasp_approach_direction_pred'] # (b, N, 3)
		grasp_lateral_direction_gt = y_true['grasp_lateral_direction_gt'] # (b, N, 3)
		grasp_lateral_direction_pred = y_pred['grasp_lateral_direction_pred'] # (b, N, 3)
		grasp_upper_direction_gt = tf.linalg.cross(grasp_approach_direction_gt, grasp_lateral_direction_gt)
		grasp_upper_direction_pred = tf.linalg.cross(grasp_approach_direction_pred, grasp_lateral_direction_pred)
		grasp_score_gt = y_true['grasp_score_gt'] # (b, N)

		# geodesic distance = acos(0.5*(trace(R^T Rhat) - 1))
		trace_RTR = tf.reduce_sum(tf.multiply(grasp_approach_direction_gt, grasp_approach_direction_pred), axis=2) + \
						tf.reduce_sum(tf.multiply(grasp_lateral_direction_gt, grasp_lateral_direction_pred), axis=2) + \
						tf.reduce_sum(tf.multiply(grasp_upper_direction_gt, grasp_upper_direction_pred), axis=2) # (b, N)
		cosine_loss = tf.constant(0.5) * (trace_RTR - tf.constant(1.))  # (b, N)
		cosine_loss = tf.math.minimum(cosine_loss,tf.constant(0.9999))
		cosine_loss = tf.math.maximum(cosine_loss,tf.constant(-0.9999))
		angle_loss = tf.math.acos(cosine_loss)
		angle_loss = tf.multiply(angle_loss, grasp_score_gt)  # (b, N)
		grasp_orientation_loss = tf.math.reduce_mean(tf.reduce_sum(angle_loss, axis=1) / pos_grasp_nums)

		return grasp_orientation_loss
		
	def _grasp_position_distance_loss(self, y_true, y_pred):
		pos_grasp_nums = y_true['pos_grasp_nums'] # (b, )
		grasp_position_distance_gt = y_true['grasp_position_distance_gt'] # (b, N)
		grasp_position_distance_pred = y_pred['grasp_position_distance_pred'] # (b, N, 1)
		grasp_score_gt = y_true['grasp_score_gt'] # (b, N)

		grasp_position_distance_loss = tf.reduce_sum(tf.math.abs(tf.expand_dims(grasp_position_distance_gt, axis=2)-grasp_position_distance_pred), axis=2) # (b, N)
		grasp_position_distance_loss = tf.multiply(grasp_position_distance_loss, grasp_score_gt)
		grasp_position_distance_loss = tf.math.reduce_mean(tf.reduce_sum(grasp_position_distance_loss, axis=1) / pos_grasp_nums)

		return grasp_position_distance_loss
		
	def _grip_width_loss(self, y_true, y_pred):
		pos_grasp_nums = y_true['pos_grasp_nums'] # (b, )
		grip_width_gt = y_true['grip_width_gt'] # (b, N)
		grip_width_pred = y_pred['grip_width_pred'] # (b, N, 1)
		grasp_score_gt = y_true['grasp_score_gt'] # (b, N)

		grip_width_loss = tf.reduce_sum(tf.math.abs(tf.expand_dims(grip_width_gt, axis=2)-grip_width_pred), axis=2) # (b, N)
		grip_width_loss = tf.multiply(grip_width_loss, grasp_score_gt)
		grip_width_loss = tf.math.reduce_mean(tf.reduce_sum(grip_width_loss, axis=1) / pos_grasp_nums)

		return grip_width_loss

	def _grasp_score_loss(self, y_true, y_pred):
		""" all """
		grasp_score_gt = y_true['grasp_score_gt'] # (b, N)
		grasp_score_head = y_pred['grasp_score_head'] # (b, N, 1)

		grasp_score_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.expand_dims(grasp_score_gt, axis=2),
																	logits=grasp_score_head)  # (b, N, 1)
		grasp_score_loss = tf.math.reduce_mean(grasp_score_loss)

		return grasp_score_loss

	def _occupancy_loss(self, y_true, y_pred):
		""" all """
		occupancy_gt = y_true['occupancy_gt'] # (b, N)
		occupancy_head = y_pred['occupancy_head'] # (b, N, 1)

		occupancy_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.expand_dims(occupancy_gt, axis=2),
																	logits=occupancy_head)  # (b, N, 1)
		occupancy_loss = tf.math.reduce_mean(occupancy_loss)

		return occupancy_loss

	def _grasp_collision_loss(self, y_true, y_pred):
		""" all """
		pos_grasp_nums = y_true['pos_grasp_nums'] # (b, )
		grasp_collision_gt = y_true['grasp_collision_gt'] # (b, N)
		grasp_collision_head = y_pred['grasp_collision_head'] # (b, N, 1)
		grasp_score_gt = y_true['grasp_score_gt'] # (b, N)

		grasp_collision_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.expand_dims(grasp_collision_gt, axis=2),
																	logits=grasp_collision_head)  # (b, N, 1)
		grasp_collision_loss = tf.reduce_sum(grasp_collision_loss, axis=2) # (b, N)										
		grasp_collision_loss = tf.multiply(grasp_collision_loss, grasp_score_gt) # (b, N)
		grasp_collision_loss = tf.math.reduce_mean(tf.reduce_sum(grasp_collision_loss, axis=1) / pos_grasp_nums)

		return grasp_collision_loss