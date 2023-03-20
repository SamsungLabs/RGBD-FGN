import tensorflow as tf
from RGBbackbone.fpn_with_backbone import FPN_W_BACKBONE
from Depthbackbone.pointnet2 import Pointnet2
from head import MLPHead
from loss import GraspLoss
import tensorflow.keras.backend as K

class GraspNetwork(tf.keras.Model):
	def __init__(self):
		super().__init__(self)
		self.loss_func = GraspLoss()
		self._build()

	def _build(self):
		self.RGB_backbone = FPN_W_BACKBONE(feature_num=64, resnet_num=50)
		self.depth_backbone = Pointnet2()
		self.grasp_approach_direction_head = MLPHead(filters_length=128, output_length=3)
		self.grasp_lateral_direction_head = MLPHead(filters_length=128, output_length=3)
		self.grasp_position_distance_head = MLPHead(filters_length=128, output_length=1)
		self.grasp_score_head = MLPHead(filters_length=128, output_length=1)
		self.grasp_collision_head = MLPHead(filters_length=128, output_length=1)
		self.grip_width_head = MLPHead(filters_length=128, output_length=1)
		self.occupancy_head = MLPHead(filters_length=128, output_length=1)

	def call(self, inputs, training=False):
		rgb = tf.slice(inputs['rgb'], [0,0,0,0], [-1,480,640,3]) # (b, 480, 640, 3)
		pc = tf.slice(inputs['pc'], [0,0,0], [-1,-1,3]) # (b, Np, 3)
		pc_mean = tf.math.reduce_mean(pc, axis=1, keepdims=True)  # (b, 1, 3)

		# backbone
		rgb_feature = self.RGB_backbone(rgb, training=training) # (b, 120, 160, N_cf)
		point_features = self.depth_backbone(pc-pc_mean, training=training)
		point_feature = point_features['features'][0] # (b, Np, N_pf)

		# feature concatenation
		point_feature = tf.reshape(point_feature, [-1,120,160,K.int_shape(point_feature)[2]]) # (b, 120, 160, N_pf)
		fused_feature = tf.concat([point_feature, rgb_feature], axis=3)  # (b, 120, 160, N_pf+N_cf)
		fused_feature = tf.reshape(fused_feature, [-1,120*160,K.int_shape(fused_feature)[3]]) # (b, Np, N_pf+N_cf)

		# head
		grasp_approach_direction_pred = self.grasp_approach_direction_head(fused_feature, training=training) # (b, N, 3)
		grasp_approach_direction_pred = tf.math.l2_normalize(grasp_approach_direction_pred, axis=2) # (b, N, 3)

		grasp_lateral_direction_pred = self.grasp_lateral_direction_head(fused_feature, training=training)
		inner_product = tf.reduce_sum(tf.multiply(grasp_lateral_direction_pred, grasp_approach_direction_pred), axis=2, keepdims=True)
		grasp_lateral_direction_pred = grasp_lateral_direction_pred - inner_product*grasp_approach_direction_pred
		grasp_lateral_direction_pred = tf.math.l2_normalize(grasp_lateral_direction_pred, axis=2)  # (b, N, 3)

		grasp_position_distance_head = self.grasp_position_distance_head(fused_feature, training=training)  # (b, N, 1)
		grasp_position_distance_pred = tf.nn.sigmoid(grasp_position_distance_head)  # (b, N, 1)

		grasp_score_head = self.grasp_score_head(fused_feature, training=training)  # (b, N, 1)
		grasp_score_pred = tf.nn.sigmoid(grasp_score_head)  # (b, N, 1)
		
		grasp_collision_head = self.grasp_collision_head(fused_feature, training=training)  # (b, N, 1)
		grasp_collision_pred = tf.nn.sigmoid(grasp_collision_head)  # (b, N, 1)

		grip_width_head = self.grip_width_head(fused_feature, training=training)  # (b, N, len)
		grip_width_pred = tf.nn.sigmoid(grip_width_head)  # (b, N, len)

		occupancy_head = self.occupancy_head(fused_feature, training=training)  # (b, N, 1)
		occupancy_pred = tf.nn.sigmoid(occupancy_head)  # (b, N, 1)

		return dict(fused_feature=fused_feature,
					grasp_approach_direction_pred=grasp_approach_direction_pred,
					grasp_lateral_direction_pred=grasp_lateral_direction_pred,
					grasp_position_distance_pred=grasp_position_distance_pred,
					grasp_score_head=grasp_score_head,
					grasp_score_pred=grasp_score_pred,
					grasp_collision_head=grasp_collision_head,
					grasp_collision_pred=grasp_collision_pred,
					grip_width_head=grip_width_head,
					grip_width_pred=grip_width_pred,
					occupancy_head=occupancy_head,
					occupancy_pred=occupancy_pred)

	def train_step(self, inputs):
		with tf.GradientTape() as tape:
			preds = self.call(inputs[0], training=True)
			losses = self.loss_func(inputs[1], preds)

		gradients = tape.gradient(losses['total_loss'], self.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

		return losses