import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet import ResNet101
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class RESNET_CUSTOM(tf.keras.Model):
	def __init__(self, model_num):
		super().__init__(self)
		self.model_num = model_num
		self.offset=(0.485, 0.456, 0.406)
		self.offset = tf.constant(self.offset)
		self.offset = tf.expand_dims(self.offset, axis=0)
		self.offset = tf.expand_dims(self.offset, axis=0)
		self.scale=(0.229, 0.224, 0.225)
		self.scale = tf.constant(self.scale)
		self.scale = tf.expand_dims(self.scale, axis=0)
		self.scale = tf.expand_dims(self.scale, axis=0)
		self._build()

	def _build(self):
		input_tensor = Input(shape=(640,640,3), name='input')
		
		if self.model_num == 50:
			resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)
		elif self.model_num == 101:
			resnet = ResNet101(include_top=False, weights='imagenet', input_tensor=input_tensor)
		
		outputs = {}
		outputs['C2'] = resnet.get_layer('conv2_block3_out').output # (B, H/4, W/4, 256)
		outputs['C3'] = resnet.get_layer('conv3_block4_out').output # (B, H/8, W/8, 512)
		if self.model_num == 50:
			outputs['C4'] = resnet.get_layer('conv4_block6_out').output # (B, H/16, W/16, 1024)
		elif self.model_num == 101:
			outputs['C4'] = resnet.get_layer('conv4_block23_out').output # (B, H/16, W/16, 1024)
		outputs['C5'] = resnet.get_layer('conv5_block3_out').output # (B, H/32, W/32, 2048)
		self.model = Model(input_tensor, outputs)

	def call(self, inputs, training=False):
		"""Normalizes the image to zero mean and unit variance."""
		inputs = tf.cast(inputs, tf.uint8)
		inputs = tf.image.convert_image_dtype(inputs, dtype=tf.float32)
		inputs -= self.offset
		inputs /= self.scale

		"""Add zero padding at the bottom"""
		inputs = tf.image.pad_to_bounding_box(inputs,0,0,640,640)

		pred = self.model(inputs, training=training)

		return pred