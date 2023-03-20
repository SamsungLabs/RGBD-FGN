import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Add, UpSampling2D, BatchNormalization, Dropout
from RGBbackbone.resnet import RESNET_CUSTOM

class FPN_W_BACKBONE(tf.keras.Model):
	def __init__(self, feature_num, resnet_num=50):
		super().__init__(self)
		self.feature_num = feature_num
		self.resnet_num = resnet_num
		self._build()

	def _build(self):
		self.backbone = RESNET_CUSTOM(model_num=self.resnet_num)

		self.m5_conv = Conv2D(self.feature_num, (1, 1))
		self.m4_conv = Conv2D(self.feature_num, (1, 1))
		self.m3_conv = Conv2D(self.feature_num, (1, 1))
		self.m2_conv = Conv2D(self.feature_num, (1, 1))

		self.m5_bn = BatchNormalization()
		self.m4_bn = BatchNormalization()
		self.m3_bn = BatchNormalization()
		self.m2_bn = BatchNormalization()

		self.m5_dp = Dropout(rate=0.3)
		self.m4_dp = Dropout(rate=0.3)
		self.m3_dp = Dropout(rate=0.3)
		self.m2_dp = Dropout(rate=0.3)

		self.m4_us = UpSampling2D(size=(2, 2))
		self.m3_us = UpSampling2D(size=(2, 2))
		self.m2_us = UpSampling2D(size=(2, 2))

		self.p2_conv = Conv2D(self.feature_num, (3, 3), padding="same")

	def call(self, inputs, training=False):
		backbone_output = self.backbone(inputs, training=training)

		M5 = self.m5_conv(backbone_output['C5']) # (b, H/32, W/32, C)
		M5 = self.m5_bn(M5, training=training)
		M5 = tf.nn.relu(M5)
		M5 = self.m5_dp(M5, training=training)

		M4 = Add()([self.m4_us(M5), self.m4_conv(backbone_output['C4'])]) # (b, H/16, W/16, C)
		M4 = self.m4_bn(M4, training=training)
		M4 = tf.nn.relu(M4)
		M4 = self.m4_dp(M4, training=training)

		M3 = Add()([self.m3_us(M4), self.m3_conv(backbone_output['C3'])]) # (b, H/8, W/8, C)
		M3 = self.m3_bn(M3, training=training)
		M3 = tf.nn.relu(M3)
		M3 = self.m3_dp(M3, training=training)

		M2 = Add()([self.m2_us(M3), self.m2_conv(backbone_output['C2'])]) # (b, H/4, W/4, C)
		M2 = self.m2_bn(M2, training=training)
		M2 = tf.nn.relu(M2)
		M2 = self.m2_dp(M2, training=training)

		P2 = self.p2_conv(M2) # (b, H/4, W/4, C)
		P2 = P2[:,:120,:,:]
		
		pred = P2

		return pred
    