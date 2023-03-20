import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dropout, BatchNormalization  

class MLPHead(tf.keras.layers.Layer):
	def __init__(self, filters_length, output_length):
		super(MLPHead, self).__init__()
		self.output_length = output_length
		self.filters_length = filters_length
		self._build()

	def _build(self):
		self.conv1 = Conv1D(filters=self.filters_length, kernel_size=1)
		self.bn = BatchNormalization()
		self.drop = Dropout(rate=0.3)
		self.conv2 = Conv1D(filters=self.output_length, kernel_size=1)

	def call(self, inputs, training=False):
		x = self.conv1(inputs)
		x = self.bn(x, training=training)
		x = tf.nn.relu(x)
		x = self.drop(x, training=training)
		x = self.conv2(x)
		return x    
