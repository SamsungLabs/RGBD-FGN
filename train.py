import tensorflow as tf
from grasp_network import GraspNetwork
from grasp_data import GraspData

BATCH_SIZE = 64
EPOCHS = 10000
USE_PREDEFINED_MODEL = False

def train():
	data_path = 'RGBD-FGN-DATA/train_data'
	logs_path = 'logs'
	if USE_PREDEFINED_MODEL:
		pretrained_logs_path = 'logs_pre'
	
	lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
		initial_learning_rate=0.001,
		decay_steps=200000,
		decay_rate=0.8,
		staircase=True) # decayed_learning_rate = initial_learning_rate * decay_rate ^ (training_step / decay_steps) (if staircase=True)
		
	# Check the gpu list.
	gpu_list = tf.config.experimental.list_physical_devices('GPU')
	for gpu in gpu_list:
		tf.config.experimental.set_memory_growth(gpu, True)
	print('{} Available gpu: {}'.format('='*10, gpu_list))
	num_gpus = len(gpu_list)
	devices = ["device:GPU:%d" % i for i in range(num_gpus)]
	strategy = tf.distribute.MirroredStrategy(devices=devices, 
                                            	cross_device_ops=tf.distribute.NcclAllReduce())

	with strategy.scope():
		model = GraspNetwork()
		if USE_PREDEFINED_MODEL:
			model.load_weights('./'+pretrained_logs_path+'/checkpoint').expect_partial()
		model.compile(
			optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
			metrics=[]
		)

	callbacks = [
		tf.keras.callbacks.TensorBoard(
            './'+logs_path,
            histogram_freq=1
            ),
		tf.keras.callbacks.ModelCheckpoint(
			'./'+logs_path+'/checkpoint',
            save_weights_only=True, 
            verbose = 1,
            ),
		tf.keras.callbacks.CSVLogger(
            './'+logs_path+'/log_train',separator=',', append=False),
	    ]

	data = GraspData(BATCH_SIZE, data_path)
	dataset = data.get_dataset()

	model.fit(
		dataset,
		epochs=EPOCHS,
		callbacks=callbacks,
		verbose=1
	)

if __name__ == '__main__':
	train()