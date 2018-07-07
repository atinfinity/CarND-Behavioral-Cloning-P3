import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EPOCHS = 10
DATA_DIR = 'data/'

def nvidia_model():
	model = Sequential()

	# Normalize
	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(66, 200, 3)))

	# Convolution
	model.add(Convolution2D(24, 5, 5, subsample=(2,2), border_mode='valid', activation='relu'))
	model.add(Convolution2D(36, 5, 5, subsample=(2,2), border_mode='valid', activation='relu'))
	model.add(Convolution2D(48, 5, 5, subsample=(2,2), border_mode='valid', activation='relu'))
	model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
	model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
	model.add(Dropout(0.5))

	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))
	return model

def preprocessing(image):
	image = cv2.resize(image, (200, 66), interpolation = cv2.INTER_AREA)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
	return image

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1:
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			correction = 0.2
			for sample in batch_samples:
				name_center = DATA_DIR + 'IMG/' +sample[0].split('/')[-1]
				name_left   = DATA_DIR + 'IMG/' +sample[1].split('/')[-1]
				name_right  = DATA_DIR + 'IMG/' +sample[2].split('/')[-1]
				center_image = cv2.imread(name_center)
				left_image = cv2.imread(name_left)
				right_image = cv2.imread(name_right)
				center_image = preprocessing(center_image)
				left_image   = preprocessing(left_image)
				right_image  = preprocessing(right_image)
				images.append(center_image)
				images.append(left_image)
				images.append(right_image)

				center_angle = float(sample[3])
				angles.append(center_angle)
				angles.append(center_angle + correction)
				angles.append(center_angle - correction) 

			# data augumentation
			augmented_images, augmented_angles = [], []
			for image, angle in zip(images, angles):
				augmented_images.append(image)
				augmented_angles.append(angle)
				augmented_images.append(cv2.flip(image, 1))
				augmented_angles.append(angle*-1.0)

			# trim image to only see section with road
			X_train = np.array(augmented_images)
			y_train = np.array(augmented_angles)
			yield sklearn.utils.shuffle(X_train, y_train)

def plot_history(history):
	plt.plot(range(1, EPOCHS+1), history.history['loss'], marker="o")
	plt.plot(range(1, EPOCHS+1), history.history['val_loss'], marker="o")
	plt.title('model mean squared error loss')
	plt.gca().xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
	plt.ylabel('mean squared error loss')
	plt.xlabel('epoch')
	plt.ylim(ymin=0)
	plt.xlim([1, EPOCHS])
	plt.legend(['training set', 'validation set'], loc='upper right')
	plt.savefig('figure.png')

if __name__ == '__main__':
	samples = []
	with open(DATA_DIR + 'driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for sample in reader:
			samples.append(sample)

	train_samples, validation_samples = train_test_split(samples, test_size=0.2)
	train_generator      = generator(train_samples)
	validation_generator = generator(validation_samples)

	model = nvidia_model()
	model.compile(loss='mse', optimizer='adam')
	history = model.fit_generator(train_generator, 
				samples_per_epoch=len(train_samples)*6, 
				validation_data=validation_generator, 
				nb_val_samples=len(validation_samples)*6, 
				nb_epoch=EPOCHS)
	plot_history(history)
	model.save('model.h5')
