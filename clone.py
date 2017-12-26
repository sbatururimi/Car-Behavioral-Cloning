import csv
import numpy as np
import cv2

lines =[]
# with open("../data-car-behavioral/driving_log.csv") as csvfile:
#     reader = csv.reader(csvfile)
#     for line in reader:
#     	lines.append(line)

# images = []
# measurements = []
# for line in lines:
# 	source_path = line[0]
# 	filename = source_path.split('/')[-1]
# 	current_path = '../data-car-behavioral/IMG/' + filename
# 	image = cv2.imread(current_path)
# 	images.append(image)
# 	# fleeping images & steering measurements
# 	measurement = float(line[3])
# 	measurements.append(measurement)

# X_train = np.array(images)
# y_train = np.array(measurements)

# images = []
# measurements = []
# for line in lines:
# 	source_path = line[0]
# 	filename = source_path.split('/')[-1]
# 	current_path = '../data-car-behavioral/IMG/' + filename
# 	image = cv2.imread(current_path)
# 	images.append(image)
# 	# fleeping images & steering measurements
# 	measurement = float(line[3])
# 	measurements.append(measurement)

# # data augmentation
# augmented_images, augmented_measurements = [], []
# for image, measurement in zip(images, measurements):
# 	augmented_images.append(image)
# 	augmented_measurements.append(measurement)

# 	augmented_images.append(cv2.flip(image, 1))
# 	augmented_measurements.append(measurement * -1.0)

# X_train = np.array(augmented_images)
# y_train = np.array(augmented_measurements)

car_images = []
steering_angles = []
with open("../data-car-behavioral/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
    	# lines.append(line)
    	steering_center = float(line[3])

    	# create adjusted steering measurements for the side camera images
    	correction = 0.2
    	steering_left = steering_center + correction
    	steering_right = steering_center - correction

    	# read in images from center, left and right cameras
    	source_path_center = line[0]
    	filename = source_path_center.split('/')[-1]
    	current_path_center = '../data-car-behavioral/IMG/' + filename
    	image_center = cv2.imread(current_path_center)

    	source_path_left = line[1]
    	filename = source_path_left.split('/')[-1]
    	current_path_left = '../data-car-behavioral/IMG/' + filename
    	image_left = cv2.imread(current_path_left)

    	source_path_right = line[2]
    	filename = source_path_right.split('/')[-1]
    	current_path_right = '../data-car-behavioral/IMG/' + filename
    	image_right = cv2.imread(current_path_right)

    	# add images and angles to data set
    	car_images.extend([image_center, image_left, image_right])
    	steering_angles.extend([steering_center, steering_left, steering_right])



# data augmentation
augmented_images, augmented_measurements = [], []
for image, measurement in zip(car_images, steering_angles):
	augmented_images.append(image)
	augmented_measurements.append(measurement)

	augmented_images.append(cv2.flip(image, 1))
	augmented_measurements.append(measurement * -1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# LeNet
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Conv2D(6, (5, 5), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(6, (5, 5), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save('model.h5')
