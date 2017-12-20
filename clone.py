import csv
import numpy as np
import cv2

lines =[]
with open("../data-car-behavioral/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
    	lines.append(line)

images = []
measurements = []
for line in lines:
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = '../data-car-behavioral/IMG/' + filename
	# print(current_path)
	image = cv2.imread(current_path)
	# print(image)
	# print(image.shape)
	images.append(image)

	# measurement = float(line[3])
	# measurements.append(measurement)

X_train = np.array(images)
print(X_train.shape)
# y_train = np.array(measurements)

# from keras.models import Sequential
# from keras.layers import Flatten, Dense

# model = Sequential()
# model.add(Flatten(input_shape=(160, 320, 3)))
# model.add(Dense(1))

# model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=7)

# model.save('model.h5')
