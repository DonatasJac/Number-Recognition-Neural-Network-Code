import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.mnist
print(data)
(trainImages, trainLabels), (testImages, testLabels) = data.load_data()
trainImages = trainImages / 255.0
testImages = testImages / 255.0

classNames = ['0','1','2','3','4','5','6','7','8','9']
network = keras.Sequential([
	keras.layers.Flatten(input_shape = (28,28)),
	keras.layers.Dense(256, activation = 'sigmoid'),
	keras.layers.Dense(10, activation = 'softmax') #Softmax used as it ensures the values of the 10 output neurons adds up to 1, which is exactly whats needed as the program
	])											   #should output to what extent it thinks the image is each number, and if the probabilities are assigned correctly, the total 
												   #would be 1. Also, there are more than 2 classes, so an activation which works well with 2 categories of target labels such as the sigmoid
												   #is not appropriate for this problem.

network.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]) #categorical entropy because multiple classes are used, binary is insufficient
network.fit(trainImages, trainLabels, epochs = 10)

prediction = network.predict(testImages)	

correct = 10000
incorrectT = []
incorrectI = []
incorrectA = []
for i in range(10000):
	if classNames[np.argmax(prediction[i])] != classNames[testLabels[i]]:
		correct = correct - 1 
		incorrectT.append(classNames[np.argmax(prediction[i])])
		incorrectI.append(testImages[i])
		incorrectA.append(classNames[testLabels[i]])
print(correct)
counter = 0
for i in incorrectT:
	plt.grid(False)
	plt.imshow(incorrectI[counter], cmap = plt.cm.binary)
	plt.title("Prediction: " + incorrectT[counter])
	plt.xlabel("Actual: " + incorrectA[counter])
	counter += 1
	plt.show()




	#plt.grid(False)
	#plt.imshow(testImages[i], cmap = plt.cm.binary)
	#plt.title("Prediction: " + classNames[np.argmax(prediction[i])])
	#plt.xlabel("Actual: " + classNames[testLabels[i]])
	#plt.show()

