from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np
#from keras.layers import SGD
import numpy
import sys
from sklearn.model_selection import train_test_split  

def normalization(X):
	#normalizacao
	for i in range(X.shape[1]):
		X[...,i] = (X[...,i] - np.min(X[...,i])) / (np.max(X[...,i]) - np.min(X[...,i]))
	
	return X
def main():

	X = np.loadtxt("cancer.data", delimiter=",") # pega o dataset
	label= open("label.data", 'r')

	normalizado = normalization(X)


	lab = np.zeros(569).reshape((569))
	c = 0
	for l in label:
		if(l == "M\n"):
			lab[c] = 1
		elif(l == "B\n"):
           
			lab[c] = 0
		c = c + 1
	lab = lab.astype(int)

	normalizado = np.array(normalizado)
	lab = np.array(lab)

	dTreino, dTeste, lTreino, lTeste = train_test_split(normalizado, lab, test_size = 0.30)



	model = Sequential()
	model.add(Dense(100, input_dim=30, activation='sigmoid'))
	model.add(Dropout(0.2))
	model.add(Dense(33, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(11, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	model.fit(dTreino, lTreino, epochs=200, batch_size=100)


	scores = model.evaluate(dTeste, lTeste)




	print("\b%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	
main()
