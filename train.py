from sklearn.datasets import load_breast_cancer
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import numpy as np

#Data Preprocessing 


d=load_breast_cancer()
data=StandardScaler().fit_transform(d.data)
X_train,X_test,Y_train,Y_test=train_test_split(data,d.target)


#Train a neural network to classify data

model=Sequential()
model.add(Dense(50,activation='relu',input_dim=30))
model.add(Dense(50,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Train the model
if not os.path.exists('mymodel.h5'):
		model.fit(X_train,Y_train,epochs=100)
		model.save('mymodel.h5')
else:
	model=load_model('mymodel.h5')
	print(model.evaluate(X_test,Y_test)[1])
