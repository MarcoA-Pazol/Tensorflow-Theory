import tensorflow as tf
from keras import models, layers

#Define the model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dense(1, activation='sigmoid')
])

#Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Print model summary
print(model.summary())

"""Loading and preprocessing data"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

#Load Iris Dataset
iris = datasets.load_iris()
x, y = iris.data, iris.target

#Preprocess the data
scaler = StandardScaler()
x = scaler.fit_transform(x)

#One-hot encode the labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y.reshape(-1, 1))

#Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

"""Training a model"""
model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))

"""Evaluate and predict"""
loss, accuracy = model.evaluatte(x_test, y_test)
print(f'Test accuracy: {accuracy}')

#Make predictions
predictions = model.predict(x_test)
