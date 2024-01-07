import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


"""Create Image Data Generators"""
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'C:/Users/raven/OneDrive/Escritorio/DataSets/oranges_training_ds',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary' #Use 'categorical' for multiple classes
)

test_generator = test_datagen.flow_from_directory(
    'C:/Users/raven/OneDrive/Escritorio/DataSets/oranges_testing_ds',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Calculate steps per epoch and validation steps
steps_per_epoch = (train_generator.samples // train_generator.batch_size) + 1
validation_steps = (test_generator.samples // test_generator.batch_size) + 1

"""Build the Model"""
model = Sequential([
    layers.Conv2D(8, (3, 3), activation='relu', input_shape = (150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid') #Use 'softmax' for multiple classes   
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], run_eagerly=True)


"""Train the Model"""
epochs_history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=20,
    validation_data=test_generator,
    validation_steps=validation_steps
)

"""Plot training history"""
plt.plot(epochs_history.history['loss'])
plt.title('Lossing progress during model training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend('Training Loss')
plt.show()


"""Evaluate and Test"""
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc}')