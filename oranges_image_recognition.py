import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model


"""Create Image Data Generators"""
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'C:/Users/raven/OneDrive/Escritorio/DataSets/oranges_training_ds', #Your Path
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary' #Use 'categorical' for multiple classes
)

test_generator = test_datagen.flow_from_directory(
    'C:/Users/raven/OneDrive/Escritorio/DataSets/oranges_testing_ds', #Your Path
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Calculate steps per epoch and validation steps
steps_per_epoch = (train_generator.samples // train_generator.batch_size) + 1
validation_steps = (test_generator.samples // test_generator.batch_size) + 1




"""##########################################################################################################################################"""
"""MODEL: I turn off this code block because the trained model has been saved in a variable, so, it is not neccesary to train again every time the app runs, but if we need to retrain the model we just have to uncoment the upper code block and modifie it."""
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
    epochs=5,
    validation_data=test_generator,
    validation_steps=validation_steps
)

"""Saving trained model"""
model.save('C:/Users/raven/OneDrive/Escritorio/AI_Trained_Models/Oranges_Image_Recognition.h5') #Your Path


"""Plot training history"""
plt.plot(epochs_history.history['accuracy'])
plt.title('Lossing progress during model training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend('Training Loss')
plt.show()
"""########################################################################################################################################"""


"""Load entire model"""
model = loaded_model = load_model('C:/Users/raven/OneDrive/Escritorio/AI_Trained_Models/Oranges_Image_Recognition.h5') #Your Path

"""Evaluate and Test"""
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc}')

def image_predict(image_path:str):
    """
    DOCSTRING:
        This method obtain a value (True or False) comprobing if the image that we give them is an orange or not using the trained model 'predict()' method.
    
    Attributes:
        image_path:str
    
    Return:
        result:str
    """
    image_path = image_path
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 #Normalizing image
    
    prediction = model.predict(img_array)
    
    #Threshold for binary clasification
    threshold = 0.5
    predicted_class = 1 if prediction[0] >= threshold else 0
    
    if predicted_class == 1:
        prediction_result = 'It´s an orange!'
    else:
        prediction_result = 'It´s not an orange.'
                 
    result = f'Accuracy:{prediction}\nResult:{prediction_result}\nImage:{image_path}'
    return result
    
    
    
#Doing predictions
print(image_predict('C:/Users/raven/OneDrive/Escritorio/DataSets/prediction_images/orange1.jpg')) #Your Path

print(image_predict('C:/Users/raven/OneDrive/Escritorio/DataSets/prediction_images/aircraft.jpg')) #Your Path

print(image_predict('C:/Users/raven/OneDrive/Escritorio/DataSets/prediction_images/apple.jpg')) #Your Path

print(image_predict('C:/Users/raven/OneDrive/Escritorio/DataSets/prediction_images/orange2.jpg')) #Your Path

print(image_predict('C:/Users/raven/OneDrive/Escritorio/DataSets/prediction_images/pumpkin.jpg')) #Your Path

print(image_predict('C:/Users/raven/OneDrive/Escritorio/DataSets/prediction_images/apple2.jpg')) #Your Path

print(image_predict('C:/Users/raven/OneDrive/Escritorio/DataSets/prediction_images/grappes.jpg')) #Your Path

print(image_predict('C:/Users/raven/OneDrive/Escritorio/DataSets/prediction_images/kiwi.jpg')) #Your Path

print(image_predict('C:/Users/raven/OneDrive/Escritorio/DataSets/prediction_images/guava.jpg')) #Your Path