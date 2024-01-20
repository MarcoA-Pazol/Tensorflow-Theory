import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np


#Use your own dataset instead or use dataset provided by me in GitHub
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'C:/Users/raven/OneDrive/Escritorio/DataSets/pets_dataset/train', #Your Path
    target_size=(150, 150),
    batch_size=6,
    class_mode='binary', #Use 'categorical' for multiple classes
    color_mode='grayscale',
    shuffle=False
)

validation_generator = validation_datagen.flow_from_directory(
    'C:/Users/raven/OneDrive/Escritorio/DataSets/pets_dataset/validation', #Your Path
    target_size=(150, 150),
    batch_size=6,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=False
)

#Calculating steps per epoch and validation steps
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size 

"""CREATING MODEL: Use 'Sigmoid' activation function instead softmax. Sigmoid ever returns data between 0-1. Realize a training to determine if the data returned is close to 0 it is a cat, but instead it is close to 1, it is a dog."""
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(150, 150, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    
    tf.keras.layers.Flatten(), #Ensure that the output shape is correct
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

print(model.summary())


"""COMPILING: Use crossentropy for binaries because we only have 2 options(dog or cat)"""
model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'],
                  run_eagerly=True)


"""TRAINING: Tensorboard variable it is delivery to 'callbacks' array. In this case it save data in the indicate directory for each epoch, this is for Tensorboard to read the data to make the plots"""
from keras.callbacks import TensorBoard

model_tensorboard = TensorBoard(log_dir='logs/model')
model.fit(train_generator,
            steps_per_epoch = steps_per_epoch,
            validation_data = validation_generator,
            validation_steps= validation_steps,
            epochs=15,
            callbacks=[model_tensorboard]
)

# #Realize Data incrementation with various transformations. 
# datagen = ImageDataGenerator(
#     rotation_range=30, #Rotate the image 30 degres to show it to the model for better understanding of the images looking towards different angles
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=15,
#     zoom_range=[0.7, 1.4], #Increment and decrement zoom from 0.7 to 1.4 range
#     horizontal_flip=True, #Turn image 180 degrees horizontaly
#     vertical_flip=True #Turn image 180 degrees verticaly
# )
# datagen.fit()

"""PREDICTION METHOD"""
def image_predict(image_path:str):
    """
    DOCSTRING:
        This method obtain a value(True or False) depending if the image is about a cat based of the training model.
        
    Attributes:
        image_path:str
        
    Return:
        result:str
    """
    try:
        image_path = image_path
        img = image.load_img(image_path, target_size=(150, 150), color_mode='grayscale')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0 #Normalizing Image
        
        prediction = model.predict(img_array)   
    except Exception as e:
        print('Exception:', e)
    
    try:
        #Threshold for binary classification
        threshold = 0.5
        predicted_class = 1 if prediction[0] >= threshold else 0

        if predicted_class == 1:
            prediction_result = 'It is a Dog'
        else:
            prediction_result = 'It is a Cat'
        result = f'Accuracy:{prediction}\nClass:{prediction_result}\nImage:{image_path}'
        return result
    except Exception as e:
        print('Exception:', e)

"""DOING PREDICTIONS"""
print(image_predict('C:/Users/raven/OneDrive/Escritorio/DataSets/prediction_images/dog1.jpg'))
print(image_predict('C:/Users/raven/OneDrive/Escritorio/DataSets/prediction_images/dog2.jpg'))
print(image_predict('C:/Users/raven/OneDrive/Escritorio/DataSets/prediction_images/cat1.jpg'))
print(image_predict('C:/Users/raven/OneDrive/Escritorio/DataSets/prediction_images/cat2.jpg'))
print(image_predict('C:/Users/raven/OneDrive/Escritorio/DataSets/prediction_images/cat3.jpg'))
print(image_predict('C:/Users/raven/OneDrive/Escritorio/DataSets/prediction_images/dog3.jpg'))
print(image_predict('C:/Users/raven/OneDrive/Escritorio/DataSets/prediction_images/dog4.jpg'))
print(image_predict('C:/Users/raven/OneDrive/Escritorio/DataSets/prediction_images/dog5.jpg'))
print(image_predict('C:/Users/raven/OneDrive/Escritorio/DataSets/prediction_images/dog6.jpg'))
print(image_predict('C:/Users/raven/OneDrive/Escritorio/DataSets/prediction_images/cat4.jpg'))
print(image_predict('C:/Users/raven/OneDrive/Escritorio/DataSets/prediction_images/cat5.jpg'))
print(image_predict('C:/Users/raven/OneDrive/Escritorio/DataSets/prediction_images/cat6.jpg'))
print(image_predict('C:/Users/raven/OneDrive/Escritorio/DataSets/prediction_images/cat7.jpg'))

