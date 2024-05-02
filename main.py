#%%
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_datasets

# Gloval variables
IMG_HEIGHT       = 28
IMG_WIDTH        = 28
NUM_CLASSES      = 24

images_per_class = 80
fixed_size       = tuple((IMG_HEIGHT, IMG_WIDTH))
train_path       = "dataset/train"
test_path        = "dataset/test"
h5_data          = 'output/data.h5'
h5_labels        = 'output/labels.h5'

test_size = 0.10
seed      = 42

initial_learning_rate = 0.001
batch_size = 32
n_epochs = 100


#%% Load datasets
# Lấy tên từng folder trong tập train (tương ứng với tên các labels)
train_labels = os.listdir(train_path)
train_labels.sort()
print(train_labels)

images, labels = load_datasets(train_path, train_labels, fixed_size)

global_features = np.array(images)
global_labels   = np.array(labels)
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                        np.array(global_labels),
                                                                                        test_size=test_size,
                                                                                        random_state=seed)

#%% Create Model and Training data
# Load and preprocess the dataset
# Code to load and preprocess the flower dataset goes here
# Define the CNN architecture
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(trainDataGlobal)

# Load and preprocess the dataset
# Code to load and preprocess the flower dataset goes here
# Define the CNN architecture
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(300, activation='relu', kernel_initializer=keras.initializers.he_normal()),
    layers.Dense(200, activation='relu', kernel_initializer=keras.initializers.he_normal()),
    layers.Dense(100, activation='relu', kernel_initializer=keras.initializers.he_normal()),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])
model.summary()
early_stopper = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20)

   
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=20,
    decay_rate=0.98,
    staircase=True)


# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(datagen.flow(trainDataGlobal, trainLabelsGlobal, batch_size=batch_size),
                    epochs=n_epochs, 
                    validation_data=(testDataGlobal, testLabelsGlobal), 
                    callbacks=[early_stopper])
model.save('models/pretrained_model.keras')
history = history.history
print(history["val_accuracy"][-1])

pd.DataFrame({'Training acc':history['accuracy'],'Validation acc':history['val_accuracy']}).plot(figsize=(8, 5))
plt.grid(True)
plt.xlim(0, n_epochs)
plt.ylim(0, 1)
plt.xlabel('epoch')
plt.xticks(np.arange(0, n_epochs + 1,5))
plt.show()

# %%
