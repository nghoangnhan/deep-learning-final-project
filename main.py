#%%
import cv2, os
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow import keras

# Gloval variables
IMG_HEIGHT       = 28
IMG_WIDTH        = 28
NUM_CLASSES      = 17

images_per_class = 80
fixed_size       = tuple((IMG_HEIGHT, IMG_WIDTH))
train_path       = "dataset/train"
h5_data          = 'output/data.h5'
h5_labels        = 'output/labels.h5'

test_size = 0.10
seed      = 42


# Lấy tên từng folder trong tập train (tương ứng với tên các labels)
train_labels = os.listdir(train_path)

# Sắp xếp các labels theo thứ tự abc
train_labels.sort()
print(train_labels)

# List để lưu trữ dữ liệu của từng dữ liệu ảnh trong từng folder
images = []
# List lưu trữ tên Labels
labels = []

# Load datasets from folder
def load_datasets():
    i = 0
    for training_name in train_labels:
        # join the training data path and each species training folder
        dir = os.path.join(train_path, training_name)

        # get the current training label
        current_label = i

        # Lấy từng ảnh trong folder
        for x in range(1, images_per_class + 1):
            # get the image file name
            file = dir + "/" + "(" + str(x) + ")" + ".jpg"

            # read the image and resize it to a fixed-size
            image = cv2.imread(file)    
            image = cv2.resize(image, fixed_size)

            # Lưu trữ lại
            labels.append(current_label)
            images.append(image)

        # Thông báo đã xử lí xong 1 folder
        print("[STATUS] processed folder: {}".format(current_label))
        i += 1

    print("Size of an image: {}", images[0].shape)

load_datasets()
global_features = np.array(images)
global_labels   = np.array(labels)

(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                        np.array(global_labels),
                                                                                        test_size=test_size,
                                                                                        random_state=seed)


# Load and preprocess the dataset
# Code to load and preprocess the flower dataset goes here
# Define the CNN architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

early_stopper = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

n_epochs = 50
# Train the model
history = model.fit(trainDataGlobal, trainLabelsGlobal, epochs=n_epochs, 
                    validation_data=(testDataGlobal, testLabelsGlobal), callbacks=[early_stopper])
history = history.history
print(history["val_accuracy"][-1])


#%%
# Evaluate the model
test_path = "dataset/test"
file1 = "dataset/test/1.jpg"
file2 = "dataset/test/6.jpg"

# read the image and resize it to a fixed-size
image1 = cv2.imread(file1)    
image1 = cv2.resize(image1, fixed_size)

# read the image and resize it to a fixed-size
image2 = cv2.imread(file2)    
image2 = cv2.resize(image2, fixed_size)

test_images = []
test_labels = []

test_images.append(image1);
test_images.append(image2);
test_labels.append(1);
test_labels.append(6);

test_images = np.array(test_images)
test_labels = np.array(test_labels)

X_new = test_images[0:2]
y_proba = model.predict(X_new).round(2) # return probabilities (output of output neurons)
print('Prediction proba: \n', y_proba)
y_pred = np.argmax(model.predict(X_new), axis=1) # return class with highest proba
print('Predicted class: ', y_pred)
print('True labels: ', test_labels[0:2])
# %%
