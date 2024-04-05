import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing import image
import numpy as np
import os

# Data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    r'dataset\train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    r'dataset\test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Model architecture
model_path = "best_model.h5"

if not os.path.exists(model_path):
    cnn = Sequential()
    
    cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size=2, strides=2))
    
    cnn.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D(pool_size=2, strides=2))
    
    cnn.add(Flatten())
    
    cnn.add(Dense(units=128, activation='relu'))
    cnn.add(Dropout(0.5))
    
    cnn.add(Dense(units=64, activation='relu'))
    cnn.add(Dropout(0.3))
    
    cnn.add(Dense(units=23, activation='softmax'))

    # Optimizer with learning rate scheduler
    optimizer = Adam(lr=0.0005)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)

    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    cnn.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit model with callbacks
    history = cnn.fit(
        x=training_set,
        validation_data=test_set,
        epochs=100,
        callbacks=[checkpoint, early_stop, lr_scheduler]
    )
else:
    cnn = load_model(model_path)

# Evaluate the model on test set and print accuracy
loss, accuracy = cnn.evaluate(test_set)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Prediction function
def make_prediction(image_path, class_indices):
    test_image = image.load_img(image_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image /= 255.0
    test_image = np.expand_dims(test_image, axis=0)
    result = cnn.predict(test_image)

    predicted_class_index = np.argmax(result)
    predicted_class_label = list(class_indices.keys())[list(class_indices.values()).index(predicted_class_index)]

    return result, predicted_class_label

result, predicted_class = make_prediction(
    r'dataset\train\Bullous Disease Photos\benign-familial-chronic-pemphigus-2.jpg', 
    training_set.class_indices
)
print(result, predicted_class)
