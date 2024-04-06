import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing import image
import numpy as np
import os
import pickle
import streamlit as st

# Data Augmentation and Normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load dataset
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

# Model Architecture
cnn = Sequential()

cnn.add(Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size=2, strides=2))

cnn.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size=2, strides=2))

cnn.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size=2, strides=2))

cnn.add(Flatten())

cnn.add(Dense(units=512, activation='relu'))
cnn.add(Dropout(0.5))

cnn.add(Dense(units=256, activation='relu'))
cnn.add(Dropout(0.3))

cnn.add(Dense(units=23, activation='softmax'))

# Optimizer with Learning Rate Scheduler
optimizer = Adam(lr=0.001)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)

cnn.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Model Checkpoint and Early Stopping
model_path = "best_model.h5"
checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1)

# Model Training with early stopping based on validation accuracy
history = cnn.fit(
    x=training_set,
    validation_data=test_set,
    epochs=100,  # Increased epochs
    callbacks=[checkpoint, lr_scheduler, 
               EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)]  # Early stopping based on validation accuracy
)

# Load class indices from pickle file
with open("class_indices.pkl", "rb") as pkl_file:
    class_indices = pickle.load(pkl_file)

# Load model architecture from JSON file
with open("model_architecture.json", "r") as json_file:
    model_json = json_file.read()

cnn = model_from_json(model_json)
cnn.load_weights("model_weights.h5")

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

# Streamlit web app
st.title("Improved Image Classification Web App")

# Navigation Bar
st.markdown("""
<style>
    .navbar {
        display: flex;
        justify-content: space-between;
        padding: 1rem;
        background-color: #f4f4f4;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="navbar"><a href="#about">About</a></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image_path = "temp_image.jpg"
    with open(image_path, "wb") as buffer:
        buffer.write(uploaded_file.getvalue())

    result, predicted_class = make_prediction(image_path, class_indices)

    st.image(uploaded_file, caption=f"Uploaded Image", use_column_width=True)
    st.write("")
    st.write(f"Prediction Result: {predicted_class}")
    st.write(f"Confidence: {result[0][np.argmax(result)]*100:.2f}%")

    os.remove(image_path)

# Footer
st.markdown("""
<style>
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #f4f4f4;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="footer"><a href="#privacy">Privacy</a> | <a href="#control">Control</a> | <a href="#about">About</a></div>', unsafe_allow_html=True)
