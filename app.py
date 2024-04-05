import streamlit as st
from cnn import make_prediction,training_set

# Set the page title and favicon
st.set_page_config(page_title="Image Classification Web App", page_icon="🌟")

# Add title and description
st.title("Image Classification Web App")
st.write("""
         This is a simple web app to demonstrate image classification using a CNN model.
         Upload an image and the model will predict its class.
         """)

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    
    # Make prediction
    result, predicted_class = make_prediction(uploaded_file, training_set.class_indices)
    
    # Display prediction result
    st.write("")
    st.write("Classifying...")
    st.write(f"Prediction Result: {result}")
    st.write(f"Predicted Class: {predicted_class}")
