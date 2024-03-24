import streamlit as st
from PIL import Image
import numpy as np



st.title('Tool Wear Detection App')

# Add a radio button or selectbox for model selection
model_choice = st.sidebar.radio('Choose Model', ('Large Model', 'Small Model'))

st.sidebar.title('Enter Data')
machine_name = st.sidebar.text_input('Machine Name')
work_cycle = st.sidebar.text_input('Work Cycle')
speed = st.sidebar.number_input('Speed', value=0)
feed = st.sidebar.number_input('Feed', value=0)

uploaded_image = st.file_uploader('Upload an image of the tool', type=['jpg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
