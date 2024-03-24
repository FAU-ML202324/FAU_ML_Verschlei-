import streamlit as st
import pickle
import numpy as np
from opencv-python import cv2

def main():
    st.title("Seminararbeit im ML Seminar, WS23/24")
    
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
        # Convert the uploaded image into a bytes object
        image_bytes = uploaded_image.read()
        # Convert the bytes object to a numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Decode the numpy array into an image using OpenCV
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Convert the image from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    


    values = []
    for var, (min_val, max_val) in variables.items():
        value = st.select_slider(f"{var.capitalize()} (Einheit)", range(min_val, max_val + 1))
        values.append(value)

    if st.button("Vorhersage machen"):
        input_values_scaled = scale_input(values, scaler)
        prediction_scaled = model.predict(input_values_scaled)
        prediction = inverse_scale_output(prediction_scaled, scaler_y)
        st.write("Prognostizierte Festigkeit Ihres Betons in MPa:")
        st.text_area("Ergebnis", f"{prediction}", height=100)

if __name__ == "__main__":
    main()
