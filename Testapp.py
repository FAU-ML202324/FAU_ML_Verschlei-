import streamlit as st
from PIL import Image
import numpy as np
import joblib

filename = 'model_augment_quiteokay.pkl'
#large_model = pickle.load(open(filename, 'rb'))
large_model = joblib.load(filename)




uploaded_image = st.file_uploader('Upload an image of the tool', type=['jpg', 'png'])
left = 125
upper = 440
right = 1475
lower = 800
bearbeitungszeit = 0
wertepaare_dict = {}

# Function to make predictions using the small model
def predict_tool_wear_large(image):
    result = large_model.predict(image)
    if max_index(result)== 0:
      modelprediction = 'Defekt'
    elif max_index(result) == 1:
      modelprediction = 'Mittel'
    else:
      modelprediction = 'Neuwertig'
    confidence = round(result[0][max_index(result)]*100,2)
    pred_wear = round(result[0]+result[0]*100,1)
    return modelprediction, confidence, pred_wear

# Function to make predictions using the small model
def predict_tool_wear_small(image):
    # Your code to predict using the small model
    return "Small model prediction"
    
def max_index(modelpred):
  a = modelpred[0]
  max_ind = max(enumerate(a),key=lambda x: x[1])[0]
  return max_ind

def main():

    st.title('Tool Wear Detection App')

    # Add a radio button or selectbox for model selection
    model_choice = st.sidebar.radio('Choose Model', ('Large Model', 'Small Model'))

    st.sidebar.title('Enter Data')
    machine_name = st.sidebar.text_input('Machine Name')
    work_cycle = st.sidebar.text_input('Bearbeitungsdauer (min)')
    speed = st.sidebar.number_input('Schnittgeschwindigkeit (m/min)', value=0)
    feed = st.sidebar.number_input('Vorschubgeschwindigkeit (mm/min)', value=0)

    if st.button("Werkzeugzustand bewerten") == True:
        image = Image.open(uploaded_image)
        #st.image(image, caption='Uploaded Image', use_column_width=True)
        cropped_image = image.crop((left, upper, right, lower))
        resized_image = cropped_image.resize((cropped_image.width // 2, cropped_image.height // 2))
        bild=[]
        bild.append(resized_image)
        image_array = np.asarray(bild)
        if model_choice == 'Large Model':
            modelprediction, confidence, pred_wear = predict_tool_wear_large(image_array)
            st.write("Werkzeugzustand:")
            st.text_area("Ergebnis", f"{modelprediction}", height=100)
            st.write("Wie sicher ist sich das Modell bei dieser Klassifizierung: ", f"{confidence}")
        elif model_choice == 'Small Model':
            toolwear_prediction = predict_tool_wear_small(image_array)
            prediction = predict_tool_wear_large(image_array)
            st.write("Werkzeugzustand:")
            st.text_area("Ergebnis", f"{toolwear_prediction}", height=100)
        bearbeitungszeit = bearbeitungszeit + work_cycle
        wertepaar = (bearbeitungszeit,pred_wear)
        st.write("Bearbeitungszeit: ", f"{Bearbeitungszeit}"," und Werkzeugzustand: ", f"{pred_wear}")
        if machine_name in wertepaare_dict:
            wertepaare_dict[machine_name].append(wertepaar)
        else:
            wertepaare_dict[machine_name] = [wertepaar]

    if st.button("Verschleißverlauf anzeigen") == True:
        st.write("Hier Diagramm mit allen Werten aus wertepaare_dict für st.sidebar.text_input('Machine Name')")

if __name__ == "__main__":
    main()
