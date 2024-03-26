import streamlit as st
from PIL import Image
import numpy as np
import joblib
import json

@st.cache_resource
def load_model():
	  return joblib.load('model_augment_quiteokay.pkl')
    
large_model = load_model()




uploaded_image = st.file_uploader('Upload an image of the tool', type=['jpg', 'png'])
left = 125
upper = 440
right = 1475
lower = 800
bearbeitungszeit = 0.0
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
    pred_wear = round((result[0][1]+result[0][2])*100,1)
    return modelprediction, confidence, pred_wear

# Function to make predictions using the small model
def predict_tool_wear_small(image):
    # Your code to predict using the small model
    return "Small model prediction"
    
def max_index(modelpred):
  a = modelpred[0]
  max_ind = max(enumerate(a),key=lambda x: x[1])[0]
  return max_ind

def load_results(filename):
    try:
        with open(filename, 'r') as file:
            results = json.load(file)
    except FileNotFoundError:
        results = {}
    return results

def save_results(results, filename):
    with open(filename, 'w') as file:
        json.dump(results, file)

  
st.title('Bestimmen des Werkzeugverschleißes')

st.sidebar.title('Daten eingeben')
machine_name = st.sidebar.text_input('Maschine')
tool_type = st.sidebar.text_input('Werkzeugtyp')
st.sidebar.title('Prozessparameter')
work_cycle = st.sidebar.text_input('Bearbeitungsdauer (min)')
speed = st.sidebar.number_input('Schnittgeschwindigkeit (m/min)', value=0)
feed = st.sidebar.number_input('Vorschubgeschwindigkeit (mm/min)', value=0)
angle = st.sidebar.number_input('Zustellung (mm)', value=0)
rotation = st.sidebar.number_input('Drehgeschwindigkeit', value=0)

def main():    
	if st.button("Werkzeugzustand bewerten") == True:
    		image = Image.open(uploaded_image)
    		#st.image(image, caption='Uploaded Image', use_column_width=True)
    		cropped_image = image.crop((left, upper, right, lower))
    		resized_image = cropped_image.resize((cropped_image.width // 2, cropped_image.height // 2))
    		bild=[]
    		bild.append(resized_image)
    		image_array = np.asarray(bild)
		modelprediction, confidence, pred_wear = predict_tool_wear_large(image_array)
    		st.write("Werkzeugzustand:")
    		st.text_area("Ergebnis", f"{modelprediction}", height=100)
    		st.write("Wie sicher ist sich das Modell bei dieser Klassifizierung: ", f"{confidence}")
       
    		filename = 'Ergebnis.json'
    		results = load_results(filename)
    		if machine_name in results:
        		results[machine_name].append({
            		'Verschleißzustand': modelprediction,
            		'Bearbeitungsdauer': work_cycle,
            		'Verschleißzustand_quantitativ': pred_wear
        		})
    		else:
        		results[machine_name] = [{
            		'Verschleißzustand': modelprediction,
            		'Bearbeitungsdauer': work_cycle,
            		'Verschleißzustand_quantitativ': pred_wear
		        }]
    		save_results(results, filename)
		st.download_button(
    		label="Download data as JSON",
    		data=filename,
    		file_name='Ergebnis.json',
    		mime='json'
		)
  
#if st.sidebar.button("Verschleißverlauf anzeigen") == True:
    #st.write("Hier Diagramm mit allen Werten aus wertepaare_dict für st.sidebar.text_input('Machine Name')")
    

