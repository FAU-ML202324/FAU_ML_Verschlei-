import streamlit as st
import pickle
import numpy as np

def main():
    st.title("Regressionsübung im ML Seminar, WS23/24")
    st.header("Prognose der Betonfestigkeit")

    # Abschnitt für SelectSlider-Elemente
    st.header("Wählen Sie die Mengen Ihrer Betoninhaltsstoffe aus")

    st.title('Tool Wear Detection App')

    # Add a radio button or selectbox for model selection
    model_choice = st.sidebar.radio('Choose Model', ('Large Model', 'Small Model'))

    st.sidebar.title('Enter Data')
    machine_name = st.sidebar.text_input('Machine Name')
    work_cycle = st.sidebar.text_input('Work Cycle')
    speed = st.sidebar.number_input('Speed', value=0)
    feed = st.sidebar.number_input('Feed', value=0)

    uploaded_image = st.file_uploader('Upload an image of the tool', type=['jpg', 'png'])

    # Variablen und ihre Bereichsgrenzen
    variables = {
        "cement": (100, 500),
        "slag": (0, 200),
        "flyash": (0, 200),
        "water": (100, 300),
        "superplasticizer": (0, 30),
        "coarseaggregate": (800, 1200),
        "fineaggregate": (600, 1000),
        "age": (1, 365)
    }


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
