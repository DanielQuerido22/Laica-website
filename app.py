import streamlit as st

import requests
import librosa
import json
import pandas as pd
import matplotlib.pyplot as plt

def main_page():
    st.markdown("# LAICA")
    # st.sidebar.markdown("# LAICA")
    st.subheader("Dog Sound Classifier")
    st.text("Text provided by Lazlo")
    # st.text("Upload a audio file and find what kind of sound the dog is making either from:")
    # st.text("1. Spectograms")
    # st.text("2. Sound Features")
    # Create three columns for the images
    col1, col2, col3 = st.columns(3)

    # Display the images in the columns
    with col1:
        st.image('image_1.jpg')
    with col2:
        st.image('image_2.jpg')
    with col3:
        st.image('image_3.jpg')
        # st.image("image_1.jpg", use_column_width=True, width=50)

def page_2():
    st.markdown("# FEATURES")
    st.subheader("Model Trainning using Sound Features")
    st.text("Text")
    # st.sidebar.markdown("# Page 2")

def page_3():
    st.markdown("# SPECTOGRAMS")
    st.subheader("Model Trainning using Sound Spectograms")
    st.text("Text")
    # st.sidebar.markdown("# Page 3")

def page_4():
    st.markdown("# MODEL")
    # st.sidebar.markdown("# Page 4")

    # st.set_option('deprecation.showfileUploaderEncoding', False)
    uploaded_file = st.file_uploader("Choose a wav file")
    def predict_from_spectograms():

        if uploaded_file is not None:

            dog_sound_api_url = 'https://laica-bmz4557psa-ew.a.run.app/predict_from_spectograms/'
            files = {'audio': uploaded_file}
            response = requests.post(dog_sound_api_url, files=files)

            #prediction = json.loads(response.headers.get('X-proba').replace("'", '"'))
            proba_str = response.headers.get('X-proba')
            prediction = dict((a.strip().replace('{','').replace('"','').replace("'",""), float(b.strip().replace('}','').replace('"','').replace("'","")))
                     for a, b in (element.split(':')
                                  for element in proba_str.split(', ')))

            image = response.content
            st.write(prediction)

            st.write(prediction.get('Probability of bark'))
            #pred = prediction['label']

            #prediction = response.json()

            st.header(f'We have just heard a:')


            df = pd.DataFrame(prediction.items()).set_index(0)
            df.plot(kind='bar')
            fig = plt.gcf()
            st.pyplot(fig)

            st.image(image, use_column_width=True)

    def predict_from_features():

        if uploaded_file is not None:

            dog_sound_api_url = 'https://laica-bmz4557psa-ew.a.run.app/predict_from_features/'
            files = {'audio': uploaded_file}
            response = requests.post(dog_sound_api_url, files=files)
            st.write(response.status_code)
            prediction = response.json()

            st.header(f'We have just heard a {prediction}')


    st.button('Predict_from_spectograms', on_click=predict_from_spectograms)
    st.button('Predict_from_features', on_click=predict_from_features)

def page_5():
    st.markdown("# TEAM")
    # st.sidebar.markdown("# Page 5")

page_names_to_funcs = {
    "Laica": main_page,
    "Features": page_2,
    "Spectograms": page_3,
    "Model": page_4,
    "Team": page_5,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()


FONT_SIZE_CSS = f"""
<style>
h1 {{
    font-size: 36px !important;
}}
</style>
"""
st.write(FONT_SIZE_CSS, unsafe_allow_html=True)
