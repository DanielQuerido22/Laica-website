import streamlit as st

import requests
import pandas as pd
import matplotlib.pyplot as plt

st.sidebar.image(
    "resources/laica_logo_v1.png",
    width=300,
)

text_style = """
    font-size: 18px;
    color: gray;
    text-align: justify;
    font-family: 'Roboto'
"""

text_style_2 = """
    font-size: 12px;
    color: gray;
    text-align: centered;
    font-family: 'Roboto'
"""
text_style_3 = """
    font-size: 15px;
    color: black;
    text-align: centered;
    font-family: 'Roboto'
"""

title_style = """
    font-size: 30px;
    color: black;
    font-weight:normal;
    font-family: 'Roboto'
"""

subtitle_style = """
    font-size: 20px;
    color: black;
    font-family: 'Roboto'
"""

subtitle_style_2 = """
    font-size: 20px;
    color: gray;
    font-family: 'Roboto'
"""

def main_page():

    title_1 = "LAICA"
    subtitle_1 = "Learning-based Artificial Intelligence on Canine Acoustics"
    text_1 = """LAICA is a collection of machine learning models that can classify four different
types of dog vocalizations: barks, growls, whines and pants. We developed LAICA as
a demo project during our Data Science training at Le Wagon Portugal. But we hope
that this can serve as a pilot project for researchers who need to rapidly and
reliably classify various dog sounds."""
    text_2 = """We received tens of gigabytes of raw audio files with corresponding labeling from
the Ethology Departement of ELTE University (Budapest). These were long recordings
of various experiments, containing different sounds. However, researchers have already
manually labeled dog vocalizations using Praat. Based on their work we managed to
splice out roughly 20,000 sound snippets. After pre-selection, cleaning,
pre-processing and augmentation, these served as the training sets of the model."""

    st.markdown(f"<p style='{title_style}'>{title_1}</p>", unsafe_allow_html=True)
    st.write(f"<p style='{subtitle_style}'>{subtitle_1}</p>", unsafe_allow_html=True)
    st.write(f"<p style='{text_style}'>{text_1}</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    # Display the images in the columns
    with col1:
        st.image('resources/image_01.jpg')
    with col2:
        st.image('resources/image_02.jpg')
    with col3:
        st.image('resources/image_03.jpg')
        # st.image("image_1.jpg", use_column_width=True, width=50)

    st.write(f"<p style='{text_style}'>{text_2}</p>", unsafe_allow_html=True)
    st.audio("resources/bark_00005.wav", format='audio/wav')

def page_2():
    title_2 = "FEATURES"
    subtitle_2 = "Model Trainning using Audio Features"
    text_2 = """This model was trained using the some of the sound features extracted from the
audio files. These features were extracted using the Librosa library. From these
features we therefore created a dataset that we trained, using the XGBoost model
from machine learning library Scikit-learn.
"""

    st.markdown(f"<p style='{title_style}'>{title_2}</p>", unsafe_allow_html=True)
    st.write(f"<p style='{subtitle_style}'>{subtitle_2}</p>", unsafe_allow_html=True)
    st.write(f"<p style='{text_style}'>{text_2}</p>", unsafe_allow_html=True)
    st.image("resources/image_6.jpeg")

def page_3():
    title_3 = "SPECTOGRAMS"
    subtitle_3 = "Model Trainning using sound Spectograms"
    text_3 = """In this approach we used the sound spectograms as features. First we extracted
the spectogram information from the audio files using the Librosa library. Then
ploted the spectograms using the library matplotlib and saved them as png images.
We then used these images as to train the model using a neural network from deep
learning library Keras.
"""

    st.markdown(f"<p style='{title_style}'>{title_3}</p>", unsafe_allow_html=True)
    st.write(f"<p style='{subtitle_style}'>{subtitle_3}</p>", unsafe_allow_html=True)
    st.write(f"<p style='{text_style}'>{text_3}</p>", unsafe_allow_html=True)
    st.image("resources/image_5.png")

def page_4():

    title_4 = "MODEL"
    st.markdown(f"<p style='{title_style}'>{title_4}</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a wav file")


    def predict_from_spectograms():

            dog_sound_api_url = 'https://laica-bmz4557psa-ew.a.run.app/predict_from_spectograms/'
            files = {'audio': uploaded_file}
            response = requests.post(dog_sound_api_url, files=files)

            proba_str = response.headers.get('X-proba')
            prediction = dict((a.strip().replace('{','').replace('"','').replace("'",""), float(b.strip().replace('}','').replace('"','').replace("'","")))
                     for a, b in (element.split(':')
                                  for element in proba_str.split(', ')))

            image = response.content

            max_key = max(prediction, key=lambda k: prediction[k])
            max_value = prediction[max_key]

            text_6 = 'This is the spectogram of the audio file:'
            st.write(f"<p style='{subtitle_style}'>{text_6}</p>", unsafe_allow_html=True)
            st.image(image, use_column_width=True)
            result_1 = (f'The {max_key} is {max_value*100:.2f}%')
            st.markdown(f"<p style='{title_style}'>{result_1}</p>", unsafe_allow_html=True)

    def predict_from_features():

            dog_sound_api_url = 'https://laica-bmz4557psa-ew.a.run.app/predict_from_features/'
            files = {'audio': uploaded_file}
            response = requests.post(dog_sound_api_url, files=files)
            st.write(response.status_code)
            prediction = response.json()
            max_key = max(prediction, key=lambda k: prediction[k])
            max_value = prediction[max_key]
            st.header(f'We are {max_value*100:.2f}% sure that we have just heard a {max_key.upper()}')

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        col4, col5 = st.columns(2)
        # Display the images in the columns
        with col4:
            if st.button('Predict from Audio Image'):
                predict_from_spectograms()
        with col5:
            if st.button('Predict from Audio Features'):
                predict_from_features()

def page_5():
    title_5 = "TEAM"
    text_daniel = "Daniel Matos Querido"
    text_laszlo = "Laszlo Robert Zsiros"
    text_mariana = "Mariana Sanin Riaño"

    st.markdown(f"<p style='{title_style}'>{title_5}</p>", unsafe_allow_html=True)

    col6, col7, col8 = st.columns(3, gap='small')
    # Display the images in the columns
    with col6:
        st.image('resources/me.png')
        st.write(f"<p style='{text_style_3}'>{text_daniel}</p>", text_align='center', unsafe_allow_html=True)
        st.write(f"<p style='{text_style_2}'>daniel.mquerido@gmail.com</p>", text_align='center', unsafe_allow_html=True)
        st.write(f"<p style='{text_style_2}'>https://www.linkedin.com/in/daniel-querido/</p>", text_align='center', unsafe_allow_html=True)
        st.write(f"<p style='{text_style_2}'>https://www.instagram.com/danielmquerido/</p>", text_align='center', unsafe_allow_html=True)

    with col7:
        st.image('resources/laszlo.png')
        st.write(f"<p style='{text_style_3}'>{text_laszlo}</p>", text_align='center', unsafe_allow_html=True)
        st.write(f"<p style='{text_style_2}'>robilaci@gmail.com</p>", text_align='center', unsafe_allow_html=True)
        st.write(f"<p style='{text_style_2}'>https://www.linkedin.com/in/laszlo-robert-zsiros-5a440250/</p>", text_align='center', unsafe_allow_html=True)
        st.write(f"<p style='{text_style_2}'>https://www.instagram.com/rblc81/</p>", text_align='center', unsafe_allow_html=True)

    with col8:
        st.image('resources/mariana.png')
        st.write(f"<p style='{text_style_3}'>{text_mariana}</p>", unsafe_allow_html=True)
        st.write(f"<p style='{text_style_2}'>mariana.sanin99@gmail.com</p>", text_align='center', unsafe_allow_html=True)
        st.write(f"<p style='{text_style_2}'>https://www.linkedin.com/in/marianasanin/</p>", text_align='center', unsafe_allow_html=True)
        st.write(f"<p style='{text_style_2}'>https://www.instagram.com/marianasanin/</p>", text_align='center', unsafe_allow_html=True)

    col6.text_align = 'center'
    col7.text_align = 'center'
    col8.text_align = 'center'


page_names_to_funcs = {
    "Laica": main_page,
    "Features": page_2,
    "Spectograms": page_3,
    "Model": page_4,
    "Team": page_5,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
