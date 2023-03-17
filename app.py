import streamlit as st

import requests
import pandas as pd
import matplotlib.pyplot as plt

text_style = """
    font-size: 18px;
    color: gray;
    text-align: justify;
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
    # subtitle_2 = "What is LAICA"
    text = """LAICA is a collection of machine learning models that can classify four different
types of dog vocalizations: barks, growls, whines and pants. We developed LAICA as
a demo project during our Data Science training at Le Wagon Portugal. But we hope
that this can serve as a pilot project for researchers who need to rapidly and
reliably classify various dog sounds."""

    st.markdown(f"<p style='{title_style}'>{title_1}</p>", unsafe_allow_html=True)
    st.write(f"<p style='{subtitle_style}'>{subtitle_1}</p>", unsafe_allow_html=True)
    # st.write(f"<p style='{subtitle_style_2}'>{subtitle_2}</p>", unsafe_allow_html=True)
    # st.sidebar.markdown("# LAICA")
    # st.header("Dog Sound Classifier")
    st.write(f"<p style='{text_style}'>{text}</p>", unsafe_allow_html=True)
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

            dog_sound_api_url = 'https://laica-bmz4557psa-ew.a.run.app/predict_from_spectograms/'
            files = {'audio': uploaded_file}
            response = requests.post(dog_sound_api_url, files=files)

            #prediction = json.loads(response.headers.get('X-proba').replace("'", '"'))
            proba_str = response.headers.get('X-proba')
            prediction = dict((a.strip().replace('{','').replace('"','').replace("'",""), float(b.strip().replace('}','').replace('"','').replace("'","")))
                     for a, b in (element.split(':')
                                  for element in proba_str.split(', ')))

            image = response.content
            #st.write(prediction)

            st.write(prediction.get('Probability of bark'))
            #pred = prediction['label']

            #prediction = response.json()

            df = pd.DataFrame(prediction.items()).set_index(0)
            df.plot(kind='bar')
            fig = plt.gcf()
            st.pyplot(fig)

            max_key = max(prediction, key=lambda k: prediction[k])
            max_value = prediction[max_key]

            st.header(max_key)
            st.header(f'the {max_key} is {max_value*100:.2f}%')

            st.image(image, use_column_width=True)

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
        col4, col5 = st.columns(2)
        # Display the images in the columns
        with col4:
            st.button('Predict_from_spectograms', on_click=predict_from_spectograms)
        with col5:
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
