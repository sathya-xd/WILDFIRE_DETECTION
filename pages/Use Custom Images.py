import streamlit as st
from PIL import Image
import torch
from going_modular.predictions import predict_single_image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title='Use Custom Images')
st.title("Use Custom Images for Wildfire Classification")

# Guidance on how to use the Demo.
st.header('Getting Started')
st.write('You can upload your own satellite image or use your camera to capture an image and run a prediction on the model to classify the area as wildfire prone or not. Before uploading an image or using the camera, please read the directions below so you can get the best possible predictions from the model. \n 1) Take an image from Bing Maps since, because it is less cluttered with store names, street names, etc. If they are cluttered you can unselect "details" in the option. Click here to go to [Bing maps](https://www.bing.com/maps?cp=47.431688%7E-53.948823&lvl=8.4&style=a) \n 2) Use the satellite terrain while taking a snip of the map or capturing an image. \n 3) Take an image of about an acre roughly, preferably with 100 meters elevation. This will improve the model performance as the dataset also has image of about 1 acre having an elevation of 100 meters.')

# Model prediction header
st.header("Model Prediction")

# Get Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.warning(f'Prediction will run on {device.upper()}')

class ImageCapture(VideoTransformerBase):
    def __init__(self):
        self.prediction = None

    def recv(self, frame):
        img = frame.to_image()
        # Assuming `predict_single_image` can take PIL images directly
        self.prediction = predict_single_image(img)
        return frame  # Return the original frame

# Option to choose input method
input_method = st.radio("Select input method:", ("Upload Image", "Use Camera"))

if input_method == "Upload Image":
    image = st.file_uploader(label='Upload a satellite image', accept_multiple_files=False)
    if image is not None:
        image = Image.open(image).convert('RGB')
        st.image(image)
        with st.spinner("Prediction Running...Please Wait.."):
            predicted_class = predict_single_image(image)
            st.info(f'Predicted Class: {predicted_class}')
elif input_method == "Use Camera":
    webrtc_ctx = webrtc_streamer(key="example", video_processor_factory=ImageCapture)
    if webrtc_ctx.video_processor:
        if webrtc_ctx.state.playing:
            # Check if a prediction has been made
            if webrtc_ctx.video_processor.prediction is not None:
                st.info(f'Predicted Class: {webrtc_ctx.video_processor.prediction}')