import streamlit as st
# import altair as alt
# import pandas as pd
import numpy as np
import config
import os, urllib, cv2
from model import *
from utils import *

# This sidebar UI lets the user select parameters for the YOLO object detector.

# This file should remain constant for the user. He should requires minimal changes here.
# This will create an overall layout of the app. It shoulnd't be something to edit.

def object_detector_ui():
    st.sidebar.markdown("# Model")
    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    overlap_threshold = st.sidebar.slider("Overlap threshold", 0.0, 1.0, 0.3, 0.01)
    return confidence_threshold, overlap_threshold

def object_selector_ui():
    st.sidebar.markdown("# Objects to detect")
    # The user can pick which type of object to search for.
    object_type = st.sidebar.selectbox("Search for which objects?", config.OBJECTS_TO_DETCT)
    # The user can select a range for how many of the selected objecgt should be present.
    min_elts, max_elts = st.sidebar.slider("How many %s s (select a range)?" % object_type, 0, 10, [3, 5])

if __name__ == "__main__":
    # A simple streamlit demo is what we want.
    # Features of the demo
    # Get the readme text from readme file
    readme_text = st.markdown(get_file_content_as_string("README.md"))
    # print(readme_text)
    # Download the model or use from system.
    download_file(config.MODEL_PATH)

    # Load Pytorch model here
    # model = load_model(config.MODEL_PATH)
    
    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["About the App", "Run the app", "Show the source code"])
    if app_mode == "About the App":
        st.sidebar.success('To continue select "Run the app".')
        # Render the Readme of the app here. Do not clear it.

    elif app_mode == "Show the source code":
        readme_text.empty()
        st.code(get_file_content_as_string(config.APP_NAME))
    elif app_mode == "Run the app":
        readme_text.empty()
        st.write("# Running the object detection App")
        st.write("- Adjust the thresholds, Upload a file and click predict")
        # run_the_app()
    
        object_detector_ui()
        object_selector_ui()
        
        st.write("# Upload an Image to get its predictions")

        img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if img_file_buffer is not None:
            image = Image.open(img_file_buffer)
            img_array = np.array(image)
            if image is not None:
                st.image(image, caption=f"You amazing image has shape {img_array.shape[0:2]}", use_column_width=True)
            else:
                print("Invalid input")