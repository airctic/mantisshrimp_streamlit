import streamlit as st
from zipfile import ZipFile 
import zipfile
from PIL import Image
import numpy as np
import config
import random
import torch
import os, urllib, cv2
from model import *
from utils import *

# This file should remain constant for the user. He should requires minimal changes here.
# This will create an overall layout of the app. It shoulnd't be something to edit.


# This sidebar UI lets the user select parameters for the object detector.
def object_detector_ui():
    st.sidebar.markdown("# Model")
    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    overlap_threshold = st.sidebar.slider("Overlap threshold", 0.0, 1.0, 0.3, 0.01)
    return confidence_threshold, overlap_threshold

def object_selector_ui():
    st.sidebar.markdown("# Objects to detect")
    # The user can pick which type of object to search for.
    object_type = st.sidebar.selectbox("Search for which objects?", config.OBJECTS_TO_DETECT)
    # The user can select a range for how many of the selected objecgt should be present.
    min_objs, max_objs = st.sidebar.slider("How many %s s (select a range)?" % object_type, 0, 10, [3, 5])

    return object_type, min_objs, max_objs

if __name__ == "__main__":
    # A simple streamlit demo is what we want.
    # Features of the demo
    # Get the readme text from readme file
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    readme_text = st.markdown(get_file_content_as_string("src/Info.md"))    
    # Once we have the dependencies, add a selector for the app mode on the sidebar.

    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["About the App", "Run the app", "Show the source code"])

    if app_mode == "About the App":
        # Render the Readme of the app here. Do not clear it.
        st.sidebar.success('To continue select "Run the app".')

    elif app_mode == "Show the source code":
        readme_text.empty()
        st.code(get_file_content_as_string(config.APP_NAME))

    elif app_mode == "Run the app":
        readme_text.empty()
        flag = 0
        chk_fg = 0
        st.write("# Running the object detection App")
        st.write("- Adjust the thresholds, Upload a file and click predict")
        st.write("- It might take time to download the model.")
        st.write("- To load a sample image just click on the load sample image")

        # Download the model or use from system.
        download_file(config.MODEL_BUCKET_URL, config.SAVE_PATH)
        # Extracts the downloaded zip model
        # You can skip these model extract step if you directly have a .pt file in data folder
        if not os.path.exists(config.MODEL_PATH):
            if(zipfile.is_zipfile(config.SAVE_PATH)):
                with zipfile.ZipFile(config.SAVE_PATH, 'r') as zip: 
                    zip.extractall(config.DATA_PATH) 

        confidence_threshold, overlap_threshold = object_detector_ui()
        object_type, min_objs, max_objs = object_selector_ui()    

        if (st.button("Load a sample Image")):
            # Just load an image from sample_images folder
            random_image = random.choice(config.SAMPLE_IMAGES)
            st.image(random_image)
            image = load_image_file(random_image)
            flag = 1

        st.write("# Upload an Image to get its predictions")

        img_file_buffer = st.file_uploader("", type=["png", "jpg", "jpeg"])
        if img_file_buffer is not None:
            image = load_image_file(img_file_buffer)
            if image is not None:    
                st.image(image, caption=f"You amazing image has shape {image.shape[0:2]}", use_column_width=True)
                flag = 1
            else:
                print("Invalid input")

        # Load Pytorch model here. You can come here automatically if you have downloaded pt file itself.
        model = load_model(config.MODEL_PATH)

        if flag == 1:
            image_out, labels, scores = predict(model, image, confidence_threshold, overlap_threshold)
            if len(labels) == 0:
                st.write("No relevant object detected in the image")
            else:
                st.image(image_out, use_column_width=True)
                st.write("- Image with detection")
                for i in range(len(labels)):
                    if config.OBJECTS_TO_DETECT[labels[i]] == object_type: 
                        st.write("Successfully Detected object {}".format(object_type))
                        chk_fg = 1
                    st.write("Detected %s, with confidence %0.2f" %(config.OBJECTS_TO_DETECT[labels[i]], scores[i]))
                
                if(chk_fg == 1):
                    st.write("Detected the required object: {}".format(object_type))

