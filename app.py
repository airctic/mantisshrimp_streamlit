# Standalone app that includes everything from src.
# This is just to make it simple to run from Github.
# For making your own project. Please follow the repo strucutre
# finally combine all your secret sauce into a simple app file like this.

import streamlit as st
from zipfile import ZipFile 
import zipfile
from PIL import Image
import numpy as np
import random
import torch
import os, urllib, cv2

from mantisshrimp.models.mantis_rcnn import MantisFasterRCNN
from mantisshrimp.visualize.show_data import show_pred
import matplotlib.pyplot as plt

APP_NAME = "app.py"
MODEL_BUCKET_URL = "https://mantisshrimp-models.s3.us-east-2.amazonaws.com/weights-384px-adam2%2B%2B.pth.zip"
SAVE_PATH = "data/demo_model.pth.zip"
DATA_PATH = "data/"
MODEL_PATH = "data//weights-384px-adam2++.pth"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Some sample images over internet that you may like to give. Enter urls of images here.
SAMPLE_IMAGES = ["sample_images//cat0.jpg", "sample_images//cat1.jpg",
"sample_images//dog0.jpg", "sample_images//dog1.jpg", "sample_images//dog2.jpg", 
"sample_images//dog3.jpg", "sample_images//dog4.jpg", "sample_images//dog5.jpg"]

NUM_CLASSES = 38 # Hyperparameters of model
# Note this might be diffrent from len(OBJECTS_TO_DETECT) as you may have an extra background class.
# Optionally the user can simply provide classes which the model was trained on

OBJECTS_TO_DETECT = ['background', 'Abyssinian', 'Bengal', 
'Birman', 'Bombay', 'British_Shorthair', 
'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 
'Russian_Blue', 'Siamese', 'Sphynx', 'american_bulldog',
'american_pit_bull_terrier', 'basset_hound', 'beagle', 'boxer', 
'chihuahua', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 
'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 
'miniature_pinscher', 'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 
'samoyed', 'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier', 
'wheaten_terrier', 'yorkshire_terrier']

# Download a single file and make its content available as a string.
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    # url = 'https://raw.githubusercontent.com/insert_path_to_repo_here' + path
    # response = urllib.request.urlopen(url)
    with open(path, encoding="utf-8", errors="ignore") as f:
        response = f.read()
    return response

@st.cache(show_spinner=False)
def load_image_file(image_path):
    image = Image.open(image_path)
    image = image.convert("RGB")
    image = np.array(image)
    image = image / 255.
    image = image.astype(np.float32)
    return image
    
# Utility to beautifully download a file from its url
def download_file(file_path, save_path):
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(save_path):
        return
    else:
        # Download from the url
        # These are handles to two visual elements to animate.
        weights_warning, progress_bar = None, None
        try:
            weights_warning = st.warning("Downloading %s..." % file_path)
            progress_bar = st.progress(0)
            with open(save_path, "wb") as output_file:
                with urllib.request.urlopen(file_path) as response:
                    length = int(response.info()["Content-Length"])
                    counter = 0.0
                    MEGABYTES = 2.0 ** 20.0
                    while True:
                        data = response.read(8192)
                        if not data:
                            break
                        counter += len(data)
                        output_file.write(data)

                        # We perform animation by overwriting the elements.
                        weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %(file_path, counter / MEGABYTES, length / MEGABYTES))
                        progress_bar.progress(min(counter / length, 1.0))
        
        # Finally, we remove these visual elements by calling .empty().
        finally:
            if weights_warning is not None:
                weights_warning.empty()
            if progress_bar is not None:
                progress_bar.empty()

# This sidebar UI lets the user select parameters for the object detector.
def object_detector_ui():
    st.sidebar.markdown("# Model")
    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    overlap_threshold = st.sidebar.slider("Overlap threshold", 0.0, 1.0, 0.3, 0.01)
    return confidence_threshold, overlap_threshold

def object_selector_ui():
    st.sidebar.markdown("# Objects to detect")
    # The user can pick which type of object to search for.
    object_type = st.sidebar.selectbox("Search for which objects?", OBJECTS_TO_DETECT)
    # The user can select a range for how many of the selected objecgt should be present.
    min_objs, max_objs = st.sidebar.slider("How many %s s (select a range)?" % object_type, 0, 10, [3, 5])
    return object_type, min_objs, max_objs

# This is to be done with mantisshrimp 
# Initialize the mantisshirmp model and return it.
def create_model():
    # model = mantisshirmp.something.something()
    model = MantisFasterRCNN(num_classes=NUM_CLASSES)
    return model

# Loading is constant code no edits needed
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    # Very important we just want to evaluate
    model.eval()
    model.to(device)
    return model

# Forward pass from the model
# This code might need customizations from user side.    
@st.cache(allow_output_mutation=True)
def predict(model, image, confidence_threshold, overlap_threshold):
    # Cumbersome PyTorch code.
    # Very important to set it to no_grad
    # with torch.no_grad():
    #     prediction = model(image_batch)[0] # Maybe 0 is needed verify once.
    #     selected = prediction["scores"] > confidence_threshold
    #     predictions = {k: v[selected] for k, v in prediction.items()}
        
    #     for pred in predictions:
    #         boxes = pred['boxes'].data.cpu().numpy()
    #         labels = pred['labels'].data.cpu().numpy()
    #         scores = pred['scores'].data.cpu().numpy()

    # Since this is a mantiss model we can directly use model.predict
    # Mantisshrimp eases out this processing.
    preds = model.predict([image], detection_threshold=confidence_threshold)
    # print(type(preds))
    # Perform NMS.
    # Once we know what to draw we can use the utility to simply draw the box
    # Just display this image
    # print(preds)
    # print(preds[0])
    labels = preds[0]['labels']
    scores = preds[0]['scores']
    # bboxes = preds[0]['bboxes']

    # Show pred helps us to vizualize the data quickly. It is a helper function in mantisshrimp

    show_pred(image, preds[0], show=False, classes=OBJECTS_TO_DETECT)
    # img = np.fromstring(canvas.to_string_rgb(), dtype='uint8')
    fig = plt.gcf()
    fig.canvas.draw()
    fig_arr = np.array(fig.canvas.renderer.buffer_rgba())
    return fig_arr, labels, scores

if __name__ == "__main__":
    # A simple streamlit demo is what we want.
    # Features of the demo
    # Get the readme text from readme file
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
        st.code(get_file_content_as_string(APP_NAME))

    elif app_mode == "Run the app":
        readme_text.empty()
        flag = 0
        chk_fg = 0
        st.write("# Running the object detection App")
        st.write("- Adjust the thresholds, Upload a file and click predict")
        st.write("- It might take time to download the model.")
        st.write("- To load a sample image just click on the load sample image")

        # Download the model or use from system.
        download_file(MODEL_BUCKET_URL, SAVE_PATH)
        # Extracts the downloaded zip model
        # You can skip these model extract step if you directly have a .pt file in data folder
        if not os.path.exists(MODEL_PATH):
            if(zipfile.is_zipfile(SAVE_PATH)):
                with zipfile.ZipFile(SAVE_PATH, 'r') as zip: 
                    zip.extractall(DATA_PATH) 

        confidence_threshold, overlap_threshold = object_detector_ui()
        object_type, min_objs, max_objs = object_selector_ui()    

        if (st.button("Load a sample Image")):
            # Just load an image from sample_images folder
            random_image = random.choice(SAMPLE_IMAGES)
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
        model = load_model(MODEL_PATH)

        if flag == 1:
            image_out, labels, scores = predict(model, image, confidence_threshold, overlap_threshold)
            if len(labels) == 0:
                st.write("No relevant object detected in the image")
            else:
                st.image(image_out, use_column_width=True)
                st.write("- Image with detection")
                for i in range(len(labels)):
                    if OBJECTS_TO_DETECT[labels[i]] == object_type: 
                        st.write("Successfully Detected object {}".format(object_type))
                        chk_fg = 1
                    st.write("Detected %s, with confidence %0.2f" %(OBJECTS_TO_DETECT[labels[i]], scores[i]))
                
                if(chk_fg == 1):
                    st.write("Detected the required object: {}".format(object_type))

