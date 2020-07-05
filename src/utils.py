import os
import config
import streamlit as st
import urllib
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as T
import torch

# Some utils which users can use.
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Draws an image with boxes overlayed
def draw_image_with_boxes(image, boxes):
    pass

# Download a single file and make its content available as a string.
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    # url = 'https://raw.githubusercontent.com/insert_path_to_repo_here' + path
    # response = urllib.request.urlopen(url)
    with open(path, encoding="utf-8", errors="ignore") as f:
        response = f.read()
    return response

def show_image(image_path):
    # image = load_image_file(config.IMAGE_PATH)
    # We need to support both JPG and PNG file format
    image = load_image_file(image_path)
    if(image_path[-3:] == "jpg"):
        st.image(image, caption="", use_column_width=True, clamp=True, format="JPEG")
    elif(image_path[-3:] == "png"):
        st.image(image, caption="", use_column_width=True, clamp=True, format="PNG")
    elif(image_path[-4:] == "jpeg"):
        st.image(image, caption="", use_column_width=True, clamp=True, format="JPEG")
    else:
        print("Invalid Image")

# This function loads an image from Streamlit public repo on S3. We use st.cache on this
# function as well, so we can reuse the images across runs.
@st.cache(show_spinner=False)
def load_image_url(url):
    with urllib.request.urlopen(url) as response:
        image = np.asarray(bytearray(response.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = image[:, :, [2, 1, 0]] # BGR -> RGB
    
    image_tensor = T.ToTensor(image)
    image_batch = [image_tensor.to(device)]
    return image_batch

@st.cache(show_spinner=False)
def load_image_tensor(image_path, device):
    image_tensor = T.ToTensor()(Image.open(image_path))
    image_batch = [image_tensor.to(device)]
    return image_batch

@st.cache(show_spinner=False)
def load_image_file(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

