# Some utility functions that are used in the app.
import os
import config
import streamlit as st
import urllib
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as T
import torch

__all__ = [
    "get_file_content_as_string",
    "show_image",
    "download_file",
    "load_image_tensor",
    "load_image_tensor",
    "load_image_file",
]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    """
    Download a single file and make its content available as a string.
    """
    # url = 'https://raw.githubusercontent.com/insert_path_to_repo_here' + path
    # response = urllib.request.urlopen(url)
    with open(path, encoding="utf-8", errors="ignore") as f:
        response = f.read()
    return response


def show_image(image_path):
    """ Show an image """
    image = load_image_file(image_path)
    if image_path[-3:] == "jpg":
        st.image(image, caption="", use_column_width=True, clamp=True, format="JPEG")
    elif image_path[-3:] == "png":
        st.image(image, caption="", use_column_width=True, clamp=True, format="PNG")
    elif image_path[-4:] == "jpeg":
        st.image(image, caption="", use_column_width=True, clamp=True, format="JPEG")
    else:
        print("Invalid Image")


@st.cache(show_spinner=False)
def load_image_url(url):
    """ Loads an image given the url """
    with urllib.request.urlopen(url) as response:
        image = np.asarray(bytearray(response.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = image[:, :, [2, 1, 0]]  # BGR -> RGB

    image_tensor = T.ToTensor(image)
    image_batch = [image_tensor.to(device)]
    return image_batch


@st.cache(show_spinner=False)
def load_image_tensor(image_path, device):
    """
    Loads an image into pytorch tensor. 
    """
    image_tensor = T.ToTensor()(Image.open(image_path))
    image_batch = [image_tensor.to(device)]
    return image_batch


@st.cache(show_spinner=False)
def load_image_file(image_path):
    """
    Loads an Image file
    """
    image = Image.open(image_path)
    image = image.convert("RGB")
    image = np.array(image)
    image = image / 255.0
    image = image.astype(np.float32)
    return image


def download_file(file_path, save_path):
    """
    Utility to beautifully download a file from its url
    """
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
                        weights_warning.warning(
                            "Downloading %s... (%6.2f/%6.2f MB)"
                            % (file_path, counter / MEGABYTES, length / MEGABYTES)
                        )
                        progress_bar.progress(min(counter / length, 1.0))

        # Finally, we remove these visual elements by calling .empty().
        finally:
            if weights_warning is not None:
                weights_warning.empty()
            if progress_bar is not None:
                progress_bar.empty()
