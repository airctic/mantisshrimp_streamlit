# Use the mantisshrimp Pytorch model here.
# Simply create the model and get its predictions.

from mantisshrimp.models.mantis_rcnn import MantisFasterRCNN
from mantisshrimp.visualize.show_data import show_pred
import torch
import numpy as np
import config
import utils
import streamlit as st
import matplotlib.pyplot as plt

__all__ = ["create_model", "load_model", "predict"]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def create_model():
    """ Instatiate your Mantis Model Here """
    model = MantisFasterRCNN(num_classes=config.NUM_CLASSES)
    return model


# We cache the loading function to make is very fast on reload.
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    """ Create the model and load state dict here """
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to eval mode
    model.to(device)
    return model


# Forward pass from the model
# You can customize this as per your needs
# We cache it for faster inference
@st.cache(allow_output_mutation=True)
def predict(model, image, confidence_threshold, overlap_threshold):
    """
    Forward pass through the model and get its predictions. 
    """
    # Cumbersome PyTorch code.
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
    labels = preds[0]["labels"]
    scores = preds[0]["scores"]
    show_pred(image, preds[0], show=False, classes=config.OBJECTS_TO_DETECT)
    fig = plt.gcf()
    fig.canvas.draw()
    fig_arr = np.array(fig.canvas.renderer.buffer_rgba())
    return fig_arr, labels, scores
