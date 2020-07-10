# Use the mantisshrimp Pytorch model here.
# This file should require minimal edits from user
# If user knows the model he has created, he can simply make use of create_model here

from mantisshrimp.models.mantis_rcnn import MantisFasterRCNN
from mantisshrimp.visualize.show_data import show_pred
import torch
import numpy as np
import config
import utils
import streamlit as st
import matplotlib.pyplot as plt

__all__ = ["create_model", "load_model", "predict"]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# This is to be done with mantisshrimp 
# Initialize the mantisshirmp model and return it.
def create_model():
    # model = mantisshirmp.something.something()
    model = MantisFasterRCNN(num_classes=config.NUM_CLASSES)
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
    # print(preds)
    # Show pred helps us to vizualize the data quickly. It is a helper function in mantisshrimp
    # print(labels)
    show_pred(image, preds[0], show=False, classes=config.OBJECTS_TO_DETECT)
    # img = np.fromstring(canvas.to_string_rgb(), dtype='uint8')
    fig = plt.gcf()
    fig.canvas.draw()
    fig_arr = np.array(fig.canvas.renderer.buffer_rgba())
    return fig_arr, labels, scores




