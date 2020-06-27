# Use the mantisshrimp Pytorch model here.
# This file should require minimal edits from user
# If user knows the model he has created, he can simply make use of create_model here


# import mantisshrimp
import torch
import utils
import streamlit as st

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Initialize the mantisshirmp model and return it.
def create_model():
    # model = mantisshirmp.something.something()
    model = None

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
def predict(model, image_batch, confidence_threshold):
    # Very important to set it to no_grad
    with torch.no_grad():
        prediction = model(image_batch)
    
    for pred in prediction:
        boxes = pred['boxes'].data.cpu().numpy()
        labels = pred['labels'].data.cpu().numpy()
        scores = pred['scores'].data.cpu().numpy()
    
    # Perform NMS and confidence thresholding.
    # Once we know what to draw we can use the utility to simply draw the box
    



