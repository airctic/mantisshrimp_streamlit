# Use the mantisshrimp Pytorch model here.
# This file should require minimal edits from user
# If user knows the model he has created, he can simply make use of create_model here


from mantisshrimp.models.mantis_rcnn import MantisFasterRCNN
import torch
import config
import utils
import streamlit as st

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
def predict(model, image_batch, confidence_threshold, overlap_threshold):
    # Very important to set it to no_grad
    # with torch.no_grad():
    #     prediction = model(image_batch)[0] # Maybe 0 is needed verify once.
    #     selected = prediction["scores"] > confidence_threshold
    #     predictions = {k: v[selected] for k, v in prediction.items()}
        
    #     for pred in predictions:
    #         boxes = pred['boxes'].data.cpu().numpy()
    #         labels = pred['labels'].data.cpu().numpy()
    #         scores = pred['scores'].data.cpu().numpy()

    # Since this is a mantics model we can directly use model.predict
    preds = model.predict(image_batch, detection_threshold=confidence_threshold)
    # Perform NMS.
    # Once we know what to draw we can use the utility to simply draw the box
    # Just display this image
    return preds




