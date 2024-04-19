### 1. Imports and class names setup ###
import gradio as gr
import os
import torch
from torchvision import transforms

from models import get_mobilenet_v2_model, get_resnet_18_model, get_vgg_16_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup class names
class_names = ["car","dragon","hourse","pegasus","ship","t-rex","tree"]

### 2. Model and transforms preparation ###

# Create EffNetB2 model
img_transforms = transforms.Compose(
    [
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
    ]
)

model_name_to_fn = {
    "mobilenet_v2": get_mobilenet_v2_model,
    "resnet_18": get_resnet_18_model,
    "vgg_16": get_vgg_16_model,
}
model_name_to_path = {
    "mobilenet_v2": "mobilenet_v2.pth",
    "resnet_18": "resnet_18.pth",
    "vgg_16": "vgg_16.pt",
}

### 3. Predict function ###


# Create predict function
def predict(img, model_name: str,) -> Tuple[Dict, float]:
    """
    Desc: Transforms and performs a prediction on img and returns prediction and time taken.
    Args:
        model_name (str): Name of the model to use for prediction.
        img (PIL.Image): Image to perform prediction on.
    Returns:
        Tuple[Dict, float]: Tuple containing a dictionary of prediction labels and probabilities and the time taken to perform the prediction.
    """
    # Start the timer
    start_time = timer()

    # Get the model function based on the model name
    model_fn = model_name_to_fn[model_name]
    model_path = model_name_to_path[model_name]

    # Create the model and load its weights
    model = model_fn().to(device)
    model.load_state_dict(
        torch.load(f"./models/{model_name}.pth", map_location=torch.device(device=device))
    )

    # Put model into evaluation mode and turn on inference mode
    model.eval()
    with torch.inference_mode():
        # Transform the target image and add a batch dimension
        img = img_transforms(img).unsqueeze(0).to(device)

        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(model(img), dim=1)

    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {
        class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))
    }

    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)

    # Return the prediction dictionary and prediction time
    return pred_labels_and_probs, pred_time


### 4. Gradio app ###

# Create title, description and article strings
title = "SketchRec Mini ✍🏻"
description = "An Mutimodel Sketch Recognition App 🎨"
article = ""

# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
model_selection_dropdown = gr.components.Dropdown(
    choices=list(model_name_to_fn.keys()), label="Select a model",
    value="mobilenet_v2"
)

demo = gr.Interface(
    fn=predict,  # mapping function from input to output
    inputs=[gr.Image(type="pil"),model_selection_dropdown],  # what are the inputs?
    outputs=[
        gr.Label(num_top_classes=7, label="Predictions"),  # what are the outputs?
        gr.Number(label="Prediction time (s)"),
    ],  # our fn has two outputs, therefore we have two outputs
    # Create examples list from "examples/" directory
    examples=example_list,
    title=title,
    description=description,
    article=article,
)

# Launch the demo!
demo.launch(
    debug=True,
)
