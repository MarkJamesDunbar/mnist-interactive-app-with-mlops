# import dependencies
import base64
import torch
import gradio as gr
import cv2
from safetensors.torch import load_file


import model.model_architecture as ma

# app title
title = "Interactive MNIST Digit Recognition"

# Convert the image to Base64
with open("./assets/mnist-classes.png", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

# Create the HTML string with the Base64 image
head = f"""
<center>
  <img src='data:image/png;base64,{encoded_image}' width=400>
  <p>The neural net was trained to classify numbers (from 0 to 9).\nTo test it, write your number in the space provided.</p>
</center>
"""

# GitHub repository link
ref = "This is a demo app for testing CI/CD builds with Deployed Machine Learning. Find my shoddy code [here](https://github.com/MarkJamesDunbar/mnist-interactive-app-with-mlops)."

# image size: 28x28
img_size = 28

# classes name (from 0 to 9)
labels = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
]

# load model
net = ma.Network()
state_dict = load_file("./model/model_weights.safetensors")
net.load_state_dict(state_dict)
net.eval()


# prediction function for sketch recognition
def predict(img):
    # Resize image to match model's input size
    img = cv2.resize(img, (img_size, img_size))

    # Convert to grayscale (if not already)
    if len(img.shape) == 3:  # If RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalise and reshape to match model input
    img = img / 255.0  # Normalise pixel values to [0, 1]
    img_tensor = (
        torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    )  # Shape: (1, 1, 28, 28)

    # Make predictions
    with torch.no_grad():
        outputs = net(img_tensor)  # Raw logits from the model
        probs = torch.nn.functional.softmax(outputs, dim=1)  # Convert to probabilities

    # Map predictions to class labels
    return {labels[i]: float(probs[0, i]) for i in range(len(labels))}


# Top labels
label = gr.Label(num_top_classes=3)

# Launch the app
interface = gr.Interface(
    fn=predict,
    inputs="sketchpad",
    outputs=label,
    title=title,
    description=head,
    article=ref,
).launch()
