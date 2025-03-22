import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision import transforms
import joblib
from PIL import Image, ImageColor

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

model_path = fr"models/face_shape_classifier.pth"

le = joblib.load("models/label_encoder.pkl")




class MyNormalize(object):
    def __init__(self, mean, std):
        """
        Initializes the MyNormalize transformation.

        Args:
            mean (list or tuple): The mean values for each channel.
            std (list or tuple): The standard deviation values for each channel.
        """
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Normalizes the input tensor using the specified mean and standard deviation.

        If the input tensor has only one channel (grayscale), it is converted to a 3-channel tensor
        by duplicating the single channel. This is useful when working with models expecting a 3-channel input.

        Args:
            tensor (torch.Tensor): The input image tensor to be normalized.

        Returns:
            torch.Tensor: The normalized image tensor.
        """
        # If the tensor has only one channel, duplicate it to create a 3-channel image.
        if tensor.size(0) == 1:
            tensor = torch.cat([tensor, tensor, tensor], 0)

        # Apply normalization using torchvision's functional API.
        tensor = transforms.functional.normalize(tensor, self.mean, self.std)
        return tensor

    



# Set the device to CPU (Change to 'cuda' if using a GPU)
device = torch.device('cpu')

# Load the EfficientNet-B4 model (pretrained=False means it won’t use ImageNet weights)
model = models.efficientnet_b4(pretrained=False)

# Define the number of output classes (for face shape classification)
num_classes = len(le.classes_)

# Modify the classifier layer of EfficientNet to match the number of output classes
# EfficientNet-B4's default classifier is a sequential layer where index [1] is the final fully connected (FC) layer.
model.classifier = nn.Sequential(
    nn.Linear(model.classifier[1].in_features, num_classes)  # Adjust FC layer to output 5 classes
)

# Load the trained model weights from a saved file
# 'map_location=device' ensures compatibility with the current device (CPU or GPU)
model.load_state_dict(torch.load(model_path, map_location=device))

# Set the model to evaluation mode (disables dropout, batch norm updates)
model.eval()



# Define a sequence of transformations to preprocess input images before feeding into the model
transform = transforms.Compose([
    transforms.Resize(256),  # Resize the shorter side of the image to 256 pixels (aspect ratio preserved)
    transforms.CenterCrop(224),  # Crop the center 224x224 pixels (required input size for EfficientNet)
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor and scale pixel values to [0, 1]
    MyNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Apply normalization using precomputed mean/std
])


def detect_face_shape(image_path):
    """
    Detects the face shape using a pre-trained CNN model.

    Parameters:
    - image_path (str): Path to the input image.

    Returns:
    - predicted_label (str): Predicted face shape category.
    """

    # Load the image
    image = Image.open(image_path)

    # Convert to RGB mode if the image is not already in RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Convert the image to a NumPy array
    image_np = np.array(image)

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Detect faces in the image using a pre-trained Haar cascade classifier
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # If at least one face is detected, process the first detected face
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_img = image.crop((x, y, x + w, y + h))  # Crop the detected face region

        # Apply preprocessing transformations to the face image
        input_image = transform(face_img).unsqueeze(0)  # Convert to tensor and add batch dimension

        # Disable gradient calculation for inference
        with torch.no_grad():
            output = model(input_image)  # Get model predictions

        # Get the predicted class index
        predicted_class_idx = torch.argmax(output).item()

        # Convert the numerical prediction to the corresponding face shape label
        predicted_label = le.inverse_transform(np.ravel(predicted_class_idx))[0]

        return predicted_label  # Return the predicted face shape

    else:
        return "No face detected."
    


img_path= fr'FaceShape_Dataset\testing_set\Square\square (61).jpg'
# prediction= detect_face_shape(img_path)
# print(f'Predicted Face Shape: {prediction}')
