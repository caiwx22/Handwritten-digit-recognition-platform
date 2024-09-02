import argparse
import os
import cv2
import torch
import numpy as np

from recognitionModel.models.lenet import LeNet


def pre_process(img, device):
    # Resize image to 28x28
    img = cv2.resize(img, (28, 28))
    # Convert image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Invert colors (make background black and text white)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    # Normalize image
    img = img / 255
    # Convert to PyTorch tensor
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img = img.unsqueeze(0)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img


def inference(model, img):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img = pre_process(img, device)
    model.to(device)
    model.eval()
    preds = model(img)
    label = preds[0].argmax()
    return label


def run_inference(num_classes, image_path, model_path):
    # Load pretrained model
    model = LeNet(num_classes)
    if not os.path.exists(model_path):
        raise ValueError(f'model_path is invalid: {model_path}')
    load_dict = torch.load(model_path)
    model.load_state_dict(load_dict['state_dict'])

    if not os.path.exists(image_path):
        raise ValueError(f'image_path is invalid: {image_path}')
    img = cv2.imread(image_path)
    label = inference(model, img)

    return "pred: " + str(label.item())
