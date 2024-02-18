# import torch
import torchvision.transforms as transforms

import onnx
import onnxruntime as ort

import numpy as np
from PIL import Image


def show_deployed(model_path: str, image_path: str):
    # Load the ONNX model
    session = ort.InferenceSession(model_path)
    # get the size which the image should be resized
    input_shape = session.get_inputs()[0].shape
    img_size = [input_shape[-2], input_shape[-1]]
    transform = transforms.Compose([transforms.Resize(img_size),
                                    # transforms.CenterCrop(img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    img = Image.open(image_path)
    transformed_img = transform(img)
    input_img = transformed_img.unsqueeze(0)

    # Convert the PyTorch tensor to a numpy array and then to a float32
    input_data = np.array(input_img.numpy(), dtype=np.float32)

    # Run inference
    ort_inputs = {session.get_inputs()[0].name: input_data}
    output = session.run(None, ort_inputs)

    print(output)
