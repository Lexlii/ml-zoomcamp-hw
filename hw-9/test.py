from torchvision import transforms
from io import BytesIO
from urllib import request
from PIL import Image
import onnxruntime as ort
import numpy as np
import onnx


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

preprocess = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ) # ImageNet normalization
])

url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"

img = download_image(url)
img = prepare_image(img, target_size=(200, 200))

# Apply the preprocessing
input_tensor = preprocess(img)

# Convert to numpy array (C x H x W)
input_array = input_tensor.numpy()

input_array = np.expand_dims(input_array, axis=0)
onnx_model_path = "hair_classifier_v1.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

inputs = session.get_inputs()
outputs = session.get_outputs()

input_name = inputs[0].name
output_name = outputs[0].name

result = session.run([output_name], {input_name: input_array})
predictions = result[0][0].tolist()
print("Predictions:", predictions)
