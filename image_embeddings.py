import os
import faiss
import torch
import skimage
import requests
import pinecone
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import IPython.display
import matplotlib.pyplot as plt
from datasets import load_dataset
from collections import OrderedDict
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
# Define the model ID
model_ID = "openai/clip-vit-base-patch32"
# Save the model to device
model = CLIPModel.from_pretrained(model_ID).to(device)
# Get the processor
processor = CLIPProcessor.from_pretrained(model_ID)
# Get the tokenizer
tokenizer = CLIPTokenizer.from_pretrained(model_ID)

def get_single_image_embedding(my_image,processor, model, device):
    image = processor(
          text = None,
          images = my_image,
          return_tensors="pt"
         )["pixel_values"].to(device)
    embedding = model.get_image_features(image)
  # convert the embeddings to numpy array
    return embedding.cpu().detach().numpy()

def get_image(image_URL):
       response = requests.get(image_URL)
       image = Image.open(BytesIO(response.content)).convert("RGB")
       return image

def get_image_embedding(image_url):
    image = get_image(image_url)
    image_embedding = get_single_image_embedding(image, processor, model, device)

    return image_embedding.tolist()

    

