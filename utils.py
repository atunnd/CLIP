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

from pkg_resources import packaging

device = "cuda" if torch.cuda.is_available() else "cpu"
# Define the model ID
model_ID = "openai/clip-vit-base-patch32"
# Save the model to device
model = CLIPModel.from_pretrained(model_ID).to(device)
# Get the processor
processor = CLIPProcessor.from_pretrained(model_ID)
# Get the tokenizer
tokenizer = CLIPTokenizer.from_pretrained(model_ID)


def get_text_embedding(prompt, clip_model=model) -> list:
    """
    Returns the embedding for a given text prompt.

    Args:
        prompt (str): the text prompt
        clip_model (fiftyone.zoo.models.CLIPModel): the CLIP model

    Returns:
        the embedding for the given prompt, as a list
    """
    tokenizer = clip_model._tokenizer

    # standard start-of-text token
    sot_token = tokenizer.encoder["<|startoftext|>"]

    # standard end-of-text token
    eot_token = tokenizer.encoder["<|endoftext|>"]

    # encode prompt with CLIP tokenizer
    prompt_tokens = tokenizer.encode(prompt)
    # add start-of-text and end-of-text tokens
    all_tokens = [[sot_token] + prompt_tokens + [eot_token]]

    # create feature vector
    text_features = torch.zeros(
        len(all_tokens),
        clip_model.config.context_length,
        dtype=dtype,
        device=device,
    )

    # insert tokens into feature vector
    text_features[0, : len(all_tokens[0])] = torch.tensor(all_tokens)

    # encode text
    embedding = clip_model._model.encode_text(text_features).to(device)

    # convert to list for Pinecone
    return embedding.tolist()
