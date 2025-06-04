import numpy as np
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind import data
import os

class Clean_Captions:
    def __init__(self, model):
        self.model = model

    def clean_captions(self, image_paths, captions, device):
        """
        Cleans captions using ImageBind.
        
        Args:
            image_paths (list of str): List of paths to image files.
            model (ImageBind): ImageBind model for caption cleaning.
            captions (list of str): List of captions to clean.
            device (str): Device to run the model on ("cuda" or "cpu").

        Returns:
            list of str: List of cleaned captions.
        """
        # Preparing Input
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(captions, device),
            ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
        }
        
        # Getting Embeddings
        with torch.no_grad():
            embeddings = self.model(inputs)

        # Cleaned Caption indices
        cleaned_caption_idx = np.argmax(torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1).detach().cpu(), axis=1)

        # Replacing original captions with cleaned
        cleaned_video_caption = [captions[idx] for idx in cleaned_caption_idx]
        
        return cleaned_video_caption