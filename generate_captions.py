# Step 1: Create captions for each frame of a video using BLIP-2 model
import os
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import numpy as np

class Caption_Generator:
    def __init__(self, model, processor): # assuming frame_paths is a list of paths to frames of a video (in order)
        # self.frame_paths = frame_paths
        self.model = model
        self.processor = processor
        # self.device = device

    def generate_captions(self, image_paths, device):
        """
        Generates captions for a batch of images using BLIP-2 (one video).
        
        Args:
            image_paths (list of str): List of paths to image files.
            processor (Blip2Processor): BLIP-2 processor for image preprocessing.
            model (Blip2ForConditionalGeneration): BLIP-2 model for caption generation.
            device (str): Device to run the model on ("cuda" or "cpu").
            max_length (int): Maximum length of the generated captions.
            num_beams (int): Number of beams for beam search.

        Returns:
            list of str: List of generated captions.
        """
        images = [Image.open(path).convert("RGB") for path in image_paths]  # Load all images
        inputs = self.processor(images, return_tensors="pt", padding=True).to(device)  # Batch processing

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs)
            captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return captions # list of captions for each frame of the video