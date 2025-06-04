# Step 3: Generate temporal summary for each video centered around each frame
import transformers
import torch
from transformers import pipeline

# Generates summaru for each temporal window of ONE video
class Window_Summary:
    def __init__(self, pipe):
        self.pipe = pipe # llm model instance

    def augment(self, video_temporal_captions): # function to prepare llm input
        captions = str([string for string in video_temporal_captions if string!='*'])
        
        messages = [
            {"role": "system", "content": "Please Summarize what happened in few sentences, based on the following temporal description of a scene. Do not include any unnecessary details or descriptions."},
            {"role": "user", "content": captions},
        ]
        return messages

    def generate(self, video_temporal_captions): # function to generate and return string of summary
        messages = self.augment(video_temporal_captions)
        
        outputs = self.pipe(
            messages,
            max_new_tokens=256,
            pad_token_id=self.pipe.tokenizer.eos_token_id,
        )

        return str(outputs[0]["generated_text"][2]["content"])