# Step 4: Create a raw anomaly score for each frame of a video, given temporal summaries (Si)
import transformers
import torch
from transformers import pipeline

class Anomaly_Score:
    def __init__(self, pipe):
        self.pipe = pipe # llm model instance

    def augment(self, summarized_caption): # function to prepare llm input
        '''summarized_caption: (str) Temporally Summarized Caption Si'''

        messages = [
            {"role": "system", "content": "If you were a law enforcement agency, how would you rate the scene described on a scale from 0 to 1, with 0 representing a standard scene and 1 denoting a scene with suspicious activities. Return nothing else but ONLY a single python float number out of the given list: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] and nothing else"},
            {"role": "user", "content": summarized_caption},
        ]
        return messages

    def generate(self, summarized_caption): # function to generate and return anomaly score
        messages = self.augment(summarized_caption)
        
        outputs = self.pipe(
            messages,
            max_new_tokens=256,
            pad_token_id=self.pipe.tokenizer.eos_token_id,
        )

        return float(outputs[0]["generated_text"][2]["content"])