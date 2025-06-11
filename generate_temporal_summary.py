# Step 3: Generate temporal summary for each video centered around each frame
import transformers
import torch
from transformers import pipeline

class Window_Summary:
    def __init__(self, pipe):
        self.pipe = pipe

    def augment(self, video_temporal_captions):
        captions = str([c for c in video_temporal_captions if c != '*'])
        return [
            {"role":"system", "content": (
                "Please Summarize what happened in few sentences, "
                "based on the following temporal description of a scene. "
                "Do not include any unnecessary details or descriptions."
            )},
            {"role":"user", "content": captions},
        ]

    def generate(self, video_temporal_captions):
        # single-window wrapper stays unchanged
        msgs = self.augment(video_temporal_captions)
        outs = self.pipe(msgs, max_new_tokens=256, pad_token_id=self.pipe.tokenizer.eos_token_id)
        return str(outs[0]["generated_text"][2]["content"])

    def batch_generate(self, windows: list[list[str]]) -> list[str]:
        # Build one list of message-lists
        batch_msgs = [self.augment(win) for win in windows]
        # Run pipeline once on the entire batch
        outs = self.pipe(batch_msgs, max_new_tokens=256, pad_token_id=self.pipe.tokenizer.eos_token_id)
        summaries = []
        for out in outs:
            entry = out[0] if isinstance(out, list) else out
            # Extract assistant content
            msg_list = entry["generated_text"]
            # messages are list[{"role", "content"}]
            assistant = next((m["content"] for m in msg_list if m["role"]=="assistant"), "")
            summaries.append(assistant)
        return summaries
